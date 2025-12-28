import torch
import math
from .consts import *

class VectorGolfEnv:
    def __init__(self, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = torch.device(device)

        # State Tensors
        self.hands = torch.zeros((num_envs, NUM_PLAYERS, 9), dtype=torch.long, device=self.device)
        self.visible = torch.zeros((num_envs, NUM_PLAYERS, 9), dtype=torch.bool, device=self.device)
        self.deck_cards = torch.zeros((num_envs, TOTAL_CARDS), dtype=torch.long, device=self.device)
        self.draw_ptr = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        
        self.top_discard = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.graveyard_counts = torch.zeros((num_envs, 26), dtype=torch.float32, device=self.device)
        
        self.stages = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.long, device=self.device)
        self.scores = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.float32, device=self.device) # Unused in single round logic but kept for compat
        self.dealer_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.current_player_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        
        self.finisher_idx = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.final_turns_taken = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.bool, device=self.device)
        self.dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

        # Lookup tables
        self.point_map = torch.tensor([POINT_VALUES[i % 13] for i in range(TOTAL_CARDS)], device=self.device)
        self.face_map = torch.tensor([i % 13 for i in range(TOTAL_CARDS)], device=self.device)
        
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        num_reset = len(env_ids)
        
        # 1. Shuffle
        canonical = torch.arange(TOTAL_CARDS, device=self.device).repeat(num_reset, 1)
        noise = torch.rand(num_reset, TOTAL_CARDS, device=self.device)
        perm = noise.argsort(dim=1)
        self.deck_cards[env_ids] = torch.gather(canonical, 1, perm)
        
        # 2. Reset Pointers
        self.draw_ptr[env_ids] = 0
        self.graveyard_counts[env_ids] = 0
        
        # 3. Deal
        flat_hands = self.deck_cards[env_ids, :45]
        self.hands[env_ids] = flat_hands.view(num_reset, NUM_PLAYERS, 9)
        self.draw_ptr[env_ids] = 45
        
        # 4. Discard
        self.top_discard[env_ids] = self.deck_cards[env_ids, 45]
        self.draw_ptr[env_ids] += 1
        
        # 5. Reset Game State
        self.visible[env_ids] = False
        self.scores[env_ids] = 0 # Reset scores every round (Single Round Mode)
        self.stages[env_ids] = STAGE_ARRANGE
        self.finisher_idx[env_ids] = -1
        self.final_turns_taken[env_ids] = False
        self.dones[env_ids] = False
        
        self.dealer_idx[env_ids] = torch.randint(0, NUM_PLAYERS, (num_reset,), device=self.device)
        self.current_player_idx[env_ids] = (self.dealer_idx[env_ids] + 1) % NUM_PLAYERS

        return self.get_obs(env_ids)

    def step(self, actions):
        curr_p = self.current_player_idx
        active_stages = self.stages.gather(1, curr_p.unsqueeze(1)).squeeze(1)
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # --- STAGE LOGIC (1-3) ---
        mask_arrange = (active_stages == STAGE_ARRANGE)
        if mask_arrange.any():
            self.stages[mask_arrange, curr_p[mask_arrange]] = STAGE_FLIP_1

        mask_flip1 = (active_stages == STAGE_FLIP_1)
        if mask_flip1.any():
            ids = mask_flip1.nonzero().squeeze(1)
            acts = actions[ids]
            self.visible[ids, curr_p[ids], acts] = True
            self.stages[ids, curr_p[ids]] = STAGE_FLIP_2

        mask_flip2 = (active_stages == STAGE_FLIP_2)
        if mask_flip2.any():
            ids = mask_flip2.nonzero().squeeze(1)
            acts = actions[ids]
            self.visible[ids, curr_p[ids], acts] = True

        # --- PLAY LOGIC (4-5) ---
        mask_play_draw = (active_stages == STAGE_PLAY_DRAW)
        if mask_play_draw.any():
            ids = mask_play_draw.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = curr_p[ids]
            
            mask_pass = (acts == 9)
            if mask_pass.any():
                pass_ids = ids[mask_pass]
                ptrs = self.draw_ptr[pass_ids].clamp(max=TOTAL_CARDS-1)
                new_cards = self.deck_cards[pass_ids, ptrs]
                old_tops = self.top_discard[pass_ids]
                self._update_graveyard(pass_ids, old_tops)
                self.top_discard[pass_ids] = new_cards
                self.draw_ptr[pass_ids] += 1
                self.stages[pass_ids, p_idxs[mask_pass]] = STAGE_PLAY_DISCARD
                
            mask_take = ~mask_pass
            if mask_take.any():
                take_ids = ids[mask_take]
                slot_idxs = acts[mask_take]
                p_take = p_idxs[mask_take]
                score_before = self._calc_score_batch(take_ids, p_take)
                self._swap_discard(take_ids, p_take, slot_idxs)
                score_after = self._calc_score_batch(take_ids, p_take)
                rewards[take_ids] += (score_before - score_after) * 0.1

        mask_play_disc = (active_stages == STAGE_PLAY_DISCARD)
        if mask_play_disc.any():
            ids = mask_play_disc.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = curr_p[ids]
            
            mask_pass = (acts == 9)
            if mask_pass.any():
                pass_ids = ids[mask_pass]
                card_to_bury = self.top_discard[pass_ids]
                self._update_graveyard(pass_ids, card_to_bury)
            
            mask_take = ~mask_pass
            if mask_take.any():
                take_ids = ids[mask_take]
                slot_idxs = acts[mask_take]
                p_take = p_idxs[mask_take]
                score_before = self._calc_score_batch(take_ids, p_take)
                self._swap_discard(take_ids, p_take, slot_idxs)
                score_after = self._calc_score_batch(take_ids, p_take)
                rewards[take_ids] += (score_before - score_after) * 0.1

        # --- ADVANCE & CHECK DONE ---
        # Logic to advance turns and detect if a player finished
        advance_mask = (mask_flip2) | ((mask_play_draw) & (actions < 9)) | (mask_play_disc)
        if advance_mask.any():
            self._advance_turn_batch(advance_mask)

        # --- END OF ROUND REWARDS (Z-Score) ---
        # If self.dones is set (by _advance_turn_batch finding round end), we calc rewards
        if self.dones.any():
            done_ids = self.dones.nonzero().squeeze(1)
            
            # 1. Calculate Scores for ALL players in done envs
            final_scores = torch.zeros((len(done_ids), NUM_PLAYERS), device=self.device)
            for p in range(NUM_PLAYERS):
                p_tensor = torch.full((len(done_ids),), p, dtype=torch.long, device=self.device)
                final_scores[:, p] = self._calc_score_batch(done_ids, p_tensor)
            
            # 2. Calculate Z-Scores
            # We want LOW scores. So we invert: (Mean - Score) / StdDev
            # If I scored 5 and mean is 20, (20-5)=15 -> Positive Reward.
            # If I scored 30 and mean is 20, (20-30)=-10 -> Negative Reward.
            
            means = final_scores.mean(dim=1, keepdim=True)
            stds = final_scores.std(dim=1, keepdim=True) + 1e-5 # Avoid div/0
            
            # Normalize rewards for the *Current Agent*
            # We identify the current agent by who was playing?
            # In self-play, we reward the agent for the result of the seat "curr_p"
            curr_p_done = self.current_player_idx[done_ids]
            my_scores = final_scores.gather(1, curr_p_done.unsqueeze(1)).squeeze(1)
            
            # Z-Score Reward
            # Multiplier to make it significant (e.g. 50pts variance)
            z_rewards = ((means.squeeze(1) - my_scores) / stds.squeeze(1)) * 20.0
            
            # Bonus for absolute winning (Rank 0)
            ranks = final_scores.argsort(dim=1).argsort(dim=1)
            my_ranks = ranks.gather(1, curr_p_done.unsqueeze(1)).squeeze(1)
            win_bonus = (my_ranks == 0).float() * 50.0 # Extra +50 for actually winning
            
            rewards[done_ids] += z_rewards + win_bonus

        # --- RESET ---
        # Reset if Deck Empty OR Round Over
        deck_empty_mask = (self.draw_ptr >= (TOTAL_CARDS - 5))
        reset_mask = deck_empty_mask | self.dones
        
        if reset_mask.any():
            reset_ids = reset_mask.nonzero().squeeze(1)
            self.reset(reset_ids)

        return self.get_obs(), rewards, self.dones, {}
    
    def _swap_discard(self, env_ids, p_idxs, slot_idxs):
        """Swaps hand card with top discard. Old hand card becomes top discard."""
        new_card = self.top_discard[env_ids]
        
        # Get old card
        # hands: (N, 5, 9). gather requires matching dims.
        # We access hands[env_ids, p_idxs, slot_idxs]
        old_card = self.hands[env_ids, p_idxs, slot_idxs]
        
        # Update Hand
        self.hands[env_ids, p_idxs, slot_idxs] = new_card
        self.visible[env_ids, p_idxs, slot_idxs] = True
        
        # Update Discard Pile (Old card becomes top)
        # The previous 'new_card' is effectively removed from pile, old_card added.
        # But we need to update graveyard with the card that was COVERED? 
        # No, the "Top Discard" is a single slot variable. 
        # The card *under* it is already in graveyard counts? 
        # Wait, the prompt says: "Graveyard... all cards in discard EXCEPT top-most".
        # So when we swap, the card we picked up leaves the pile. The card we dropped enters.
        # The 'Graveyard' count doesn't change yet. The Pile just swaps top cards.
        self.top_discard[env_ids] = old_card

    def _update_graveyard(self, env_ids, cards):
        """Adds cards to the graveyard count tensor."""
        # cards: (N,) card IDs 0-103
        # We map to 0-25 (Ace-King Red, Ace-King Blue)
        # 0-12: Red A-K. 13-25: Blue A-K.
        
        # Logic: 
        # If card < 52 (Red): index = card % 13
        # If card >= 52 (Blue): index = 13 + (card % 13)
        
        is_blue = (cards >= 52).long()
        faces = cards % 13
        indices = (is_blue * 13) + faces
        
        # Scatter add
        # graveyard: (N, 26)
        # We add 1 to the specific columns
        self.graveyard_counts[env_ids, indices] += 1

    # CRITICAL: We need _advance_turn_batch to set self.dones when round ends!
    def _advance_turn_batch(self, mask):
        ids = mask.nonzero().squeeze(1)
        curr = self.current_player_idx[ids]
        
        # Check if current player just finished (all visible)
        # We need to peek at visibility for 'ids' and 'curr'
        # vis: (K, 9)
        # gather is complex here, simpler to check logical condition
        # Actually, standard Golf rules: Round ends if someone finishes?
        # Your rule: "Play continues until one player reveals final card... then every other player gets one final turn."
        # This logic is complex for Vector.
        # SIMPLIFIED: If anyone is fully visible, the round is flagged "Final Turn Phase".
        # For training efficiency, we can end the round IMMEDIATELY when someone finishes.
        # This rewards speed and closing out games. 
        # Let's try Immediate End for faster cycles.
        
        # Check if the player who just moved is fully visible
        # We need to construct indexer [ids, curr, :]
        # But 'curr' is a vector.
        # vis_subset = self.visible[ids, curr, :] -> Shape (K, 9)
        # This requires advanced indexing: self.visible[ids, curr] works in PyTorch
        
        vis_subset = self.visible[ids, curr]
        is_done = vis_subset.all(dim=1) # (K,) booleans
        
        # If done, mark env as done
        # We can't set self.dones immediately because we need to finish the step logic
        # We set it for the NEXT reset cycle? No, reset happens at end of step.
        
        # We update self.dones for the matching envs
        if is_done.any():
            done_sub_ids = ids[is_done]
            self.dones[done_sub_ids] = True
        
        # Only advance turns for NOT done envs?
        # If env is done, it will reset anyway.
        # Advancing turn doesn't hurt.
        
        next_p = (curr + 1) % NUM_PLAYERS
        # ... [Stage transition logic from previous code] ...
        next_stages = self.stages[ids, next_p]
        stay_init_mask = (next_stages == STAGE_ARRANGE)
        start_play_mask = ~stay_init_mask & (self.stages[ids, curr] <= STAGE_FLIP_2)
        normal_play_mask = (self.stages[ids, curr] >= STAGE_PLAY_DRAW)
        
        if stay_init_mask.any():
            sub = ids[stay_init_mask]
            self.current_player_idx[sub] = next_p[stay_init_mask]
        if start_play_mask.any():
            sub = ids[start_play_mask]
            self.stages[sub, :] = STAGE_PLAY_DRAW
            d = self.dealer_idx[sub]
            self.current_player_idx[sub] = (d + 1) % NUM_PLAYERS
        if normal_play_mask.any():
            sub = ids[normal_play_mask]
            self.current_player_idx[sub] = next_p[normal_play_mask]
            self.stages[sub, self.current_player_idx[sub]] = STAGE_PLAY_DRAW

    def _calc_score_batch(self, env_ids, p_idxs):
        """Calculates score for specific player in specific envs."""
        # hands: (N, 5, 9)
        # Relevant hands: (K, 9)
        h = self.hands[env_ids, p_idxs] # Shape (K, 9)
        
        # Columns: [0,3,6], [1,4,7], [2,5,8]
        # We need values: (K, 3) for each col
        c1 = h[:, [0,3,6]]
        c2 = h[:, [1,4,7]]
        c3 = h[:, [2,5,8]]
        
        # Helper to calc column score
        def score_col(col_cards):
            # col_cards: (K, 3)
            faces = self.face_map[col_cards] # (K, 3)
            # Check equality: f1==f2 AND f2==f3
            eq = (faces[:, 0] == faces[:, 1]) & (faces[:, 1] == faces[:, 2])
            
            # Points
            pts = self.point_map[col_cards].sum(dim=1) # (K,)
            
            # If eq, score is 0
            return torch.where(eq, torch.zeros_like(pts), pts)

        total = score_col(c1) + score_col(c2) + score_col(c3)
        return total.float()

    def get_action_masks(self):
        """Returns boolean mask (N, 10). True = Invalid."""
        # Start with EVERYTHING Invalid (True)
        masks = torch.ones((self.num_envs, 10), dtype=torch.bool, device=self.device)
        
        curr = self.current_player_idx
        # Get stage for every environment's current player
        st = self.stages.gather(1, curr.unsqueeze(1)).squeeze(1)
        
        # 1. STAGE ARRANGE
        # Rule: Cannot choose 9. Must choose 0-8.
        is_arrange = (st == STAGE_ARRANGE)
        if is_arrange.any():
            # Unmask 0-8 (Make them False/Valid)
            masks[is_arrange, :9] = False
            # 9 remains True (Invalid)
            
        # 2. FLIP PHASES (1 & 2)
        # Rule: Cannot choose 9. Cannot choose already visible cards.
        is_flip = (st == STAGE_FLIP_1) | (st == STAGE_FLIP_2)
        if is_flip.any():
            # Get visible flags for current player: (N, 9)
            vis = self.visible[torch.arange(self.num_envs), curr]
            
            # If card is visible, mask is True (Invalid)
            # If card is hidden (False), mask is False (Valid) -- wait, we need to copy 'vis' exactly
            # vis has True for Visible. masks needs True for Invalid.
            # So masking visible cards is just direct assignment.
            masks[is_flip, :9] = vis[is_flip]
            
            # Action 9 is always Invalid in Flip
            masks[is_flip, 9] = True
            
        # 3. PLAY PHASES (Draw & Discard)
        # Rule: All options valid (0-9)
        is_play = (st == STAGE_PLAY_DRAW) | (st == STAGE_PLAY_DISCARD)
        if is_play.any():
            masks[is_play, :] = False # All valid
            
        return masks

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        N = len(env_ids)
        curr = self.current_player_idx[env_ids]
        
        # We need to construct the 741 vector.
        # We will build list of tensors and cat them.
        obs_list = []
        
        # 1. Stage Flags (5 bits)
        # One-hot encoding of stage
        st = self.stages[env_ids, curr]
        st_oh = torch.nn.functional.one_hot(st, num_classes=5).float()
        obs_list.append(st_oh)
        
        # 2. Positions (5 bits) - My Seat Index
        pos_oh = torch.nn.functional.one_hot(curr, num_classes=NUM_PLAYERS).float()
        obs_list.append(pos_oh)
        
        # --- RELATIVE ROTATION INDICES ---
        # We need indices to gather data: [My, My+1, My+2...]
        # shape (N, 5)
        offsets = torch.arange(NUM_PLAYERS, device=self.device).repeat(N, 1)
        rot_idxs = (offsets + curr.unsqueeze(1)) % NUM_PLAYERS
        
        # 3. Cumulative Scores (5 floats)
        # Gather scores
        sc = self.scores[env_ids].gather(1, rot_idxs)
        obs_list.append(sc / 100.0)
        
        # 4. Counts (Red Cards) (5 floats/ints)
        # Count reds in hands
        # hands: (N, 5, 9).
        is_red = (self.hands[env_ids] < 52).float()
        red_counts = is_red.sum(dim=2) # (N, 5)
        # Rotate
        obs_list.append(red_counts.gather(1, rot_idxs))
        
        # 5. Finisher Trigger (5 bits)
        # Who finished relative to me?
        # finisher_idx is (N,). We need one hot relative.
        # If -1, all zeros.
        fin = self.finisher_idx[env_ids]
        is_fin = (fin != -1)
        # relative pos of finisher
        rel_fin = (fin - curr + NUM_PLAYERS) % NUM_PLAYERS
        fin_oh = torch.nn.functional.one_hot(rel_fin, num_classes=5).float()
        # Mask out where fin == -1
        fin_oh = fin_oh * is_fin.unsqueeze(1).float()
        obs_list.append(fin_oh)
        
        # 6. Top Discard (14 bits)
        # Color (1), Face (13 one hot)
        td = self.top_discard[env_ids]
        is_blue = (td >= 52).float().unsqueeze(1)
        faces = td % 13
        face_oh = torch.nn.functional.one_hot(faces, num_classes=13).float()
        obs_list.append(is_blue)
        obs_list.append(face_oh)
        
        # 7. Draw Pile Color (1 bit)
        # We need to peek at deck[draw_ptr].
        # Note: If deck empty, this might crash if not handled. 
        # But we handle reset before this.
        ptrs = self.draw_ptr[env_ids]
        # Clamp to avoid out of bounds on done envs
        ptrs = ptrs.clamp(max=TOTAL_CARDS-1) 
        next_cards = self.deck_cards[env_ids, ptrs]
        is_draw_blue = (next_cards >= 52).float().unsqueeze(1)
        obs_list.append(is_draw_blue)
        
        # 8. Graveyard Counts (26 ints)
        obs_list.append(self.graveyard_counts[env_ids])
        
        # 9. Known Hands (675 bits)
        # 5 players * 9 cards * 15 bits.
        # We need to respect "Visibility".
        # If visible=False, all 15 bits are 0.
        # If visible=True, set Color and Face.
        
        # Gather hands in relative order: (N, 5, 9)
        # We need to expand rot_idxs to (N, 5, 9) to gather
        rot_exp = rot_idxs.unsqueeze(2).expand(-1, -1, 9)
        rel_hands = torch.gather(self.hands[env_ids], 1, rot_exp) # (N, 5, 9)
        rel_vis = torch.gather(self.visible[env_ids], 1, rot_exp) # (N, 5, 9)
        
        # Flatten to (N, 45) cards to process easily
        flat_hands = rel_hands.view(N, 45)
        flat_vis = rel_vis.view(N, 45)
        
        # Build 15-bit vector for each of 45 cards
        # Bit 0: Red, Bit 1: Blue, Bits 2-14: Face
        # (N, 45, 15)
        
        card_vecs = torch.zeros((N, 45, 15), device=self.device)
        
        # Masks
        is_b = (flat_hands >= 52)
        is_r = ~is_b
        f = flat_hands % 13
        
        # Set Red/Blue bits
        card_vecs[:, :, 0] = is_r.float()
        card_vecs[:, :, 1] = is_b.float()
        
        # Set Face bits (scatter)
        # We need to scatter 1 into indices 2 + f
        # scatter expects index tensor same dims
        target_idx = (2 + f).unsqueeze(2) # (N, 45, 1)
        src = torch.ones_like(target_idx, dtype=torch.float)
        card_vecs.scatter_(2, target_idx, src)
        
        # MASK HIDDEN CARDS
        # If visible is false, zero out the whole vector for that card
        # flat_vis: (N, 45) -> (N, 45, 1)
        mask_vis = flat_vis.unsqueeze(2).float()
        card_vecs = card_vecs * mask_vis
        
        # Flatten to (N, 675)
        obs_list.append(card_vecs.view(N, -1))
        
        # Concatenate
        return torch.cat(obs_list, dim=1)