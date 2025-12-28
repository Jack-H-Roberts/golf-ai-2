import torch
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
        self.scores = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.float32, device=self.device) 
        
        self.dealer_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.current_player_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        
        # Arrangement Buffer: Tracks which slots the agent has chosen to be RED so far
        self.arrange_mask = torch.zeros((num_envs, 9), dtype=torch.bool, device=self.device)
        
        # Track who finished first and who has taken their final turn
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
        self.scores[env_ids] = 0
        self.stages[env_ids] = STAGE_ARRANGE
        self.finisher_idx[env_ids] = -1
        self.final_turns_taken[env_ids] = False
        self.dones[env_ids] = False
        self.arrange_mask[env_ids] = False 
        
        self.dealer_idx[env_ids] = torch.randint(0, NUM_PLAYERS, (num_reset,), device=self.device)
        
        # Start with player after dealer
        start_p = (self.dealer_idx[env_ids] + 1) % NUM_PLAYERS
        self.current_player_idx[env_ids] = start_p

        # Check for players with 0 Red cards (Skip Arrange)
        self._check_skip_arrange(env_ids)

        return self.get_obs(env_ids)

    def step(self, actions):
        curr_p = self.current_player_idx
        active_stages = self.stages.gather(1, curr_p.unsqueeze(1)).squeeze(1)
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # --- STAGE 1: SEQUENTIAL ARRANGE ---
        mask_arrange = (active_stages == STAGE_ARRANGE)
        if mask_arrange.any():
            ids = mask_arrange.nonzero().squeeze(1)
            p_idxs = curr_p[ids]
            acts = actions[ids]
            
            # Record the choice (Action = Index to place a Red card)
            self.arrange_mask[ids, acts] = True
            
            # Check completion
            hand_cards = self.hands[ids, p_idxs] # (K, 9)
            num_reds = (hand_cards < 52).sum(dim=1)
            num_selected = self.arrange_mask[ids].sum(dim=1)
            
            done_arranging = (num_selected >= num_reds)
            
            if done_arranging.any():
                fin_ids = ids[done_arranging]
                fin_p = p_idxs[done_arranging]
                
                # Apply the physical sort based on the chosen mask
                self._apply_arrangement(fin_ids, fin_p, self.arrange_mask[fin_ids])
                
                self.stages[fin_ids, fin_p] = STAGE_FLIP_1
                self.arrange_mask[fin_ids] = False

        # --- STAGE 2: FLIP 1 ---
        mask_flip1 = (active_stages == STAGE_FLIP_1)
        if mask_flip1.any():
            ids = mask_flip1.nonzero().squeeze(1)
            acts = actions[ids]
            self.visible[ids, curr_p[ids], acts] = True
            self.stages[ids, curr_p[ids]] = STAGE_FLIP_2

        # --- STAGE 3: FLIP 2 ---
        mask_flip2 = (active_stages == STAGE_FLIP_2)
        if mask_flip2.any():
            ids = mask_flip2.nonzero().squeeze(1)
            acts = actions[ids]
            self.visible[ids, curr_p[ids], acts] = True
            # Advance happens at end of step

        # --- STAGE 4: PLAY DRAW ---        
        mask_play_draw = (active_stages == STAGE_PLAY_DRAW)
        if mask_play_draw.any():
            ids = mask_play_draw.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = curr_p[ids]
            
            mask_pass = (acts == 9)
            if mask_pass.any():
                pass_ids = ids[mask_pass]
                ptrs = self.draw_ptr[pass_ids].clamp(max=TOTAL_CARDS-1)
                new_top = self.deck_cards[pass_ids, ptrs]
                old_top = self.top_discard[pass_ids]
                self._update_graveyard(pass_ids, old_top)
                self.top_discard[pass_ids] = new_top
                self.draw_ptr[pass_ids] += 1
                self.stages[pass_ids, p_idxs[mask_pass]] = STAGE_PLAY_DISCARD
                
            mask_take = ~mask_pass
            if mask_take.any():
                take_ids = ids[mask_take]
                slot_idxs = acts[mask_take]
                p_take = p_idxs[mask_take]
                self._swap_discard(take_ids, p_take, slot_idxs)

        # --- STAGE 5: PLAY DISCARD ---
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
                self._swap_discard(take_ids, p_take, slot_idxs)

        # --- TURN ADVANCEMENT ---
        turn_end_mask = (mask_flip2) | ((mask_play_draw) & (actions < 9)) | (mask_play_disc)
        
        if turn_end_mask.any():
            self._advance_turn_batch(turn_end_mask)

        # --- REWARDS & RESETS ---
        if self.dones.any():
            done_ids = self.dones.nonzero().squeeze(1)
            
            # 1. Calc scores
            final_scores = torch.zeros((len(done_ids), NUM_PLAYERS), device=self.device)
            for p in range(NUM_PLAYERS):
                p_tensor = torch.full((len(done_ids),), p, dtype=torch.long, device=self.device)
                final_scores[:, p] = self._calc_score_batch(done_ids, p_tensor)
            
            # 2. Ranks (Ascending: Lowest score = Rank 0)
            ranks = final_scores.argsort(dim=1).argsort(dim=1)
            curr_p_done = self.current_player_idx[done_ids]
            my_ranks = ranks.gather(1, curr_p_done.unsqueeze(1)).squeeze(1)
            
            # 3. Rewards (+1 for Rank 0/Lowest Score, -1 for others)
            is_winner = (my_ranks == 0)
            flat_rewards = torch.where(is_winner, torch.tensor(1.0, device=self.device), torch.tensor(-1.0, device=self.device))
            rewards[done_ids] = flat_rewards

            info = {
                "avg_winner": final_scores.min(dim=1)[0].mean().item(),
                "avg_score": final_scores.mean().item()
            }
            
            self.reset(done_ids)
            return self.get_obs(), rewards, self.dones, info
        
        # Deck Empty Reset
        deck_empty_mask = (self.draw_ptr >= (TOTAL_CARDS - 5))
        if deck_empty_mask.any():
            self.reset(deck_empty_mask.nonzero().squeeze(1))

        return self.get_obs(), rewards, self.dones, {}

    def _apply_arrangement(self, env_ids, p_idxs, mask):
        """
        Sorts hand cards: Reds go to 'mask' slots, Blues go to non-mask slots.
        """
        hands = self.hands[env_ids, p_idxs] # (K, 9)
        
        # 1. Sort current hand so Reds are at front [R, R, R, B, B...]
        is_red = (hands < 52)
        _, sorted_src_idxs = (~is_red).sort(dim=1) 
        sorted_hands = torch.gather(hands, 1, sorted_src_idxs) 
        
        # 2. Sort the MASK to map the Reds to the chosen locations
        _, dest_idxs = mask.long().sort(dim=1, descending=True)
        
        # 3. Scatter sorted cards into destination slots
        final_hand = torch.zeros_like(hands)
        final_hand.scatter_(1, dest_idxs, sorted_hands)
        
        self.hands[env_ids, p_idxs] = final_hand

    def _advance_turn_batch(self, mask):
        ids = mask.nonzero().squeeze(1)
        curr = self.current_player_idx[ids]
        
        # Check Finisher
        vis_subset = self.visible[ids, curr]
        just_finished = vis_subset.all(dim=1)
        
        fin_idx = self.finisher_idx[ids]
        no_finisher_yet = (fin_idx == -1)
        
        new_finishers = just_finished & no_finisher_yet
        if new_finishers.any():
            sub = ids[new_finishers]
            self.finisher_idx[sub] = curr[new_finishers]
            self.final_turns_taken[sub, curr[new_finishers]] = True
            
        fin_idx_updated = self.finisher_idx[ids]
        has_finisher = (fin_idx_updated != -1)
        if has_finisher.any():
            sub = ids[has_finisher]
            self.final_turns_taken[sub, curr[has_finisher]] = True
            
        # Check Game Over
        all_turns_taken = self.final_turns_taken[ids].all(dim=1)
        if all_turns_taken.any():
            self.dones[ids[all_turns_taken]] = True
            
        # Next Player
        not_done = ~all_turns_taken
        if not_done.any():
            cont_ids = ids[not_done]
            next_p = (curr[not_done] + 1) % NUM_PLAYERS
            self.current_player_idx[cont_ids] = next_p
            
            # Transition Logic
            next_stages = self.stages[cont_ids, next_p]
            
            # If next player is already in play/flip, move them to Play Draw
            ready_for_play = (next_stages >= STAGE_FLIP_2)
            if ready_for_play.any():
                self.stages[cont_ids[ready_for_play], next_p[ready_for_play]] = STAGE_PLAY_DRAW

            self._check_skip_arrange(cont_ids)

    def _check_skip_arrange(self, env_ids):
        curr = self.current_player_idx[env_ids]
        st = self.stages[env_ids, curr]
        
        is_arrange = (st == STAGE_ARRANGE)
        if is_arrange.any():
            sub = env_ids[is_arrange]
            p_sub = curr[is_arrange]
            
            hands = self.hands[sub, p_sub]
            reds = (hands < 52).sum(dim=1)
            
            zero_reds = (reds == 0)
            if zero_reds.any():
                z_ids = sub[zero_reds]
                z_p = p_sub[zero_reds]
                self.stages[z_ids, z_p] = STAGE_FLIP_1

    def _swap_discard(self, env_ids, p_idxs, slot_idxs):
        new_card = self.top_discard[env_ids]
        old_card = self.hands[env_ids, p_idxs, slot_idxs]
        self.hands[env_ids, p_idxs, slot_idxs] = new_card
        self.visible[env_ids, p_idxs, slot_idxs] = True
        self.top_discard[env_ids] = old_card

    def _update_graveyard(self, env_ids, cards):
        is_blue = (cards >= 52).long()
        faces = cards % 13
        indices = (is_blue * 13) + faces
        grad_updates = torch.nn.functional.one_hot(indices, num_classes=26).float()
        self.graveyard_counts[env_ids] += grad_updates
        
    def _calc_score_batch(self, env_ids, p_idxs):
        h = self.hands[env_ids, p_idxs]
        c1, c2, c3 = h[:, [0,3,6]], h[:, [1,4,7]], h[:, [2,5,8]]
        def score_col(col):
            f = self.face_map[col]
            eq = (f[:, 0] == f[:, 1]) & (f[:, 1] == f[:, 2])
            pts = self.point_map[col].sum(dim=1)
            return torch.where(eq, torch.zeros_like(pts), pts)
        return (score_col(c1) + score_col(c2) + score_col(c3)).float()

    def get_action_masks(self):
        masks = torch.ones((self.num_envs, 10), dtype=torch.bool, device=self.device)
        curr = self.current_player_idx
        st = self.stages.gather(1, curr.unsqueeze(1)).squeeze(1)
        
        # Arrange: Allow 0-8, but mask ALREADY CHOSEN slots
        is_arrange = (st == STAGE_ARRANGE)
        if is_arrange.any():
            masks[is_arrange, :9] = False
            already_chosen = self.arrange_mask[is_arrange]
            masks[is_arrange, :9] = masks[is_arrange, :9] | already_chosen
            masks[is_arrange, 9] = True
            
        # Flip: Allow 0-8 if hidden
        is_flip = (st == STAGE_FLIP_1) | (st == STAGE_FLIP_2)
        if is_flip.any():
            vis = self.visible[torch.arange(self.num_envs), curr]
            masks[is_flip, :9] = vis[is_flip]
            masks[is_flip, 9] = True
            
        # Play: Allow 0-9
        is_play = (st == STAGE_PLAY_DRAW) | (st == STAGE_PLAY_DISCARD)
        if is_play.any():
            masks[is_play, :] = False
            
        return masks

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        N = len(env_ids)
        curr = self.current_player_idx[env_ids]
        
        obs_list = []
        
        # 1. Stages (5)
        st = self.stages[env_ids, curr]
        obs_list.append(torch.nn.functional.one_hot(st, num_classes=5).float())
        
        # 2. Pos (5)
        obs_list.append(torch.nn.functional.one_hot(curr, num_classes=NUM_PLAYERS).float())
        
        # 3. Rotation indices (Counts/Graveyard)
        offsets = torch.arange(NUM_PLAYERS, device=self.device).repeat(N, 1)
        rot_idxs = (offsets + curr.unsqueeze(1)) % NUM_PLAYERS
        
        # 4. Counts (5) - Red Card Counts
        is_red = (self.hands[env_ids] < 52).float()
        obs_list.append(is_red.sum(dim=2).gather(1, rot_idxs))
        
        # 5. Finisher (5)
        fin = self.finisher_idx[env_ids]
        is_fin = (fin != -1)
        rel_fin = (fin - curr + NUM_PLAYERS) % NUM_PLAYERS
        fin_oh = torch.nn.functional.one_hot(rel_fin, num_classes=5).float() * is_fin.unsqueeze(1).float()
        obs_list.append(fin_oh)
        
        # 6. Top Discard (1+13)
        td = self.top_discard[env_ids]
        obs_list.append((td >= 52).float().unsqueeze(1)) # Color
        obs_list.append(torch.nn.functional.one_hot(td % 13, num_classes=13).float()) # Face
        
        # 7. Draw Color (1)
        ptrs = self.draw_ptr[env_ids].clamp(max=TOTAL_CARDS-1)
        next_cards = self.deck_cards[env_ids, ptrs]
        obs_list.append((next_cards >= 52).float().unsqueeze(1))
        
        # 8. Graveyard (26)
        obs_list.append(self.graveyard_counts[env_ids])
        
        # 9. Known Hands
        rot_exp = rot_idxs.unsqueeze(2).expand(-1, -1, 9)
        rel_hands = torch.gather(self.hands[env_ids], 1, rot_exp)
        rel_vis = torch.gather(self.visible[env_ids], 1, rot_exp)
        rel_stages = torch.gather(self.stages[env_ids], 1, rot_idxs) # (N, 5)
        
        flat_hands = rel_hands.view(N, 45)
        flat_vis = rel_vis.view(N, 45) # Boolean
        flat_stages = rel_stages.unsqueeze(2).expand(-1, -1, 9).reshape(N, 45)
        
        card_vecs = torch.zeros((N, 45, 15), device=self.device)
        is_b = (flat_hands >= 52)
        f = flat_hands % 13
        
        # 9a. Color: Visible OR (Stage > ARRANGE)
        show_color = flat_vis | (flat_stages > STAGE_ARRANGE)
        card_vecs[:, :, 0] = (~is_b).float() * show_color.float()
        card_vecs[:, :, 1] = is_b.float() * show_color.float()
        
        # 9b. Face: Visible ONLY
        target_idx = (2 + f).unsqueeze(2)
        src = torch.ones_like(target_idx, dtype=torch.float)
        face_vecs = torch.zeros_like(card_vecs)
        face_vecs.scatter_(2, target_idx, src)
        
        mask_vis_broad = flat_vis.unsqueeze(2).float()
        card_vecs = card_vecs + (face_vecs * mask_vis_broad)
        
        obs_list.append(card_vecs.view(N, -1))
        
        return torch.cat(obs_list, dim=1)