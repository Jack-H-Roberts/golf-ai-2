import torch
from .consts import *


class VectorGolfEnv:
    """
    Vectorized Golf environment.

    Step API:
        next_obs, rewards, env_dones, player_dones, acting_players, info = env.step(actions)

    Note on rewards: this env always returns reward = 0. Reward attribution
    (the +1/-1 binary signal) is computed in train.py at env_done time using
    info["all_scores"] and info["done_env_ids"], then written back into the
    rollout buffer at each player's last_player_done step.

    get_obs(env_ids=None, current_player_override=None):
        If current_player_override is provided, the obs is computed AS IF those
        players were currently acting. Used for value bootstrap from each player's
        perspective at rollout boundaries.

    simplified:
        When True (recommended for the current obs encoding), the env skips
        STAGE_ARRANGE / STAGE_FLIP_1 / STAGE_FLIP_2. At reset time, each player
        is dealt cards (random order from the shuffled deck), 2 random cards per
        player are auto-flipped, and all players start in STAGE_PLAY_DRAW.

        The 728-dim obs encoding is designed for the play stages only; it is
        semantically meaningful only when stage is PLAY_DRAW or PLAY_DISCARD.
        Running with simplified=False against this obs is not supported.
    """

    def __init__(self, num_envs, device="cuda", simplified=False):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.simplified = simplified

        self.hands = torch.zeros((num_envs, NUM_PLAYERS, 9), dtype=torch.long, device=self.device)
        self.visible = torch.zeros((num_envs, NUM_PLAYERS, 9), dtype=torch.bool, device=self.device)
        self.deck_cards = torch.zeros((num_envs, TOTAL_CARDS), dtype=torch.long, device=self.device)
        self.draw_ptr = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.top_discard = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        # graveyard_counts tracks BURIED cards (discards no longer on top).
        # Shape: (num_envs, 26): first 13 are red faces, next 13 are blue faces.
        self.graveyard_counts = torch.zeros((num_envs, 26), dtype=torch.float32, device=self.device)
        self.stages = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.long, device=self.device)
        self.dealer_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.current_player_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.arrange_mask = torch.zeros((num_envs, 9), dtype=torch.bool, device=self.device)
        self.finisher_idx = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.final_turns_taken = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.bool, device=self.device)
        self.dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

        # Lookup tables
        self.point_map = torch.tensor([POINT_VALUES[i % 13] for i in range(TOTAL_CARDS)], device=self.device)
        self.face_map = torch.tensor([i % 13 for i in range(TOTAL_CARDS)], device=self.device)

        # Cached
        self._env_idx = torch.arange(num_envs, device=self.device)

        self.reset()

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        num_reset = len(env_ids)

        canonical = torch.arange(TOTAL_CARDS, device=self.device).repeat(num_reset, 1)
        noise = torch.rand(num_reset, TOTAL_CARDS, device=self.device)
        perm = noise.argsort(dim=1)
        self.deck_cards[env_ids] = torch.gather(canonical, 1, perm)

        self.draw_ptr[env_ids] = 0
        self.graveyard_counts[env_ids] = 0

        flat_hands = self.deck_cards[env_ids, :45]
        self.hands[env_ids] = flat_hands.view(num_reset, NUM_PLAYERS, 9)
        self.draw_ptr[env_ids] = 45

        self.top_discard[env_ids] = self.deck_cards[env_ids, 45]
        self.draw_ptr[env_ids] += 1

        self.visible[env_ids] = False

        if self.simplified:
            # 2 random cards per player auto-flipped; all players start in PLAY_DRAW.
            flip_noise = torch.rand(num_reset, NUM_PLAYERS, 9, device=self.device)
            flip_slots = flip_noise.argsort(dim=2)[:, :, :2]
            new_visible = torch.zeros((num_reset, NUM_PLAYERS, 9), dtype=torch.bool, device=self.device)
            src = torch.ones_like(flip_slots, dtype=torch.bool)
            new_visible.scatter_(2, flip_slots, src)
            self.visible[env_ids] = new_visible
            self.stages[env_ids] = STAGE_PLAY_DRAW
        else:
            self.stages[env_ids] = STAGE_ARRANGE

        self.finisher_idx[env_ids] = -1
        self.final_turns_taken[env_ids] = False
        self.dones[env_ids] = False
        self.arrange_mask[env_ids] = False

        self.dealer_idx[env_ids] = torch.randint(0, NUM_PLAYERS, (num_reset,), device=self.device)
        self.current_player_idx[env_ids] = (self.dealer_idx[env_ids] + 1) % NUM_PLAYERS

        if not self.simplified:
            self._check_skip_arrange(env_ids)

        return self.get_obs(env_ids)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------
    def step(self, actions):
        acting_player = self.current_player_idx.clone()
        active_stages = self.stages.gather(1, acting_player.unsqueeze(1)).squeeze(1)
        rewards = torch.zeros(self.num_envs, device=self.device)
        player_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- ARRANGE (only used when simplified=False) ---
        mask_arrange = (active_stages == STAGE_ARRANGE)
        if mask_arrange.any():
            ids = mask_arrange.nonzero().squeeze(1)
            p_idxs = acting_player[ids]
            acts = actions[ids]
            self.arrange_mask[ids, acts] = True

            hand_cards = self.hands[ids, p_idxs]
            num_reds = (hand_cards < 52).sum(dim=1)
            num_selected = self.arrange_mask[ids].sum(dim=1)
            done_arranging = (num_selected >= num_reds)

            if done_arranging.any():
                fin_ids = ids[done_arranging]
                fin_p = p_idxs[done_arranging]
                self._apply_arrangement(fin_ids, fin_p, self.arrange_mask[fin_ids])
                self.stages[fin_ids, fin_p] = STAGE_FLIP_1
                self.arrange_mask[fin_ids] = False

        # --- FLIP 1 (only used when simplified=False) ---
        mask_flip1 = (active_stages == STAGE_FLIP_1)
        if mask_flip1.any():
            ids = mask_flip1.nonzero().squeeze(1)
            self.visible[ids, acting_player[ids], actions[ids]] = True
            self.stages[ids, acting_player[ids]] = STAGE_FLIP_2

        # --- FLIP 2 (only used when simplified=False) ---
        mask_flip2 = (active_stages == STAGE_FLIP_2)
        if mask_flip2.any():
            ids = mask_flip2.nonzero().squeeze(1)
            self.visible[ids, acting_player[ids], actions[ids]] = True

        # --- PLAY DRAW ---
        mask_play_draw = (active_stages == STAGE_PLAY_DRAW)
        if mask_play_draw.any():
            ids = mask_play_draw.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = acting_player[ids]

            mask_pass = (acts == 9)
            if mask_pass.any():
                pass_ids = ids[mask_pass]
                ptrs = self.draw_ptr[pass_ids].clamp(max=TOTAL_CARDS - 1)
                new_top = self.deck_cards[pass_ids, ptrs]
                old_top = self.top_discard[pass_ids]
                # Old top discard moves to graveyard (buried by new draw card).
                self._update_graveyard(pass_ids, old_top)
                self.top_discard[pass_ids] = new_top
                self.draw_ptr[pass_ids] += 1
                self.stages[pass_ids, p_idxs[mask_pass]] = STAGE_PLAY_DISCARD

            mask_take = ~mask_pass
            if mask_take.any():
                self._swap_discard(ids[mask_take], p_idxs[mask_take], acts[mask_take])

        # --- PLAY DISCARD ---
        mask_play_disc = (active_stages == STAGE_PLAY_DISCARD)
        if mask_play_disc.any():
            ids = mask_play_disc.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = acting_player[ids]

            # Pass in PLAY_DISCARD: card stays as top discard for the next player.
            # No state change; turn just ends (handled by turn_end_mask below).
            # (Previous code added this card to graveyard while ALSO leaving it as
            #  top_discard, double-counting it as seen. Removed.)
            mask_take = (acts < 9)
            if mask_take.any():
                self._swap_discard(ids[mask_take], p_idxs[mask_take], acts[mask_take])

        # --- TURN ADVANCEMENT ---
        turn_end_mask = (
            mask_flip2
            | (mask_play_draw & (actions < 9))
            | mask_play_disc
        )
        if turn_end_mask.any():
            new_done_mask = self._advance_turn(turn_end_mask, acting_player)
            player_dones = player_dones | new_done_mask

        # --- ENV-DONE: capture, then reset ---
        env_done_snapshot = self.dones.clone()
        info = {}
        if env_done_snapshot.any():
            done_ids = env_done_snapshot.nonzero().squeeze(1)
            all_scores = self._compute_all_scores(done_ids)
            info = {
                "avg_winner": all_scores.min(dim=1)[0].mean().item(),
                "avg_score": all_scores.mean().item(),
                "all_scores": all_scores,
                "done_env_ids": done_ids,
            }
            self.reset(done_ids)

        # --- DECK-EMPTY: also marks env_done so trajectories don't bridge across the reset ---
        deck_empty_mask = (self.draw_ptr >= (TOTAL_CARDS - 5))
        if deck_empty_mask.any():
            deck_empty_ids = deck_empty_mask.nonzero().squeeze(1)
            env_done_snapshot[deck_empty_ids] = True
            self.reset(deck_empty_ids)

        return (
            self.get_obs(),
            rewards,
            env_done_snapshot,
            player_dones,
            acting_player,
            info,
        )

    # ------------------------------------------------------------------
    # TURN ADVANCEMENT
    # ------------------------------------------------------------------
    def _advance_turn(self, mask, acting_player):
        """Returns player_done_mask (NUM_ENVS,) — True iff the acting player just took their final action."""
        ids = mask.nonzero().squeeze(1)
        curr = acting_player[ids]

        new_done_full = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        vis_subset = self.visible[ids, curr]
        just_finished = vis_subset.all(dim=1)

        fin_idx = self.finisher_idx[ids]
        no_finisher_yet = (fin_idx == -1)
        had_finisher_already = ~no_finisher_yet

        new_finishers = just_finished & no_finisher_yet
        if new_finishers.any():
            sub = ids[new_finishers]
            self.finisher_idx[sub] = curr[new_finishers]
            self.final_turns_taken[sub, curr[new_finishers]] = True

        if had_finisher_already.any():
            sub = ids[had_finisher_already]
            self.final_turns_taken[sub, curr[had_finisher_already]] = True

        player_done_for_acting = new_finishers | had_finisher_already
        if player_done_for_acting.any():
            sub_ids = ids[player_done_for_acting]
            sub_curr = curr[player_done_for_acting]
            # Reveal the player's full hand (so other players see it during their final turns)
            self.visible[sub_ids, sub_curr] = True
            new_done_full[sub_ids] = True

        # Game over if all 5 players have taken final turn
        all_turns_taken = self.final_turns_taken[ids].all(dim=1)
        if all_turns_taken.any():
            self.dones[ids[all_turns_taken]] = True

        # Advance current_player for envs not done
        not_done = ~all_turns_taken
        if not_done.any():
            cont_ids = ids[not_done]
            next_p = (curr[not_done] + 1) % NUM_PLAYERS
            self.current_player_idx[cont_ids] = next_p

            next_stages = self.stages[cont_ids, next_p]
            ready_for_play = (next_stages >= STAGE_FLIP_2)
            if ready_for_play.any():
                self.stages[cont_ids[ready_for_play], next_p[ready_for_play]] = STAGE_PLAY_DRAW

            if not self.simplified:
                self._check_skip_arrange(cont_ids)

        return new_done_full

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------
    def _compute_all_scores(self, env_ids):
        K = len(env_ids)
        all_scores = torch.zeros((K, NUM_PLAYERS), device=self.device)
        for p in range(NUM_PLAYERS):
            p_tensor = torch.full((K,), p, dtype=torch.long, device=self.device)
            all_scores[:, p] = self._calc_score_batch(env_ids, p_tensor)
        return all_scores

    def _calc_score_batch(self, env_ids, p_idxs):
        h = self.hands[env_ids, p_idxs]
        c1, c2, c3 = h[:, [0, 3, 6]], h[:, [1, 4, 7]], h[:, [2, 5, 8]]

        def score_col(col):
            f = self.face_map[col]
            eq = (f[:, 0] == f[:, 1]) & (f[:, 1] == f[:, 2])
            pts = self.point_map[col].sum(dim=1)
            return torch.where(eq, torch.zeros_like(pts), pts)

        return (score_col(c1) + score_col(c2) + score_col(c3)).float()

    # ------------------------------------------------------------------
    # ARRANGEMENT (only used when simplified=False)
    # ------------------------------------------------------------------
    def _apply_arrangement(self, env_ids, p_idxs, mask):
        hands = self.hands[env_ids, p_idxs]
        is_red = (hands < 52)
        _, sorted_src_idxs = (~is_red).sort(dim=1)
        sorted_hands = torch.gather(hands, 1, sorted_src_idxs)

        _, dest_idxs = mask.long().sort(dim=1, descending=True)
        final_hand = torch.zeros_like(hands)
        final_hand.scatter_(1, dest_idxs, sorted_hands)
        self.hands[env_ids, p_idxs] = final_hand

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
                self.stages[sub[zero_reds], p_sub[zero_reds]] = STAGE_FLIP_1

    # ------------------------------------------------------------------
    # CARD HELPERS
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ACTION MASKS
    # ------------------------------------------------------------------
    def get_action_masks(self):
        masks = torch.ones((self.num_envs, 10), dtype=torch.bool, device=self.device)
        curr = self.current_player_idx
        st = self.stages.gather(1, curr.unsqueeze(1)).squeeze(1)

        is_arrange = (st == STAGE_ARRANGE)
        if is_arrange.any():
            masks[is_arrange, :9] = False
            masks[is_arrange, :9] = masks[is_arrange, :9] | self.arrange_mask[is_arrange]
            masks[is_arrange, 9] = True

        is_flip = (st == STAGE_FLIP_1) | (st == STAGE_FLIP_2)
        if is_flip.any():
            vis = self.visible[self._env_idx, curr]
            masks[is_flip, :9] = vis[is_flip]
            masks[is_flip, 9] = True

        is_play = (st == STAGE_PLAY_DRAW) | (st == STAGE_PLAY_DISCARD)
        if is_play.any():
            masks[is_play, :] = False

        return masks

    # ------------------------------------------------------------------
    # OBSERVATION (728-dim, simplified env)
    # ------------------------------------------------------------------
    def get_obs(self, env_ids=None, current_player_override=None):
        """728-dim observation. See consts.py for the layout breakdown.

        Semantically valid only when stage is PLAY_DRAW or PLAY_DISCARD
        (i.e., simplified env, or post-init players in full env).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        N = len(env_ids)
        if current_player_override is not None:
            curr = current_player_override
        else:
            curr = self.current_player_idx[env_ids]

        obs_list = []

        # 1. Stage flag (1 dim): 1 = PLAY_DRAW (passing reveals next), 0 = PLAY_DISCARD (passing ends turn)
        st = self.stages[env_ids, curr]
        stage_bit = (st == STAGE_PLAY_DRAW).float().unsqueeze(1)
        obs_list.append(stage_bit)

        # 2. Relative finisher one-hot (5 dims), all-zero if no finisher yet
        fin = self.finisher_idx[env_ids]
        is_fin = (fin != -1)
        rel_fin = (fin - curr + NUM_PLAYERS) % NUM_PLAYERS
        fin_oh = torch.nn.functional.one_hot(rel_fin, num_classes=5).float() * is_fin.unsqueeze(1).float()
        obs_list.append(fin_oh)

        # 3. Graveyard distribution (26 dims): P(face | color) for both colors,
        #    computed from the unseen pool = initial deck - (buried + top discard + visible hand cards).
        seen_count = self.graveyard_counts[env_ids].clone()  # buried cards: [N, 26]

        # Add top discard
        td = self.top_discard[env_ids]
        td_idx = ((td >= 52).long() * 13) + (td % 13)
        seen_count.scatter_add_(
            1, td_idx.unsqueeze(1), torch.ones((N, 1), device=self.device)
        )

        # Add visible hand cards
        hands_flat = self.hands[env_ids].reshape(N, NUM_PLAYERS * 9)
        vis_flat = self.visible[env_ids].reshape(N, NUM_PLAYERS * 9).float()
        hand_idx = ((hands_flat >= 52).long() * 13) + (hands_flat % 13)
        seen_count.scatter_add_(1, hand_idx, vis_flat)

        # Unseen pool, normalized to per-color probabilities
        unseen_count = (INITIAL_PER_COLOR_FACE - seen_count).clamp(min=0)
        unseen_per_color = unseen_count.view(N, 2, 13)
        total_per_color = unseen_per_color.sum(dim=2, keepdim=True).clamp(min=1.0)
        grav_obs = (unseen_per_color / total_per_color).view(N, 26)
        obs_list.append(grav_obs)

        # 4. Top discard (15 dims): always visible; 1 color + 13 face one-hot + 1 normalized value
        td_color = (td >= 52).float().unsqueeze(1)
        td_face = torch.nn.functional.one_hot(td % 13, num_classes=13).float()
        td_value = (self.point_map[td].float() / 10.0).unsqueeze(1)
        obs_list.append(torch.cat([td_color, td_face, td_value], dim=1))

        # 5. Top draw color (1 dim): face is unknown, distribution lives in the graveyard block
        ptrs = self.draw_ptr[env_ids].clamp(max=TOTAL_CARDS - 1)
        next_cards = self.deck_cards[env_ids, ptrs]
        draw_color = (next_cards >= 52).float().unsqueeze(1)
        obs_list.append(draw_color)

        # 6. Hand cards (45 cards x 15 dims = 675 dims), ego-rotated.
        #    Per card: 1 color + 13 face (one-hot if visible, all-zero if face-down) + 1 value (or 0).
        offsets = torch.arange(NUM_PLAYERS, device=self.device).repeat(N, 1)
        rot_idxs = (offsets + curr.unsqueeze(1)) % NUM_PLAYERS
        rot_exp = rot_idxs.unsqueeze(2).expand(-1, -1, 9)

        rel_hands = torch.gather(self.hands[env_ids], 1, rot_exp)
        rel_vis = torch.gather(self.visible[env_ids], 1, rot_exp)

        flat_hands = rel_hands.reshape(N, 45)
        flat_vis = rel_vis.reshape(N, 45).float()

        is_blue = (flat_hands >= 52).float()
        faces = flat_hands % 13

        color_dim = is_blue.unsqueeze(2)
        face_one_hot = torch.nn.functional.one_hot(faces, num_classes=13).float()
        face_one_hot = face_one_hot * flat_vis.unsqueeze(2)  # zero for face-down
        point_values = self.point_map[flat_hands].float()
        value_dim = (point_values / 10.0 * flat_vis).unsqueeze(2)  # zero for face-down

        card_vecs = torch.cat([color_dim, face_one_hot, value_dim], dim=2)  # [N, 45, 15]
        obs_list.append(card_vecs.reshape(N, 675))

        # 7. Per-player face-down count (5 dims), ego-rotated, normalized /9
        fd_count = (~self.visible[env_ids]).float().sum(dim=2)  # [N, 5]
        fd_rotated = fd_count.gather(1, rot_idxs)
        obs_list.append(fd_rotated / 9.0)

        return torch.cat(obs_list, dim=1)
