"""Vectorized Golf environment — Stage 2.

CHANGES FROM STAGE 1
====================
1. Hands are stored in CANONICAL form per player (not deal order). Every
   state-modifying action re-canonicalizes the affected player's hand. See
   src/canonicalize.py for the sort spec.
2. Observation is EGO-ROTATED: per-player blocks are ordered with acting
   player at index 0, followed by dealer-cycle order. The whose-turn one-hot
   is dropped (always [1,0,0,0,0] after rotation).
3. Per-card encoding ZEROS color bits for visible cards (visible-color
   collapse). Top discard and top draw retain explicit colors.

OBS_SIZE = 886 (was 891 in Stage 1).

Game flow per env (unchanged):
   1. Deal 9 cards to each of 5 players. Top discard revealed. Top draw's
      color visible (face still hidden).
   2. SETUP, in dealer order — for each player k:
        STAGE_ARRANGE: choose how to place R reds and 9-R blues across the 3x3.
        STAGE_FLIP1:   choose first slot to flip (reveals face).
        STAGE_FLIP2:   choose second slot to flip (must be face-down).
   3. PLAY, in dealer order:
        STAGE_PLAY_DRAW:    take top discard / pass and reveal draw card.
        STAGE_PLAY_DISCARD: (after pass) take revealed draw / pass.
   4. End: when finisher's clock runs out or deck runs out.

State conventions:
   - hands[env, player, slot]: card_id (0-103). Player axis is DEAL ORDER;
     slot axis is CANONICAL ORDER within each player.
   - visible[env, player, slot]: same indexing as hands.
   - stages[env, player]: ARRANGE → FLIP1 → FLIP2 → PLAY_DRAW (or _DISCARD).
   - "Placed" status: derived as (stages > STAGE_ARRANGE). Unplaced player
     blocks are zeroed in the obs.

After each canonicalize: card-slot positions may move, but the multiset of
(card_id, visibility) pairs is preserved. Action slot S always refers to
slot S of the CURRENT canonical state.
"""

import torch
import torch.nn.functional as F
from .consts import (
    NUM_PLAYERS, CARDS_PER_PLAYER, NUM_DECKS, TOTAL_CARDS,
    POINT_VALUES, COLUMNS, INITIAL_PER_COLOR_FACE,
    STAGE_ARRANGE, STAGE_FLIP1, STAGE_FLIP2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD,
    NUM_STAGES, OBS_SIZE, ACTION_SIZE,
    MAX_ARRANGEMENT_OPTIONS, _build_red_slot_mask_table,
)
from .canonicalize import canonicalize_hands, ego_rotate, rotated_player_index


class VectorGolfEnv:
    def __init__(self, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = torch.device(device)

        # --- Persistent state ---
        self.hands = torch.zeros((num_envs, NUM_PLAYERS, 9), dtype=torch.long, device=self.device)
        self.visible = torch.zeros((num_envs, NUM_PLAYERS, 9), dtype=torch.bool, device=self.device)
        self.deck_cards = torch.zeros((num_envs, TOTAL_CARDS), dtype=torch.long, device=self.device)
        self.draw_ptr = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.top_discard = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.graveyard_counts = torch.zeros((num_envs, 26), dtype=torch.float32, device=self.device)
        self.stages = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.long, device=self.device)
        self.dealer_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.current_player_idx = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        self.finisher_idx = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.final_turns_taken = torch.zeros((num_envs, NUM_PLAYERS), dtype=torch.bool, device=self.device)
        self.dones = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)
        self.action_count = torch.zeros((num_envs,), dtype=torch.long, device=self.device)

        # Lookup tables
        self.point_map = torch.tensor(
            [POINT_VALUES[i % 13] for i in range(TOTAL_CARDS)], device=self.device,
        )
        self.face_map = torch.tensor([i % 13 for i in range(TOTAL_CARDS)], device=self.device)
        self.point_per_face = torch.tensor(
            [POINT_VALUES[f] for f in range(13)], dtype=torch.float32, device=self.device,
        )
        self._env_idx = torch.arange(num_envs, device=self.device)
        self._col_idx = torch.tensor(COLUMNS, device=self.device, dtype=torch.long)
        self._red_slot_mask_table = _build_red_slot_mask_table().to(self.device)

        self.reset()

    # ==================================================================
    # RESET
    # ==================================================================
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        num_reset = len(env_ids)

        # Shuffle decks
        canonical = torch.arange(TOTAL_CARDS, device=self.device).repeat(num_reset, 1)
        noise = torch.rand(num_reset, TOTAL_CARDS, device=self.device)
        perm = noise.argsort(dim=1)
        self.deck_cards[env_ids] = torch.gather(canonical, 1, perm)

        # Deal 9 cards/player + 1 starting top-discard
        self.hands[env_ids] = self.deck_cards[env_ids, :45].view(num_reset, NUM_PLAYERS, 9)
        self.top_discard[env_ids] = self.deck_cards[env_ids, 45]
        self.draw_ptr[env_ids] = 46

        self.visible[env_ids] = False

        # Canonicalize each player's hand into the sorted face-down layout.
        # With all cards face-down, sort puts blues (key 13) before reds (key 14).
        self.hands[env_ids], self.visible[env_ids] = canonicalize_hands(
            self.hands[env_ids], self.visible[env_ids]
        )

        self.stages[env_ids] = STAGE_ARRANGE
        self.graveyard_counts[env_ids] = 0
        self.finisher_idx[env_ids] = -1
        self.final_turns_taken[env_ids] = False
        self.dones[env_ids] = False
        self.action_count[env_ids] = 0

        self.dealer_idx[env_ids] = torch.randint(0, NUM_PLAYERS, (num_reset,), device=self.device)
        self.current_player_idx[env_ids] = (self.dealer_idx[env_ids] + 1) % NUM_PLAYERS

        return self.get_obs(env_ids)

    # ==================================================================
    # STEP
    # ==================================================================
    def step(self, actions):
        """actions: [num_envs] long, values 0..9.
          ARRANGE: action = partition index (clamped per R).
          FLIP1/FLIP2: action = canonical slot 0..8.
          PLAY_DRAW: action 9 = pass; 0..8 = take top discard into canonical slot.
          PLAY_DISCARD: action 9 = pass; 0..8 = take revealed draw into canonical slot.
        """
        acting_player = self.current_player_idx.clone()
        active_stages = self.stages.gather(1, acting_player.unsqueeze(1)).squeeze(1)
        self.action_count += 1

        # --- ARRANGE ---
        mask_arr = (active_stages == STAGE_ARRANGE)
        if mask_arr.any():
            ids = mask_arr.nonzero().squeeze(1)
            ap = acting_player[ids]
            self._apply_arrange(ids, ap, actions[ids])
            self.stages[ids, ap] = STAGE_FLIP1
            # _apply_arrange canonicalizes internally.

        # --- FLIP1 ---
        mask_f1 = (active_stages == STAGE_FLIP1)
        if mask_f1.any():
            ids = mask_f1.nonzero().squeeze(1)
            ap = acting_player[ids]
            slot_choice = self._sanitize_flip_slot(ids, ap, actions[ids])
            self.visible[ids, ap, slot_choice] = True
            self.stages[ids, ap] = STAGE_FLIP2
            # Canonicalize the acting player's hand after the visibility flip.
            self.hands[ids, ap], self.visible[ids, ap] = canonicalize_hands(
                self.hands[ids, ap], self.visible[ids, ap]
            )

        # --- FLIP2 ---
        mask_f2 = (active_stages == STAGE_FLIP2)
        if mask_f2.any():
            ids = mask_f2.nonzero().squeeze(1)
            ap = acting_player[ids]
            slot_choice = self._sanitize_flip_slot(ids, ap, actions[ids])
            self.visible[ids, ap, slot_choice] = True
            self.stages[ids, ap] = STAGE_PLAY_DRAW
            self.hands[ids, ap], self.visible[ids, ap] = canonicalize_hands(
                self.hands[ids, ap], self.visible[ids, ap]
            )
            self._advance_current_player(ids, ap)

        # --- PLAY_DRAW (first choice) ---
        mask_draw = (active_stages == STAGE_PLAY_DRAW)
        if mask_draw.any():
            ids = mask_draw.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = acting_player[ids]

            mask_pass = (acts == 9)
            if mask_pass.any():
                pass_ids = ids[mask_pass]
                deck_empty_mask = (self.draw_ptr[pass_ids] >= TOTAL_CARDS)
                empty_now = pass_ids[deck_empty_mask]
                ok_now = pass_ids[~deck_empty_mask]

                if empty_now.numel() > 0:
                    self.dones[empty_now] = True

                if ok_now.numel() > 0:
                    ptrs = self.draw_ptr[ok_now]
                    new_top = self.deck_cards[ok_now, ptrs]
                    old_top = self.top_discard[ok_now]
                    self._update_graveyard(ok_now, old_top)
                    self.top_discard[ok_now] = new_top
                    self.draw_ptr[ok_now] += 1
                    p_pass_ok = p_idxs[mask_pass][~deck_empty_mask]
                    self.stages[ok_now, p_pass_ok] = STAGE_PLAY_DISCARD

            mask_take = ~mask_pass
            if mask_take.any():
                self._swap_discard(ids[mask_take], p_idxs[mask_take], acts[mask_take])
                # _swap_discard canonicalizes internally.

        # --- PLAY_DISCARD (second choice) ---
        mask_disc = (active_stages == STAGE_PLAY_DISCARD)
        if mask_disc.any():
            ids = mask_disc.nonzero().squeeze(1)
            acts = actions[ids]
            p_idxs = acting_player[ids]

            mask_take = (acts < 9)
            if mask_take.any():
                self._swap_discard(ids[mask_take], p_idxs[mask_take], acts[mask_take])

        # --- Turn-end logic ---
        play_turn_end_mask = (
            (mask_draw & (actions != 9))
            | mask_disc
            | (self.dones & mask_draw & (actions == 9))
        )
        if play_turn_end_mask.any():
            self._advance_play_turn(play_turn_end_mask, acting_player)

        # --- Snapshot, build info, reset ---
        env_done_snapshot = self.dones.clone()
        info = {}
        if env_done_snapshot.any():
            done_ids = env_done_snapshot.nonzero().squeeze(1)
            all_scores = self._compute_all_scores(done_ids)
            winners_one_hot = self._compute_winners_one_hot(all_scores)
            decisions_per_player = self.action_count[done_ids].float() / NUM_PLAYERS
            info = {
                "all_scores": all_scores,
                "done_env_ids": done_ids,
                "winners_one_hot": winners_one_hot,
                "decisions_per_player": decisions_per_player,
                "final_hands": self.hands[done_ids].clone(),
                "final_visible": self.visible[done_ids].clone(),
                "avg_winner": all_scores.min(dim=1)[0].mean().item(),
                "avg_score": all_scores.mean().item(),
            }
            self.reset(done_ids)

        return self.get_obs(), env_done_snapshot, acting_player, info

    # ==================================================================
    # ACTION MASKS / VALIDATION HELPERS
    # ==================================================================
    def _sanitize_flip_slot(self, ids, p_idxs, slot_actions):
        slot_actions = slot_actions.clamp(min=0, max=8)
        already_vis = self.visible[ids, p_idxs, slot_actions]
        if already_vis.any():
            fix_mask = already_vis
            invalid_ids = ids[fix_mask]
            invalid_p_idxs = p_idxs[fix_mask]
            facedown = ~self.visible[invalid_ids, invalid_p_idxs]
            first_fd = facedown.long().argmax(dim=-1)
            slot_actions = slot_actions.clone()
            slot_actions[fix_mask] = first_fd
        return slot_actions

    # ==================================================================
    # ARRANGEMENT APPLICATION (vectorized)
    # ==================================================================
    def _apply_arrange(self, ids, p_idxs, actions):
        K = ids.numel()
        if K == 0:
            return

        hand = self.hands[ids, p_idxs]
        is_red = (hand < 52).long()
        R = is_red.sum(dim=-1)

        actions_clamped = actions.clamp(min=0, max=MAX_ARRANGEMENT_OPTIONS - 1)
        M = self._red_slot_mask_table[R, actions_clamped]

        is_blue = 1 - is_red
        sort_idx = torch.argsort(is_blue, dim=-1, stable=True)
        sorted_hand = hand.gather(1, sort_idx)

        cum_M = M.cumsum(dim=-1)
        cum_notM = (1 - M).cumsum(dim=-1)
        red_position = cum_M - 1
        blue_position = R.unsqueeze(-1) + cum_notM - 1
        position = torch.where(M.bool(), red_position, blue_position)
        position = position.clamp(min=0, max=8)
        new_hand = sorted_hand.gather(1, position)

        self.hands[ids, p_idxs] = new_hand
        # Canonicalize the modified hand. Visibility unchanged (all False during arrangement).
        self.hands[ids, p_idxs], self.visible[ids, p_idxs] = canonicalize_hands(
            self.hands[ids, p_idxs], self.visible[ids, p_idxs]
        )

    # ==================================================================
    # OBS
    # ==================================================================
    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = self._env_idx
        return self._compute_obs_from_state(
            env_ids=env_ids,
            hands=self.hands[env_ids],
            visible=self.visible[env_ids],
            top_discard=self.top_discard[env_ids],
            graveyard_counts=self.graveyard_counts[env_ids],
            draw_ptr=self.draw_ptr[env_ids],
            stages=self.stages[env_ids],
            current_player_idx=self.current_player_idx[env_ids],
            finisher_idx=self.finisher_idx[env_ids],
            final_turns_taken=self.final_turns_taken[env_ids],
            dealer_idx=self.dealer_idx[env_ids],
        )

    def _compute_obs_from_state(
        self, env_ids, hands, visible, top_discard, graveyard_counts,
        draw_ptr, stages, current_player_idx, finisher_idx, final_turns_taken,
        dealer_idx, draw_color_override=None,
    ):
        N = hands.shape[0]
        device = self.device

        # --- Rotated copies of per-player tensors (acting at index 0) ---
        rotated_hands = ego_rotate(hands, current_player_idx)
        rotated_visible = ego_rotate(visible, current_player_idx)
        rotated_stages = ego_rotate(stages, current_player_idx)
        rotated_final_turns_taken = ego_rotate(final_turns_taken, current_player_idx)
        rotated_placed = (rotated_stages > STAGE_ARRANGE)

        obs_blocks = []

        # 1. Finisher (5), rotated.
        is_fin = (finisher_idx >= 0)
        rot_fin_idx = rotated_player_index(finisher_idx.clamp(min=0), current_player_idx)
        fin_oh = F.one_hot(rot_fin_idx, num_classes=NUM_PLAYERS).float()
        fin_oh = fin_oh * is_fin.unsqueeze(1).float()
        obs_blocks.append(fin_oh)

        # 2. Has-taken-final-turn (5), rotated.
        obs_blocks.append(rotated_final_turns_taken.float())

        # 3. Acting player's stage one-hot (5). Acting at rotated index 0.
        acting_stage = rotated_stages[:, 0]
        stage_oh = F.one_hot(acting_stage, num_classes=NUM_STAGES).float()
        obs_blocks.append(stage_oh)

        # 4. Remaining reds across players-after-acting still in ARRANGE (1, sum/9).
        players_arr = torch.arange(NUM_PLAYERS, device=device).unsqueeze(0).expand(N, -1)
        dealer_pos_per_player = (players_arr - dealer_idx.unsqueeze(1) - 1) % NUM_PLAYERS
        current_dealer_pos = dealer_pos_per_player.gather(1, current_player_idx.unsqueeze(1))
        is_after = dealer_pos_per_player > current_dealer_pos
        is_arrange_p = (stages == STAGE_ARRANGE)
        to_count_p = is_after & is_arrange_p
        reds_per_player = (hands < 52).long().sum(dim=-1)
        remaining_reds = (reds_per_player * to_count_p.long()).sum(dim=1).float() / 9.0
        obs_blocks.append(remaining_reds.unsqueeze(1))

        # 5. Graveyard distribution (26).
        seen_count = graveyard_counts.clone()
        td_idx = ((top_discard >= 52).long() * 13) + (top_discard % 13)
        seen_count.scatter_add_(1, td_idx.unsqueeze(1), torch.ones((N, 1), device=device))
        hands_flat = hands.reshape(N, NUM_PLAYERS * 9)
        vis_flat_all = visible.reshape(N, NUM_PLAYERS * 9).float()
        hand_idx_all = ((hands_flat >= 52).long() * 13) + (hands_flat % 13)
        seen_count.scatter_add_(1, hand_idx_all, vis_flat_all)
        unseen_count = (INITIAL_PER_COLOR_FACE - seen_count).clamp(min=0)
        unseen_per_color = unseen_count.view(N, 2, 13)
        total_per_color = unseen_per_color.sum(dim=2, keepdim=True).clamp(min=1.0)
        grav_dist_per_color = unseen_per_color / total_per_color
        obs_blocks.append(grav_dist_per_color.reshape(N, 26))

        # 6. Top discard (17). Always visible; color retained.
        td = top_discard
        td_color_idx = (td >= 52).long()
        td_color_oh = F.one_hot(td_color_idx, num_classes=2).float()
        td_face = F.one_hot(td % 13, num_classes=13).float()
        td_value = (self.point_map[td].float() / 10.0).unsqueeze(1)
        td_visible = torch.ones((N, 1), device=device)
        obs_blocks.append(torch.cat([td_color_oh, td_face, td_value, td_visible], dim=1))

        # 7. Top draw (17). Color visible, face from grav.
        deck_empty = (draw_ptr >= TOTAL_CARDS)
        if draw_color_override is not None:
            draw_color_idx = draw_color_override.long()
        else:
            ptrs_clamped = draw_ptr.clamp(max=TOTAL_CARDS - 1)
            next_cards = torch.gather(self.deck_cards[env_ids], 1, ptrs_clamped.unsqueeze(1)).squeeze(1)
            draw_color_idx = (next_cards >= 52).long()
        draw_color_oh = F.one_hot(draw_color_idx, num_classes=2).float()
        batch_idx = torch.arange(N, device=device)
        draw_face_dist = grav_dist_per_color[batch_idx, draw_color_idx]
        draw_value = torch.zeros((N, 1), device=device)
        draw_visible = torch.zeros((N, 1), device=device)
        empty_zero = (~deck_empty).float().unsqueeze(1)
        draw_color_oh = draw_color_oh * empty_zero
        draw_face_dist = draw_face_dist * empty_zero
        obs_blocks.append(torch.cat([draw_color_oh, draw_face_dist, draw_value, draw_visible], dim=1))

        # 8. Hand cards (765). Rotated; visible-color collapsed.
        flat_hands = rotated_hands.reshape(N, 45)
        flat_vis = rotated_visible.reshape(N, 45).float()
        flat_placed = rotated_placed.unsqueeze(-1).expand(-1, -1, 9).reshape(N, 45).float()
        is_blue_card = (flat_hands >= 52).long()
        is_red_card = 1 - is_blue_card
        color_oh = torch.stack([is_red_card.float(), is_blue_card.float()], dim=-1)
        # Visible-color collapse:
        color_oh = color_oh * (1.0 - flat_vis).unsqueeze(-1)
        faces = flat_hands % 13
        face_one_hot_visible = F.one_hot(faces, num_classes=13).float() * flat_vis.unsqueeze(2)
        batch_n_45 = torch.arange(N, device=device).unsqueeze(1).expand(-1, 45)
        face_dist_per_card = grav_dist_per_color[batch_n_45, is_blue_card]
        face_dist_facedown = face_dist_per_card * (1.0 - flat_vis).unsqueeze(2)
        face_dim = face_one_hot_visible + face_dist_facedown
        point_values = self.point_map[flat_hands].float()
        value_dim = (point_values / 10.0 * flat_vis).unsqueeze(2)
        visibility_dim = flat_vis.unsqueeze(2)
        card_vecs = torch.cat([color_oh, face_dim, value_dim, visibility_dim], dim=2)
        card_vecs = card_vecs * flat_placed.unsqueeze(2)
        obs_blocks.append(card_vecs.reshape(N, 45 * 17))

        # 9/10. Per-column features (15 + 15) from rotated tensors.
        col_ev, col_match_p = self._column_features(
            hands=rotated_hands, visible=rotated_visible,
            grav_dist_per_color=grav_dist_per_color,
        )
        placed_3 = rotated_placed.unsqueeze(-1).float()
        col_ev = col_ev * placed_3
        col_match_p = col_match_p * placed_3
        obs_blocks.append(col_ev.reshape(N, 15))
        obs_blocks.append(col_match_p.reshape(N, 15))

        # 11. Per-player hand EV (5), rotated.
        hand_ev = (col_ev * (1.0 - col_match_p)).sum(dim=2)
        obs_blocks.append(hand_ev)

        # 12. Per-player face-down count / 9 (5), rotated.
        fd_count = (~rotated_visible).float().sum(dim=2) / 9.0
        fd_count = fd_count * rotated_placed.float()
        obs_blocks.append(fd_count)

        # 13. Per-player score gap (5), rotated.
        hand_ev_for_min = hand_ev.masked_fill(~rotated_placed, float("inf"))
        any_placed = rotated_placed.any(dim=1, keepdim=True)
        min_ev = hand_ev_for_min.min(dim=1, keepdim=True).values
        min_ev = torch.where(any_placed, min_ev, torch.zeros_like(min_ev))
        score_gap = (hand_ev - min_ev) * rotated_placed.float()
        obs_blocks.append(score_gap)

        return torch.cat(obs_blocks, dim=1)

    def _column_features(self, hands, visible, grav_dist_per_color):
        N = hands.shape[0]
        device = self.device
        col_indices = self._col_idx
        cols_cards = hands[:, :, col_indices]
        cols_vis = visible[:, :, col_indices]
        cols_color = (cols_cards >= 52).long()
        cols_face = cols_cards % 13
        oh = F.one_hot(cols_face, num_classes=13).float()
        batch_idx = torch.arange(N, device=device).view(N, 1, 1, 1).expand(N, 5, 3, 3)
        fd_dist = grav_dist_per_color[batch_idx, cols_color]
        vis_f = cols_vis.float().unsqueeze(-1)
        per_card_face_dist = oh * vis_f + fd_dist * (1.0 - vis_f)
        per_card_ev = (per_card_face_dist * self.point_per_face).sum(dim=-1)
        col_ev = per_card_ev.sum(dim=-1)
        joint_face = (
            per_card_face_dist[:, :, :, 0, :]
            * per_card_face_dist[:, :, :, 1, :]
            * per_card_face_dist[:, :, :, 2, :]
        )
        col_match_p = joint_face.sum(dim=-1)
        return col_ev, col_match_p

    # ==================================================================
    # WHAT-IF OBS HELPERS
    # ==================================================================
    def whatif_obs(
        self, env_ids,
        new_hands=None, new_visible=None, new_top_discard=None,
        new_graveyard_counts=None, new_draw_ptr=None, new_stages=None,
        new_current_player_idx=None, new_finisher_idx=None,
        new_final_turns_taken=None, new_dealer_idx=None,
        draw_color_override=None,
    ):
        hands = new_hands if new_hands is not None else self.hands[env_ids]
        visible = new_visible if new_visible is not None else self.visible[env_ids]
        top_discard = new_top_discard if new_top_discard is not None else self.top_discard[env_ids]
        graveyard_counts = new_graveyard_counts if new_graveyard_counts is not None else self.graveyard_counts[env_ids]
        draw_ptr = new_draw_ptr if new_draw_ptr is not None else self.draw_ptr[env_ids]
        stages = new_stages if new_stages is not None else self.stages[env_ids]
        current_player_idx = new_current_player_idx if new_current_player_idx is not None else self.current_player_idx[env_ids]
        finisher_idx = new_finisher_idx if new_finisher_idx is not None else self.finisher_idx[env_ids]
        final_turns_taken = new_final_turns_taken if new_final_turns_taken is not None else self.final_turns_taken[env_ids]
        dealer_idx = new_dealer_idx if new_dealer_idx is not None else self.dealer_idx[env_ids]

        # Canonicalize: idempotent on already-canonical input; necessary on
        # decision-module modifications (placed card at slot S, etc.).
        hands, visible = canonicalize_hands(hands, visible)

        return self._compute_obs_from_state(
            env_ids=env_ids,
            hands=hands, visible=visible,
            top_discard=top_discard, graveyard_counts=graveyard_counts,
            draw_ptr=draw_ptr, stages=stages,
            current_player_idx=current_player_idx,
            finisher_idx=finisher_idx, final_turns_taken=final_turns_taken,
            dealer_idx=dealer_idx,
            draw_color_override=draw_color_override,
        )

    # ==================================================================
    # INTERNAL HELPERS
    # ==================================================================
    def _swap_discard(self, env_ids, p_idxs, slot_idxs):
        new_card = self.top_discard[env_ids]
        old_card = self.hands[env_ids, p_idxs, slot_idxs]
        self.hands[env_ids, p_idxs, slot_idxs] = new_card
        self.visible[env_ids, p_idxs, slot_idxs] = True
        self.top_discard[env_ids] = old_card
        self.hands[env_ids, p_idxs], self.visible[env_ids, p_idxs] = canonicalize_hands(
            self.hands[env_ids, p_idxs], self.visible[env_ids, p_idxs]
        )

    def _update_graveyard(self, env_ids, cards):
        is_blue = (cards >= 52).long()
        faces = cards % 13
        indices = (is_blue * 13) + faces
        updates = F.one_hot(indices, num_classes=26).float()
        self.graveyard_counts[env_ids] += updates

    def _advance_current_player(self, ids, curr):
        next_p = (curr + 1) % NUM_PLAYERS
        self.current_player_idx[ids] = next_p

    def _advance_play_turn(self, mask, acting_player):
        ids = mask.nonzero().squeeze(1)
        curr = acting_player[ids]

        vis_subset = self.visible[ids, curr]
        just_finished = vis_subset.all(dim=1)
        fin_idx = self.finisher_idx[ids]
        no_fin_yet = (fin_idx == -1)

        new_finishers = just_finished & no_fin_yet
        if new_finishers.any():
            sub = ids[new_finishers]
            self.finisher_idx[sub] = curr[new_finishers]
            self.final_turns_taken[sub, curr[new_finishers]] = True

        had_fin = ~no_fin_yet
        if had_fin.any():
            sub = ids[had_fin]
            self.final_turns_taken[sub, curr[had_fin]] = True

        any_done = new_finishers | had_fin
        if any_done.any():
            sub_ids = ids[any_done]
            sub_curr = curr[any_done]
            self.visible[sub_ids, sub_curr] = True
            self.hands[sub_ids, sub_curr], self.visible[sub_ids, sub_curr] = canonicalize_hands(
                self.hands[sub_ids, sub_curr], self.visible[sub_ids, sub_curr]
            )

        all_taken = self.final_turns_taken[ids].all(dim=1)
        if all_taken.any():
            self.dones[ids[all_taken]] = True

        self.stages[ids, curr] = STAGE_PLAY_DRAW

        not_done = ~all_taken
        if not_done.any():
            cont_ids = ids[not_done]
            next_p = (curr[not_done] + 1) % NUM_PLAYERS
            self.current_player_idx[cont_ids] = next_p

    # ==================================================================
    # SCORING
    # ==================================================================
    def _compute_all_scores(self, env_ids):
        K = env_ids.numel()
        scores = torch.zeros((K, NUM_PLAYERS), device=self.device)
        for p in range(NUM_PLAYERS):
            p_tensor = torch.full((K,), p, dtype=torch.long, device=self.device)
            scores[:, p] = self._calc_score_batch(env_ids, p_tensor)
        return scores

    def _calc_score_batch(self, env_ids, p_idxs):
        h = self.hands[env_ids, p_idxs]
        c1, c2, c3 = h[:, [0, 3, 6]], h[:, [1, 4, 7]], h[:, [2, 5, 8]]

        def col_score(col):
            f = self.face_map[col]
            eq = (f[:, 0] == f[:, 1]) & (f[:, 1] == f[:, 2])
            pts = self.point_map[col].sum(dim=1)
            return torch.where(eq, torch.zeros_like(pts), pts)

        return (col_score(c1) + col_score(c2) + col_score(c3)).float()

    def _compute_winners_one_hot(self, all_scores):
        min_scores = all_scores.min(dim=1, keepdim=True).values
        is_min = (all_scores == min_scores).float()
        winner_count = is_min.sum(dim=1, keepdim=True).clamp(min=1.0)
        return is_min / winner_count