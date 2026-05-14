"""Decision-making for the value-network architecture, all 5 stages.

Stage 3 changes
===============
1. `_own_winp` now reads `probs[:, rotated_acting_idx]` where the rotation pivot
   used to build the obs determines where the acting player sits:
     - ARRANGE, FLIP1 → obs pivot = acting → acting at rotated position 0.
     - FLIP2, PLAY_DRAW branches A/B/C, PLAY_DISCARD → obs pivot = (acting+1)%5
       → acting at rotated position 4.
   This was wrong in Stage 2: the old code gathered `probs[:, acting_per_row]`
   (deal-order index) but the obs has been ego-rotated since Stage 2, so the
   network output is in rotated order.

2. `make_decisions` accepts optional `env_ids` for league play subsetting.
   When provided, only those envs are processed; the returned actions tensor
   has size [num_envs] with non-subset entries at default value 9 (pass).
"""

import torch
import torch.nn.functional as F
from .consts import (
    NUM_PLAYERS, ACTION_SIZE, TOTAL_CARDS, INITIAL_PER_COLOR_FACE,
    STAGE_ARRANGE, STAGE_FLIP1, STAGE_FLIP2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD,
    NUM_OPTIONS_PER_R, MAX_ARRANGEMENT_OPTIONS,
)


# Rotated-position indices for the acting player in the network output.
# Derived from how each stage builds its what-if obs (see _own_winp docstring).
_ROT_ACTING_AT_0 = 0                       # ARRANGE, FLIP1
_ROT_ACTING_AT_PREV = NUM_PLAYERS - 1      # FLIP2, play branches (pivot = next, so acting = -1 % 5 = 4)


@torch.no_grad()
def make_decisions(env, network, epsilon=0.0, return_diagnostics=False, env_ids=None):
    """Returns actions [num_envs], optionally a diagnostics dict.

    If env_ids is provided, only those envs are processed. The returned tensor
    has shape [num_envs]; non-subset entries are left at default value 9 (pass).
    Caller should select only env_ids when merging.

    diagnostics dict (per-env tensors, size [num_envs]; entries outside env_ids
    are NaN / -1):
      stage : stage of each env (0..4 or -1 if not in subset).
      top1  : best action's predicted own-win-prob.
      top2  : runner-up action's predicted own-win-prob.
      gap   : top1 - top2.
    """
    device = env.device
    actions = torch.full((env.num_envs,), 9, dtype=torch.long, device=device)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)

    if return_diagnostics:
        top1_all = torch.full((env.num_envs,), float('nan'), device=device)
        top2_all = torch.full((env.num_envs,), float('nan'), device=device)
        stage_all = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)

    if env_ids.numel() == 0:
        if return_diagnostics:
            return actions, {
                "stage": stage_all,
                "top1": top1_all,
                "top2": top2_all,
                "gap": top1_all - top2_all,
            }
        return actions

    acting_sub = env.current_player_idx[env_ids]
    stages_sub = env.stages[env_ids].gather(1, acting_sub.unsqueeze(1)).squeeze(1)

    if return_diagnostics:
        stage_all[env_ids] = stages_sub

    handlers = [
        (STAGE_ARRANGE,      _decide_arrange),
        (STAGE_FLIP1,        _decide_flip1),
        (STAGE_FLIP2,        _decide_flip2),
        (STAGE_PLAY_DRAW,    _decide_play_draw),
        (STAGE_PLAY_DISCARD, _decide_play_discard),
    ]
    for stage_val, fn in handlers:
        mask = (stages_sub == stage_val)
        if mask.any():
            local_idx = mask.nonzero().squeeze(1)
            global_ids = env_ids[local_idx]
            if return_diagnostics:
                acts_here, t1, t2 = fn(env, network, global_ids, epsilon, return_diagnostics=True)
                actions[global_ids] = acts_here
                top1_all[global_ids] = t1
                top2_all[global_ids] = t2
            else:
                actions[global_ids] = fn(env, network, global_ids, epsilon)

    if return_diagnostics:
        return actions, {
            "stage": stage_all,
            "top1": top1_all,
            "top2": top2_all,
            "gap": top1_all - top2_all,
        }
    return actions


# ---------- shared helpers ----------
def _grav_dist(env, env_ids):
    """Returns (unseen_counts, dist) for the unseen-pool by color × face."""
    K = env_ids.numel()
    device = env.device
    seen = env.graveyard_counts[env_ids].clone()
    td = env.top_discard[env_ids]
    seen.scatter_add_(1, ((td >= 52).long() * 13 + td % 13).unsqueeze(1),
                      torch.ones((K, 1), device=device))
    hf = env.hands[env_ids].reshape(K, NUM_PLAYERS * 9)
    vf = env.visible[env_ids].reshape(K, NUM_PLAYERS * 9).float()
    seen.scatter_add_(1, ((hf >= 52).long() * 13 + hf % 13), vf)
    unseen = (INITIAL_PER_COLOR_FACE - seen).clamp(min=0).view(K, 2, 13)
    dist = unseen / unseen.sum(dim=2, keepdim=True).clamp(min=1.0)
    return unseen, dist


def _own_winp(network, obs, rotated_acting_idx):
    """Forward the win head and return P(acting wins) per row.

    The obs is ego-rotated around new_current_player_idx (the player who will
    act NEXT in the what-if state). The network's per-player output is in that
    rotation. `rotated_acting_idx` is where the (current) acting player sits
    in that rotation:
        - For obs built with new_current = acting (ARRANGE, FLIP1):
          acting at position 0.
        - For obs built with new_current = (acting+1) % 5 (FLIP2, play branches):
          acting at position (acting - new_current) % 5 = -1 % 5 = 4.

    Uses float16 autocast on CUDA. Logits cast to FP32 for softmax accuracy.
    Score head's output is discarded here (used only during training).
    """
    device_type = obs.device.type
    use_amp = (device_type == 'cuda')
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        logits, _score = network(obs)
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[:, rotated_acting_idx]


def _sample_from_mask(mask):
    noise = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1.0)
    return noise.argmax(dim=-1)


def _topk2(values):
    """Returns (top1, top2). When only one finite entry exists per row,
    top2 is set equal to top1 (gap = 0)."""
    topk = values.topk(2, dim=-1)
    top1 = topk.values[:, 0]
    top2 = topk.values[:, 1]
    top2 = torch.where(torch.isinf(top2), top1, top2)
    return top1, top2


# ---------- ARRANGE ----------
@torch.no_grad()
def _decide_arrange(env, network, env_ids, epsilon, return_diagnostics=False):
    K = env_ids.numel()
    device = env.device
    M = MAX_ARRANGEMENT_OPTIONS
    acting = env.current_player_idx[env_ids]
    acting_hands = env.hands[env_ids, acting]
    is_red = (acting_hands < 52).long()
    R = is_red.sum(dim=-1)

    sort_idx = torch.argsort(1 - is_red, dim=-1, stable=True)
    sorted_hand = acting_hands.gather(1, sort_idx)

    options = torch.arange(M, device=device).view(1, M).expand(K, M)
    mask_M = env._red_slot_mask_table[R.view(K, 1).expand(K, M), options]

    cum_M = mask_M.cumsum(dim=-1)
    cum_notM = (1 - mask_M).cumsum(dim=-1)
    pos = torch.where(
        mask_M.bool(), cum_M - 1, R.view(K, 1, 1) + cum_notM - 1
    ).clamp(min=0, max=8)
    new_acting = sorted_hand.unsqueeze(1).expand(K, M, 9).gather(2, pos)

    full_hands = env.hands[env_ids].view(K, 1, NUM_PLAYERS, 9).expand(K, M, NUM_PLAYERS, 9).clone()
    rK = torch.arange(K, device=device).view(K, 1).expand(K, M)
    rM = torch.arange(M, device=device).view(1, M).expand(K, M)
    aK = acting.view(K, 1).expand(K, M)
    full_hands[rK, rM, aK] = new_acting

    new_stages = env.stages[env_ids].view(K, 1, NUM_PLAYERS).expand(K, M, NUM_PLAYERS).clone()
    new_stages[rK, rM, aK] = STAGE_FLIP1

    BATCH = K * M
    obs = env.whatif_obs(
        env_ids=env_ids.view(K, 1).expand(K, M).reshape(BATCH),
        new_hands=full_hands.reshape(BATCH, NUM_PLAYERS, 9),
        new_stages=new_stages.reshape(BATCH, NUM_PLAYERS),
    )
    # ARRANGE: new_current defaults to acting → acting at rotated position 0.
    own = _own_winp(network, obs, _ROT_ACTING_AT_0).view(K, M)

    nopts = torch.tensor(NUM_OPTIONS_PER_R, device=device)[R]
    valid = torch.arange(M, device=device).view(1, M) < nopts.view(K, 1)
    own = own.masked_fill(~valid, float('-inf'))
    greedy = own.argmax(dim=-1)

    if epsilon > 0:
        rand = torch.minimum(torch.randint(0, M, (K,), device=device), nopts - 1)
        is_random = torch.rand(K, device=device) < epsilon
        result = torch.where(is_random, rand, greedy)
    else:
        result = greedy

    if return_diagnostics:
        top1, top2 = _topk2(own)
        return result, top1, top2
    return result


# ---------- FLIP1 / FLIP2 ----------
def _decide_flip1(env, network, env_ids, epsilon, return_diagnostics=False):
    return _decide_flip(env, network, env_ids, epsilon, is_flip2=False,
                        return_diagnostics=return_diagnostics)


def _decide_flip2(env, network, env_ids, epsilon, return_diagnostics=False):
    return _decide_flip(env, network, env_ids, epsilon, is_flip2=True,
                        return_diagnostics=return_diagnostics)


@torch.no_grad()
def _decide_flip(env, network, env_ids, epsilon, is_flip2, return_diagnostics=False):
    K = env_ids.numel()
    device = env.device
    acting = env.current_player_idx[env_ids]
    acting_hands = env.hands[env_ids, acting]
    slot_colors = (acting_hands >= 52).long()

    _, grav = _grav_dist(env, env_ids)
    rK = torch.arange(K, device=device)
    p_F = grav[rK.view(K, 1).expand(-1, 9), slot_colors]

    rK3 = rK.view(K, 1, 1).expand(K, 9, 13)
    rS3 = torch.arange(9, device=device).view(1, 9, 1).expand(K, 9, 13)
    rF3 = torch.arange(13, device=device).view(1, 1, 13).expand(K, 9, 13)
    a3 = acting.view(K, 1, 1).expand(K, 9, 13)

    hyp = (slot_colors.view(K, 9, 1).expand(K, 9, 13) * 52 + rF3)
    hands_b = env.hands[env_ids].view(K, 1, 1, NUM_PLAYERS, 9).expand(K, 9, 13, NUM_PLAYERS, 9).clone()
    hands_b[rK3, rS3, rF3, a3, rS3] = hyp

    vis_b = env.visible[env_ids].view(K, 1, 1, NUM_PLAYERS, 9).expand(K, 9, 13, NUM_PLAYERS, 9).clone()
    vis_b[rK3, rS3, rF3, a3, rS3] = True

    target_stage = STAGE_PLAY_DRAW if is_flip2 else STAGE_FLIP2
    stages_b = env.stages[env_ids].view(K, 1, 1, NUM_PLAYERS).expand(K, 9, 13, NUM_PLAYERS).clone()
    stages_b[rK3, rS3, rF3, a3] = target_stage

    if is_flip2:
        cur_b = ((acting + 1) % NUM_PLAYERS).view(K, 1, 1).expand(K, 9, 13).contiguous()
    else:
        cur_b = acting.view(K, 1, 1).expand(K, 9, 13).contiguous()

    BATCH = K * 9 * 13
    obs = env.whatif_obs(
        env_ids=env_ids.view(K, 1, 1).expand(K, 9, 13).reshape(BATCH),
        new_hands=hands_b.reshape(BATCH, NUM_PLAYERS, 9),
        new_visible=vis_b.reshape(BATCH, NUM_PLAYERS, 9),
        new_stages=stages_b.reshape(BATCH, NUM_PLAYERS),
        new_current_player_idx=cur_b.reshape(BATCH),
    )
    # FLIP1: new_current = acting → acting at 0.
    # FLIP2: new_current = (acting+1)%5 → acting at 4.
    rot_idx = _ROT_ACTING_AT_PREV if is_flip2 else _ROT_ACTING_AT_0
    own = _own_winp(network, obs, rot_idx).view(K, 9, 13)
    ev_per_S = (own * p_F).sum(dim=-1)

    visible_mask = env.visible[env_ids, acting]
    ev_per_S = ev_per_S.masked_fill(visible_mask, float('-inf'))
    greedy = ev_per_S.argmax(dim=-1)

    if epsilon > 0:
        rand = _sample_from_mask(~visible_mask)
        is_random = torch.rand(K, device=device) < epsilon
        result = torch.where(is_random, rand, greedy)
    else:
        result = greedy

    if return_diagnostics:
        top1, top2 = _topk2(ev_per_S)
        return result, top1, top2
    return result


# ---------- PLAY_DRAW: full A + B + C ----------
@torch.no_grad()
def _decide_play_draw(env, network, env_ids, epsilon, return_diagnostics=False):
    K = env_ids.numel()
    device = env.device
    acting = env.current_player_idx[env_ids]
    rK = torch.arange(K, device=device)

    unseen, grav = _grav_dist(env, env_ids)
    color_marg_count = unseen.sum(dim=-1)
    p_color_marg = color_marg_count / color_marg_count.sum(dim=-1, keepdim=True).clamp(min=1.0)

    deck_empty = (env.draw_ptr[env_ids] >= TOTAL_CARDS)
    next_ptr = env.draw_ptr[env_ids].clamp(max=TOTAL_CARDS - 1)
    next_card = env.deck_cards[env_ids, next_ptr]
    draw_color = (next_card >= 52).long()
    p_F_given_draw = grav[rK, draw_color]

    slot_cards = env.hands[env_ids, acting]
    slot_colors = (slot_cards >= 52).long()
    slot_visible = env.visible[env_ids, acting]
    slot_actual_face = slot_cards % 13
    p_DF = grav[rK.view(K, 1).expand(-1, 9), slot_colors]
    DF_oh = F.one_hot(slot_actual_face, num_classes=13).float()
    p_DF = torch.where(slot_visible.unsqueeze(-1).expand(K, 9, 13), DF_oh, p_DF)

    branch_a_ev = _branch_a(env, network, env_ids, acting, slot_colors, p_DF)
    branch_a_best, branch_a_arg = branch_a_ev.max(dim=-1)
    branch_c_ev = _branch_c(env, network, env_ids, acting, draw_color, p_F_given_draw, p_color_marg)
    branch_b_ev = _branch_b(env, network, env_ids, acting, draw_color,
                            p_F_given_draw, p_color_marg, slot_colors, p_DF)
    branch_b_best, _ = branch_b_ev.max(dim=-1)

    pass_ev = torch.maximum(branch_b_best, branch_c_ev)
    take_better = (branch_a_best >= pass_ev)
    greedy = torch.where(take_better, branch_a_arg,
                         torch.full((K,), 9, dtype=torch.long, device=device))
    if deck_empty.any():
        greedy = torch.where(deck_empty, branch_a_arg, greedy)

    if epsilon > 0:
        rand = torch.randint(0, 10, (K,), device=device)
        if deck_empty.any():
            rand = torch.where(deck_empty, rand.clamp(max=8), rand)
        is_random = torch.rand(K, device=device) < epsilon
        result = torch.where(is_random, rand, greedy)
    else:
        result = greedy

    if return_diagnostics:
        pass_col = pass_ev.clone()
        if deck_empty.any():
            pass_col = torch.where(deck_empty, torch.full_like(pass_col, float('-inf')), pass_col)
        action_evs = torch.cat([branch_a_ev, pass_col.unsqueeze(1)], dim=1)
        top1, top2 = _topk2(action_evs)
        return result, top1, top2
    return result


def _branch_a(env, network, env_ids, acting, slot_colors, p_DF):
    K = env_ids.numel()
    placed = env.top_discard[env_ids]
    next_p = (acting + 1) % NUM_PLAYERS
    obs = _build_play_obs(env, env_ids, acting, next_p, placed, slot_colors,
                          bury_old_top=False, advance_draw_ptr=False, draw_color_override=None)
    # _build_play_obs pivots around next_p → acting at rotated position 4.
    own = _own_winp(network, obs, _ROT_ACTING_AT_PREV).view(9, 13, K)
    p_DF_arr = p_DF.permute(1, 2, 0)
    return (own * p_DF_arr).sum(dim=1).t()


def _branch_b(env, network, env_ids, acting, draw_color, p_F_given_draw, p_color_marg, slot_colors, p_DF):
    """Pass-then-take EV per slot. Single forward pass over (F=13, C=2) chance
    outcomes vectorized across slots and DF. See Stage 2 for full memory notes."""
    K = env_ids.numel()
    device = env.device
    next_p = (acting + 1) % NUM_PLAYERS
    F_TOTAL = 13
    C_TOTAL = 2
    FCK = F_TOTAL * C_TOTAL * K

    F_arange = torch.arange(F_TOTAL, device=device)
    placed_FCK = (
        draw_color.view(1, 1, K) * 52 + F_arange.view(F_TOTAL, 1, 1)
    ).expand(F_TOTAL, C_TOTAL, K).reshape(-1)

    env_ids_FCK   = env_ids.view(1, 1, K).expand(F_TOTAL, C_TOTAL, K).reshape(-1)
    acting_FCK    = acting.view(1, 1, K).expand(F_TOTAL, C_TOTAL, K).reshape(-1)
    next_p_FCK    = next_p.view(1, 1, K).expand(F_TOTAL, C_TOTAL, K).reshape(-1)
    slot_colors_FCK = slot_colors.view(1, 1, K, 9).expand(F_TOTAL, C_TOTAL, K, 9).reshape(FCK, 9)

    C_arange = torch.arange(C_TOTAL, device=device)
    override = C_arange.view(1, 1, 1, C_TOTAL, 1).expand(9, 13, F_TOTAL, C_TOTAL, K).reshape(-1)

    obs = _build_play_obs(
        env, env_ids_FCK, acting_FCK, next_p_FCK, placed_FCK, slot_colors_FCK,
        bury_old_top=True, advance_draw_ptr=True, draw_color_override=override,
    )
    # _build_play_obs pivots around next_p → acting at rotated position 4.
    own = _own_winp(network, obs, _ROT_ACTING_AT_PREV).view(9, 13, F_TOTAL, C_TOTAL, K)

    p_DF_arr = p_DF.permute(1, 2, 0).view(9, 13, 1, 1, K)
    ev_FCK_per_S = (own * p_DF_arr).sum(dim=1)

    p_F_FK = p_F_given_draw.t().unsqueeze(1)
    p_C_CK = p_color_marg.t().unsqueeze(0)
    p_FC = (p_F_FK * p_C_CK).unsqueeze(0)

    ev_K = (ev_FCK_per_S * p_FC).sum(dim=(1, 2))
    return ev_K.t()


def _branch_c(env, network, env_ids, acting, draw_color, p_F_given_draw, p_color_marg):
    K = env_ids.numel()
    device = env.device
    next_p = (acting + 1) % NUM_PLAYERS

    new_grav = env.graveyard_counts[env_ids].clone()
    old_td = env.top_discard[env_ids]
    new_grav.scatter_add_(1, ((old_td >= 52).long() * 13 + old_td % 13).unsqueeze(1),
                          torch.ones((K, 1), device=device))

    rF = torch.arange(13, device=device)
    rC = torch.arange(2, device=device)
    new_top = (draw_color.view(1, 1, K) * 52 + rF.view(13, 1, 1)).expand(13, 2, K).reshape(-1)
    override = rC.view(1, 2, 1).expand(13, 2, K).reshape(-1)

    grav_b = new_grav.unsqueeze(0).unsqueeze(0).expand(13, 2, K, 26).reshape(-1, 26)
    drawptr_b = (env.draw_ptr[env_ids] + 1).unsqueeze(0).unsqueeze(0).expand(13, 2, K).reshape(-1)
    hands_b = env.hands[env_ids].unsqueeze(0).unsqueeze(0).expand(13, 2, K, NUM_PLAYERS, 9).reshape(-1, NUM_PLAYERS, 9)
    vis_b = env.visible[env_ids].unsqueeze(0).unsqueeze(0).expand(13, 2, K, NUM_PLAYERS, 9).reshape(-1, NUM_PLAYERS, 9)
    stages_b = env.stages[env_ids].unsqueeze(0).unsqueeze(0).expand(13, 2, K, NUM_PLAYERS).reshape(-1, NUM_PLAYERS)
    cur_b = next_p.unsqueeze(0).unsqueeze(0).expand(13, 2, K).reshape(-1)
    env_b = env_ids.unsqueeze(0).unsqueeze(0).expand(13, 2, K).reshape(-1)

    obs = env.whatif_obs(
        env_ids=env_b, new_hands=hands_b, new_visible=vis_b,
        new_top_discard=new_top, new_graveyard_counts=grav_b,
        new_draw_ptr=drawptr_b, new_stages=stages_b,
        new_current_player_idx=cur_b, draw_color_override=override,
    )
    # Pivot around next_p → acting at rotated position 4.
    own = _own_winp(network, obs, _ROT_ACTING_AT_PREV).view(13, 2, K)

    p_FC = p_F_given_draw.t().unsqueeze(1) * p_color_marg.t().unsqueeze(0)
    return (own * p_FC).sum(dim=(0, 1))


def _build_play_obs(env, env_ids, acting, next_player, placed_card, slot_colors,
                    bury_old_top, advance_draw_ptr, draw_color_override):
    """Build [9*13*K, OBS_SIZE] for "place placed_card at slot S; old card → top discard".
    Includes finishing logic. Sets new_current_player_idx = next_player, so the
    output obs is ego-rotated around next_player."""
    K = env_ids.numel()
    device = env.device
    rS = torch.arange(9, device=device)
    rDF = torch.arange(13, device=device)
    rK = torch.arange(K, device=device)

    hands_base = env.hands[env_ids]
    vis_base = env.visible[env_ids]
    new_hands_S = hands_base.unsqueeze(0).expand(9, K, NUM_PLAYERS, 9).clone()
    new_vis_S = vis_base.unsqueeze(0).expand(9, K, NUM_PLAYERS, 9).clone()
    s_idx = rS.view(9, 1).expand(9, K)
    k_idx = rK.view(1, K).expand(9, K)
    a_2 = acting.view(1, K).expand(9, K)
    p_2 = placed_card.view(1, K).expand(9, K)
    new_hands_S[s_idx, k_idx, a_2, s_idx] = p_2
    new_vis_S[s_idx, k_idx, a_2, s_idx] = True

    fd_mask = ~vis_base[rK, acting]
    fd_count = fd_mask.sum(dim=1)
    finished_SK = fd_mask.t() & (fd_count == 1).unsqueeze(0)
    had_fin = (env.finisher_idx[env_ids] != -1)
    takes_final_SK = had_fin.unsqueeze(0).expand(9, K) | finished_SK

    rNP = torch.arange(NUM_PLAYERS, device=device)
    is_acting_KP = (rNP.unsqueeze(0) == acting.unsqueeze(1))
    reveal_SKP = takes_final_SK.unsqueeze(2) & is_acting_KP.unsqueeze(0)
    new_vis_S = new_vis_S | reveal_SKP.unsqueeze(3).expand(9, K, NUM_PLAYERS, 9)

    minus_one = torch.full((9, K), -1, dtype=torch.long, device=device)
    new_finisher_S = torch.where(
        had_fin.unsqueeze(0).expand(9, K),
        env.finisher_idx[env_ids].unsqueeze(0).expand(9, K),
        torch.where(finished_SK, acting.unsqueeze(0).expand(9, K), minus_one),
    )
    new_final_turns_S = (
        env.final_turns_taken[env_ids].unsqueeze(0).expand(9, K, NUM_PLAYERS)
        | (takes_final_SK.unsqueeze(2) & is_acting_KP.unsqueeze(0))
    )

    new_hands_b = new_hands_S.unsqueeze(1).expand(9, 13, K, NUM_PLAYERS, 9).reshape(9 * 13 * K, NUM_PLAYERS, 9)
    new_vis_b = new_vis_S.unsqueeze(1).expand(9, 13, K, NUM_PLAYERS, 9).reshape(9 * 13 * K, NUM_PLAYERS, 9)

    slot_color_SK = slot_colors.t()
    new_top_b = (slot_color_SK.unsqueeze(1) * 52 + rDF.view(1, 13, 1)).reshape(9 * 13 * K)

    new_grav_K = env.graveyard_counts[env_ids].clone()
    if bury_old_top:
        old_top = env.top_discard[env_ids]
        old_idx = ((old_top >= 52).long() * 13) + (old_top % 13)
        new_grav_K.scatter_add_(1, old_idx.unsqueeze(1), torch.ones((K, 1), device=device))
    new_grav_b = new_grav_K.unsqueeze(0).unsqueeze(0).expand(9, 13, K, 26).reshape(9 * 13 * K, 26)

    new_draw_ptr_K = env.draw_ptr[env_ids] + (1 if advance_draw_ptr else 0)
    new_draw_ptr_b = new_draw_ptr_K.unsqueeze(0).unsqueeze(0).expand(9, 13, K).reshape(-1)

    new_stages_b = env.stages[env_ids].unsqueeze(0).unsqueeze(0).expand(9, 13, K, NUM_PLAYERS).reshape(9 * 13 * K, NUM_PLAYERS)
    new_current_b = next_player.unsqueeze(0).unsqueeze(0).expand(9, 13, K).reshape(-1)
    new_finisher_b = new_finisher_S.unsqueeze(1).expand(9, 13, K).reshape(-1)
    new_final_b = new_final_turns_S.unsqueeze(1).expand(9, 13, K, NUM_PLAYERS).reshape(9 * 13 * K, NUM_PLAYERS)
    env_ids_b = env_ids.unsqueeze(0).unsqueeze(0).expand(9, 13, K).reshape(-1)

    return env.whatif_obs(
        env_ids=env_ids_b,
        new_hands=new_hands_b, new_visible=new_vis_b,
        new_top_discard=new_top_b, new_graveyard_counts=new_grav_b,
        new_draw_ptr=new_draw_ptr_b, new_stages=new_stages_b,
        new_current_player_idx=new_current_b,
        new_finisher_idx=new_finisher_b, new_final_turns_taken=new_final_b,
        draw_color_override=draw_color_override,
    )


# ---------- PLAY_DISCARD ----------
@torch.no_grad()
def _decide_play_discard(env, network, env_ids, epsilon, return_diagnostics=False):
    """Second choice: take revealed draw card to slot S, or pass (it stays as top discard)."""
    K = env_ids.numel()
    device = env.device
    acting = env.current_player_idx[env_ids]
    rK = torch.arange(K, device=device)

    drawn = env.top_discard[env_ids]
    slot_cards = env.hands[env_ids, acting]
    slot_colors = (slot_cards >= 52).long()
    slot_visible = env.visible[env_ids, acting]
    slot_actual = slot_cards % 13

    _, grav = _grav_dist(env, env_ids)
    p_DF = grav[rK.view(K, 1).expand(-1, 9), slot_colors]
    DF_oh = F.one_hot(slot_actual, num_classes=13).float()
    p_DF = torch.where(slot_visible.unsqueeze(-1).expand(K, 9, 13), DF_oh, p_DF)

    next_p = (acting + 1) % NUM_PLAYERS
    obs_take = _build_play_obs(env, env_ids, acting, next_p, drawn, slot_colors,
                                bury_old_top=False, advance_draw_ptr=False,
                                draw_color_override=None)
    # _build_play_obs pivots around next_p → acting at 4.
    own_take = _own_winp(network, obs_take, _ROT_ACTING_AT_PREV).view(9, 13, K)
    take_ev = (own_take * p_DF.permute(1, 2, 0)).sum(dim=1).t()

    pass_obs = env.whatif_obs(env_ids=env_ids, new_current_player_idx=next_p)
    # Same: pivot = next_p → acting at 4.
    own_pass = _own_winp(network, pass_obs, _ROT_ACTING_AT_PREV)

    take_best, take_arg = take_ev.max(dim=-1)
    take_better = (take_best >= own_pass)
    greedy = torch.where(take_better, take_arg, torch.full((K,), 9, dtype=torch.long, device=device))

    if epsilon > 0:
        rand = torch.randint(0, 10, (K,), device=device)
        is_random = torch.rand(K, device=device) < epsilon
        result = torch.where(is_random, rand, greedy)
    else:
        result = greedy

    if return_diagnostics:
        action_evs = torch.cat([take_ev, own_pass.unsqueeze(1)], dim=1)
        top1, top2 = _topk2(action_evs)
        return result, top1, top2
    return result
