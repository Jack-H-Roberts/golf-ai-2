import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

from src.model import GolfNet
from src.vector_env import VectorGolfEnv
from src.consts import *

# --- HYPERPARAMETERS ---
NUM_ENVS = 4096
LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 2_000_000_000
NUM_STEPS = 128
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 8192
NUM_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# --- POPULATION TRAINING ---
NUM_CURRENT_SLOTS = 2          # Out of NUM_PLAYERS=5; the rest are sampled from pool.
SAVE_EVERY_N_UPDATES = 20

# --- FLAGS ---
USE_MIXED_PRECISION = True
USE_SIMPLIFIED_ENV = True
LOAD_MODEL = False
MODEL_PATH = "latest_model.pt"
POOL_DIR = "model_pool"

MIN_ACTIONS_PER_PLAYER = 7


def step_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"model_(\d+)\.pt", name)
    return int(m.group(1)) if m else -1


def load_pool_models(pool_dir, device):
    """Load all checkpoints from pool_dir as frozen opponent models.
    Skips any that fail to load (e.g., from a previous OBS_SIZE)."""
    if not os.path.exists(pool_dir):
        return []
    paths = sorted(
        glob.glob(os.path.join(pool_dir, "*.pt")),
        key=lambda p: step_from_filename(p) if step_from_filename(p) >= 0 else float("inf"),
    )
    models = []
    skipped = 0
    for p in paths:
        try:
            m = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            models.append(m)
        except Exception as e:
            skipped += 1
            print(f"  Skipping incompatible checkpoint {p}: {type(e).__name__}")
    if skipped > 0:
        print(f"  ({skipped} checkpoint(s) skipped — likely from a different OBS_SIZE.)")
    return models


def assign_envs(num_envs, num_pool, num_current, device):
    """For each env, pick num_current random slots for the learning agent.
    Remaining slots get random pool model indices (ignored if pool empty)."""
    rand = torch.rand(num_envs, NUM_PLAYERS, device=device)
    sorted_slots = rand.argsort(dim=1)
    is_current = torch.zeros(num_envs, NUM_PLAYERS, dtype=torch.bool, device=device)
    cur_slots = sorted_slots[:, :num_current]
    is_current.scatter_(1, cur_slots, True)
    pool_idx = torch.randint(0, max(num_pool, 1), (num_envs, NUM_PLAYERS), device=device)
    return is_current, pool_idx


def compute_next_action_step(acting_player, env_done, num_envs, num_steps, device):
    """For each (step, env), returns the next step at which the SAME slot acts in the
    SAME episode, or -1 if no further action this episode."""
    last_p = torch.full((NUM_PLAYERS, num_envs), -1, dtype=torch.long, device=device)
    next_action_step = torch.full((num_steps, num_envs), -1, dtype=torch.long, device=device)
    env_idx = torch.arange(num_envs, device=device)
    neg_one_full = torch.full_like(last_p, -1)

    for t in reversed(range(num_steps)):
        p_t = acting_player[t]
        ed_mask = env_done[t].unsqueeze(0)
        nxt = torch.where(ed_mask.expand_as(last_p), neg_one_full, last_p)
        next_action_step[t] = nxt[p_t, env_idx]
        last_p = nxt
        last_p[p_t, env_idx] = t

    return next_action_step


def compute_per_player_gae(rewards, values, player_done, next_action_step,
                           acting_player, boundary_values,
                           num_envs, num_steps, device, gamma, gae_lambda):
    """Per-slot GAE. Bridges across other slots' actions using next_action_step.
    For population training: pool-slot positions get values=0; their advantages are
    computed but never used (we filter to current-policy positions before PPO).
    Within a slot's trajectory, the slot is either always-current or always-pool
    for the whole episode, so cross-contamination cannot happen across slots."""
    advantages = torch.zeros_like(rewards)
    env_idx = torch.arange(num_envs, device=device)
    zero = torch.zeros(num_envs, device=device)

    for t in reversed(range(num_steps)):
        nat = next_action_step[t]
        has_next = (nat >= 0)
        nat_clamped = nat.clamp(min=0)

        v_at_nat = values[nat_clamped, env_idx]
        adv_at_nat = advantages[nat_clamped, env_idx]
        v_at_boundary = boundary_values[acting_player[t], env_idx]

        v_next = torch.where(has_next, v_at_nat, v_at_boundary)
        adv_next = torch.where(has_next, adv_at_nat, zero)

        nonterm = (~player_done[t]).float()
        delta = rewards[t] + gamma * v_next * nonterm - values[t]
        advantages[t] = delta + gamma * gae_lambda * adv_next * nonterm

    return advantages


def train():
    if not os.path.exists(POOL_DIR):
        os.makedirs(POOL_DIR)

    run_name = f"golf_ppo_pop{NUM_CURRENT_SLOTS}of{NUM_PLAYERS}_obs{OBS_SIZE}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_MIXED_PRECISION and (device.type == "cuda")

    print(f"Training on device: {device}")
    print(f"Mixed precision: {'enabled (fp16 autocast)' if use_amp else 'disabled'}")
    print(f"Simplified env: {USE_SIMPLIFIED_ENV} (skip arrange/flip stages)")
    print(f"OBS_SIZE: {OBS_SIZE}")

    # --- POOL ---
    print(f"Loading pool models from {POOL_DIR}/...")
    pool_models = load_pool_models(POOL_DIR, device)
    print(f"Loaded {len(pool_models)} pool models")
    if len(pool_models) == 0:
        print("Pool empty: starting in pure self-play mode (all slots = current policy).")
        print(f"After first save in {SAVE_EVERY_N_UPDATES} updates, switches to "
              f"{NUM_CURRENT_SLOTS}-current + {NUM_PLAYERS - NUM_CURRENT_SLOTS}-pool.")
    else:
        print(f"Population mode: {NUM_CURRENT_SLOTS}-current + "
              f"{NUM_PLAYERS - NUM_CURRENT_SLOTS}-pool per env.")

    effective_num_current = NUM_CURRENT_SLOTS if len(pool_models) > 0 else NUM_PLAYERS

    print(f"NUM_ENVS={NUM_ENVS} | NUM_STEPS={NUM_STEPS} | MINIBATCH_SIZE={MINIBATCH_SIZE}")

    env = VectorGolfEnv(NUM_ENVS, device=device, simplified=USE_SIMPLIFIED_ENV)
    agent = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)

    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        print(f"Resuming from {MODEL_PATH}...")
        agent.load_state_dict(torch.load(MODEL_PATH))

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    scaler = GradScaler("cuda", enabled=use_amp)

    is_current_assignment, pool_idx_assignment = assign_envs(
        NUM_ENVS, len(pool_models), effective_num_current, device
    )

    obs_buf = torch.zeros((NUM_STEPS, NUM_ENVS, OBS_SIZE), device=device)
    actions_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values_buf = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    invalid_masks_buf = torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SIZE), dtype=torch.bool, device=device)
    acting_player_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.long, device=device)
    player_done_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.bool, device=device)
    env_done_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.bool, device=device)
    is_current_buf = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.bool, device=device)

    last_player_done_step = torch.full((NUM_ENVS, NUM_PLAYERS), -1, dtype=torch.long, device=device)

    global_step = 0
    cumulative_games = 0
    start_time = time.time()
    next_obs = env.get_obs()

    env_idx_all = torch.arange(NUM_ENVS, device=device)
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE

    for update in range(1, num_updates + 1):
        agent.eval()

        epoch_winner_scores = []
        epoch_avg_scores = []
        rollout_terminations = 0
        rollout_completed_games = 0

        # === ROLLOUT ===
        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs_buf[step] = next_obs

            with torch.no_grad():
                masks = env.get_action_masks()
                invalid_masks_buf[step] = masks

                acting_slot = env.current_player_idx
                acting_is_current = is_current_assignment.gather(
                    1, acting_slot.unsqueeze(1)
                ).squeeze(1)
                acting_pool_idx = pool_idx_assignment.gather(
                    1, acting_slot.unsqueeze(1)
                ).squeeze(1)

                actions = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
                logprobs = torch.zeros(NUM_ENVS, device=device)
                values = torch.zeros(NUM_ENVS, device=device)

                if acting_is_current.any():
                    cur_envs = acting_is_current.nonzero().squeeze(1)
                    with autocast(device_type="cuda", enabled=use_amp):
                        a, lp, _, v = agent.get_action(
                            next_obs[cur_envs], action_masks=masks[cur_envs]
                        )
                    actions[cur_envs] = a
                    logprobs[cur_envs] = lp.float()
                    values[cur_envs] = v.flatten().float()

                if len(pool_models) > 0 and (~acting_is_current).any():
                    pool_acting = ~acting_is_current
                    for m_idx in range(len(pool_models)):
                        this_mask = pool_acting & (acting_pool_idx == m_idx)
                        if not this_mask.any():
                            continue
                        env_idxs = this_mask.nonzero().squeeze(1)
                        with autocast(device_type="cuda", enabled=use_amp):
                            a, _, _, _ = pool_models[m_idx].get_action(
                                next_obs[env_idxs], action_masks=masks[env_idxs]
                            )
                        actions[env_idxs] = a

            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            values_buf[step] = values
            is_current_buf[step] = acting_is_current

            (
                next_obs,
                _reward_unused,
                env_done,
                player_done,
                acting_player,
                info,
            ) = env.step(actions)

            env_done_buf[step] = env_done
            player_done_buf[step] = player_done
            acting_player_buf[step] = acting_player

            if player_done.any():
                pd_envs = player_done.nonzero().squeeze(1)
                last_player_done_step[pd_envs, acting_player[pd_envs]] = step
                rollout_terminations += pd_envs.numel()

            if "all_scores" in info:
                all_scores = info["all_scores"]
                done_env_ids = info["done_env_ids"]
                K = done_env_ids.numel()
                rollout_completed_games += K

                min_scores = all_scores.min(dim=1, keepdim=True).values
                is_min = (all_scores == min_scores)
                num_min = is_min.sum(dim=1, keepdim=True)
                is_solo_winner = is_min & (num_min == 1)
                is_tied_winner = is_min & (num_min > 1)

                ones = torch.ones((K, NUM_PLAYERS), device=device)
                zeros = torch.zeros_like(ones)
                neg_ones = -ones
                per_player_rewards = torch.where(
                    is_solo_winner, ones,
                    torch.where(is_tied_winner, zeros, neg_ones),
                )

                final_steps = last_player_done_step[done_env_ids]
                done_is_current = is_current_assignment[done_env_ids]
                # ONLY attribute reward to current-policy slots (pool-slot rewards are unused)
                valid = (final_steps >= 0) & done_is_current

                if valid.any():
                    valid_steps = final_steps[valid]
                    env_ids_expanded = done_env_ids.unsqueeze(1).expand(-1, NUM_PLAYERS)
                    valid_envs = env_ids_expanded[valid]
                    valid_rewards = per_player_rewards[valid]
                    rewards_buf[valid_steps, valid_envs] = valid_rewards

                new_is_cur, new_pool = assign_envs(
                    K, len(pool_models), effective_num_current, device
                )
                is_current_assignment[done_env_ids] = new_is_cur
                pool_idx_assignment[done_env_ids] = new_pool

            if env_done.any():
                last_player_done_step[env_done] = -1

            if "avg_winner" in info:
                epoch_winner_scores.append(info["avg_winner"])
                epoch_avg_scores.append(info["avg_score"])

        cumulative_games += rollout_completed_games

        # === BOUNDARY VALUES ===
        with torch.no_grad():
            obs_per_player = []
            for p in range(NUM_PLAYERS):
                p_idxs = torch.full((NUM_ENVS,), p, dtype=torch.long, device=device)
                obs_per_player.append(env.get_obs(env_ids=env_idx_all, current_player_override=p_idxs))
            stacked_obs = torch.cat(obs_per_player, dim=0)
            with autocast(device_type="cuda", enabled=use_amp):
                _, stacked_values = agent(stacked_obs, action_masks=None)
            boundary_values = stacked_values.view(NUM_PLAYERS, NUM_ENVS).float()

        # === GAE ===
        with torch.no_grad():
            next_action_step = compute_next_action_step(
                acting_player_buf, env_done_buf, NUM_ENVS, NUM_STEPS, device
            )
            advantages = compute_per_player_gae(
                rewards_buf, values_buf, player_done_buf, next_action_step,
                acting_player_buf, boundary_values,
                NUM_ENVS, NUM_STEPS, device, GAMMA, GAE_LAMBDA,
            )
            returns = advantages + values_buf

        # === FILTER TO CURRENT-POLICY POSITIONS ===
        cur_flat = is_current_buf.reshape(-1)
        cur_indices = cur_flat.nonzero().squeeze(1)
        NUM_CURRENT = cur_indices.numel()

        if NUM_CURRENT == 0:
            print(f"Update {update}: no current-policy steps in rollout. Skipping.")
            continue

        b_obs = obs_buf.reshape((-1, OBS_SIZE))[cur_indices]
        b_logprobs = logprobs_buf.reshape(-1)[cur_indices]
        b_actions = actions_buf.reshape(-1)[cur_indices]
        b_advantages = advantages.reshape(-1)[cur_indices]
        b_returns = returns.reshape(-1)[cur_indices]
        b_masks = invalid_masks_buf.reshape((-1, ACTION_SIZE))[cur_indices]

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # === PPO UPDATE ===
        agent.train()
        b_inds = np.arange(NUM_CURRENT)

        last_pg_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0

        for _ in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_CURRENT, MINIBATCH_SIZE):
                mb_inds = b_inds[start:start + MINIBATCH_SIZE]

                with autocast(device_type="cuda", enabled=use_amp):
                    newlogits, newvalue = agent(b_obs[mb_inds], b_masks[mb_inds])
                    probs = torch.distributions.Categorical(logits=newlogits)
                    newlogprob = probs.log_prob(b_actions[mb_inds])
                    entropy = probs.entropy()
                    newvalue = newvalue.view(-1)

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    pg_loss1 = -b_advantages[mb_inds] * ratio
                    pg_loss2 = -b_advantages[mb_inds] * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                last_pg_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_entropy = entropy_loss.item()

        # === LOGGING ===
        fps = int(global_step / (time.time() - start_time))
        total_actions = NUM_STEPS * NUM_ENVS
        avg_actions_per_player = total_actions / max(rollout_completed_games * NUM_PLAYERS, 1)
        cum_str = (
            f"{cumulative_games / 1000:.1f}k"
            if cumulative_games < 1_000_000
            else f"{cumulative_games / 1_000_000:.2f}M"
        )
        cur_frac = NUM_CURRENT / total_actions

        print(
            f"Update {update} | FPS: {fps} | "
            f"Games: {rollout_completed_games} (cum: {cum_str}) | "
            f"Actions/player: {avg_actions_per_player:.2f} (min={MIN_ACTIONS_PER_PLAYER}) | "
            f"Pool: {len(pool_models)} | Cur frac: {cur_frac:.2f}"
        )

        writer.add_scalar("charts/fps", fps, global_step)
        writer.add_scalar("charts/games_per_rollout", rollout_completed_games, global_step)
        writer.add_scalar("charts/cumulative_games", cumulative_games, global_step)
        writer.add_scalar("charts/avg_actions_per_player", avg_actions_per_player, global_step)
        writer.add_scalar("charts/terminations_per_rollout", rollout_terminations, global_step)
        writer.add_scalar("charts/value_loss", last_v_loss, global_step)
        writer.add_scalar("charts/policy_loss", last_pg_loss, global_step)
        writer.add_scalar("charts/entropy", last_entropy, global_step)
        writer.add_scalar("charts/pool_size", len(pool_models), global_step)
        writer.add_scalar("charts/current_action_fraction", cur_frac, global_step)
        writer.add_scalar("charts/current_action_count", NUM_CURRENT, global_step)

        if len(epoch_winner_scores) > 0:
            avg_win = float(np.mean(epoch_winner_scores))
            avg_per_player = float(np.mean(epoch_avg_scores))
            avg_total_per_game = avg_per_player * NUM_PLAYERS
            writer.add_scalar("game/avg_winner_score", avg_win, global_step)
            writer.add_scalar("game/avg_per_player_score", avg_per_player, global_step)
            writer.add_scalar("game/avg_total_per_game", avg_total_per_game, global_step)
            print(
                f"   -> Avg Winner ~{avg_win:.2f} | "
                f"Avg per player ~{avg_per_player:.2f} | "
                f"Avg total per game ~{avg_total_per_game:.1f}"
            )

        if update % SAVE_EVERY_N_UPDATES == 0:
            torch.save(agent.state_dict(), MODEL_PATH)
            pool_name = os.path.join(POOL_DIR, f"model_{global_step}.pt")
            torch.save(agent.state_dict(), pool_name)

            new_model = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
            new_model.load_state_dict(agent.state_dict())
            new_model.eval()
            for param in new_model.parameters():
                param.requires_grad = False
            pool_models.append(new_model)

            effective_num_current = NUM_CURRENT_SLOTS if len(pool_models) > 0 else NUM_PLAYERS

            print(f"  -> Saved {pool_name} | pool size now: {len(pool_models)}")

    writer.close()


if __name__ == "__main__":
    train()
