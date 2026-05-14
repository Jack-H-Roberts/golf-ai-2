"""Training loop for the value-network architecture — Stage 3 (league play).

Stage 3 changes
===============
1. League play. Each env's 5 seats are randomly assigned to 5 roles:
     role 0: latest network, ε = 0.0   (greedy)
     role 1: latest network, ε = 0.05  (exploratory)
     role 2: ~1-hour-old snapshot, ε = 0.0
     role 3: ~5-hour-old snapshot, ε = 0.0
     role 4: earliest available snapshot, ε = 0.0
   Cold-start: missing snapshots fall back to the latest network. Seat
   assignment is re-rolled each time an env resets (per-game reshuffle).
   The training loop only saves obs from envs where the acting player is at
   role 0 or 1 (i.e., where the latest network is the next decision-maker).

2. Label rotation. PendingStates tracks the acting player at obs-save time.
   At flush, the winner one-hot and per-player score targets are ego-rotated
   to match the obs's ego frame (position 0 = acting at obs time). This was
   a latent bug in Stage 2: the obs got ego-rotated but the label didn't.

3. Refresh league networks on every save (cheap; reuses already-loaded
   networks when their paths haven't changed).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import shutil
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.consts import (
    OBS_SIZE, VALUE_OUTPUT_DIM, NUM_PLAYERS,
    OBS_OFFSET_GRAVEYARD, OBS_OFFSET_TOP_DISCARD, OBS_OFFSET_TOP_DRAW,
    OBS_OFFSET_HAND_CARDS, OBS_HAND_CARD_DIM, OBS_NUM_HAND_CARDS,
)
from src.model import ValueNet
from src.vector_env import VectorGolfEnv
from src.decision import make_decisions


# --- HYPERPARAMETERS ---
NUM_ENVS = 512
TOTAL_ENV_STEPS = 50_000_000

BUFFER_SIZE = 1_000_000
MIN_BUFFER_BEFORE_TRAIN = 20_000

TRAIN_BATCH_SIZE = 4096
TRAIN_STEPS_PER_UPDATE = 2
LR_START = 3e-4
LR_END = 3e-5

SCORE_MIN = -12.0
SCORE_MAX = 90.0
SCORE_RANGE = SCORE_MAX - SCORE_MIN
SCORE_LOSS_WEIGHT = 0

# --- LEAGUE CONFIG ---
LEAGUE_NUM_ROLES = 5
# Per-role ε. Roles 0/1 share the latest network with different ε; roles 2-4 use
# older snapshots and play greedily.
LEAGUE_ROLE_EPS = [0.0, 0.05, 0.0, 0.0, 0.0]
# Role-2/3/4 source within sorted league_pool: most recent / 5 back / earliest.
LEAGUE_5HR_OFFSET = 5
# Roles whose acted-from obs we save into the training buffer. The latest
# network (roles 0 and 1) is the one being trained; older models' POVs would
# introduce off-policy bias on the value function we care about at deploy.
LEAGUE_LATEST_ROLES = (0, 1)

EPS_START = 0.30
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000

LOG_INTERVAL_SECONDS = 60
SAVE_INTERVAL_SECONDS = 600
LEAGUE_KEEP_EVERY_N_SAVES = 6
WINDOW_GAMES = 100
MODEL_PATH = "latest_model.pt"
LEAGUE_POOL_DIR = "league_pool"
TEMP_POOL_DIR = "model_pool"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_amp_dtype():
    if DEVICE.type != 'cuda':
        return None, False
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
    except Exception:
        pass
    return torch.float16, True


AMP_DTYPE, NEEDS_SCALER = detect_amp_dtype()
USE_AMP = AMP_DTYPE is not None


def cosine_lr(step, total=TOTAL_ENV_STEPS, lr_start=LR_START, lr_end=LR_END):
    if step >= total:
        return lr_end
    frac = step / total
    cos = 0.5 * (1.0 + math.cos(math.pi * frac))
    return lr_end + (lr_start - lr_end) * cos


def epsilon_at(step):
    if step >= EPS_DECAY_STEPS:
        return EPS_END
    frac = step / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)


def format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_color_swap_indices(device):
    """Red ↔ blue permutation. Uses OBS_OFFSET_* from consts so it auto-updates."""
    idx = torch.arange(OBS_SIZE, device=device).clone()
    g = OBS_OFFSET_GRAVEYARD
    g_first = torch.arange(g, g + 13, device=device)
    g_second = torch.arange(g + 13, g + 26, device=device)
    idx[g : g + 13] = g_second
    idx[g + 13 : g + 26] = g_first

    td = OBS_OFFSET_TOP_DISCARD
    idx[td] = td + 1
    idx[td + 1] = td

    tr = OBS_OFFSET_TOP_DRAW
    idx[tr] = tr + 1
    idx[tr + 1] = tr

    hc = OBS_OFFSET_HAND_CARDS
    for i in range(OBS_NUM_HAND_CARDS):
        base = hc + i * OBS_HAND_CARD_DIM
        idx[base] = base + 1
        idx[base + 1] = base

    return idx


# ==================================================================
# REPLAY BUFFER + PENDING (with label rotation)
# ==================================================================
class ReplayBuffer:
    def __init__(self, capacity, obs_size, num_players, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_size), dtype=torch.float32, device=device)
        self.targets = torch.zeros((capacity, num_players * 2), dtype=torch.float32, device=device)
        self.size = 0
        self.ptr = 0

    def add_batch(self, obs_batch, target_batch):
        n = obs_batch.shape[0]
        if n == 0:
            return
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr:end] = obs_batch
            self.targets[self.ptr:end] = target_batch
        else:
            split = self.capacity - self.ptr
            self.obs[self.ptr:] = obs_batch[:split]
            self.targets[self.ptr:] = target_batch[:split]
            rem = n - split
            self.obs[:rem] = obs_batch[split:]
            self.targets[:rem] = target_batch[split:]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.targets[idx]


class PendingStates:
    """Per-env list of (obs, acting) waiting to be labeled. At flush, the
    deal-order winner+score target is ego-rotated to each obs's frame via
    the saved acting.
    """
    MAX_PER_ENV = 400

    def __init__(self, num_envs, obs_size, num_players, device):
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.num_players = num_players
        self.device = device
        self.obs_buffer = torch.zeros((num_envs, self.MAX_PER_ENV, obs_size), dtype=torch.float32, device=device)
        self.acting_buffer = torch.zeros((num_envs, self.MAX_PER_ENV), dtype=torch.long, device=device)
        self.counts = torch.zeros((num_envs,), dtype=torch.long, device=device)
        self._arange_p = torch.arange(num_players, device=device)

    def append(self, env_ids, obs, acting):
        """env_ids: [K] long. obs: [K, obs_size]. acting: [K] long (deal-order)."""
        K = env_ids.numel()
        if K == 0:
            return
        idx = self.counts[env_ids]
        valid = (idx < self.MAX_PER_ENV)
        if valid.any():
            valid_envs = env_ids[valid]
            valid_idx = idx[valid]
            self.obs_buffer[valid_envs, valid_idx] = obs[valid]
            self.acting_buffer[valid_envs, valid_idx] = acting[valid]
            self.counts[valid_envs] = valid_idx + 1

    def flush(self, env_ids, targets):
        """targets: [K, NUM_PLAYERS*2] = [winner_one_hot (deal), scores (deal)].
        Returns (obs_out, target_out_rotated) for all saved obs in env_ids.
        Each obs's target is ego-rotated to its own frame via acting_buffer.
        """
        K = env_ids.numel()
        target_dim = self.num_players * 2
        empty = (torch.zeros((0, self.obs_size), device=self.device),
                 torch.zeros((0, target_dim), device=self.device))
        if K == 0:
            return empty

        counts = self.counts[env_ids]
        out_obs = []
        out_tgt = []
        for i in range(K):
            n = int(counts[i].item())
            if n == 0:
                continue
            eid = env_ids[i]
            obs_slice = self.obs_buffer[eid, :n]               # [n, obs_size]
            acting_slice = self.acting_buffer[eid, :n]         # [n]
            tgt_row = targets[i]                               # [target_dim]

            # Build per-obs rotation indices: rot_idx[i, j] = (acting[i] + j) % NUM_PLAYERS.
            rot_idx = (acting_slice.unsqueeze(1) + self._arange_p.unsqueeze(0)) % self.num_players  # [n, 5]
            winners_deal = tgt_row[:self.num_players].unsqueeze(0).expand(n, -1)   # [n, 5]
            scores_deal  = tgt_row[self.num_players:].unsqueeze(0).expand(n, -1)   # [n, 5]
            winners_rot = winners_deal.gather(1, rot_idx)
            scores_rot  = scores_deal.gather(1, rot_idx)
            tgt_rot = torch.cat([winners_rot, scores_rot], dim=1)  # [n, target_dim]

            out_obs.append(obs_slice)
            out_tgt.append(tgt_rot)

        self.counts[env_ids] = 0

        if not out_obs:
            return empty
        return torch.cat(out_obs, dim=0), torch.cat(out_tgt, dim=0)


# ==================================================================
# LEAGUE MANAGEMENT
# ==================================================================
def list_league_checkpoints():
    """Returns sorted [(global_step, path), ...] for files in LEAGUE_POOL_DIR."""
    if not os.path.exists(LEAGUE_POOL_DIR):
        return []
    files = os.listdir(LEAGUE_POOL_DIR)
    out = []
    for f in files:
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                step = int(f[len("model_"):-len(".pt")])
                out.append((step, os.path.join(LEAGUE_POOL_DIR, f)))
            except ValueError:
                continue
    out.sort()
    return out


def select_league_paths_by_role():
    """Returns {2: path | None, 3: path | None, 4: path | None}.

      role 2: most recent league_pool checkpoint (~1 hour old)
      role 3: 5 league_pool checkpoints back (~5 hours old, clamped to earliest)
      role 4: earliest league_pool checkpoint
    """
    paths = list_league_checkpoints()
    n = len(paths)
    if n == 0:
        return {2: None, 3: None, 4: None}
    role3_idx = max(n - LEAGUE_5HR_OFFSET, 0)
    return {
        2: paths[-1][1],
        3: paths[role3_idx][1],
        4: paths[0][1],
    }


def load_league_network(path, device):
    """Load a checkpoint into a fresh ValueNet (eval mode, no grad)."""
    net = ValueNet(OBS_SIZE, num_players=VALUE_OUTPUT_DIM).to(device)
    state = torch.load(path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def refresh_league_networks(latest_network, current_state, device):
    """Returns updated league state dict: {2: net, 3: net, 4: net, "paths": {...}}.
    Reuses already-loaded networks when their source path hasn't changed.
    Cold-start: missing roles fall back to latest_network.
    """
    paths_for_role = select_league_paths_by_role()
    current_paths = current_state.get("paths", {}) if current_state else {}
    current_nets = current_state if current_state else {}

    # Path → network (deduped, so two roles pointing to same file share one net).
    path_to_net = {}
    unique_paths = {p for p in paths_for_role.values() if p is not None}
    for path in unique_paths:
        # Reuse if any current role already had this path.
        reused = None
        for role, cur_path in current_paths.items():
            if cur_path == path and role in current_nets:
                reused = current_nets[role]
                break
        path_to_net[path] = reused if reused is not None else load_league_network(path, device)

    out = {"paths": dict(paths_for_role)}
    for role, path in paths_for_role.items():
        out[role] = path_to_net[path] if path is not None else latest_network
    return out


def reshuffle_seat_assignment(num_envs, device):
    """Per env, a random permutation of [0..LEAGUE_NUM_ROLES). Shape [num_envs, NUM_PLAYERS].
    Entry [env, seat] = role assigned to that deal-order seat.
    """
    rand = torch.rand(num_envs, LEAGUE_NUM_ROLES, device=device)
    return rand.argsort(dim=1)


def league_make_decisions(env, role_networks, role_eps, role_for_env):
    """role_networks: list of LEAGUE_NUM_ROLES networks.
    role_eps:      list of LEAGUE_NUM_ROLES floats.
    role_for_env:  [num_envs] long, the role of the current acting player per env.

    Returns: actions [num_envs] long.

    Dispatches each role's envs to a separate make_decisions call. Roles 0 and 1
    share the latest network but have different ε, so they run as two passes.
    """
    device = env.device
    actions = torch.full((env.num_envs,), 9, dtype=torch.long, device=device)
    for role_idx in range(LEAGUE_NUM_ROLES):
        mask = (role_for_env == role_idx)
        if not mask.any():
            continue
        ids = mask.nonzero().squeeze(1)
        net_actions = make_decisions(
            env, role_networks[role_idx], epsilon=role_eps[role_idx], env_ids=ids
        )
        actions[ids] = net_actions[ids]
    return actions


# ==================================================================
# CHECKPOINT HELPERS
# ==================================================================
def save_checkpoint(path, network, optimizer, global_step, games_completed,
                    save_count, scaler=None):
    state = {
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "games_completed": games_completed,
        "save_count": save_count,
        "obs_size": OBS_SIZE,
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def maybe_resume(path, network, optimizer, scaler=None):
    if not os.path.exists(path):
        return 0, 0, 0
    print(f"Resuming from {path}")
    state = torch.load(path, map_location=DEVICE)

    saved_obs = state.get("obs_size", None) if isinstance(state, dict) else None
    if saved_obs is not None and saved_obs != OBS_SIZE:
        print(f"  WARNING: checkpoint OBS_SIZE={saved_obs} ≠ current OBS_SIZE={OBS_SIZE}. "
              f"Starting fresh.")
        return 0, 0, 0

    if not isinstance(state, dict) or "model" not in state:
        try:
            network.load_state_dict(state)
            print(f"  loaded legacy state_dict (no step/optimizer info available)")
            return 0, 0, 0
        except Exception as e:
            print(f"  WARNING: failed to load legacy state_dict ({e}); starting fresh.")
            return 0, 0, 0

    network.load_state_dict(state["model"])
    if "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            print(f"  WARNING: optimizer state failed to load ({e}); reinitializing.")
    if scaler is not None and "scaler" in state:
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"  WARNING: scaler state failed to load ({e}); reinitializing.")
    g = state.get("global_step", 0)
    gc = state.get("games_completed", 0)
    sc = state.get("save_count", 0)
    print(f"  resumed at step={g:,}, games={gc:,}, save_count={sc}")
    return g, gc, sc


def manage_pools(global_step, save_count):
    """Keep transient TEMP_POOL_DIR to the most-recent save; copy every Nth save
    permanently into LEAGUE_POOL_DIR."""
    for fname in os.listdir(TEMP_POOL_DIR):
        try:
            os.remove(os.path.join(TEMP_POOL_DIR, fname))
        except OSError:
            pass

    transient_path = os.path.join(TEMP_POOL_DIR, f"model_{global_step}.pt")
    try:
        shutil.copy2(MODEL_PATH, transient_path)
    except Exception:
        pass

    if save_count > 0 and save_count % LEAGUE_KEEP_EVERY_N_SAVES == 0:
        permanent_path = os.path.join(LEAGUE_POOL_DIR, f"model_{global_step}.pt")
        try:
            shutil.copy2(MODEL_PATH, permanent_path)
        except Exception as e:
            print(f"  WARNING: failed to save permanent checkpoint: {e}")


# ==================================================================
# MAIN
# ==================================================================
def train():
    for d in (LEAGUE_POOL_DIR, TEMP_POOL_DIR):
        if not os.path.exists(d):
            os.makedirs(d)

    run_name = f"golf_value_net_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    print(f"Device: {DEVICE} | NUM_ENVS={NUM_ENVS} | OBS_SIZE={OBS_SIZE}")
    print(f"AMP: dtype={AMP_DTYPE}, scaler={NEEDS_SCALER}")
    print(f"Buffer={BUFFER_SIZE:,} (min {MIN_BUFFER_BEFORE_TRAIN:,} before train) | "
          f"batch={TRAIN_BATCH_SIZE} (×2 with color swap) | grad_steps_per_iter={TRAIN_STEPS_PER_UPDATE}")
    print(f"LR: cosine {LR_START} → {LR_END} over {TOTAL_ENV_STEPS:,} steps")
    print(f"Score head: range=[{SCORE_MIN}, {SCORE_MAX}], weight={SCORE_LOSS_WEIGHT}")
    print(f"League: roles={LEAGUE_NUM_ROLES}, ε per role={LEAGUE_ROLE_EPS}, "
          f"save-only-when-role-in={LEAGUE_LATEST_ROLES}")
    print()
    header = (f"{'elapsed':>10}  {'games':>9}  {'fps':>6}  "
              f"{'winner':>7}  {'other':>7}  {'dec/p':>6}  {'loss':>7}  {'league':>6}")
    print(header)
    print("-" * len(header))

    env = VectorGolfEnv(NUM_ENVS, device=DEVICE)
    network = ValueNet(OBS_SIZE, num_players=VALUE_OUTPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=LR_START)
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and NEEDS_SCALER))

    global_step, games_completed, save_count = maybe_resume(
        MODEL_PATH, network, optimizer, scaler if NEEDS_SCALER else None
    )

    buffer = ReplayBuffer(BUFFER_SIZE, OBS_SIZE, NUM_PLAYERS, DEVICE)
    pending = PendingStates(NUM_ENVS, OBS_SIZE, NUM_PLAYERS, DEVICE)
    color_swap_idx = make_color_swap_indices(DEVICE)

    # League state: dict mapping role 2/3/4 → network, plus "paths" → {role: path}.
    league_state = refresh_league_networks(network, {}, DEVICE)
    role_networks = [
        network,                # role 0: latest greedy
        network,                # role 1: latest exploratory
        league_state[2],        # role 2: ~1hr old
        league_state[3],        # role 3: ~5hr old
        league_state[4],        # role 4: earliest
    ]

    seat_assignment = reshuffle_seat_assignment(NUM_ENVS, DEVICE)

    win_q = deque(maxlen=WINDOW_GAMES)
    other_q = deque(maxlen=WINDOW_GAMES)
    dec_q = deque(maxlen=WINDOW_GAMES)
    loss_q = deque(maxlen=WINDOW_GAMES)

    start_time = time.time()
    last_log_time = start_time
    last_save_time = start_time
    last_log_step = global_step

    next_obs = env.get_obs()

    while global_step < TOTAL_ENV_STEPS:
        # --- Determine role per env for the current acting player. ---
        acting_at_save = env.current_player_idx.clone()
        role_for_env = seat_assignment.gather(1, acting_at_save.unsqueeze(1)).squeeze(1)

        # --- Save obs into pending. While league_pool is empty, all roles point
        # to the latest network, so the role-based filter would just waste data;
        # save everything. Once league_pool has snapshots, restrict saves to the
        # roles that use the latest (training) network.
        if len(list_league_checkpoints()) > 0:
            latest_mask = torch.zeros_like(role_for_env, dtype=torch.bool)
            for r in LEAGUE_LATEST_ROLES:
                latest_mask = latest_mask | (role_for_env == r)
        else:
            latest_mask = torch.ones_like(role_for_env, dtype=torch.bool)
        latest_ids = latest_mask.nonzero().squeeze(1)
        if latest_ids.numel() > 0:
            pending.append(latest_ids, next_obs[latest_ids], acting_at_save[latest_ids])

        # --- Decisions via league dispatch. Each role uses its own (network, ε). ---
        network.eval()
        actions = league_make_decisions(env, role_networks, LEAGUE_ROLE_EPS, role_for_env)

        # --- Step. ---
        next_obs, env_done, _acting, info = env.step(actions)
        global_step += NUM_ENVS

        # --- Flush completed games; reshuffle their seat assignments for next game. ---
        if "all_scores" in info:
            done_ids = info["done_env_ids"]
            winners = info["winners_one_hot"]
            scores = info["all_scores"]
            decs = info["decisions_per_player"]
            K = done_ids.numel()
            games_completed += K

            combined = torch.cat([winners, scores], dim=1)
            obs_to_add, tgt_to_add = pending.flush(done_ids, combined)
            buffer.add_batch(obs_to_add, tgt_to_add)

            # Re-roll seat assignment per-env that just reset.
            seat_assignment[done_ids] = reshuffle_seat_assignment(K, DEVICE)

            # Game-summary stats (deal-order analysis is fine: we just need score
            # spread; per-seat identity doesn't matter for these aggregates).
            min_scores = scores.min(dim=1, keepdim=True).values
            is_winner = (scores == min_scores).float()
            num_winners = is_winner.sum(dim=1).clamp(min=1.0)
            sum_scores = scores.sum(dim=1)
            sum_winner_scores = (scores * is_winner).sum(dim=1)
            non_winner_avg = (sum_scores - sum_winner_scores) / (NUM_PLAYERS - num_winners).clamp(min=1.0)
            winner_avg = sum_winner_scores / num_winners

            for i in range(K):
                win_q.append(winner_avg[i].item())
                other_q.append(non_winner_avg[i].item())
                dec_q.append(decs[i].item())

        # --- Train ---
        if buffer.size >= MIN_BUFFER_BEFORE_TRAIN:
            lr_now = cosine_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            network.train()
            iter_loss = 0.0
            iter_winp_loss = 0.0
            iter_score_loss = 0.0
            for _ in range(TRAIN_STEPS_PER_UPDATE):
                obs_b, tgt_b = buffer.sample(TRAIN_BATCH_SIZE)
                # Color-swap augmentation: stack native + swapped (labels color-invariant).
                obs_swap = obs_b.index_select(1, color_swap_idx)
                obs_stacked = torch.cat([obs_b, obs_swap], dim=0)
                tgt_stacked = torch.cat([tgt_b, tgt_b], dim=0)

                winner_t = tgt_stacked[:, :NUM_PLAYERS]  # already ego-rotated at flush
                score_t = (tgt_stacked[:, NUM_PLAYERS:] - SCORE_MIN) / SCORE_RANGE  # rotated too

                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits, score_pred = network(obs_stacked)

                log_probs = torch.log_softmax(logits.float(), dim=-1)
                winp_loss = -(winner_t * log_probs).sum(dim=1).mean()
                score_loss = F.mse_loss(score_pred.float(), score_t)
                loss = winp_loss + SCORE_LOSS_WEIGHT * score_loss

                optimizer.zero_grad()
                if NEEDS_SCALER:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    optimizer.step()

                iter_loss += loss.item()
                iter_winp_loss += winp_loss.item()
                iter_score_loss += score_loss.item()

            iter_loss /= TRAIN_STEPS_PER_UPDATE
            iter_winp_loss /= TRAIN_STEPS_PER_UPDATE
            iter_score_loss /= TRAIN_STEPS_PER_UPDATE
            loss_q.append(iter_loss)

        # --- Time-based save + league refresh ---
        now = time.time()
        if buffer.size >= MIN_BUFFER_BEFORE_TRAIN and (now - last_save_time) >= SAVE_INTERVAL_SECONDS:
            save_count += 1
            save_checkpoint(MODEL_PATH, network, optimizer, global_step,
                            games_completed, save_count,
                            scaler if NEEDS_SCALER else None)
            manage_pools(global_step, save_count)
            # Reload older-snapshot networks if league_pool has new entries.
            league_state = refresh_league_networks(network, league_state, DEVICE)
            role_networks = [
                network, network,
                league_state[2], league_state[3], league_state[4],
            ]
            last_save_time = now

        # --- Time-based log ---
        if (now - last_log_time) >= LOG_INTERVAL_SECONDS:
            elapsed = now - start_time
            since_last = now - last_log_time
            recent_steps = global_step - last_log_step
            recent_fps = int(recent_steps / max(since_last, 1e-6))
            last_log_step = global_step
            last_log_time = now

            league_size = len(list_league_checkpoints())

            def avg_or_dash(q, fmt):
                if not q:
                    return "  -  "
                return fmt.format(sum(q) / len(q))

            line = (
                f"{format_elapsed(elapsed):>10}  "
                f"{games_completed:>9,}  "
                f"{recent_fps:>6}  "
                f"{avg_or_dash(win_q, '{:>7.2f}')}  "
                f"{avg_or_dash(other_q, '{:>7.2f}')}  "
                f"{avg_or_dash(dec_q, '{:>6.2f}')}  "
                f"{avg_or_dash(loss_q, '{:>7.4f}')}  "
                f"{league_size:>6}"
            )
            print(line, flush=True)

            writer.add_scalar("metrics/fps", recent_fps, global_step)
            writer.add_scalar("metrics/league_pool_size", league_size, global_step)
            if win_q:
                writer.add_scalar("metrics/winner_score", sum(win_q) / len(win_q), global_step)
                writer.add_scalar("metrics/non_winner_score", sum(other_q) / len(other_q), global_step)
                writer.add_scalar("metrics/decisions_per_player", sum(dec_q) / len(dec_q), global_step)
            if loss_q:
                writer.add_scalar("train/loss", sum(loss_q) / len(loss_q), global_step)
                writer.add_scalar("train/loss_winp", iter_winp_loss, global_step)
                writer.add_scalar("train/loss_score", iter_score_loss, global_step)
                writer.add_scalar("train/lr", cosine_lr(global_step), global_step)

    writer.close()


if __name__ == "__main__":
    train()
