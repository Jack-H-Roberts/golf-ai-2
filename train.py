"""Training loop for the value-network architecture.

Pure self-play with chance-node expectimax + ε-greedy. MC targets:
each visited state in a completed game gets the eventual winner one-hot AND
the final score vector as labels, supervising the win-prob and score heads
respectively.

LOGGING (column format):
   elapsed       games       fps     winner    other     dec/p     loss
   00:01:23     12,500      350      14.2     32.1     11.3     0.0450

All numbers (except elapsed and games) are 100-game moving averages.

TENSORBOARD CHARTS:
   train/loss            (combined)
   train/loss_winp       (win-prob CE only)
   train/loss_score      (score MSE on normalized targets)
   metrics/winner_score
   metrics/non_winner_score
   metrics/decisions_per_player
   metrics/fps
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.consts import OBS_SIZE, VALUE_OUTPUT_DIM, NUM_PLAYERS
from src.model import ValueNet
from src.vector_env import VectorGolfEnv
from src.decision import make_decisions


# --- HYPERPARAMETERS ---
NUM_ENVS = 64
TOTAL_ENV_STEPS = 50_000_000

# Replay buffer
BUFFER_SIZE = 200_000
MIN_BUFFER_BEFORE_TRAIN = 10_000

# Training
TRAIN_BATCH_SIZE = 4096
TRAIN_STEPS_PER_UPDATE = 2  # was 4 — reduces buffer-replay rate
LEARNING_RATE = 3e-4

# Score-head auxiliary loss
# Theoretical bounds on a single player's final score: min = -12 (three 8s + a
# 0-scoring column + …), max = 90 (worst-case face-up). Normalize to [0, 1].
SCORE_MIN = -12.0
SCORE_MAX = 90.0
SCORE_RANGE = SCORE_MAX - SCORE_MIN  # 102.0
# With normalized targets in [0, 1], MSE values are small (~0.01 once trained).
# Bumped from 1.0 so the score loss provides comparable gradient magnitude to CE.
SCORE_LOSS_WEIGHT = 10.0

# ε-greedy schedule
EPS_START = 0.30
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000

# Save / log — time-based now
LOG_INTERVAL_SECONDS = 60     # ~1 minute logs
SAVE_INTERVAL_SECONDS = 600   # ~10 minute checkpoints
WINDOW_GAMES = 100
MODEL_PATH = "latest_model.pt"
POOL_DIR = "model_pool"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Turing (RTX 20-series) has FP16 Tensor Cores but not BF16. Use float16
# with GradScaler. For Ampere+ (RTX 30-series and later) bfloat16 is preferred
# and you can drop the scaler.
USE_AMP = (DEVICE.type == 'cuda')
AMP_DTYPE = torch.float16


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


class ReplayBuffer:
    """FIFO ring buffer of (obs, target) where target is [winner_one_hot (5), scores (5)]."""

    def __init__(self, capacity, obs_size, num_players, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros((capacity, obs_size), dtype=torch.float32, device=device)
        # 10 = 5 winner one-hot + 5 final scores (raw, will be normalized at loss-time)
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
    """Per-env list of saved obs that haven't been labeled yet (game not done)."""

    MAX_PER_ENV = 400

    def __init__(self, num_envs, obs_size, device):
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.device = device
        self.buffer = torch.zeros((num_envs, self.MAX_PER_ENV, obs_size), dtype=torch.float32, device=device)
        self.counts = torch.zeros((num_envs,), dtype=torch.long, device=device)

    def append(self, env_ids, obs):
        K = env_ids.numel()
        if K == 0:
            return
        idx = self.counts[env_ids]
        valid = (idx < self.MAX_PER_ENV)
        if valid.any():
            valid_envs = env_ids[valid]
            valid_idx = idx[valid]
            valid_obs = obs[valid]
            self.buffer[valid_envs, valid_idx] = valid_obs
            self.counts[valid_envs] = valid_idx + 1

    def flush(self, env_ids, targets):
        """targets: [K, 2*NUM_PLAYERS] = [winner_one_hot (5), scores (5)]."""
        K = env_ids.numel()
        target_dim = NUM_PLAYERS * 2
        if K == 0:
            return (torch.zeros((0, self.obs_size), device=self.device),
                    torch.zeros((0, target_dim), device=self.device))

        counts = self.counts[env_ids]
        out_obs = []
        out_tgt = []
        for i in range(K):
            n = counts[i].item()
            if n == 0:
                continue
            eid = env_ids[i]
            out_obs.append(self.buffer[eid, :n])
            out_tgt.append(targets[i].unsqueeze(0).expand(n, -1))
        self.counts[env_ids] = 0

        if not out_obs:
            return (torch.zeros((0, self.obs_size), device=self.device),
                    torch.zeros((0, target_dim), device=self.device))
        return torch.cat(out_obs, dim=0), torch.cat(out_tgt, dim=0)


def train():
    if not os.path.exists(POOL_DIR):
        os.makedirs(POOL_DIR)

    run_name = f"golf_value_net_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    print(f"Device: {DEVICE} | NUM_ENVS={NUM_ENVS} | OBS_SIZE={OBS_SIZE} | AMP={AMP_DTYPE if USE_AMP else 'off'}")
    print(f"Buffer={BUFFER_SIZE} (min {MIN_BUFFER_BEFORE_TRAIN} before train) | "
          f"batch={TRAIN_BATCH_SIZE} | grad_steps_per_iter={TRAIN_STEPS_PER_UPDATE}")
    print(f"Score head: range=[{SCORE_MIN}, {SCORE_MAX}], weight={SCORE_LOSS_WEIGHT}")
    print()
    header = f"{'elapsed':>10}  {'games':>9}  {'fps':>6}  {'winner':>7}  {'other':>7}  {'dec/p':>6}  {'loss':>7}"
    print(header)
    print("-" * len(header))

    env = VectorGolfEnv(NUM_ENVS, device=DEVICE)
    network = ValueNet(OBS_SIZE, num_players=VALUE_OUTPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    # GradScaler for FP16. No-op for BF16 / FP32. The new API takes the device.
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and AMP_DTYPE == torch.float16))
    buffer = ReplayBuffer(BUFFER_SIZE, OBS_SIZE, NUM_PLAYERS, DEVICE)
    pending = PendingStates(NUM_ENVS, OBS_SIZE, DEVICE)

    win_q = deque(maxlen=WINDOW_GAMES)
    other_q = deque(maxlen=WINDOW_GAMES)
    dec_q = deque(maxlen=WINDOW_GAMES)
    loss_q = deque(maxlen=WINDOW_GAMES)

    global_step = 0
    iter_count = 0
    update_count = 0
    games_completed = 0
    start_time = time.time()
    last_log_time = start_time
    last_save_time = start_time
    last_log_step = 0

    next_obs = env.get_obs()

    while global_step < TOTAL_ENV_STEPS:
        # Save the obs the network is about to evaluate, for every env.
        all_ids = torch.arange(NUM_ENVS, device=DEVICE)
        pending.append(all_ids, next_obs)

        eps = epsilon_at(global_step)
        network.eval()
        actions = make_decisions(env, network, epsilon=eps)
        next_obs, env_done, _acting, info = env.step(actions)
        global_step += NUM_ENVS
        iter_count += 1

        if "all_scores" in info:
            done_ids = info["done_env_ids"]
            winners = info["winners_one_hot"]                  # [K, 5]
            scores = info["all_scores"]                        # [K, 5]
            decs = info["decisions_per_player"]
            K = done_ids.numel()
            games_completed += K

            # Combined target: [winner_one_hot, raw_scores].
            combined = torch.cat([winners, scores], dim=1)     # [K, 10]
            obs_to_add, tgt_to_add = pending.flush(done_ids, combined)
            buffer.add_batch(obs_to_add, tgt_to_add)

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

        # ----- Train -----
        if buffer.size >= MIN_BUFFER_BEFORE_TRAIN:
            network.train()
            iter_loss = 0.0
            iter_winp_loss = 0.0
            iter_score_loss = 0.0
            for _ in range(TRAIN_STEPS_PER_UPDATE):
                obs_b, tgt_b = buffer.sample(TRAIN_BATCH_SIZE)
                winner_t = tgt_b[:, :NUM_PLAYERS]                                     # [B, 5]
                # Normalize raw scores [SCORE_MIN, SCORE_MAX] → [0, 1].
                score_t = (tgt_b[:, NUM_PLAYERS:] - SCORE_MIN) / SCORE_RANGE          # [B, 5]

                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    logits, score_pred = network(obs_b)

                # Loss in FP32.
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                winp_loss = -(winner_t * log_probs).sum(dim=1).mean()
                score_loss = F.mse_loss(score_pred.float(), score_t)
                loss = winp_loss + SCORE_LOSS_WEIGHT * score_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                iter_loss += loss.item()
                iter_winp_loss += winp_loss.item()
                iter_score_loss += score_loss.item()
            iter_loss /= TRAIN_STEPS_PER_UPDATE
            iter_winp_loss /= TRAIN_STEPS_PER_UPDATE
            iter_score_loss /= TRAIN_STEPS_PER_UPDATE
            loss_q.append(iter_loss)
            update_count += 1

        # ----- Time-based save -----
        now = time.time()
        if buffer.size >= MIN_BUFFER_BEFORE_TRAIN and (now - last_save_time) >= SAVE_INTERVAL_SECONDS:
            torch.save(network.state_dict(), MODEL_PATH)
            pool_path = os.path.join(POOL_DIR, f"model_{global_step}.pt")
            torch.save(network.state_dict(), pool_path)
            last_save_time = now

        # ----- Time-based log -----
        if (now - last_log_time) >= LOG_INTERVAL_SECONDS:
            elapsed = now - start_time
            since_last = now - last_log_time
            recent_steps = global_step - last_log_step
            recent_fps = int(recent_steps / max(since_last, 1e-6))
            last_log_step = global_step
            last_log_time = now

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
                f"{avg_or_dash(loss_q, '{:>7.4f}')}"
            )
            print(line, flush=True)

            writer.add_scalar("metrics/fps", recent_fps, global_step)
            if win_q:
                writer.add_scalar("metrics/winner_score", sum(win_q) / len(win_q), global_step)
                writer.add_scalar("metrics/non_winner_score", sum(other_q) / len(other_q), global_step)
                writer.add_scalar("metrics/decisions_per_player", sum(dec_q) / len(dec_q), global_step)
            if loss_q:
                writer.add_scalar("train/loss", sum(loss_q) / len(loss_q), global_step)
                writer.add_scalar("train/loss_winp", iter_winp_loss, global_step)
                writer.add_scalar("train/loss_score", iter_score_loss, global_step)

    writer.close()


if __name__ == "__main__":
    train()
