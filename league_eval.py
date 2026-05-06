"""League-style evaluation comparing all checkpoint models against each other.

Each parallel env has 5 different models randomly assigned to its 5 player slots
from the model pool. Over many games, every model plays many games against varied
opponent mixes. Outputs a leaderboard with win rates and 95% CIs, plus a
chronological view to detect cyclic dynamics in self-play.

Defaults to simplified=True (matching the new obs encoding and current training).
Old checkpoints from a different OBS_SIZE will be skipped automatically.

Usage:
    python league_eval.py
    python league_eval.py --pool model_pool --games 20000
    python league_eval.py --include-latest
    python league_eval.py --no-simplified           # for old-env checkpoints
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import glob
import math
import re
import time

import torch
from torch.amp import autocast

from src.model import GolfNet
from src.vector_env import VectorGolfEnv
from src.consts import OBS_SIZE, ACTION_SIZE, NUM_PLAYERS


def step_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"model_(\d+)\.pt", name)
    return int(m.group(1)) if m else -1


def load_compatible_models(model_paths, device):
    """Load models, skipping any that fail (e.g. wrong OBS_SIZE)."""
    models = []
    valid_paths = []
    for path in model_paths:
        try:
            m = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
            valid_paths.append(path)
        except Exception as e:
            print(f"  Skipping {path}: {type(e).__name__}")
    return models, valid_paths


def run_league(model_paths, num_games=20000, num_envs=2048,
               use_amp=True, simplified=True, device=None, verbose=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and (device.type == "cuda")

    if verbose:
        print(f"Loading {len(model_paths)} models...")

    models, model_paths = load_compatible_models(model_paths, device)
    NUM_MODELS = len(models)

    if NUM_MODELS < 2:
        raise RuntimeError(f"Need at least 2 compatible models for a league; got {NUM_MODELS}")

    if verbose:
        print(f"Loaded {NUM_MODELS} compatible models")
        print(f"Running league: {num_games} games, {num_envs} parallel envs")
        print(f"Env mode: {'simplified' if simplified else 'full'}")
        print()

    env = VectorGolfEnv(num_envs, device=device, simplified=simplified)

    assignments = torch.randint(0, NUM_MODELS, (num_envs, NUM_PLAYERS), device=device)

    wins = torch.zeros(NUM_MODELS, device=device, dtype=torch.float32)
    ties = torch.zeros(NUM_MODELS, device=device, dtype=torch.float32)
    games = torch.zeros(NUM_MODELS, device=device, dtype=torch.float32)

    next_obs = env.get_obs()
    games_completed = 0
    start_time = time.time()
    last_print = start_time

    while games_completed < num_games:
        with torch.no_grad():
            masks = env.get_action_masks()
            acting = env.current_player_idx
            acting_model = assignments.gather(1, acting.unsqueeze(1)).squeeze(1)

            actions = torch.zeros(num_envs, dtype=torch.long, device=device)
            for m_idx in range(NUM_MODELS):
                this_mask = (acting_model == m_idx)
                if not this_mask.any():
                    continue
                env_idxs = this_mask.nonzero().squeeze(1)
                with autocast(device_type="cuda", enabled=use_amp):
                    a, _, _, _ = models[m_idx].get_action(
                        next_obs[env_idxs], action_masks=masks[env_idxs]
                    )
                actions[env_idxs] = a

        next_obs, _, env_done, _, _, info = env.step(actions)

        if "all_scores" in info:
            done_ids = info["done_env_ids"]
            all_scores = info["all_scores"]
            K = done_ids.numel()

            min_scores = all_scores.min(dim=1, keepdim=True).values
            is_min = (all_scores == min_scores)
            num_min = is_min.sum(dim=1, keepdim=True)
            is_solo = is_min & (num_min == 1)
            is_tied = is_min & (num_min > 1)

            env_assignments = assignments[done_ids]

            ones_K = torch.ones(K, device=device)
            for p in range(NUM_PLAYERS):
                model_ids = env_assignments[:, p]
                games.scatter_add_(0, model_ids, ones_K)
                wins.scatter_add_(0, model_ids, is_solo[:, p].float())
                ties.scatter_add_(0, model_ids, is_tied[:, p].float())

            games_completed += K

            assignments[done_ids] = torch.randint(
                0, NUM_MODELS, (K, NUM_PLAYERS), device=device
            )

            now = time.time()
            if verbose and (now - last_print > 5.0):
                rate = games_completed / max(now - start_time, 1e-6)
                print(f"  Progress: {games_completed}/{num_games} games "
                      f"({rate:.0f}/s)")
                last_print = now

    if verbose:
        elapsed = time.time() - start_time
        print(f"  Done: {games_completed} games in {elapsed:.1f}s "
              f"({games_completed/elapsed:.0f}/s)")

    return wins.cpu().numpy(), ties.cpu().numpy(), games.cpu().numpy(), model_paths


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def report(model_paths, wins, ties, games):
    print()
    total_games = int(games.sum() / NUM_PLAYERS)
    print(f"=== League Results: {total_games} games, {len(model_paths)} models ===")
    print(f"Pool-equilibrium baseline (all models equal skill): {100/NUM_PLAYERS:.2f}%")
    print()

    rows = []
    for i, path in enumerate(model_paths):
        n = int(games[i])
        w = int(wins[i])
        t = int(ties[i])
        if n == 0:
            continue
        wr = w / n
        tr = t / n
        lo, hi = wilson_ci(wr, n)
        rows.append({
            "path": path, "step": step_from_filename(path),
            "n": n, "w": w, "t": t,
            "wr": wr, "tr": tr,
            "ci_low": lo, "ci_high": hi,
        })

    print("--- Ranked by win rate ---")
    print(f"{'Rank':<5}{'Model':<30}{'Step':>14}{'Games':>8}{'Wins':>7}{'Ties':>7}"
          f"{'Win%':>9}{'CI':>22}")
    print("-" * 102)
    by_wr = sorted(rows, key=lambda r: -r["wr"])
    for rank, r in enumerate(by_wr, 1):
        name = os.path.basename(r["path"])
        ci = f"[{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}]"
        step_str = f"{r['step']:,}" if r['step'] >= 0 else "n/a"
        print(f"{rank:<5}{name:<30}{step_str:>14}"
              f"{r['n']:>8}{r['w']:>7}{r['t']:>7}{r['wr']*100:>8.2f}%{ci:>22}")

    chronological = [r for r in rows if r["step"] >= 0]
    if chronological:
        print()
        print("--- Chronological (by training step) ---")
        chronological.sort(key=lambda r: r["step"])
        print(f"{'Step':>14}{'Win%':>10}{'CI':>22}")
        print("-" * 50)
        prev_wr = None
        regressions = 0
        for r in chronological:
            ci = f"[{r['ci_low']*100:.1f}, {r['ci_high']*100:.1f}]"
            arrow = ""
            if prev_wr is not None:
                if r["wr"] > prev_wr:
                    arrow = " ↑"
                elif r["wr"] < prev_wr:
                    arrow = " ↓"
                    regressions += 1
            print(f"{r['step']:>14,}{r['wr']*100:>9.2f}%{ci:>22}{arrow}")
            prev_wr = r["wr"]

        print()
        n_steps = len(chronological)
        first, last = chronological[0], chronological[-1]
        delta = (last["wr"] - first["wr"]) * 100
        print(f"First → Last: {first['wr']*100:.2f}% → {last['wr']*100:.2f}% "
              f"(Δ = {delta:+.2f} pp)")
        print(f"Step-to-step regressions: {regressions} / {n_steps - 1}")
        if regressions == 0:
            print("Strictly monotonic improvement → real accumulating skill.")
        elif regressions <= (n_steps - 1) * 0.2:
            print("Mostly monotonic → real skill growth with some noise.")
        else:
            print("Frequent regressions → likely cyclic dynamics; pure self-play "
                  "may have saturated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=str, default="model_pool")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--include-latest", action="store_true")
    parser.add_argument("--no-simplified", action="store_true",
                        help="Use full env (only for evaluating old-format checkpoints)")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.pool, "*.pt")),
        key=lambda p: step_from_filename(p) if step_from_filename(p) >= 0 else float("inf"),
    )
    if args.include_latest and os.path.exists("latest_model.pt"):
        paths.append("latest_model.pt")

    if len(paths) < 2:
        print(f"Need at least 2 models in {args.pool}/ (found {len(paths)})")
        exit(1)

    print(f"Found {len(paths)} checkpoint files")

    wins, ties, games, valid_paths = run_league(
        paths,
        num_games=args.games,
        num_envs=args.num_envs,
        use_amp=not args.no_amp,
        simplified=not args.no_simplified,
    )

    report(valid_paths, wins, ties, games)
