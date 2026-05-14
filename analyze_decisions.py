"""Analyze the network's decision-making confidence by logging top-1/top-2 win-prob gaps.

Run:
    python analyze_decisions.py [path_to_model.pt] [num_games]

Defaults: latest_model.pt, 1000 games.

Plays games at epsilon=0 and aggregates the gap between the best and second-best
action's predicted win probability, broken down by stage. A small gap means the
network is roughly indifferent between the top choices (lookahead is picking near
noise); a large gap means it has clear conviction.

Interpretation guide (rough):
  median PLAY_DRAW gap < 0.5%  → value function is at MC-label noise floor
  median PLAY_DRAW gap 0.5-3%  → some real signal but lookahead resolution is mixed
  median PLAY_DRAW gap > 3%    → value function has meaningful conviction;
                                  if score is still plateaued, the bottleneck is
                                  decision quality (depth/symmetries), not value learning.
"""

import sys
import time
from collections import defaultdict
import torch

from src.consts import (
    OBS_SIZE, NUM_PLAYERS,
    STAGE_ARRANGE, STAGE_FLIP1, STAGE_FLIP2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD,
)
from src.model import ValueNet
from src.vector_env import VectorGolfEnv
from src.decision import make_decisions


NUM_ENVS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE_NAMES = {
    STAGE_ARRANGE: "ARRANGE",
    STAGE_FLIP1: "FLIP1",
    STAGE_FLIP2: "FLIP2",
    STAGE_PLAY_DRAW: "PLAY_DRAW",
    STAGE_PLAY_DISCARD: "PLAY_DISC",
}
STAGE_ORDER = [STAGE_ARRANGE, STAGE_FLIP1, STAGE_FLIP2, STAGE_PLAY_DRAW, STAGE_PLAY_DISCARD]


def percentile(t, q):
    if t.numel() == 0:
        return float('nan')
    return torch.quantile(t, q / 100.0).item()


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "latest_model.pt"
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"Loading {model_path}")
    network = ValueNet(OBS_SIZE, num_players=NUM_PLAYERS).to(DEVICE)
    network.load_state_dict(torch.load(model_path, map_location=DEVICE))
    network.eval()

    env = VectorGolfEnv(NUM_ENVS, device=DEVICE)
    gap_by_stage = defaultdict(list)
    games_completed = 0
    start = time.time()

    while games_completed < num_games:
        actions, diag = make_decisions(env, network, epsilon=0.0, return_diagnostics=True)
        stages = diag["stage"]
        gap = diag["gap"]
        for stage_val in STAGE_ORDER:
            mask = (stages == stage_val)
            if mask.any():
                gap_by_stage[stage_val].append(gap[mask].cpu())

        _next_obs, _dones, _acting, info = env.step(actions)
        if "all_scores" in info:
            games_completed += info["done_env_ids"].numel()

    elapsed = time.time() - start

    print()
    print(f"Games: {games_completed:,}, elapsed: {elapsed:.1f}s")
    print()
    print("Top-1 vs top-2 win-prob gap per stage. Single-option ARRANGE cases get gap=0.")
    print()
    print(f"{'Stage':<11} {'Count':>10} {'Mean':>9} {'p10':>9} {'p25':>9} {'Median':>9} {'p75':>9} {'p90':>9} {'p99':>9}")
    print("-" * 90)

    for stage_val in STAGE_ORDER:
        if not gap_by_stage[stage_val]:
            continue
        gaps = torch.cat(gap_by_stage[stage_val])
        n = gaps.numel()
        if n == 0:
            continue
        mean = gaps.mean().item()
        p10 = percentile(gaps, 10)
        p25 = percentile(gaps, 25)
        med = percentile(gaps, 50)
        p75 = percentile(gaps, 75)
        p90 = percentile(gaps, 90)
        p99 = percentile(gaps, 99)
        print(f"{STAGE_NAMES[stage_val]:<11} {n:>10,} "
              f"{mean:>9.4f} {p10:>9.4f} {p25:>9.4f} {med:>9.4f} "
              f"{p75:>9.4f} {p90:>9.4f} {p99:>9.4f}")

    # Also: fraction of decisions where gap < 0.005, 0.01, 0.03 (rough thresholds)
    print()
    print("Fraction of decisions with very small gap:")
    print(f"{'Stage':<11} {'gap<0.005':>11} {'gap<0.01':>10} {'gap<0.03':>10}")
    print("-" * 50)
    for stage_val in STAGE_ORDER:
        if not gap_by_stage[stage_val]:
            continue
        gaps = torch.cat(gap_by_stage[stage_val])
        n = gaps.numel()
        if n == 0:
            continue
        f_005 = (gaps < 0.005).float().mean().item()
        f_01 = (gaps < 0.01).float().mean().item()
        f_03 = (gaps < 0.03).float().mean().item()
        print(f"{STAGE_NAMES[stage_val]:<11} {f_005:>11.3f} {f_01:>10.3f} {f_03:>10.3f}")


if __name__ == "__main__":
    main()
