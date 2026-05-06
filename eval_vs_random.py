"""Evaluate a trained checkpoint vs uniform-random opponents.

Agent plays as `agent_player` (default 0); the other 4 player slots play
uniform-random over valid actions. Reports win/tie/loss rate over `target_games`.

Defaults to simplified=True (matching the new obs encoding and current training).

Usage:
    python eval_vs_random.py
    python eval_vs_random.py --model model_pool/model_83886080.pt --games 10000
    python eval_vs_random.py --player 2
    python eval_vs_random.py --no-simplified           # for old-env checkpoints
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import math
import time

import torch
from torch.amp import autocast

from src.model import GolfNet
from src.vector_env import VectorGolfEnv
from src.consts import OBS_SIZE, ACTION_SIZE, NUM_PLAYERS


def evaluate_vs_random(
    model_path,
    num_envs=2048,
    target_games=5000,
    agent_player=0,
    use_mixed_precision=True,
    simplified=True,
    device=None,
    verbose=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_mixed_precision and (device.type == "cuda")

    if verbose:
        print(f"Evaluating: {model_path}")
        print(f"  Agent in slot {agent_player} vs uniform random in other {NUM_PLAYERS - 1} slots")
        print(f"  Device: {device} | Mixed precision: {use_amp}")
        print(f"  Env mode: {'simplified' if simplified else 'full'}")
        print(f"  NUM_ENVS={num_envs} | target games: {target_games}")
        print()

    env = VectorGolfEnv(num_envs, device=device, simplified=simplified)
    agent = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    next_obs = env.get_obs()

    games_completed = 0
    agent_wins = 0
    agent_ties = 0
    agent_losses = 0

    start_time = time.time()
    last_print = start_time

    while games_completed < target_games:
        with torch.no_grad():
            masks = env.get_action_masks()
            acting_now = env.current_player_idx

            with autocast(device_type="cuda", enabled=use_amp):
                agent_action, _, _, _ = agent.get_action(next_obs, action_masks=masks)

            random_logits = torch.zeros_like(masks, dtype=torch.float32)
            random_logits.masked_fill_(masks, float("-inf"))
            random_action = torch.distributions.Categorical(logits=random_logits).sample()

            is_agent_turn = (acting_now == agent_player)
            action = torch.where(is_agent_turn, agent_action, random_action)

        next_obs, _, env_done, _, _, info = env.step(action)

        if "all_scores" in info:
            all_scores = info["all_scores"]
            K = all_scores.shape[0]

            agent_scores = all_scores[:, agent_player]
            min_scores = all_scores.min(dim=1).values

            is_agent_min = (agent_scores == min_scores)
            num_with_min = (all_scores == min_scores.unsqueeze(1)).sum(dim=1)

            agent_solo_win = is_agent_min & (num_with_min == 1)
            agent_tie = is_agent_min & (num_with_min > 1)
            agent_loss = ~is_agent_min

            agent_wins += int(agent_solo_win.sum().item())
            agent_ties += int(agent_tie.sum().item())
            agent_losses += int(agent_loss.sum().item())
            games_completed += K

            now = time.time()
            if verbose and (now - last_print > 5.0):
                elapsed = now - start_time
                rate = games_completed / max(elapsed, 1e-6)
                wr = agent_wins / max(games_completed, 1)
                print(f"  Progress: {games_completed}/{target_games} games "
                      f"({rate:.0f}/s) | win rate so far: {wr*100:.2f}%")
                last_print = now

    total = max(games_completed, 1)
    win_rate = agent_wins / total
    tie_rate = agent_ties / total
    loss_rate = agent_losses / total

    z = 1.96
    p, n = win_rate, total
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    ci_low = max(0.0, center - half)
    ci_high = min(1.0, center + half)

    if verbose:
        elapsed = time.time() - start_time
        print()
        print(f"=== Results over {games_completed} games ({elapsed:.1f}s) ===")
        print(f"  Solo wins:  {agent_wins:>6}  ({win_rate * 100:6.2f}%)")
        print(f"  Ties:       {agent_ties:>6}  ({tie_rate * 100:6.2f}%)")
        print(f"  Losses:     {agent_losses:>6}  ({loss_rate * 100:6.2f}%)")
        print(f"  Win rate 95% CI: [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%]")
        print(f"  Random baseline: {100.0 / NUM_PLAYERS:.2f}%")
        if ci_low > 1.0 / NUM_PLAYERS:
            margin = (win_rate - 1.0 / NUM_PLAYERS) * 100
            print(f"  >> Agent BEATS random baseline by {margin:+.2f} pp (95% CI excludes baseline)")
        elif ci_high < 1.0 / NUM_PLAYERS:
            margin = (win_rate - 1.0 / NUM_PLAYERS) * 100
            print(f"  !! Agent LOSES to random baseline by {margin:+.2f} pp (anti-learning)")
        else:
            print(f"  ?? Agent indistinguishable from random at this sample size")

    return {
        "games": games_completed,
        "wins": agent_wins,
        "ties": agent_ties,
        "losses": agent_losses,
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "loss_rate": loss_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="latest_model.pt")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-simplified", action="store_true",
                        help="Use full env (only for evaluating old-format checkpoints)")
    args = parser.parse_args()

    evaluate_vs_random(
        model_path=args.model,
        num_envs=args.num_envs,
        target_games=args.games,
        agent_player=args.player,
        use_mixed_precision=not args.no_amp,
        simplified=not args.no_simplified,
    )
