"""Evaluate the trained network by replacing one of the 5 self-play seats with
a random-action opponent. If the network is making intelligent decisions, the
4 network seats should win in aggregate at much greater than 80% (i.e., random
should win significantly less than 20%).

Run:
    python eval_vs_random.py [path_to_model.pt] [num_games]

Defaults: latest_model.pt, 1000 games.
"""

import sys
import time
import torch

from src.consts import OBS_SIZE, NUM_PLAYERS, ACTION_SIZE, STAGE_PLAY_DRAW
from src.model import ValueNet
from src.vector_env import VectorGolfEnv
from src.decision import make_decisions


NUM_ENVS = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def make_decisions_mixed(env, network, random_seat):
    """Network for all seats except `random_seat`, which uses uniform-random actions.
    Note: we run the full lookahead for all envs (some compute is wasted on envs whose
    acting player is the random seat) but it keeps the code simple and parallel.
    """
    actions_net = make_decisions(env, network, epsilon=0.0)
    actions_rand = torch.randint(0, ACTION_SIZE, (env.num_envs,), device=env.device)
    use_random = (env.current_player_idx == random_seat)
    return torch.where(use_random, actions_rand, actions_net)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "latest_model.pt"
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print(f"Loading {model_path}")
    network = ValueNet(OBS_SIZE, num_players=NUM_PLAYERS).to(DEVICE)
    network.load_state_dict(torch.load(model_path, map_location=DEVICE))
    network.eval()

    env = VectorGolfEnv(NUM_ENVS, device=DEVICE)

    # Track wins per seat (random vs network seats), aggregated across many games.
    # We rotate which seat is random across games to avoid first-player bias.
    random_wins = 0
    network_wins = 0
    total_games = 0
    score_random_when_random = []   # random's own score when random was the random seat
    score_network_when_random = []  # network seats' avg score when random was the random seat
    decisions_per_game = []

    start = time.time()
    # Each pass: pick a random seat to be "the random opponent" (rotates per env).
    # Track over completed games.
    random_seat_per_env = torch.randint(0, NUM_PLAYERS, (NUM_ENVS,), device=DEVICE)

    while total_games < num_games:
        # We can't easily vary random_seat across envs in make_decisions_mixed because
        # the network call is shared. Use a single random seat for the batch and re-roll
        # when games complete. Simpler.
        # Pick the most common seat for "this batch" — actually just use seat 0 throughout
        # but vary by re-rolling on game-done. Simplest: keep a per-env random_seat tensor
        # and gate by it.
        # Since make_decisions_mixed compares acting==random_seat per env, and we have
        # a per-env seat tensor, this generalizes naturally.
        actions_net = make_decisions(env, network, epsilon=0.0)
        actions_rand = torch.randint(0, ACTION_SIZE, (NUM_ENVS,), device=DEVICE)
        use_random = (env.current_player_idx == random_seat_per_env)
        actions = torch.where(use_random, actions_rand, actions_net)

        next_obs, dones, _acting, info = env.step(actions)

        if "all_scores" in info:
            done_ids = info["done_env_ids"]
            scores = info["all_scores"]            # [K, 5]
            winners = info["winners_one_hot"]      # [K, 5] (rows sum to 1; ties split)
            decs = info["decisions_per_player"]    # [K]
            K = done_ids.numel()

            random_seats_for_done = random_seat_per_env[done_ids]                          # [K]
            seat_idx = random_seats_for_done.unsqueeze(1)                                  # [K, 1]
            random_seat_winshare = winners.gather(1, seat_idx).squeeze(1)                  # [K]
            network_winshare = 1.0 - random_seat_winshare                                  # [K]
            random_wins += random_seat_winshare.sum().item()
            network_wins += network_winshare.sum().item()
            total_games += K

            random_scores = scores.gather(1, seat_idx).squeeze(1)                          # [K]
            other_mask = torch.ones_like(scores, dtype=torch.bool)
            other_mask.scatter_(1, seat_idx, False)
            network_scores_mean = scores[other_mask].view(K, NUM_PLAYERS - 1).mean(dim=1)  # [K]

            score_random_when_random.extend(random_scores.tolist())
            score_network_when_random.extend(network_scores_mean.tolist())
            decisions_per_game.extend(decs.tolist())

            # Reroll random seat for completed envs
            random_seat_per_env[done_ids] = torch.randint(0, NUM_PLAYERS, (K,), device=DEVICE)

    elapsed = time.time() - start

    # Wins are fractional (ties split). A fully-uniform draw would give random 20% winshare.
    random_winrate = random_wins / total_games
    network_winrate_per_seat = network_wins / total_games / (NUM_PLAYERS - 1)

    avg_random_score = sum(score_random_when_random) / len(score_random_when_random)
    avg_network_score = sum(score_network_when_random) / len(score_network_when_random)
    avg_dec = sum(decisions_per_game) / len(decisions_per_game)

    print()
    print("=" * 60)
    print(f"Games:                           {total_games:>10,}")
    print(f"Elapsed:                         {elapsed:>10.1f}s")
    print()
    print("WIN RATE  (uniform baseline = 20%; lower for random = better network)")
    print(f"  random seat:                   {random_winrate*100:>9.2f}%")
    print(f"  network seat (avg of 4):       {network_winrate_per_seat*100:>9.2f}%")
    print()
    print("AVG SCORE  (lower = better)")
    print(f"  random seat:                   {avg_random_score:>10.2f}")
    print(f"  network seat (avg of 4):       {avg_network_score:>10.2f}")
    print()
    print(f"Decisions / player / game:       {avg_dec:>10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
