import torch
from src.vector_env import VectorGolfEnv
from src.consts import OBS_SIZE


def test_env():
    print("Initializing Vector Environment...")
    num_envs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = VectorGolfEnv(num_envs=num_envs, device=device)

    print("Resetting...")
    obs = env.reset()

    assert obs.shape == (num_envs, OBS_SIZE), f"Bad obs shape: {obs.shape}, expected ({num_envs}, {OBS_SIZE})"
    print(f"Observation shape verified: {obs.shape}")

    print("Running steps with valid masks...")
    games_completed = 0
    nonzero_rewards = 0

    for step in range(500):
        masks = env.get_action_masks()

        # Sample uniformly over valid actions
        logits = torch.zeros((num_envs, 10), device=device)
        logits[masks] = -1e9
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        next_obs, rewards, env_dones, player_dones, acting_players, info = env.step(actions)

        assert not torch.isnan(next_obs).any(), "NaN in observations"
        assert not torch.isnan(rewards).any(), "NaN in rewards"
        assert next_obs.shape == (num_envs, OBS_SIZE)
        assert rewards.shape == (num_envs,)
        assert env_dones.shape == (num_envs,) and env_dones.dtype == torch.bool
        assert player_dones.shape == (num_envs,) and player_dones.dtype == torch.bool
        assert acting_players.shape == (num_envs,) and acting_players.dtype == torch.long

        # When player_done is True, reward should be nonzero (the terminal -final_score signal)
        # When player_done is False, reward should be 0
        terminal_steps = player_dones.nonzero().squeeze(-1)
        if terminal_steps.numel() > 0:
            term_rewards = rewards[terminal_steps]
            assert (term_rewards != 0).all() or (term_rewards == 0).all(), \
                "Mixed reward state at terminal steps"

        nonterminal_steps = (~player_dones).nonzero().squeeze(-1)
        if nonterminal_steps.numel() > 0:
            nonterm_rewards = rewards[nonterminal_steps]
            assert (nonterm_rewards == 0).all(), \
                f"Nonzero reward at non-terminal step: max={nonterm_rewards.abs().max().item()}"

        games_completed += int(env_dones.sum().item())
        nonzero_rewards += int((rewards != 0).sum().item())

        if step % 50 == 0:
            print(
                f"Step {step}: "
                f"games_completed={games_completed}, "
                f"player_dones_this_step={int(player_dones.sum().item())}, "
                f"nonzero_rewards_total={nonzero_rewards}"
            )

    print("\nSUCCESS: vector env ran 500 steps without crashing.")
    print(f"Total games completed: {games_completed}")
    print(f"Total nonzero rewards: {nonzero_rewards}")

    # Sanity: roughly 5 nonzero rewards per game (one per player's final action)
    if games_completed > 0:
        ratio = nonzero_rewards / games_completed
        print(f"Nonzero-rewards-per-game: {ratio:.2f} (expect ~5.0)")


if __name__ == "__main__":
    test_env()
