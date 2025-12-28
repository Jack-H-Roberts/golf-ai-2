import torch
from src.vector_env import VectorGolfEnv

def test_env():
    print("Initializing Vector Environment (CUDA)...")
    num_envs = 100 
    env = VectorGolfEnv(num_envs=num_envs, device="cuda")
    
    print("Resetting...")
    obs = env.reset()
    
    assert obs.shape == (num_envs, 741), f"Bad Obs Shape: {obs.shape}"
    print(f"Observation Shape Verified: {obs.shape}")
    
    print("Running Steps with Valid Masks...")
    for step in range(200):
        # 1. Get Valid Moves (True = Invalid, False = Valid)
        masks = env.get_action_masks()
        
        # 2. Sample Random VALID Actions
        # We create logits where invalid moves are -Infinity
        logits = torch.zeros((num_envs, 10), device="cuda")
        logits[masks] = -1e9 
        
        # Sample from this distribution ensures we never pick an invalid move
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        
        # 3. Step
        next_obs, rewards, dones, info = env.step(actions)
        
        # 4. Assertions
        assert not torch.isnan(next_obs).any(), "NaN detected in Observations!"
        assert not torch.isnan(rewards).any(), "NaN detected in Rewards!"
        
        if step % 20 == 0:
            mean_score = env.scores.mean().item()
            mean_reward = rewards.mean().item()
            print(f"Step {step}: Mean Score {mean_score:.2f} | Mean Reward {mean_reward:.4f}")
            
        if dones.any():
            # If a game finishes, reset is handled automatically by the env
            pass

    print("\nSUCCESS: Vector Env ran 200 steps without crashing.")

if __name__ == "__main__":
    test_env()