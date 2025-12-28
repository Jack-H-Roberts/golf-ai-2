import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
import glob

from src.model import GolfNet
from src.vector_env import VectorGolfEnv
from src.consts import *

# --- HYPERPARAMETERS ---
NUM_ENVS = 4096          
LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 2_000_000_000 
NUM_STEPS = 128          
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 4096    
NUM_EPOCHS = 4           
GAMMA = 0.99             
GAE_LAMBDA = 0.95        
CLIP_COEF = 0.2          
ENT_COEF = 0.02          
VF_COEF = 0.5            
MAX_GRAD_NORM = 0.5      

# --- FLAGS ---
LOAD_MODEL = False
MODEL_PATH = "latest_model.pt"
POOL_DIR = "model_pool"

def train():
    if not os.path.exists(POOL_DIR):
        os.makedirs(POOL_DIR)

    run_name = f"golf_ppo_single_round_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = VectorGolfEnv(NUM_ENVS, device=device)
    agent = GolfNet(OBS_SIZE, ACTION_SIZE).to(device)
    
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        print(f"Resuming from {MODEL_PATH}...")
        agent.load_state_dict(torch.load(MODEL_PATH))
    
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, OBS_SIZE), device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    invalid_masks = torch.zeros((NUM_STEPS, NUM_ENVS, ACTION_SIZE), dtype=torch.bool, device=device)

    global_step = 0
    start_time = time.time()
    
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE

    for update in range(1, num_updates + 1):
        agent.eval() 
        
        # --- NEW: Metric containers for this epoch ---
        epoch_winner_scores = []
        epoch_avg_scores = []

        for step in range(NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                masks = env.get_action_masks()
                invalid_masks[step] = masks
                action, logprob, _, value = agent.get_action(next_obs, action_masks=masks)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, next_done, info = env.step(action)
            rewards[step] = reward
            
            # --- NEW: Capture Stats from Info ---
            if "avg_winner" in info:
                epoch_winner_scores.append(info["avg_winner"])
                epoch_avg_scores.append(info["avg_score"])

        # GAE
        with torch.no_grad():
            next_masks = env.get_action_masks()
            _, next_value = agent(next_obs, next_masks)
            next_value = next_value.reshape(1, -1)
            
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # Train
        agent.train()
        b_obs = obs.reshape((-1, OBS_SIZE))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = invalid_masks.reshape((-1, ACTION_SIZE))

        b_inds = np.arange(BATCH_SIZE)
        
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newvalue = agent(b_obs[mb_inds], b_masks[mb_inds])
                newlogits, _ = agent(b_obs[mb_inds], b_masks[mb_inds]) 
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
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # Logging
        fps = int(global_step / (time.time() - start_time))
        print(f"Update {update} | FPS: {fps} | Mean Reward: {rewards.mean().item():.4f}")
        
        writer.add_scalar("charts/reward", rewards.mean().item(), global_step)
        writer.add_scalar("charts/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/fps", fps, global_step)
        
        # --- NEW: Log Game Scores ---
        if len(epoch_winner_scores) > 0:
            avg_win = np.mean(epoch_winner_scores)
            avg_tot = np.mean(epoch_avg_scores)
            writer.add_scalar("game/avg_winner_score", avg_win, global_step)
            writer.add_scalar("game/avg_total_score", avg_tot, global_step)
            print(f"   -> Game Stats: Avg Winner ~{avg_win:.1f} pts | Avg ~{avg_tot:.1f} pts")
        
        if update % 20 == 0:
            torch.save(agent.state_dict(), MODEL_PATH)
            # Save to pool for history
            pool_name = os.path.join(POOL_DIR, f"model_{global_step}.pt")
            torch.save(agent.state_dict(), pool_name)
            print(f"Saved checkpoint to {pool_name}")

    writer.close()

if __name__ == "__main__":
    train()