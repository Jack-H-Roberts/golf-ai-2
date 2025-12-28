import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class GolfNet(nn.Module):
    def __init__(self, obs_size, action_size, hidden_dims=[1024, 512, 256]):
        super(GolfNet, self).__init__()
        
        # --- SHARED FEATURE EXTRACTOR ---
        # We use a shared trunk to process the game state
        layers = []
        input_dim = obs_size
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)) # Normalization helps stability
            layers.append(nn.ReLU())
            input_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # --- ACTOR HEAD (Policy) ---
        # Outputs logits for the 10 possible actions
        self.actor_head = nn.Linear(hidden_dims[-1], action_size)
        
        # --- CRITIC HEAD (Value) ---
        # Outputs a single scalar: "How many points do I expect to get?"
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, action_masks=None):
        """
        x: [Batch, 741] Observation Tensor
        action_masks: [Batch, 10] Boolean Tensor (True = Invalid/Illegal move)
        """
        features = self.feature_extractor(x)
        
        # 1. Calculate Value (Critic)
        value = self.critic_head(features)
        
        # 2. Calculate Action Logits (Actor)
        logits = self.actor_head(features)
        
        # 3. Apply Action Masking
        # We set the logits of invalid actions to negative infinity
        # so the probability becomes 0 after Softmax.
        if action_masks is not None:
            # Create a very small number
            HugeNeg = -1e9
            logits = torch.where(action_masks, torch.tensor(HugeNeg, device=x.device), logits)
            
        return logits, value

    def get_action(self, x, action_masks=None, deterministic=False):
        logits, value = self(x, action_masks)
        probs = Categorical(logits=logits)
        
        if deterministic:
            # For evaluation: Pick the absolute highest probability
            action = torch.argmax(logits, dim=1)
        else:
            # For training: Sample from the distribution
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value