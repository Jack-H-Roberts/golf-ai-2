import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Value network for Golf with two heads on a shared trunk.

    Inputs an OBS_SIZE-dim state vector. Returns:
      - win_logits: [B, 5] logits over players (apply softmax for win-prob).
      - score_pred: [B, 5] predicted *normalized* final scores per player.
                    Multiply by SCORE_SCALE in train.py to recover raw scores.

    The score head is auxiliary: the win-prob head owns decision-making, but the
    score-regression loss flows back through the shared trunk and provides a
    cleaner gradient than win-classification alone, accelerating convergence on
    subtle state-quality differences.

    Hidden architecture: [1024, 512, 256] with LayerNorm + ReLU.
    """

    def __init__(self, obs_size, num_players=5, hidden_dims=(1024, 512, 256)):
        super().__init__()
        layers = []
        in_dim = obs_size
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.win_head = nn.Linear(in_dim, num_players)
        self.score_head = nn.Linear(in_dim, num_players)

    def forward(self, x):
        """Returns (win_logits, score_pred), both shape [B, 5]."""
        h = self.trunk(x)
        return self.win_head(h), self.score_head(h)

    @torch.no_grad()
    def predict_probs(self, x):
        """Returns softmax win probabilities of shape [B, 5]. Score head ignored."""
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)
