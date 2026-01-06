# marl/shared/networks/centralized_critic.py
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


def ortho_init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO.
    Input: concatenated observations of all agents.
    Output: scalar value V(s_global).
    """

    def __init__(
        self,
        global_obs_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        h1, h2 = hidden_sizes

        self.net = nn.Sequential(
            ortho_init(nn.Linear(global_obs_dim, h1), gain=np.sqrt(2)),
            nn.Tanh(),
            ortho_init(nn.Linear(h1, h2), gain=np.sqrt(2)),
            nn.Tanh(),
            ortho_init(nn.Linear(h2, 1), gain=1.0),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        global_obs: [B, global_obs_dim]
        returns: [B]
        """
        return self.net(global_obs).squeeze(-1)
