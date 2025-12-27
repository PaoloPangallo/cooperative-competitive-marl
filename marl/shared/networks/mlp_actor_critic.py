# marl/shared/networks/mlp_actor_critic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def ortho_init(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


@dataclass
class ACOutput:
    action: torch.Tensor      # [B, act_dim]
    logprob: torch.Tensor     # [B]
    value: torch.Tensor       # [B]
    entropy: torch.Tensor     # [B]


class MLPActorCritic(nn.Module):
    """
    Actor-Critic for continuous actions (Normal policy).
    - Actor outputs mean; log_std is a learned parameter vector.
    - Actions are squashed with tanh to fit [-1, 1].
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, int] = (256, 256),
        log_std_init: float = -0.5,
    ):
        super().__init__()

        h1, h2 = hidden_sizes

        self.actor = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, h1), gain=np.sqrt(2)),
            nn.Tanh(),
            ortho_init(nn.Linear(h1, h2), gain=np.sqrt(2)),
            nn.Tanh(),
            ortho_init(nn.Linear(h2, act_dim), gain=0.01),
        )

        self.critic = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, h1), gain=np.sqrt(2)),
            nn.Tanh(),
            ortho_init(nn.Linear(h1, h2), gain=np.sqrt(2)),
            nn.Tanh(),
            ortho_init(nn.Linear(h2, 1), gain=1.0),
        )

        # one log_std per action dimension (shared across states)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: [B, act_dim]
            value: [B]
        """
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return mean, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> ACOutput:
        """
        Sample action from policy.
        obs: [B, obs_dim]
        """
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)

        dist = Normal(mean, std)
        raw_action = dist.rsample()  # reparameterized sample
        squashed_action = torch.tanh(raw_action)  # [-1, 1]

        # logprob correction for tanh squashing
        # log_prob(raw) - sum log(1 - tanh(raw)^2)
        logprob_raw = dist.log_prob(raw_action).sum(-1)
        correction = torch.log(1.0 - squashed_action.pow(2) + 1e-6).sum(-1)
        logprob = logprob_raw - correction

        entropy = dist.entropy().sum(-1)

        return ACOutput(
            action=squashed_action,
            logprob=logprob,
            value=value,
            entropy=entropy,
        )
