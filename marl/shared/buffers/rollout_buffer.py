# marl/shared/buffers/rollout_buffer.py
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor        # [N, obs_dim]
    actions: torch.Tensor    # [N, act_dim]
    logprobs: torch.Tensor   # [N]
    values: torch.Tensor     # [N]
    rewards: torch.Tensor    # [N]
    dones: torch.Tensor      # [N]


class RolloutBuffer:
    """
    Minimal rollout buffer for PPO / IPPO.
    Stores data flattened over (time * agents).
    """

    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim


        # declare attributes for static analyzers
        self.obs: List[torch.Tensor]
        self.actions: List[torch.Tensor]
        self.logprobs: List[torch.Tensor]
        self.values: List[torch.Tensor]
        self.rewards: List[torch.Tensor]
        self.dones: List[torch.Tensor]

        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(
        self,
        obs: torch.Tensor,        # [n_agents, obs_dim]
        actions: torch.Tensor,    # [n_agents, act_dim]
        logprobs: torch.Tensor,   # [n_agents]
        values: torch.Tensor,     # [n_agents]
        rewards: torch.Tensor,    # [n_agents]
        dones: torch.Tensor,      # [n_agents]
    ):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def get(self) -> RolloutBatch:
        obs = torch.cat(self.obs, dim=0)
        actions = torch.cat(self.actions, dim=0)
        logprobs = torch.cat(self.logprobs, dim=0)
        values = torch.cat(self.values, dim=0)
        rewards = torch.cat(self.rewards, dim=0)
        dones = torch.cat(self.dones, dim=0)

        return RolloutBatch(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            values=values,
            rewards=rewards,
            dones=dones,
        )

