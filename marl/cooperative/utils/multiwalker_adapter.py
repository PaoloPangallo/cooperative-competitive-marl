# marl/cooperative/utils/multiwalker_adapter.py
from typing import Dict, List, Tuple
import numpy as np
import torch

from marl.shared.networks.mlp_actor_critic import MLPActorCritic, ACOutput


class MultiWalkerAdapter:
    """
    Bridge between:
      - PettingZoo MultiWalker observations (dict[agent -> np])
      - Torch policy (batched)

    Keeps agent ordering stable.
    """

    def __init__(
        self,
        agents: List[str],
        obs_dim: int,
        act_dim: int,
        device: str = "cpu",
    ):
        self.agents = agents
        self.device = device

        self.policy = MLPActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
        ).to(device)

    def obs_dict_to_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        dict[agent -> (obs_dim,)] -> torch.Tensor [n_agents, obs_dim]
        """
        obs_batch = np.stack([obs[a] for a in self.agents], axis=0)
        return torch.tensor(obs_batch, dtype=torch.float32, device=self.device)

    def tensor_to_action_dict(
        self,
        actions: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        torch.Tensor [n_agents, act_dim] -> dict[agent -> (act_dim,)]
        """
        actions = actions.cpu().numpy()
        return {
            agent: actions[i]
            for i, agent in enumerate(self.agents)
        }

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns:
          actions_dict,
          logprob [n_agents],
          value   [n_agents],
          entropy [n_agents]
        """
        obs_tensor = self.obs_dict_to_tensor(obs)
        out: ACOutput = self.policy.act(obs_tensor)

        actions_dict = self.tensor_to_action_dict(out.action)

        return actions_dict, out.logprob, out.value, out.entropy
