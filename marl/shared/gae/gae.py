# marl/shared/gae/gae.py
from typing import Tuple
import torch


def compute_gae(
    rewards: torch.Tensor,      # [N]
    values: torch.Tensor,       # [N]
    dones: torch.Tensor,        # [N]
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Inputs are flattened over (time * agents).
    We assume the rollout is time-major concatenated.

    Returns:
        advantages: [N]
        returns:    [N]
    """
    assert rewards.ndim == 1
    assert values.ndim == 1
    assert dones.ndim == 1

    N = rewards.size(0)
    advantages = torch.zeros(N, device=rewards.device)

    last_adv = 0.0
    for t in reversed(range(N)):
        if t == N - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_adv = delta + gamma * lam * next_non_terminal * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns
