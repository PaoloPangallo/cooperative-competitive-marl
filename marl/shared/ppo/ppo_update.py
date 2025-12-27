# marl/shared/ppo/ppo_update.py
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from marl.shared.networks.mlp_actor_critic import MLPActorCritic


def ppo_update(
    policy: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,          # [N, obs_dim]
    actions: torch.Tensor,      # [N, act_dim]
    old_logprobs: torch.Tensor, # [N]
    returns: torch.Tensor,      # [N]
    advantages: torch.Tensor,   # [N]
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 10,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Perform PPO update (IPPO).
    Returns mean losses for logging.
    """

    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(
        obs,
        actions,
        old_logprobs,
        returns,
        advantages,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    policy.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    n_updates = 0

    for _ in range(epochs):
        for batch in loader:
            b_obs, b_actions, b_old_logp, b_returns, b_adv = batch

            mean, values = policy(b_obs)
            std = torch.exp(policy.log_std).expand_as(mean)

            dist = torch.distributions.Normal(mean, std)

            # inverse tanh for logprob
            eps = 1e-6
            raw_action = torch.atanh(torch.clamp(b_actions, -1 + eps, 1 - eps))
            logp = dist.log_prob(raw_action).sum(-1)
            logp -= torch.log(1 - b_actions.pow(2) + eps).sum(-1)

            ratio = torch.exp(logp - b_old_logp)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred_clipped = b_returns + (values - b_returns).clamp(-0.2, 0.2)
            value_loss_unclipped = (values - b_returns).pow(2)
            value_loss_clipped = (value_pred_clipped - b_returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy = dist.entropy().sum(-1).mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            n_updates += 1

    return {
        "loss": total_loss / n_updates,
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }
