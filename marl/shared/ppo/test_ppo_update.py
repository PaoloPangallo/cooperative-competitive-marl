# marl/shared/ppo/test_ppo_update.py
import torch

from marl.shared.networks.mlp_actor_critic import MLPActorCritic
from marl.shared.ppo.ppo_update import ppo_update


def main():
    obs_dim = 31
    act_dim = 4
    N = 128

    policy = MLPActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    obs = torch.randn(N, obs_dim)
    actions = torch.tanh(torch.randn(N, act_dim))
    old_logprobs = torch.randn(N)
    returns = torch.randn(N)
    advantages = torch.randn(N)

    stats = ppo_update(
        policy=policy,
        optimizer=optimizer,
        obs=obs,
        actions=actions,
        old_logprobs=old_logprobs,
        returns=returns,
        advantages=advantages,
        epochs=3,
        batch_size=32,
    )

    print("PPO update stats:")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    print("PPO update OK")


if __name__ == "__main__":
    main()
