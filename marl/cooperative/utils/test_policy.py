# marl/cooperative/utils/test_policy.py
import numpy as np
import torch

from marl.shared.networks.mlp_actor_critic import MLPActorCritic


def main():
    obs_dim = 31
    act_dim = 4

    net = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    net.eval()

    # batch finto di 3 agenti
    obs = torch.tensor(np.random.randn(3, obs_dim), dtype=torch.float32)

    out = net.act(obs)

    print("action shape:", out.action.shape)  # (3,4)
    print("logprob shape:", out.logprob.shape)  # (3,)
    print("value shape:", out.value.shape)  # (3,)
    print("entropy shape:", out.entropy.shape)  # (3,)
    print("action min/max:", out.action.min().item(), out.action.max().item())


if __name__ == "__main__":
    main()
