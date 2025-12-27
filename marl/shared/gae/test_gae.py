# marl/shared/gae/test_gae.py
import torch
from marl.shared.gae.gae import compute_gae


def main():
    # fake rollout of 5 steps * 3 agents = 15
    rewards = torch.tensor(
        [-0.1] * 15, dtype=torch.float32
    )
    values = torch.linspace(0.0, 1.0, 15)
    dones = torch.zeros(15)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.99,
        lam=0.95,
    )

    print("Advantages shape:", advantages.shape)
    print("Returns shape:", returns.shape)
    print("Adv mean/std:", advantages.mean().item(), advantages.std().item())

    # sanity checks
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape

    print("GAE test OK")


if __name__ == "__main__":
    main()
