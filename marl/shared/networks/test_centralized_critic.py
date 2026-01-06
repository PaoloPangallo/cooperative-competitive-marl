import torch
from marl.shared.networks.centralized_critic import CentralizedCritic

def main():
    critic = CentralizedCritic(global_obs_dim=93)
    x = torch.randn(10, 93)
    v = critic(x)

    print("Value shape:", v.shape)
    assert v.shape == (10,)
    print("Centralized critic OK")

if __name__ == "__main__":
    main()
