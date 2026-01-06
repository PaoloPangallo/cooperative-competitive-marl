import torch
from marl.shared.gae.gae import compute_gae
from marl.shared.networks.centralized_critic import CentralizedCritic

def main():
    T = 5
    n_agents = 3
    global_obs_dim = 93

    critic = CentralizedCritic(global_obs_dim)

    global_obs = torch.randn(T * n_agents, global_obs_dim)
    rewards = torch.randn(T * n_agents)
    dones = torch.zeros(T * n_agents)

    with torch.no_grad():
        values = critic(global_obs)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=0.99,
        lam=0.95,
    )

    print("Advantages:", advantages.shape)
    print("Returns:", returns.shape)

    assert advantages.shape == returns.shape
    print("GAE MAPPO OK")

if __name__ == "__main__":
    main()
