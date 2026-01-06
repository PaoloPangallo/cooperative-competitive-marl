import torch
from marl.shared.buffers.rollout_buffer import RolloutBuffer

def main():
    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    for _ in range(5):
        buffer.add(
            obs=torch.randn(3, 31),
            actions=torch.randn(3, 4),
            logprobs=torch.randn(3),
            values=torch.randn(3),
            rewards=torch.randn(3),
            dones=torch.zeros(3),
            global_obs=torch.randn(1, 93),
        )

    batch = buffer.get()

    print("Obs:", batch.obs.shape)
    print("Global obs:", batch.global_obs.shape)

    assert batch.global_obs.shape == (15, 93)
    print("MAPPO buffer OK")

if __name__ == "__main__":
    main()
