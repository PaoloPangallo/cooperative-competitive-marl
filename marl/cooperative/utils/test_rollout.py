# marl/cooperative/utils/test_rollout.py
import torch

from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter
from marl.shared.buffers.rollout_buffer import RolloutBuffer

import numpy as np







def main():
    env = MultiWalkerEnv(n_walkers=3)
    obs = env.reset()

    adapter = MultiWalkerAdapter(
        agents=env.agents,
        obs_dim=31,
        act_dim=4,
    )

    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    for t in range(5):
        actions_dict, logp, value, entropy = adapter.act(obs)

        obs_tensor = adapter.obs_dict_to_tensor(obs)
        actions_np = np.stack([actions_dict[a] for a in env.agents], axis=0)
        actions_tensor = torch.from_numpy(actions_np).float()

        rewards_tensor = torch.tensor(
            [0.0 for _ in env.agents], dtype=torch.float32
        )

        dones_tensor = torch.zeros(len(env.agents))

        buffer.add(
            obs=obs_tensor,
            actions=actions_tensor,
            logprobs=logp,
            values=value,
            rewards=rewards_tensor,
            dones=dones_tensor,
        )

        obs, rewards, done, infos = env.step(actions_dict)
        if done:
            break

    batch = buffer.get()

    print("Obs batch:", batch.obs.shape)
    print("Actions batch:", batch.actions.shape)
    print("Logprobs batch:", batch.logprobs.shape)
    print("Values batch:", batch.values.shape)

    env.close()
    print("Rollout buffer OK")


if __name__ == "__main__":
    main()
