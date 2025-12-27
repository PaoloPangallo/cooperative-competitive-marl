# marl/cooperative/ippo/train_ippo_multiwalker.py
import numpy as np
import torch

from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter
from marl.shared.buffers.rollout_buffer import RolloutBuffer
from marl.shared.gae.gae import compute_gae
from marl.shared.ppo.ppo_update import ppo_update


def train_ippo(
    n_iters: int = 50,
    rollout_steps: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    env = MultiWalkerEnv(n_walkers=3)
    obs = env.reset()

    adapter = MultiWalkerAdapter(
        agents=env.agents,
        obs_dim=31,
        act_dim=4,
    )

    policy = adapter.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    for it in range(1, n_iters + 1):
        buffer.reset()
        ep_rewards = []

        for _ in range(rollout_steps):
            actions_dict, logp, values, entropy = adapter.act(obs)

            obs_tensor = adapter.obs_dict_to_tensor(obs)

            actions_np = np.stack([actions_dict[a] for a in env.agents], axis=0)
            actions_tensor = torch.from_numpy(actions_np).float()

            next_obs, rewards, done, infos = env.step(actions_dict)

            rewards_tensor = torch.tensor(
                [rewards[a] for a in env.agents],
                dtype=torch.float32,
            )
            dones_tensor = torch.tensor(
                [float(done) for _ in env.agents]
            )
            rewards_tensor = rewards_tensor / 10.0

            buffer.add(
                obs=obs_tensor,
                actions=actions_tensor,
                logprobs=logp,
                values=values,
                rewards=rewards_tensor,
                dones=dones_tensor,
            )

            ep_rewards.append(rewards_tensor.mean().item())

            obs = next_obs
            if done:
                obs = env.reset()

        batch = buffer.get()

        advantages, returns = compute_gae(
            rewards=batch.rewards,
            values=batch.values,
            dones=batch.dones,
            gamma=gamma,
            lam=lam,
        )

        stats = ppo_update(
            policy=policy,
            optimizer=optimizer,
            obs=batch.obs,
            actions=batch.actions,
            old_logprobs=batch.logprobs,
            returns=returns,
            advantages=advantages,
            epochs=5,
            batch_size=128,
        )

        print(
            f"[Iter {it:03d}] "
            f"Reward: {np.mean(ep_rewards):.3f} | "
            f"Loss: {stats['loss']:.3f} | "
            f"Entropy: {stats['entropy']:.3f}"
        )

    env.close()


if __name__ == "__main__":
    train_ippo()
