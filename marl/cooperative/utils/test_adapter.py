# marl/cooperative/utils/test_adapter.py
from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter


def main():
    env = MultiWalkerEnv(n_walkers=3)
    obs = env.reset()

    adapter = MultiWalkerAdapter(
        agents=env.agents,
        obs_dim=31,
        act_dim=4,
        device="cpu",
    )

    actions, logprob, value, entropy = adapter.act(obs)

    print("Actions keys:", actions.keys())
    print("Action shape:", actions[env.agents[0]].shape)
    print("Logprob shape:", logprob.shape)
    print("Value shape:", value.shape)
    print("Entropy shape:", entropy.shape)

    # step env once
    obs, rewards, done, infos = env.step(actions)
    print("Rewards:", rewards)

    env.close()
    print("Adapter test OK")


if __name__ == "__main__":
    main()
