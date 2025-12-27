from marl.environments.multiwalker.env import MultiWalkerEnv
import numpy as np

env = MultiWalkerEnv(n_walkers=3)
obs = env.reset()

print("Agents:", env.agents)
print("Obs shape:", obs[env.agents[0]].shape)

done = False
step = 0

while not done and step < 10:
    actions = {
        agent: env.act_spaces[agent].sample()
        for agent in env.agents
    }

    obs, rewards, done, infos = env.step(actions)
    print(f"Step {step}, rewards: {rewards}")
    step += 1

env.close()
print("Test OK")

