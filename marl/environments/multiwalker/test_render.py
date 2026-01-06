from pettingzoo.sisl import multiwalker_v9
import time

env = multiwalker_v9.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)
    env.render()
    time.sleep(0.02)

env.close()

