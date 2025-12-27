# marl/environments/multiwalker/env.py

from pettingzoo.sisl import multiwalker_v9
import numpy as np


class MultiWalkerEnv:
    """
    Thin wrapper around PettingZoo MultiWalker (parallel API).

    - Handles reset / step
    - Exposes agent list
    - Returns numpy arrays (no torch here)
    """

    def __init__(
        self,
        n_walkers=3,
        shared_reward=True,
        terminate_on_fall=True,
        remove_on_fall=True,
        terrain_length=200,
        max_cycles=500,
        render_mode=None,
        seed=None,
    ):
        self.env = multiwalker_v9.parallel_env(
            n_walkers=n_walkers,
            shared_reward=shared_reward,
            terminate_on_fall=terminate_on_fall,
            remove_on_fall=remove_on_fall,
            terrain_length=terrain_length,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )

        self.seed = seed
        self.agents = None
        self.obs_spaces = None
        self.act_spaces = None

    def reset(self):
        observations, infos = self.env.reset(seed=self.seed)

        self.agents = list(observations.keys())
        self.obs_spaces = {
            agent: self.env.observation_space(agent)
            for agent in self.agents
        }
        self.act_spaces = {
            agent: self.env.action_space(agent)
            for agent in self.agents
        }

        return observations

    def step(self, actions):
        """
        actions: dict(agent -> action)
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        done = all(terminations.values()) or all(truncations.values())

        return observations, rewards, done, infos

    def close(self):
        self.env.close()
