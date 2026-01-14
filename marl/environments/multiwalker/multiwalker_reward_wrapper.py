from collections import deque
import numpy as np


class MAPPOTeamRewardWrapper:
    """
    MAPPO + GDPO-like Team Reward Wrapper for MultiWalker
    con SOFT CURRICULUM basato sul COMPORTAMENTO.

    Curriculum:
    - curriculum ∈ [0,1] (EMA smoothed)
    - 0   → focus STABILITY
    - 1   → focus LOCOMOTION

    Principi:
    - CTDE-safe
    - reward decoupled
    - nessun hard-coding per iterazioni
    - nessuno switch discreto
    """

    def __init__(
        self,
        env,

        # windows
        progress_window: int = 30,
        misalign_window: int = 30,

        # scales
        progress_scale: float = 1.0,
        misalign_scale: float = 0.2,

        # curriculum
        target_dx: float = 0.05,
        curriculum_tau: float = 0.02,
        forward_bias_value: float = 0.008,

        # stagnation
        stagnation_eps: float = 0.002,
        stagnation_penalty: float = -0.03,
        stagnation_warmup: int = 100,

        # events
        fall_penalty: float = -0.3,
        success_bonus: float = 5.0,
        failure_penalty: float = -5.0,
    ):
        self.env = env
        self.agents = env.agents

        # windows
        self.progress_window = progress_window
        self.misalign_window = misalign_window

        # scales
        self.progress_scale = progress_scale
        self.misalign_scale = misalign_scale

        # curriculum params
        self.target_dx = target_dx
        self.curriculum_tau = curriculum_tau
        self.forward_bias_value = forward_bias_value

        # penalties
        self.stagnation_eps = stagnation_eps
        self.stagnation_penalty = stagnation_penalty
        self.stagnation_warmup = stagnation_warmup
        self.fall_penalty = fall_penalty
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty

        # buffers
        self._progress_buffer = deque(maxlen=progress_window)
        self._misalign_buffer = deque(maxlen=misalign_window)

        # internal state
        self._last_package_x = None
        self._step_count = 0

        # SOFT CURRICULUM STATE
        self.curriculum = 0.0

    # --------------------------------------------------
    # Proxy
    # --------------------------------------------------
    def __getattr__(self, name):
        return getattr(self.env, name)

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------
    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)

        self.agents = list(obs.keys())
        self._last_package_x = self._extract_package_x(obs)
        self._progress_buffer.clear()
        self._misalign_buffer.clear()
        self._step_count = 0
        self.curriculum = 0.0

        return obs

    # --------------------------------------------------
    # Step
    # --------------------------------------------------
    def step(self, actions):
        obs, _, terminations, truncations, infos = self.env.step(actions)
        self._step_count += 1

        # ==================================================
        # BASIC SIGNALS
        # ==================================================
        pkg_x = self._extract_package_x(obs)
        delta_x = 0.0
        if pkg_x is not None and self._last_package_x is not None:
            delta_x = pkg_x - self._last_package_x
        self._last_package_x = pkg_x

        self._progress_buffer.append(delta_x)
        mean_dx = np.mean(self._progress_buffer)

        fallen = any(
            infos.get(a, {}).get("fallen", False) for a in self.agents
        )

        alive = len(self.agents) - sum(
            1 for a in self.agents if infos.get(a, {}).get("fallen", False)
        )
        alive_ratio = alive / max(1, len(self.agents))

        # ==================================================
        # SOFT CURRICULUM UPDATE (EMA)
        # ==================================================
        movement_score = np.clip(mean_dx / self.target_dx, 0.0, 1.0)
        target_curriculum = 0.7 * alive_ratio + 0.3 * movement_score

        self.curriculum += self.curriculum_tau * (
            target_curriculum - self.curriculum
        )

        # ==================================================
        # 1. PROGRESS (gated by curriculum)
        # ==================================================
        progress_reward = (
            mean_dx * self.progress_scale * self.curriculum
        )
        progress_reward = np.clip(progress_reward, -0.2, 0.2)

        # ==================================================
        # 2. FORWARD BIAS (softly enabled)
        # ==================================================
        forward_bias = (
            self.forward_bias_value
            if delta_x > 0
            else 0.0
        ) * self.curriculum

        # ==================================================
        # 3. MISALIGNMENT (GDPO-style)
        # ==================================================
        xs = self._extract_walkers_x(obs)
        misalign_raw = np.std(xs) if xs else 0.0
        self._misalign_buffer.append(misalign_raw)

        misalign_weight = 0.3 + 0.7 * self.curriculum
        misalign_reward = -abs(
            self._normalized_window_reward(self._misalign_buffer)
        ) * self.misalign_scale * misalign_weight

        # ==================================================
        # 4. STAGNATION (only when curriculum high)
        # ==================================================
        stagnation_reward = 0.0
        if (
            self._step_count > self.stagnation_warmup
            and abs(mean_dx) < self.stagnation_eps
        ):
            stagnation_reward = self.stagnation_penalty * self.curriculum

        # ==================================================
        # 5. FALL
        # ==================================================
        fall_reward = self.fall_penalty if fallen else 0.0

        # ==================================================
        # 6. TERMINAL
        # ==================================================
        done = (
            (all(terminations.values()) if terminations else False)
            or (all(truncations.values()) if truncations else False)
        )

        terminal_reward = 0.0
        if done:
            terminal_reward = (
                self.success_bonus
                if pkg_x is not None and pkg_x > 0.99
                else self.failure_penalty
            )

        # ==================================================
        # FINAL TEAM REWARD
        # ==================================================
        team_reward = (
            progress_reward
            + forward_bias
            + misalign_reward
            + stagnation_reward
            + fall_reward
            + terminal_reward
        )

        team_reward = float(np.clip(team_reward, -1.0, 1.0))
        rewards = {a: team_reward for a in self.agents}

        # ==================================================
        # DEBUG / FRONTEND
        # ==================================================
        infos["_team"] = {
            "curriculum": float(self.curriculum),
            "alive_ratio": float(alive_ratio),
            "delta_x": float(delta_x),
            "mean_dx": float(mean_dx),
            "progress": float(progress_reward),
            "forward_bias": float(forward_bias),
            "misalign": float(misalign_reward),
            "stagnation": float(stagnation_reward),
            "fallen": bool(fallen),
        }

        return obs, rewards, terminations, truncations, infos

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _normalized_window_reward(self, buffer: deque) -> float:
        if len(buffer) < 2:
            return 0.0
        arr = np.asarray(buffer)
        return float((arr[-1] - arr.mean()) / (arr.std() + 1e-6))

    def _extract_package_x(self, obs):
        if obs is None:
            return 0.0
        xs = [o[28] for o in obs.values() if o is not None and len(o) > 28]
        return float(np.mean(xs)) if xs else 0.0

    def _extract_walkers_x(self, obs):
        xs = []
        for a in self.agents:
            o = obs.get(a)
            if o is not None and len(o) > 28:
                xs.append(o[28])
        return xs
