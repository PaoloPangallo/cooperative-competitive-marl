from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Mapping
import numpy as np


@dataclass
class _TrackerState:
    fallen_agents: int = 0
    alive_agents: int = 0

    mean_x_raw: float = 0.0
    mean_x: float = 0.0  # smoothed
    delta_x: float = 0.0
    is_advancing: bool = False

    reward_mean: float = 0.0
    done: bool = False


class MultiWalkerStatusTracker:
    """
    Tracker "source of truth" per telemetry MultiWalker.

    Uso consigliato:
        tracker.update(obs, rewards, terminations, done)
        status = tracker.snapshot()

    Compatibilità:
        tracker.extract(...) -> fa update + snapshot
    """

    def __init__(
        self,
        n_agents: int,
        *,
        ema_alpha: float = 0.25,  # smoothing mean_x (0..1). 0 = no smoothing
        eps_advancing: float = 1e-4,  # soglia per considerare advancing
        x_index: int = 0,  # dove leggere X dentro obs[a]
    ):
        self.n_agents = int(n_agents)
        self.ema_alpha = float(ema_alpha)
        self.eps_advancing = float(eps_advancing)
        self.x_index = int(x_index)

        self._prev_mean_x: Optional[float] = None
        self._state = _TrackerState()

    # ----------------------------
    # Public API
    # ----------------------------
    def update(
        self,
        *,
        obs: Mapping[str, np.ndarray],
        rewards: Optional[Mapping[str, float]] = None,
        terminations: Optional[Mapping[str, bool]] = None,
        done: bool = False,
    ) -> None:
        """
        Aggiorna lo stato interno. Chiamala ad ogni step (o quando vuoi).
        """
        obs = obs or {}
        terminations = terminations or {}

        # ---- HEALTH: fallen/alive robusto ----
        fallen, alive = self._compute_health(obs=obs, terminations=terminations)

        # ---- PROGRESS: mean_x e delta_x robusti ----
        mean_x_raw = self._compute_mean_x(obs=obs, terminations=terminations)

        # smoothing (EMA) opzionale
        if self.ema_alpha <= 0.0:
            mean_x = mean_x_raw
        else:
            # inizializzazione morbida
            if self._prev_mean_x is None:
                mean_x = mean_x_raw
            else:
                mean_x = (1.0 - self.ema_alpha) * self._state.mean_x + self.ema_alpha * mean_x_raw

        # delta_x su mean_x (smoothed)
        if self._prev_mean_x is None:
            delta_x = 0.0
        else:
            delta_x = float(mean_x - self._prev_mean_x)

        is_advancing = bool(delta_x > self.eps_advancing)

        # ---- EPISODE: reward_mean robusto ----
        reward_mean = 0.0
        if rewards:
            try:
                reward_mean = float(np.mean([float(v) for v in rewards.values()]))
            except Exception:
                reward_mean = 0.0

        # commit state
        self._state.fallen_agents = int(fallen)
        self._state.alive_agents = int(alive)

        self._state.mean_x_raw = float(mean_x_raw)
        self._state.mean_x = float(mean_x)
        self._state.delta_x = float(delta_x)
        self._state.is_advancing = bool(is_advancing)

        self._state.reward_mean = float(reward_mean)
        self._state.done = bool(done)

        self._prev_mean_x = float(mean_x)

    def snapshot(self) -> Dict[str, Any]:
        """
        Ritorna lo stato in formato JSON-friendly, coerente col frontend.
        """
        s = self._state
        return {
            "health": {
                "fallen_agents": s.fallen_agents,
                "alive_agents": s.alive_agents,
            },
            "progress": {
                "mean_x": s.mean_x,
                "delta_x": s.delta_x,
                "is_advancing": s.is_advancing,
                # se vuoi debug:
                # "mean_x_raw": s.mean_x_raw,
            },
            "episode": {
                "reward_mean": s.reward_mean,
                "done": s.done,
            },
        }

    def extract(
        self,
        obs: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        done: bool,
    ) -> Dict[str, Any]:
        """
        Compatibilità con il vecchio codice: aggiorna e ritorna snapshot.
        """
        self.update(obs=obs, rewards=rewards, terminations=terminations, done=done)
        return self.snapshot()

    # ----------------------------
    # Internals
    # ----------------------------
    def _compute_health(
        self,
        *,
        obs: Mapping[str, np.ndarray],
        terminations: Mapping[str, bool],
    ) -> tuple[int, int]:
        """
        Strategia robusta:
        1) se terminations ha chiavi -> usa quelle come "ground truth" dei caduti (True)
           e considera vivi i non-terminati.
        2) se terminations è vuoto o inutile -> inferisci vivi da obs (remove_on_fall)
        3) clamp sempre su [0, n_agents]
        """
        fallen = 0
        alive = 0

        if terminations:
            try:
                fallen = sum(1 for v in terminations.values() if bool(v))
                alive = self.n_agents - fallen
            except Exception:
                fallen = 0
                alive = len(obs)
        else:
            # fallback: obs contiene solo agenti presenti (spesso vivi)
            alive = len(obs)
            fallen = self.n_agents - alive

        fallen = int(max(0, min(self.n_agents, fallen)))
        alive = int(max(0, min(self.n_agents, alive)))
        return fallen, alive

    def _compute_mean_x(
        self,
        *,
        obs: Mapping[str, np.ndarray],
        terminations: Mapping[str, bool],
    ) -> float:
        """
        Calcola mean_x sui soli agenti "vivi".
        - se terminations disponibile: esclude terminated=True
        - altrimenti: usa tutti quelli in obs
        """
        xs = []
        for agent, ob in obs.items():
            # se abbiamo terminations, scarta terminated
            if terminations and bool(terminations.get(agent, False)):
                continue

            try:
                # ob può essere list/np.ndarray; serve indice x_index
                x = float(ob[self.x_index])
                xs.append(x)
            except Exception:
                continue

        return float(np.mean(xs)) if xs else 0.0
