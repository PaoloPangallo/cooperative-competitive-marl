import threading
from typing import Dict, Any, List, Optional
from copy import deepcopy


class TrainMetricsRecorder:
    """
    Accumulatore thread-safe delle metriche di training.

    È la SOURCE OF TRUTH per:
    - dashboard
    - confronto algoritmi
    - ricostruzione offline
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: List[Dict[str, Any]] = []

    # ------------------------
    # WRITE
    # ------------------------
    def reset(self):
        with self._lock:
            self._metrics.clear()

    def record(self, payload: Dict[str, Any]):
        """
        payload = {
          "iter": int,
          "reward_mean": float,
          "alive_agents": int,
          "fallen_agents": int,
          "mean_x": float,
          "delta_x": float,
        }
        """
        if not payload or "iter" not in payload:
            return

        with self._lock:
            # garantiamo immutabilità verso l'esterno
            self._metrics.append(deepcopy(payload))

    # ------------------------
    # READ
    # ------------------------
    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return deepcopy(self._metrics)

    def get_last(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._metrics:
                return None
            return deepcopy(self._metrics[-1])

    def summary(self) -> Dict[str, Any]:
        """
        Riassunto finale utile a fine training.
        """
        with self._lock:
            if not self._metrics:
                return {}

            rewards = [m.get("reward_mean", 0.0) for m in self._metrics]

            return {
                "iters": len(self._metrics),
                "best_reward": max(rewards),
                "final_reward": rewards[-1],
            }
