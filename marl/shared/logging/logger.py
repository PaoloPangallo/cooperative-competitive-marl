# marl/shared/logging/logger.py
from typing import Dict, Optional


class Logger:
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        raise NotImplementedError

    def close(self):
        pass
