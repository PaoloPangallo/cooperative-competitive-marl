# marl/shared/logging/wandb_logger.py
from typing import Dict, Optional
import wandb

from marl.shared.logging.logger import Logger


class WandbLogger(Logger):
    def __init__(
        self,
        project: str,
        config: dict,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                config=config,
            )

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def close(self):
        if self.enabled:
            wandb.finish()
