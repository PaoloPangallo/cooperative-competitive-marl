# marl/cooperative/utils/log.py

import logging
import os
import random
import sys
import numpy as np
import torch

# logger BASE, indipendente
logger = logging.getLogger("marl")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def log_info_safe(msg: str):
    try:
        logger.info(msg)
    except UnicodeEncodeError:
        logger.info(msg.encode("ascii", errors="ignore").decode())


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
