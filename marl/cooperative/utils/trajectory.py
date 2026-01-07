import os
import json
import numpy as np
import torch
import logging
from marl.environments.multiwalker.env import MultiWalkerEnv

# --- Setup logging coerente con il resto del progetto ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "algorithms", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "trajectory_eval.log")

logger = logging.getLogger("trajectory_eval")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    logger.addHandler(fh)


def _to_numpy_action(out):
    """Supporta vari formati di output policy: Tensor, tuple, namedtuple."""
    try:
        # Se è un ACOutput o namedtuple con campo action
        if hasattr(out, "action"):
            a = out.action
        elif isinstance(out, tuple):
            a = out[0]
        else:
            a = out

        if isinstance(a, np.ndarray):
            return a
        if not torch.is_tensor(a):
            raise TypeError(f"Unsupported action type: {type(a)}")

        if a.dim() > 1:
            a = a.squeeze(0)
        return a.detach().cpu().numpy()
    except Exception as e:
        logger.error(f"Failed to convert policy output to numpy: {e}")
        raise


def evaluate_and_save(policy, filename="trajectory.json", n_steps=300, seed=42):
    """
    Esegue una valutazione di policy su MultiWalkerEnv e salva le azioni/reward.

    Args:
        policy: rete actor o wrapper con metodo act(obs)
        filename: percorso file JSON di output
        n_steps: numero massimo di passi da simulare
        seed: seme per reset deterministico
    """
    logger.info(f"Starting trajectory evaluation: file={filename}, n_steps={n_steps}")

    env = MultiWalkerEnv(render_mode=None, seed=seed)
    obs = env.reset()
    trajectory = []

    try:
        for t in range(n_steps):
            actions = {}

            for agent, o in obs.items():
                if policy is not None:
                    try:
                        obs_t = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            if hasattr(policy, "act") and callable(policy.act):
                                out = policy.act(obs_t)
                            else:
                                out = policy(obs_t)
                        a = _to_numpy_action(out)
                    except Exception as e:
                        logger.warning(f"[t={t}] Policy act failed for {agent}: {e}, fallback random.")
                        a = env.act_spaces[agent].sample()
                else:
                    a = env.act_spaces[agent].sample()

                actions[agent] = a

            next_obs, rewards, _terminations, done, infos = env.step(actions)


            trajectory.append({
                "t": t,
                "actions": {a: actions[a].tolist() for a in actions},
                "rewards": {a: float(rewards[a]) for a in rewards},
            })

            obs = next_obs
            if done:
                logger.info(f"Episode ended early at step {t}")
                break

    except Exception as e:
        logger.error(f"Trajectory generation failed: {e}")
        raise
    finally:
        env.close()

    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(trajectory, f, indent=2)
        # salva anche come latest
        latest_path = os.path.join(os.path.dirname(__file__), "..", "cooperative", "trajectories", "latest_trajectory.json")
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        with open(latest_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        logger.info(f"Saved trajectory to {filename}")
        logger.info(f"Updated latest trajectory -> {latest_path}")
    except Exception as e:
        logger.error(f"Failed to save trajectory: {e}")
        raise


if __name__ == "__main__":
    # Test: run with random actions
    evaluate_and_save(policy=None, filename="traj_mappo_test.json", n_steps=100)

