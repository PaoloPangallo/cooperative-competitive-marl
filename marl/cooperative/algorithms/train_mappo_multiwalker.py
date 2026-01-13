import inspect
import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Dict, Any

import numpy as np
import torch

from marl.cooperative.utils.log import set_seed, log_info_safe
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter
from marl.cooperative.utils.multiwalker_status import MultiWalkerStatusTracker
from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.shared.buffers.rollout_buffer import RolloutBuffer
from marl.shared.gae.gae import compute_gae
from marl.shared.logging.wandb_logger import WandbLogger
from marl.shared.networks.centralized_critic import CentralizedCritic
from marl.shared.ppo.ppo_update import ppo_update

from marl.webapi.train_metrics_recorder import TrainMetricsRecorder

# ==========================================================
# LOGGING CONFIG
# ==========================================================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "train_mappo_multiwalker.log")

logger_py = logging.getLogger("mappo_gold")
logger_py.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

if not logger_py.handlers:
    logger_py.addHandler(ch)
    logger_py.addHandler(fh)

# ==========================================================
# PPO Hyperparameters
# ==========================================================
@dataclass
class PPOHyper:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 5
    ppo_bs: int = 128
    clip_range: float = 0.2
    value_clip: float = 0.2


# ==========================================================
# TRAINING FUNCTION (MAPPO MULTIWALKER)
# ==========================================================
def train_mappo_gold(
    n_iters: int = 40,
    rollout_steps: int = 256,
    gamma: float = 0.99,
    lam: float = 0.95,
    seed: int = 42,
    eval_every: int = 10,
    on_status_update=None,
    stop_event=None,
    metrics_recorder=None,   # 🔥 INIETTATO DAL BACKEND
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info_safe(f"Device: {device} | seed={seed}")

    # ================= RESET METRICS (UNA SOLA SOURCE OF TRUTH) =================
    if metrics_recorder is not None:
        metrics_recorder.reset()

    # ================= INITIAL STATUS =================
    if on_status_update:
        on_status_update({
            "state": "running",
            "algo": "mappo",
            "env": "multiwalker",
            "iter": 0,
            "max_iter": n_iters,
            "progress": 0.0,
            "message": "Training started",
        })

    # ================= ENV =================
    env = MultiWalkerEnv(n_walkers=3)
    obs = env.reset()
    n_agents = len(env.agents)

    status_tracker = MultiWalkerStatusTracker(n_agents=n_agents)
    log_info_safe(f"Agents: {env.agents}")

    # ================= POLICY / CRITIC =================
    adapter = MultiWalkerAdapter(agents=env.agents, obs_dim=31, act_dim=4)
    policy = adapter.policy.to(device)
    critic = CentralizedCritic(global_obs_dim=155).to(device)

    # ================= OPTIMIZERS =================
    hp = PPOHyper()
    actor_opt = torch.optim.Adam(policy.parameters(), lr=hp.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=hp.critic_lr, eps=1e-5)
    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    wb = WandbLogger(
        project="marl-from-scratch",
        name="mappo-gold-multiwalker",
        config={
            "algo": "MAPPO",
            "env": "MultiWalker",
            "n_walkers": n_agents,
            "gamma": gamma,
            "lambda": lam,
            "rollout_steps": rollout_steps,
            "seed": seed,
        },
    )

    try:
        for it in range(1, n_iters + 1):

            # ================= STOP =================
            if stop_event is not None and stop_event.is_set():
                log_info_safe("[STOP] Training interrotto da API")
                if on_status_update:
                    on_status_update({
                        "state": "stopping",
                        "iter": it,
                        "message": "Training stopped by user",
                    })
                return

            buffer.reset()
            ep_rewards = []

            # ================= ROLLOUT =================
            for step_idx in range(rollout_steps):
                actions_dict, logp, _, _ = adapter.act(obs)

                obs_tensor = adapter.obs_dict_to_tensor(obs).to(device)
                global_obs = adapter.global_obs_dict_to_tensor(obs).to(device)

                actions_np = np.stack([actions_dict[a] for a in env.agents], axis=0)
                actions_tensor = torch.from_numpy(actions_np).float().to(device)

                next_obs, rewards, terminations, done, infos = env.step(actions_dict)

                # ---------- UPDATE TRACKER ----------
                status_tracker.update(
                    obs=obs,
                    rewards=rewards,
                    terminations=terminations,
                    done=done,
                )

                # ---------- LIVE STATUS ----------
                if on_status_update and (step_idx % 10 == 0):
                    on_status_update({
                        "state": "running",
                        "iter": it,
                        "max_iter": n_iters,
                        "progress": it / n_iters,
                        "multiwalker": status_tracker.snapshot(),
                    })

                # ---------- BUFFER ----------
                rewards_tensor = torch.tensor(
                    [rewards[a] for a in env.agents],
                    dtype=torch.float32,
                    device=device,
                ) / 10.0

                dones_tensor = torch.full(
                    (n_agents,),
                    float(done),
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    v_t = critic(global_obs)

                buffer.add(
                    obs=obs_tensor,
                    actions=actions_tensor,
                    logprobs=logp.to(device),
                    values=v_t,
                    rewards=rewards_tensor,
                    dones=dones_tensor,
                    global_obs=global_obs,
                )

                ep_rewards.append(rewards_tensor.mean().item())
                obs = next_obs if not done else env.reset()

                if done and on_status_update:
                    on_status_update({
                        "state": "running",
                        "iter": it,
                        "multiwalker": status_tracker.snapshot(),
                    })

            # ================= ITER METRICS =================
            snapshot = status_tracker.snapshot()

            if metrics_recorder is not None:
                metrics_recorder.record({
                    "iter": it,
                    "reward_mean": snapshot["episode"]["reward_mean"],
                    "alive_agents": snapshot["health"]["alive_agents"],
                    "fallen_agents": snapshot["health"]["fallen_agents"],
                    "mean_x": snapshot["progress"]["mean_x"],
                    "delta_x": snapshot["progress"]["delta_x"],
                })

            if on_status_update:
                on_status_update({
                    "state": "running",
                    "iter": it,
                    "message": f"Iteration {it}/{n_iters}",
                    "progress": it / n_iters,
                })

            # ================= PPO UPDATE =================
            batch = buffer.get()
            with torch.no_grad():
                last_v = critic(adapter.global_obs_dict_to_tensor(obs).to(device))

            T = rollout_steps
            rewards_team = batch.rewards.view(T, n_agents).mean(dim=1)
            dones_team = batch.dones.view(T, n_agents).max(dim=1).values
            values_team = batch.values.view(T)
            last_v_team = last_v.view(-1).mean()

            advantages, returns = compute_gae(
                rewards=rewards_team,
                values=values_team,
                dones=dones_team,
                gamma=gamma,
                lam=lam,
                last_value=last_v_team,
            )

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ppo_update(
                policy=policy,
                optimizer=actor_opt,
                obs=batch.obs.to(device),
                actions=batch.actions.to(device),
                old_logprobs=batch.logprobs.to(device),
                returns=returns.repeat_interleave(n_agents).to(device),
                advantages=advantages.repeat_interleave(n_agents).to(device),
                entropy_coef=0.01,
                epochs=hp.ppo_epochs,
                batch_size=hp.ppo_bs,
            )

            log_info_safe(f"[Iter {it:03d}] R={np.mean(ep_rewards):.3f}")

        # ================= FINISHED =================
        if on_status_update:
            on_status_update({
                "state": "finished",
                "iter": n_iters,
                "progress": 1.0,
                "message": "Training completed",
                "summary": metrics_recorder.summary() if metrics_recorder else None,
                "multiwalker": status_tracker.snapshot(),
            })

    finally:
        wb.close()
        env.close()
        log_info_safe("[DONE] Training terminato")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    train_mappo_gold()
