# marl/cooperative/algorithms/train_mappo_multiwalker.py
# MAPPO (classic CTDE) for MultiWalker: decentralized actor + centralized critic
# Web-backend friendly: keeps logging, status updates, metrics recorder

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from marl.cooperative.utils.log import set_seed, log_info_safe
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter
from marl.cooperative.utils.multiwalker_status import MultiWalkerStatusTracker
from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.environments.multiwalker.multiwalker_reward_wrapper import MAPPOTeamRewardWrapper
from marl.shared.buffers.rollout_buffer import RolloutBuffer
from marl.shared.gae.gae import compute_gae
from marl.shared.logging.wandb_logger import WandbLogger
from marl.shared.networks.centralized_critic import CentralizedCritic


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
fh.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

if not logger_py.handlers:
    logger_py.addHandler(ch)
    logger_py.addHandler(fh)


# ==========================================================
# HYPERPARAMETERS (stable defaults for MultiWalker)
# ==========================================================
@dataclass
class PPOHyper:
    # optimizer
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    # PPO
    ppo_epochs: int = 10
    ppo_bs: int = 256
    clip_eps: float = 0.2

    # regularization
    max_grad_norm: float = 0.5

    # entropy schedule
    entropy_start: float = 0.02
    entropy_end: float = 0.001

    # critic update
    critic_epochs: int = 5
    critic_bs: int = 256


# ==========================================================
# Helpers
# ==========================================================
def to_team_global_obs(global_obs: torch.Tensor, n_agents: int) -> torch.Tensor:
    """
    Ensure centralized critic input is a SINGLE team vector [1, D].
    Some adapters may return [n_agents, D]. We reduce to team state via mean.
    """
    if global_obs is None:
        raise ValueError("global_obs is None")

    if global_obs.dim() == 1:
        return global_obs.unsqueeze(0)  # [1, D]

    if global_obs.dim() == 2:
        if global_obs.shape[0] == 1:
            return global_obs  # [1, D]
        if global_obs.shape[0] == n_agents:
            return global_obs.mean(dim=0, keepdim=True)  # [1, D]
        # fallback: mean over first dim
        return global_obs.mean(dim=0, keepdim=True)

    # fallback: flatten anything else
    return global_obs.view(1, -1)


def normalize_buffer_global_obs(
    gobs: torch.Tensor, T: int, n_agents: int, global_obs_dim: int
) -> torch.Tensor:
    """
    Normalize batch.global_obs to shape [T, global_obs_dim].
    Handles common buffer layouts:
      - [T, D]
      - [T*n_agents, D]
      - [T, n_agents, D]
      - [T, D*n_agents] (rare / flattened)
    """
    if gobs is None:
        raise RuntimeError("batch.global_obs is None")

    if gobs.dim() == 3:
        # [T, n_agents, D] -> [T, D]
        if gobs.shape[0] == T and gobs.shape[1] == n_agents and gobs.shape[2] == global_obs_dim:
            return gobs.mean(dim=1)
        # fallback
        return gobs.view(T, -1)[:, :global_obs_dim]

    if gobs.dim() == 2:
        if gobs.shape[0] == T and gobs.shape[1] == global_obs_dim:
            return gobs

        if gobs.shape[0] == T * n_agents and gobs.shape[1] == global_obs_dim:
            return gobs.view(T, n_agents, global_obs_dim).mean(dim=1)

        if gobs.shape[0] == T and gobs.shape[1] == global_obs_dim * n_agents:
            return gobs.view(T, n_agents, global_obs_dim).mean(dim=1)

        # last-resort: try reshape to [T, -1]
        resh = gobs.view(T, -1)
        if resh.shape[1] == global_obs_dim:
            return resh
        if resh.shape[1] == global_obs_dim * n_agents:
            return resh.view(T, n_agents, global_obs_dim).mean(dim=1)

        raise RuntimeError(f"Unexpected global_obs shape: {tuple(gobs.shape)} (T={T}, n_agents={n_agents}, D={global_obs_dim})")

    if gobs.dim() == 1:
        resh = gobs.view(T, -1)
        if resh.shape[1] == global_obs_dim:
            return resh
        if resh.shape[1] == global_obs_dim * n_agents:
            return resh.view(T, n_agents, global_obs_dim).mean(dim=1)
        raise RuntimeError(f"Unexpected 1D global_obs shape after reshape: {tuple(resh.shape)}")

    raise RuntimeError(f"Unsupported global_obs tensor rank: {gobs.dim()}")


# ==========================================================
# MAPPO CORE: Actor-only PPO update (CTDE)
# ==========================================================
def ppo_update_actor_only(
    policy,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,           # [N, obs_dim]
    actions: torch.Tensor,       # [N, act_dim] tanh-squashed
    old_logprobs: torch.Tensor,  # [N]
    advantages: torch.Tensor,    # [N] normalized
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    epochs: int = 10,
    batch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    dataset = TensorDataset(obs, actions, old_logprobs, advantages)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy.train()
    total_policy_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    n_updates = 0

    for _ in range(epochs):
        for b_obs, b_actions, b_old_logp, b_adv in loader:
            mean, _ = policy(b_obs)  # value head ignored on purpose

            std = torch.exp(policy.log_std).expand_as(mean)
            dist = torch.distributions.Normal(mean, std)

            # log-prob with tanh correction (must match adapter.act)
            eps = 1e-6
            raw_action = torch.atanh(torch.clamp(b_actions, -1 + eps, 1 - eps))
            logp = dist.log_prob(raw_action).sum(-1)
            logp -= torch.log(1 - b_actions.pow(2) + eps).sum(-1)

            ratio = torch.exp(logp - b_old_logp)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            n_updates += 1

    if n_updates == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "entropy": 0.0}

    return {
        "loss": total_loss / n_updates,
        "policy_loss": total_policy_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }


def critic_update(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_obs: torch.Tensor,  # [T, D]
    returns: torch.Tensor,     # [T]
    epochs: int = 5,
    batch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    dataset = TensorDataset(global_obs, returns)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    critic.train()
    total_vloss = 0.0
    n_updates = 0

    for _ in range(epochs):
        for b_g, b_ret in loader:
            v = critic(b_g).view(-1)
            loss = 0.5 * (v - b_ret).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer.step()

            total_vloss += loss.item()
            n_updates += 1

    if n_updates == 0:
        return {"value_loss": 0.0}
    return {"value_loss": total_vloss / n_updates}


# ==========================================================
# TRAINING FUNCTION (MAPPO MULTIWALKER)
# ==========================================================
def train_mappo_gold(
    n_iters: int = 400,
    rollout_steps: int = 1024,
    gamma: float = 0.99,
    lam: float = 0.95,
    seed: int = 42,
    eval_every: int = 10,
    on_status_update=None,
    stop_event=None,
    metrics_recorder=None,  # injected by backend
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info_safe(f"Device: {device} | seed={seed}")

    if metrics_recorder is not None:
        metrics_recorder.reset()

    if on_status_update:
        on_status_update(
            {
                "state": "running",
                "algo": "mappo",
                "env": "multiwalker",
                "iter": 0,
                "max_iter": n_iters,
                "progress": 0.0,
                "message": "Training started",
            }
        )

    base_env = MultiWalkerEnv(n_walkers=3)
    env = MAPPOTeamRewardWrapper(base_env)

    obs = env.reset()

    # PettingZoo-safe: agents may not be set yet
    agents = list(obs.keys())
    n_agents = len(agents)

    # opzionale ma consigliato: sincronizza
    env.agents = agents
    adapter = MultiWalkerAdapter(agents=agents, obs_dim=31, act_dim=4)
    policy = adapter.policy.to(device)

    status_tracker = MultiWalkerStatusTracker(n_agents=n_agents)
    log_info_safe(f"Agents: {agents}")

    # Critic (centralized) – infer global state dim robustly
    with torch.no_grad():
        _g0 = adapter.global_obs_dict_to_tensor(obs)
    g0 = to_team_global_obs(_g0, n_agents)
    global_obs_dim = int(g0.shape[-1])

    critic = CentralizedCritic(global_obs_dim=global_obs_dim).to(device)
    log_info_safe(f"CentralizedCritic global_obs_dim = {global_obs_dim}")

    hp = PPOHyper()
    actor_opt = torch.optim.Adam(policy.parameters(), lr=hp.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=hp.critic_lr, eps=1e-5)

    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    wb = WandbLogger(
        project="marl-from-scratch",
        name="mappo-classic-multiwalker",
        config={
            "algo": "MAPPO-classic-CTDE",
            "env": "MultiWalker",
            "n_walkers": n_agents,
            "gamma": gamma,
            "lambda": lam,
            "rollout_steps": rollout_steps,
            "seed": seed,
            "actor_lr": hp.actor_lr,
            "critic_lr": hp.critic_lr,
            "clip_eps": hp.clip_eps,
            "ppo_epochs": hp.ppo_epochs,
            "ppo_bs": hp.ppo_bs,
        },
    )

    try:
        for it in range(1, n_iters + 1):

            if stop_event is not None and stop_event.is_set():
                log_info_safe("[STOP] Training interrotto da API")
                if on_status_update:
                    on_status_update(
                        {"state": "stopping", "iter": it, "message": "Training stopped by user"}
                    )
                return

            buffer.reset()
            ep_rewards = []

            # entropy annealing
            frac = it / max(1, n_iters)
            entropy_coef = hp.entropy_start * (1 - frac) + hp.entropy_end * frac

            # ---------------- ROLLOUT ----------------
            for step_idx in range(rollout_steps):
                actions_dict, logp, _, _ = adapter.act(obs)

                obs_tensor = adapter.obs_dict_to_tensor(obs).to(device)  # [n_agents, obs_dim]

                g_raw = adapter.global_obs_dict_to_tensor(obs).to(device)
                global_obs = to_team_global_obs(g_raw, n_agents)  # [1, D] strictly team state

                actions_np = np.stack([actions_dict[a] for a in agents], axis=0)
                actions_tensor = torch.from_numpy(actions_np).float().to(device)

                next_obs, rewards, terminations, done, infos = env.step(actions_dict)

                # status tracker expects keyword args in your codebase
                status_tracker.update(
                    obs=obs,
                    rewards=rewards,
                    terminations=terminations,
                    done=done,
                )

                if on_status_update and (step_idx % 10 == 0):
                    on_status_update(
                        {
                            "state": "running",
                            "iter": it,
                            "max_iter": n_iters,
                            "progress": it / n_iters,
                            "multiwalker": status_tracker.snapshot(),
                        }
                    )

                rewards_tensor = torch.tensor(
                    [rewards[a] for a in env.agents], dtype=torch.float32, device=device
                )
                dones_tensor = torch.full(
                    (n_agents,), float(done), dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    v_t = critic(global_obs).view(-1)  # [1]

                # IMPORTANT: store ONLY team global obs vector [D] per step
                buffer.add(
                    obs=obs_tensor,
                    actions=actions_tensor,
                    logprobs=logp.to(device),
                    values=v_t,  # team V(s): [1]
                    rewards=rewards_tensor,
                    dones=dones_tensor,
                    global_obs=global_obs.squeeze(0),  # [D]
                )

                ep_rewards.append(rewards_tensor.mean().item())
                obs = next_obs if not done else env.reset()

                if done and on_status_update:
                    on_status_update(
                        {"state": "running", "iter": it, "multiwalker": status_tracker.snapshot()}
                    )

            # ---------------- ITER METRICS ----------------
            snapshot = status_tracker.snapshot()


            if on_status_update:
                on_status_update(
                    {
                        "state": "running",
                        "iter": it,
                        "message": f"Iteration {it}/{n_iters}",
                        "progress": it / n_iters,
                    }
                )

            # ---------------- MAPPO UPDATE ----------------
            batch = buffer.get()
            T = rollout_steps

            rewards_team = batch.rewards.view(T, n_agents).mean(dim=1)           # [T]
            dones_team = batch.dones.view(T, n_agents).max(dim=1).values         # [T]

            # values: should be [T] (one per step). If duplicated, reduce.
            v = batch.values
            if v.numel() == T:
                values_team = v.view(T)
            elif v.numel() == T * n_agents:
                values_team = v.view(T, n_agents).mean(dim=1)
            else:
                values_team = v.view(-1)[:T]

            with torch.no_grad():
                g_last_raw = adapter.global_obs_dict_to_tensor(obs).to(device)
                g_last = to_team_global_obs(g_last_raw, n_agents)
                last_v_team = critic(g_last).view(-1)[0]

            advantages, returns = compute_gae(
                rewards=rewards_team,
                values=values_team,
                dones=dones_team,
                gamma=gamma,
                lam=lam,
                last_value=last_v_team,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # actor: repeat team advantage per agent timestep
            adv_agents = advantages.repeat_interleave(n_agents).to(device)

            actor_stats = ppo_update_actor_only(
                policy=policy,
                optimizer=actor_opt,
                obs=batch.obs.to(device),
                actions=batch.actions.to(device),
                old_logprobs=batch.logprobs.to(device),
                advantages=adv_agents,
                clip_eps=hp.clip_eps,
                entropy_coef=entropy_coef,
                epochs=hp.ppo_epochs,
                batch_size=hp.ppo_bs,
                max_grad_norm=hp.max_grad_norm,
            )

            # critic: normalize global obs to [T, D]
            gobs = normalize_buffer_global_obs(batch.global_obs, T, n_agents, global_obs_dim).to(device)

            critic_stats = critic_update(
                critic=critic,
                optimizer=critic_opt,
                global_obs=gobs,
                returns=returns.to(device),
                epochs=hp.critic_epochs,
                batch_size=hp.critic_bs,
                max_grad_norm=hp.max_grad_norm,
            )

            log_info_safe(
                f"[Iter {it:03d}] R={np.mean(ep_rewards):.3f} | "
                f"actor_loss={actor_stats['policy_loss']:.4f} | "
                f"critic_loss={critic_stats['value_loss']:.4f} | "
                f"entropy={actor_stats['entropy']:.4f} (coef={entropy_coef:.4f})"
            )

            # ---------------- METRICS RECORDER ----------------
            if metrics_recorder is not None:
                metrics_recorder.record(
                    {
                        "iter": it,

                        # performance
                        "reward_mean": snapshot["episode"]["reward_mean"],
                        "alive_agents": snapshot["health"]["alive_agents"],
                        "fallen_agents": snapshot["health"]["fallen_agents"],
                        "mean_x": snapshot["progress"]["mean_x"],
                        "delta_x": snapshot["progress"]["delta_x"],

                        # MAPPO learning signals
                        "actor_loss": actor_stats["policy_loss"],
                        "critic_loss": critic_stats["value_loss"],
                        "entropy": actor_stats["entropy"],
                    }
                )

        # ---------------- FINISHED ----------------
        if on_status_update:
            on_status_update(
                {
                    "state": "finished",
                    "iter": n_iters,
                    "progress": 1.0,
                    "message": "Training completed",
                    "summary": metrics_recorder.summary() if metrics_recorder else None,
                    "multiwalker": status_tracker.snapshot(),
                }
            )

    finally:
        wb.close()
        env.close()
        log_info_safe("[DONE] Training terminato")


if __name__ == "__main__":
    train_mappo_gold()
