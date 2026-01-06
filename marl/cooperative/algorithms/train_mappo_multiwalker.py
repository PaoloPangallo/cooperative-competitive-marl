# marl/cooperative/algorithms/train_mappo_multiwalker.py

import os
import sys
import random
import inspect
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter
from marl.shared.buffers.rollout_buffer import RolloutBuffer
from marl.shared.gae.gae import compute_gae
from marl.shared.ppo.ppo_update import ppo_update
from marl.shared.networks.centralized_critic import CentralizedCritic
from marl.shared.logging.wandb_logger import WandbLogger

# ------------------------- Logging setup -------------------------
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "train_mappo_multiwalker.log")

logger_py = logging.getLogger("mappo_gold")
logger_py.setLevel(logging.DEBUG)

# Console: INFO+
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

# File: DEBUG (rotante)
fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

if not logger_py.handlers:
    logger_py.addHandler(ch)
    logger_py.addHandler(fh)


# --- helper: messaggi ASCII-safe per la console Windows ---
def ascii_msg(s: str) -> str:
    # evita crash cp1252: niente ± / caratteri greci
    return (s.replace("±", "+/-")
            .replace("π", "pi")
            .replace("Π", "Pi")
            .replace("μ", "mu")
            .replace("σ", "sigma"))


def log_info_safe(s: str):
    try:
        logger_py.info(s)
    except UnicodeEncodeError:
        logger_py.info(ascii_msg(s))


# ------------------------- Utils -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    assert y_pred.shape == y_true.shape, f"EV shape mismatch: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}"
    var_y = torch.var(y_true)
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8))


def tinfo(x: torch.Tensor, name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
        "mean": float(x.mean().item()) if x.numel() else 0.0,
    }


def log_tensor_debug_wandb(wlogger: WandbLogger, payload: Dict[str, Any], step: int):
    try:
        wlogger.log({f"debug/{k}": v for k, v in payload.items()}, step=step)
    except Exception as e:
        logger_py.debug(f"W&B debug log failed: {e}")


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
    return float(total ** 0.5)


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


# ------------------------- Eval (deterministico se supportato) -------------------------
@torch.no_grad()
def evaluate_policy(env, adapter, episodes: int = 5):
    rewards, lens, success = [], [], []
    # rileva se l'adapter supporta deterministic
    sig = None
    try:
        sig = inspect.signature(adapter.act)
    except Exception:
        sig = None
    can_det = sig is not None and ("deterministic" in sig.parameters)

    for _ in range(episodes):
        obs = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        while not done:
            if can_det:
                actions_dict, _, _, _ = adapter.act(obs, deterministic=True)
            else:
                actions_dict, _, _, _ = adapter.act(obs)
            obs, rew, done, _ = env.step(actions_dict)
            ep_ret += np.mean([rew[a] for a in env.agents])
            ep_len += 1
        rewards.append(ep_ret)
        lens.append(ep_len)
        success.append(1.0 if ep_ret > 0 else 0.0)

    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/reward_std": float(np.std(rewards) + 1e-8),
        "eval/ep_len": float(np.mean(lens)),
        "eval/success_rate": float(np.mean(success)),
    }


# ------------------------- Training -------------------------
def train_mappo_gold(
        n_iters: int = 40,
        rollout_steps: int = 256,
        gamma: float = 0.99,
        lam: float = 0.95,
        seed: int = 42,
        eval_every: int = 10,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info_safe(f"Device: {device} | seed={seed} | rollout_steps={rollout_steps}")

    # -------- Env --------
    env = MultiWalkerEnv(n_walkers=3)
    obs = env.reset()
    log_info_safe(f"Agents: {env.agents}")
    n_agents = len(env.agents)

    # -------- Adapter & Actor --------
    adapter = MultiWalkerAdapter(agents=env.agents, obs_dim=31, act_dim=4)
    policy = adapter.policy.to(device)

    # -------- Centralized Critic --------
    critic = CentralizedCritic(global_obs_dim=155).to(device)

    # -------- Opt --------
    hp = PPOHyper()
    actor_opt = torch.optim.Adam(policy.parameters(), lr=hp.actor_lr, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=hp.critic_lr, eps=1e-5)

    # -------- Buffer --------
    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    # -------- W&B Logger --------
    wb = WandbLogger(
        project="marl-from-scratch",
        name="mappo-gold-multiwalker",
        config={
            "algo": "MAPPO-GOLD",
            "env": "MultiWalker",
            "n_walkers": n_agents,
            "gamma": gamma,
            "lambda": lam,
            "rollout_steps": rollout_steps,
            "actor_lr": hp.actor_lr,
            "critic_lr": hp.critic_lr,
            "vf_coef": hp.vf_coef,
            "clip": hp.clip_range,
            "value_clip": hp.value_clip,
            "seed": seed,
        },
    )

    # -------- Training Loop --------
    try:
        for it in range(1, n_iters + 1):
            buffer.reset()
            ep_rewards = []
            entropy_coef = max(1e-3, 0.02 * (0.995 ** it))

            # -------- COLLECT --------
            for _ in range(rollout_steps):
                actions_dict, logp, _, _ = adapter.act(obs)  # stochastic
                obs_tensor = adapter.obs_dict_to_tensor(obs).to(device)
                global_obs = adapter.global_obs_dict_to_tensor(obs).to(device)

                actions_np = np.stack([actions_dict[a] for a in env.agents], axis=0)
                actions_tensor = torch.from_numpy(actions_np).float().to(device)

                next_obs, rewards, done, _ = env.step(actions_dict)

                rewards_tensor = torch.tensor(
                    [rewards[a] for a in env.agents], dtype=torch.float32, device=device
                ) / 10.0
                dones_tensor = torch.tensor(
                    [float(done) for _ in env.agents], dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    v_t = critic(global_obs)  # può essere [1] o [n_agents]

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
                obs = next_obs
                if done:
                    obs = env.reset()

            # -------- BATCH --------
            batch = buffer.get()
            debug_shapes = {
                "rewards_len": int(batch.rewards.shape[0]),
                "values_len": int(batch.values.shape[0]),
                "dones_len": int(batch.dones.shape[0]),
                "obs_len": int(batch.obs.shape[0]),
                "actions_len": int(batch.actions.shape[0]),
                "logprobs_len": int(batch.logprobs.shape[0]),
                "global_obs_len": int(batch.global_obs.shape[0]),
                "n_agents_inferred": n_agents,
            }
            logger_py.debug(f"[Iter {it}] Shapes: {debug_shapes}")
            log_tensor_debug_wandb(wb, debug_shapes, step=it)

            # Critic bootstrap sull’ultimo stato
            with torch.no_grad():
                last_v = critic(adapter.global_obs_dict_to_tensor(obs).to(device))

            def to_team_scalar(x: torch.Tensor) -> torch.Tensor:
                x = x.to(device).view(-1)
                return x.mean() if x.numel() > 1 else x

            Rlen = batch.rewards.shape[0]
            Vlen = batch.values.shape[0]

            # ---------------- ALIGNAMENTO TEAM-SEQUENCE PER GAE ----------------
            try:
                if Vlen == Rlen:
                    T = Vlen
                    values_team = batch.values.view(T, -1).mean(dim=1) if batch.values.dim() > 1 else batch.values.view(
                        T)
                    rewards_team = batch.rewards.view(T, -1).mean(
                        dim=1) if batch.rewards.dim() > 1 else batch.rewards.view(T)
                    if batch.dones.dim() > 1:
                        dones_team = batch.dones.view(T, -1).max(dim=1).values
                    else:
                        dones_team = batch.dones.view(T)

                elif Vlen % Rlen == 0:
                    n_agents_vals = Vlen // Rlen
                    T = Rlen
                    values_team = batch.values.view(T, n_agents_vals).mean(dim=1)
                    rewards_team = batch.rewards.view(T) if batch.rewards.dim() == 1 else batch.rewards.view(T,
                                                                                                             -1).mean(
                        dim=1)
                    if batch.dones.numel() == T * n_agents_vals:
                        dones_team = batch.dones.view(T, n_agents_vals).max(dim=1).values
                    else:
                        dones_team = batch.dones.view(T)

                elif Rlen % Vlen == 0:
                    n_agents_rew = Rlen // Vlen
                    T = Vlen
                    values_team = batch.values.view(T, -1).mean(dim=1) if batch.values.dim() > 1 else batch.values.view(
                        T)
                    rewards_team = batch.rewards.view(T, n_agents_rew).mean(dim=1)
                    if batch.dones.numel() == T * n_agents_rew:
                        dones_team = batch.dones.view(T, n_agents_rew).max(dim=1).values
                    else:
                        dones_team = batch.dones.view(T)
                else:
                    raise RuntimeError(
                        f"Cannot align shapes for GAE: rewards={tuple(batch.rewards.shape)} "
                        f"values={tuple(batch.values.shape)}"
                    )
            except Exception as e:
                logger_py.error(f"[Iter {it}] Align-Error: {e}")
                logger_py.error(f"[Iter {it}] rewards: {tinfo(batch.rewards, 'rewards')}")
                logger_py.error(f"[Iter {it}] values:  {tinfo(batch.values, 'values')}")
                logger_py.error(f"[Iter {it}] dones:   {tinfo(batch.dones, 'dones')}")
                raise

            last_v_team = to_team_scalar(last_v)

            # ---------------- GAE (TEAM) ----------------
            advantages_team, returns_team = compute_gae(
                rewards=rewards_team,
                values=values_team,
                dones=dones_team,
                gamma=gamma,
                lam=lam,
                last_value=last_v_team,
            )
            advantages_team = (advantages_team - advantages_team.mean()) / (advantages_team.std() + 1e-8)

            # Replica per-agente se l’actor ha T*n_agents samples
            T_actor = batch.actions.shape[0]
            if T_actor == T:
                advantages = advantages_team
                returns = returns_team
            else:
                if T_actor % T != 0:
                    msg = f"Actor batch {T_actor} not divisible by T={T}"
                    logger_py.error(f"[Iter {it}] {msg}")
                    raise RuntimeError(msg)
                n_agents_actor = T_actor // T
                advantages = advantages_team.repeat_interleave(n_agents_actor, dim=0)
                returns = returns_team.repeat_interleave(n_agents_actor, dim=0)

            # ---------------- ACTOR (PPO) — firma compatibile ----------------
            ppo_variant = "unknown"
            try:
                ppo_stats = ppo_update(
                    policy=policy,
                    optimizer=actor_opt,
                    obs=batch.obs.to(device),
                    actions=batch.actions.to(device),
                    old_logprobs=batch.logprobs.to(device),
                    returns=returns.to(device),
                    advantages=advantages.to(device),
                    entropy_coef=entropy_coef,
                    epochs=hp.ppo_epochs,
                    batch_size=hp.ppo_bs,
                    clip_coef=hp.clip_range,
                    max_grad_norm=hp.max_grad_norm,
                )
                ppo_variant = "A_clipcoef+maxgrad"
            except TypeError:
                try:
                    ppo_stats = ppo_update(
                        policy=policy,
                        optimizer=actor_opt,
                        obs=batch.obs.to(device),
                        actions=batch.actions.to(device),
                        old_logprobs=batch.logprobs.to(device),
                        returns=returns.to(device),
                        advantages=advantages.to(device),
                        entropy_coef=entropy_coef,
                        epochs=hp.ppo_epochs,
                        batch_size=hp.ppo_bs,
                        max_grad_norm=hp.max_grad_norm,
                    )
                    ppo_variant = "B_noclip+maxgrad"
                except TypeError:
                    ppo_stats = ppo_update(
                        policy=policy,
                        optimizer=actor_opt,
                        obs=batch.obs.to(device),
                        actions=batch.actions.to(device),
                        old_logprobs=batch.logprobs.to(device),
                        returns=returns.to(device),
                        advantages=advantages.to(device),
                        entropy_coef=entropy_coef,
                        epochs=hp.ppo_epochs,
                        batch_size=hp.ppo_bs,
                    )
                    ppo_variant = "C_minimal"

            try:
                actor_gn = grad_norm(policy.parameters())
            except Exception:
                actor_gn = 0.0

            # ---------------- CRITIC (MSE + VALUE CLIP) ----------------
            # Vista TEAM di global_obs: prendo il primo agente di ogni timestep (assunzione valida in MultiWalker).
            try:
                assert batch.global_obs.shape[0] % n_agents == 0, \
                    f"global_obs len {batch.global_obs.shape[0]} non multiplo di n_agents={n_agents}"
                T_global = batch.global_obs.shape[0] // n_agents
                assert T_global >= values_team.shape[0], \
                    f"T_global={T_global} < T(values_team)={values_team.shape[0]}"

                go_team_full = batch.global_obs[::n_agents].to(device)  # [T_global, G]
                go_team = go_team_full[:values_team.shape[0]]  # [T, G]
                wb.log({"debug/critic_team_view": "stride_first_agent"}, step=it)
            except Exception as e:
                logger_py.warning(f"[Iter {it}] Fallback critic team view (pick_first): {e}")
                Gdim = batch.global_obs.shape[-1]
                go_team = batch.global_obs.view(-1, n_agents, Gdim)[:, 0, :].to(device)
                wb.log({"debug/critic_team_view": "fallback_pick_first"}, step=it)

            with torch.no_grad():
                v_old_team = critic(go_team).view(-1)  # [T]

            critic_losses = []
            value_clip = hp.value_clip
            bs = hp.ppo_bs
            T_team = values_team.shape[0]

            for start in range(0, T_team, bs):
                end = start + bs
                go = go_team[start:end]
                v = critic(go).view(-1)
                v_prev = v_old_team[start:end]
                ret = returns_team[start:end].to(device).view(-1)

                v_clipped = v_prev + (v - v_prev).clamp(-value_clip, value_clip)
                v_loss = torch.max(F.mse_loss(v, ret), F.mse_loss(v_clipped, ret))
                loss = hp.vf_coef * v_loss

                critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                critic_opt.step()
                critic_losses.append(loss.item())

            try:
                critic_gn = grad_norm(critic.parameters())
            except Exception:
                critic_gn = 0.0

            # ---------------- LOGGING ----------------
            with torch.no_grad():
                v_now_team = critic(go_team).view(-1)
                ev = explained_variance(v_now_team, returns_team.to(device).view(-1))

            kl = float(ppo_stats.get("kl", 0.0))
            clipfrac = float(ppo_stats.get("clipfrac", 0.0))
            ent = float(ppo_stats.get("entropy", 0.0))
            pol_loss = float(ppo_stats.get("policy_loss", 0.0))
            total_actor_loss = float(ppo_stats.get("loss", 0.0))
            critic_loss_mean = float(np.mean(critic_losses) if critic_losses else 0.0)

            msg = (
                f"[Iter {it:03d}] "
                f"R:{float(np.mean(ep_rewards)) if ep_rewards else 0.0:.3f}+/-{float(np.std(ep_rewards)) if ep_rewards else 0.0:.3f} "
                f"pi:{pol_loss:.3f} V:{critic_loss_mean:.3f} "
                f"KL:{kl:.3f} cf:{clipfrac:.2f} H:{ent:.3f} EV:{ev:.3f} "
                f"| gpi:{actor_gn:.3f} gV:{critic_gn:.3f} "
                f"| ppo_variant:{ppo_variant}"
            )
            log_info_safe(msg)

            wb.log({
                "train/reward_mean": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "train/reward_std": float(np.std(ep_rewards)) if ep_rewards else 0.0,
                "train/entropy": ent,
                "train/policy_loss": pol_loss,
                "train/critic_loss": critic_loss_mean,
                "train/total_loss": total_actor_loss + critic_loss_mean,
                "train/entropy_coef": entropy_coef,
                "train/kl": kl,
                "train/clipfrac": clipfrac,
                "train/explained_variance": ev,
                "debug/actor_grad_norm": actor_gn,
                "debug/critic_grad_norm": critic_gn,
                "debug/T_collect": rollout_steps,
                "debug/T_team": int(values_team.shape[0]),
                "debug/T_actor": int(T_actor),
                "debug/ppo_update_variant": ppo_variant,
                "debug/n_agents_inferred": n_agents,
            }, step=it)

            # -------- Periodic eval --------
            if it % eval_every == 0:
                eval_stats = evaluate_policy(env, adapter, episodes=5)
                wb.log(eval_stats, step=it)
                log_info_safe(
                    f"  -> EVAL: R={eval_stats['eval/reward_mean']:.3f} "
                    f"SR={eval_stats['eval/success_rate']:.2f} "
                    f"L={eval_stats['eval/ep_len']:.1f}"
                )


    except Exception as e:

        logger_py.exception(f"Fatal error in training loop: {e}")

        raise

    finally:

        # ================== TRAJECTORY ARCHIVE & FINAL SAVE ==================

        try:

            from datetime import datetime

            from marl.cooperative.utils.trajectory import evaluate_and_save

            log_info_safe("[INFO] Generating and archiving sample trajectories for MAPPO policy...")

            # Directory principale dove salvare le trajectory

            TRAJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "cooperative", "trajectories")

            os.makedirs(TRAJ_ROOT, exist_ok=True)

            # ---- Archivia tutte le iteration logs ----

            for it_archive in range(eval_every, n_iters + 1, eval_every):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                traj_name = f"traj_mappo_iter{it_archive:03d}_{timestamp}.json"

                traj_path = os.path.join(TRAJ_ROOT, "iter_archive", traj_name)

                os.makedirs(os.path.dirname(traj_path), exist_ok=True)

                log_info_safe(f"[ARCHIVE] Saving trajectory for iteration {it_archive} -> {traj_path}")

                evaluate_and_save(

                    policy=policy,

                    filename=traj_path,

                    n_steps=300,

                    seed=seed,

                )

                wb.save(traj_path)

                logger_py.info(f"[ARCHIVE] Saved {traj_path}")

            # ---- Salvataggio finale per analisi interattiva ----

            final_traj_path = os.path.join(LOG_DIR, "traj_mappo_final.json")

            evaluate_and_save(

                policy=policy,

                filename=final_traj_path,

                n_steps=300,

                seed=seed,

            )

            wb.save(final_traj_path)

            log_info_safe(f"[INFO] Saved final trajectory -> {final_traj_path}")

            log_info_safe("[INFO] Trajectory archive update complete.")

        except Exception as e:

            logger_py.exception(f"Failed to generate or archive trajectories: {e}")

        wb.close()

        env.close()

        log_info_safe(f"Logs written to {LOG_PATH}")


if __name__ == "__main__":
    train_mappo_gold()
