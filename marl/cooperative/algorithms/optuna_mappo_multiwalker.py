import optuna
import numpy as np
import torch
import random

from marl.environments.multiwalker.env import MultiWalkerEnv
from marl.cooperative.utils.multiwalker_adapter import MultiWalkerAdapter
from marl.shared.buffers.rollout_buffer import RolloutBuffer
from marl.shared.gae.gae import compute_gae
from marl.shared.ppo.ppo_update import ppo_update
from marl.shared.networks.centralized_critic import CentralizedCritic


# --------------------------------------------------
# Utils
# --------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Minimal MAPPO (Optuna-only)
# --------------------------------------------------

def run_mappo_trial(
    *,
    lr: float,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    rollout_steps: int,
    ppo_epochs: int,
    batch_size: int,
    seed: int,
    n_iters: int = 200,
) -> float:
    """
    Minimal MAPPO loop.
    Returns scalar score for Optuna.
    """

    set_seed(seed)

    env = MultiWalkerEnv(n_walkers=3)
    obs = env.reset()

    adapter = MultiWalkerAdapter(
        agents=env.agents,
        obs_dim=31,
        act_dim=4,
    )
    policy = adapter.policy

    # build critic AFTER first global observation
    first_global_obs = adapter.global_obs_dict_to_tensor(obs)
    global_obs_dim = first_global_obs.shape[1]

    critic = CentralizedCritic(global_obs_dim=global_obs_dim)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(critic.parameters()),
        lr=lr,
    )

    buffer = RolloutBuffer(obs_dim=31, act_dim=4)

    reward_history = []

    for _ in range(n_iters):
        buffer.reset()
        iter_rewards = []

        for _ in range(rollout_steps):
            actions_dict, logp, _, _ = adapter.act(obs)

            obs_tensor = adapter.obs_dict_to_tensor(obs)
            global_obs = adapter.global_obs_dict_to_tensor(obs)

            actions = torch.tensor(
                np.stack([actions_dict[a] for a in env.agents]),
                dtype=torch.float32,
            )

            next_obs, rewards, done, _ = env.step(actions_dict)

            rewards_tensor = torch.tensor(
                [rewards[a] for a in env.agents],
                dtype=torch.float32,
            )

            dones_tensor = torch.tensor(
                [float(done)] * len(env.agents)
            )

            buffer.add(
                obs=obs_tensor,
                actions=actions,
                logprobs=logp,
                values=torch.zeros_like(logp),
                rewards=rewards_tensor,
                dones=dones_tensor,
                global_obs=global_obs,
            )

            iter_rewards.append(rewards_tensor.mean().item())

            obs = next_obs if not done else env.reset()

        batch = buffer.get()

        with torch.no_grad():
            values = critic(batch.global_obs)

        advantages, returns = compute_gae(
            rewards=batch.rewards,
            values=values,
            dones=batch.dones,
            gamma=0.99,
            lam=0.95,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ppo_update(
            policy=policy,
            optimizer=optimizer,
            obs=batch.obs,
            actions=batch.actions,
            old_logprobs=batch.logprobs,
            returns=returns,
            advantages=advantages,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            epochs=ppo_epochs,
            batch_size=batch_size,
        )

        reward_history.append(np.mean(iter_rewards))

    env.close()

    reward_history = np.array(reward_history)

    # ---- OPTUNA SCORE ----
    score = reward_history[-10:].mean()
    score -= 0.1 * reward_history[-10:].std()

    return float(score)


# --------------------------------------------------
# Optuna Objective
# --------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    return run_mappo_trial(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        clip_eps=trial.suggest_float("clip_eps", 0.1, 0.3),
        value_coef=trial.suggest_float("value_coef", 0.3, 1.0),
        entropy_coef=trial.suggest_float("entropy_coef", 0.001, 0.02),
        rollout_steps=trial.suggest_categorical("rollout_steps", [64, 128]),
        ppo_epochs=trial.suggest_int("ppo_epochs", 3, 6),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        seed=trial.number,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    study = optuna.create_study(
        study_name="mappo_multiwalker_gold",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=40,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print("\n=========== BEST TRIAL ===========")
    print(f"Score: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print("=================================\n")

    study.trials_dataframe().to_csv(
        "optuna_mappo_multiwalker_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

