# marl/shared/gae/gae.py

from __future__ import annotations
import torch

@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: torch.Tensor | float | None = None,
):
    """
    GAE su una singola sequenza temporale (TEAM o per-agente).
    Tutti i tensori devono essere 1D di lunghezza T (dopo eventuale allineamento a monte).

    Args:
        rewards: [T]
        values:  [T] stime del V_t
        dones:   [T] (0. o 1.) done al passo t (episodio finito -> bootstrap interrotto)
        gamma:   sconto
        lam:     lambda GAE
        last_value: opzionale V_{T} per bootstrap (stato successivo all’ultimo step). Se None, assume 0.

    Returns:
        advantages: [T]
        returns:    [T] = advantages + values
    """
    # squeeze e cast coerenti
    rewards = rewards.view(-1).detach()
    values  = values.view(-1).detach()
    dones   = dones.view(-1).detach()

    T = rewards.shape[0]
    device = rewards.device
    dtype  = rewards.dtype

    if last_value is None:
        next_value = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        if not torch.is_tensor(last_value):
            last_value = torch.tensor(float(last_value), device=device, dtype=dtype)
        # se viene passato un vettore (es. per-agente), usa media come team bootstrap
        if last_value.numel() > 1:
            last_value = last_value.mean()
        next_value = last_value.to(device=device, dtype=dtype).view(())

    advantages = torch.zeros(T, device=device, dtype=dtype)
    gae = torch.tensor(0.0, device=device, dtype=dtype)

    for t in reversed(range(T)):
        # done=1 ⇒ next_non_terminal=0 ⇒ non bootstrappo
        next_non_terminal = 1.0 - (dones[t] > 0.5).float()
        v_t   = values[t]
        v_tp1 = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * v_tp1 * next_non_terminal - v_t
        gae   = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns
