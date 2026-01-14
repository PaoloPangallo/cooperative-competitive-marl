import { useEffect, useState, useRef } from "react";
import { getTrainMetrics } from "../api/metric.api.js";

/* ================================
   Normalizzazione metrica ROBUSTA
   Compatibile MAPPO classico
   ================================ */
function normalizeMetric(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;

  const iter =
    raw.iter ??
    raw.iteration ??
    raw.step ??
    raw.timestep ??
    null;

  const reward_mean =
    raw.reward_mean ??
    raw.reward ??
    raw.mean_reward ??
    raw.avg_reward ??
    null;

  const mean_x =
    raw.mean_x ??
    raw.team_mean_x ??
    raw.x_mean ??
    raw.forward_mean ??
    null;

  const delta_x =
    raw.delta_x ??
    raw.dx ??
    raw.delta ??
    raw.progress_delta ??
    null;

  const alive_agents =
    raw.alive_agents ??
    raw.alive ??
    raw.n_alive ??
    raw.num_alive ??
    null;

  const fallen_agents =
    raw.fallen_agents ??
    raw.fallen ??
    raw.n_fallen ??
    raw.num_fallen ??
    null;

  /* 🔥 MAPPO learning signals */
  const actor_loss =
    raw.actor_loss ??
    raw.policy_loss ??
    raw.actor ??
    null;

  const critic_loss =
    raw.critic_loss ??
    raw.value_loss ??
    raw.critic ??
    null;

  const entropy =
    raw.entropy ??
    raw.ent ??
    null;

  if (typeof iter !== "number") return null;

  return {
    iter,

    // performance
    reward_mean: typeof reward_mean === "number" ? reward_mean : undefined,
    mean_x: typeof mean_x === "number" ? mean_x : undefined,
    delta_x: typeof delta_x === "number" ? delta_x : undefined,
    alive_agents: typeof alive_agents === "number" ? alive_agents : undefined,
    fallen_agents: typeof fallen_agents === "number" ? fallen_agents : undefined,

    // learning
    actor_loss: typeof actor_loss === "number" ? actor_loss : undefined,
    critic_loss: typeof critic_loss === "number" ? critic_loss : undefined,
    entropy: typeof entropy === "number" ? entropy : undefined,
  };


}

/* ================================
   Hook useTrainMetrics
   ================================ */
export function useTrainMetrics(interval = 2000) {
  const [metrics, setMetrics] = useState(null); // null = loading
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const phase = inferPhase(metrics);


  // evita duplicati per iterazione
  const seenIters = useRef(new Set());

  useEffect(() => {
    let alive = true;

    const tick = async () => {
      try {
        const data = await getTrainMetrics();
        if (!alive || !data) return;

        /* ================================
           CASO 1: metrics è UNA LISTA
           ================================ */
        if (Array.isArray(data.metrics)) {
          let added = false;

          data.metrics.forEach((raw) => {
            const m = normalizeMetric(raw);
            if (!m) return;

            if (seenIters.current.has(m.iter)) return;
            seenIters.current.add(m.iter);

            setMetrics((prev) => {
              if (!prev) return [m];
              return [...prev, m];
            });

            added = true;
          });

          if (added) {
            setSummary(data.summary ?? null);
          }
          return;
        }

        /* ================================
           CASO 2: metrics è UN OGGETTO
           ================================ */
        const m = normalizeMetric(data.metrics);
        if (!m) return;

        if (seenIters.current.has(m.iter)) return;
        seenIters.current.add(m.iter);

        setMetrics((prev) => {
          if (!prev) return [m];
          return [...prev, m];
        });

        setSummary(data.summary ?? null);

      } catch (e) {
        if (alive) {
          setError(e?.message ?? "Metrics fetch error");
        }
      }
    };

    tick();
    const id = setInterval(tick, interval);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [interval]);

  return { metrics, summary, phase, error };

}


function inferPhase(metrics, window = 20) {
  if (!Array.isArray(metrics) || metrics.length < window) {
    return "BOOTSTRAP";
  }

  const recent = metrics.slice(-window);

  const aliveRate =
    mean(recent.map((m) => safeNum(m.alive_agents, 0))) /
    Math.max(1, mean(recent.map((m) => safeNum(m.alive_agents, 0))) +
                mean(recent.map((m) => safeNum(m.fallen_agents, 0))));

  const deltaXMA = mean(recent.map((m) => safeNum(m.delta_x, 0)));

  // criterio concettuale (non magico):
  // 1) prima devono stare in piedi
  // 2) poi devono avanzare
  if (aliveRate < 0.9) return "UNSTABLE";
  if (Math.abs(deltaXMA) < 0.01) return "STABILITY";
  return "LOCOMOTION";
}

function mean(arr) {
  const v = arr.filter((x) => Number.isFinite(x));
  if (v.length === 0) return 0;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

