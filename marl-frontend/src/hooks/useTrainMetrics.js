import { useEffect, useState, useRef } from "react";
import { getTrainMetrics } from "../api/metric.api.js";

/* ================================
   Normalizzazione metrica ROBUSTA
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

  if (typeof iter !== "number" || typeof reward_mean !== "number") {
    return null;
  }

  return {
    iter,
    reward_mean,
    mean_x: typeof mean_x === "number" ? mean_x : undefined,
    alive_agents: typeof alive_agents === "number" ? alive_agents : undefined,
    fallen_agents: typeof fallen_agents === "number" ? fallen_agents : undefined,
  };
}

/* ================================
   Hook useTrainMetrics
   ================================ */
export function useTrainMetrics(interval = 2000) {
  const [metrics, setMetrics] = useState(null); // null = loading
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

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

          data.metrics.forEach(raw => {
            const m = normalizeMetric(raw);
            if (!m) return;

            if (seenIters.current.has(m.iter)) return;
            seenIters.current.add(m.iter);

            setMetrics(prev => {
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

        setMetrics(prev => {
          if (!prev) return [m];
          return [...prev, m];
        });

        setSummary(data.summary ?? null);

      } catch (e) {
        if (alive) setError(e?.message ?? "Metrics fetch error");
      }
    };

    tick();
    const id = setInterval(tick, interval);

    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [interval]);

  return { metrics, summary, error };
}
