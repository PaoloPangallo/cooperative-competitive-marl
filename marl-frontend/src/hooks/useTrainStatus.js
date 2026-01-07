import { useEffect, useRef, useState } from "react";
import { getTrainStatus } from "../api/train";

const DEFAULT_STATUS = {
  state: "idle",
  iter: 0,
  message: "Loading...",
  multiwalker: {
    health: { alive_agents: 0, fallen_agents: 0 },
    progress: { mean_x: 0, delta_x: 0, is_advancing: false },
    episode: { reward_mean: 0, done: false },
  },
};

export default function useTrainStatus(interval = 1000) {
  const [status, setStatus] = useState(DEFAULT_STATUS);
  const [error, setError] = useState(null);
  const mountedRef = useRef(true);
  const inflightRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;

    const fetchStatus = async () => {
      if (!mountedRef.current || inflightRef.current) return;

      inflightRef.current = true;
      try {
        const data = await getTrainStatus();
        if (mountedRef.current) {
          setStatus(data);
          setError(null); // 🔑 reset errore se ok
        }
      } catch (e) {
        if (mountedRef.current) {
          setError(e?.message || "Network error");
        }
      } finally {
        inflightRef.current = false;
      }
    };

    fetchStatus();
    const timer = setInterval(fetchStatus, interval);

    return () => {
      mountedRef.current = false;
      clearInterval(timer);
    };
  }, [interval]);

  return { status, error };
}
