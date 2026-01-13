import { useEffect, useState } from "react";
import { getTrainStatus } from "../api/train.api";

export function useTrainStatus(interval = 1000) {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;

    const tick = async () => {
      try {
        const data = await getTrainStatus();
        if (alive) setStatus(data);
      } catch (e) {
        if (alive) setError(e.message);
      }
    };

    tick();
    const id = setInterval(tick, interval);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [interval]);

  return { status, error };
}
