const BASE = "http://localhost:8090";

export async function getTrainMetrics() {
  const r = await fetch(`${BASE}/train/metrics`);
  if (!r.ok) {
    throw new Error("Failed to fetch train metrics");
  }
  return r.json();
}
