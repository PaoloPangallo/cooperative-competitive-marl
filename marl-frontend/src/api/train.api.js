const BASE = "http://localhost:8090";

export async function startTraining() {
  const r = await fetch(`${BASE}/train/mappo`, { method: "POST" });
  if (!r.ok) throw new Error("Failed to start training");
  return r.json();
}

export async function stopTraining() {
  const r = await fetch(`${BASE}/train/stop`, { method: "POST" });
  if (!r.ok) throw new Error("Failed to stop training");
  return r.json();
}

export async function getTrainStatus() {
  const r = await fetch(`${BASE}/train/status`);
  if (!r.ok) throw new Error("Failed to fetch status");
  return r.json();
}
