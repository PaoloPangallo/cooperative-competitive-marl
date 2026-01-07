import { startTraining } from "./api/train";
import useTrainStatus from "./hooks/useTrainStatus";
import "./App.css";

function App() {
  const { status, error } = useTrainStatus(1000);

  const onStart = async () => {
    await startTraining();
  };

  const fmt = (v, digits = 3) =>
  typeof v === "number" && Number.isFinite(v)
    ? v.toFixed(digits)
    : "—";


  return (
    <div style={{ padding: 20 }}>
      <h1>🧠 MARL – MAPPO MultiWalker</h1>

      <button onClick={onStart}>▶ Start Training</button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {status && (
        <div style={{ marginTop: 20 }}>
          <p><b>State:</b> {status.state}</p>
          <p><b>Iteration:</b> {status.iter}</p>
          <p><b>Message:</b> {status.message}</p>

          {status.multiwalker && (
            <>
              <h3>🤖 MultiWalker Status</h3>
              <p>Alive: {status.multiwalker.health.alive_agents}</p>
              <p>Fallen: {status.multiwalker.health.fallen_agents}</p>
              <p>
                Advancing:{" "}
                {status.multiwalker.progress.is_advancing ? "✅" : "❌"}
              </p>
              <p>Δx: {fmt(status.multiwalker.progress.delta_x, 3)}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
