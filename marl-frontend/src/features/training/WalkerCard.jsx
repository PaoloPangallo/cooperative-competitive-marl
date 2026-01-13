export default function WalkerCard({ index, alive, progress }) {
  return (
    <div
      style={{
        flex: 1,
        padding: 12,
        borderRadius: 10,
        background: alive ? "#1e272e" : "#3d3d3d",
        color: "#fff",
      }}
    >
      <h4>Walker {index}</h4>
      <div>Status: {alive ? "🟢 Alive" : "🔴 Fallen"}</div>
      <div>Mean X: {progress.mean_x.toFixed(2)}</div>
      <div>Δx: {progress.delta_x.toFixed(3)}</div>
    </div>
  );
}
