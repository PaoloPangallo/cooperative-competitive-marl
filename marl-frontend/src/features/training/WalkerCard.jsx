import { Paper, Typography, Chip, Stack, Box } from "@mui/material";

/**
 * WalkerCard
 * Low-level per-walker physical status.
 *
 * Responsibilities:
 * - Alive / Fallen
 * - Instantaneous position (mean_x)
 * - Instantaneous motion (Δx)
 *
 * NO learning judgement.
 */
export default function WalkerCard({ index, alive, progress }) {
  const meanX = safeNum(progress?.mean_x, 0);
  const deltaX = safeNum(progress?.delta_x, 0);

  const motion = getMotion(deltaX);

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        borderRadius: 3,
        background: alive
          ? "linear-gradient(135deg, rgba(52,199,89,0.08) 0%, rgba(48,209,88,0.08) 100%)"
          : "linear-gradient(135deg, rgba(255,59,48,0.08) 0%, rgba(255,69,58,0.08) 100%)",
        border: `2px solid ${
          alive ? "rgba(52,199,89,0.25)" : "rgba(255,59,48,0.25)"
        }`,
      }}
    >
      {/* HEADER */}
      <Stack
        direction="row"
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: 1 }}
      >
        <Typography sx={{ fontWeight: 800 }}>
          Walker {index}
        </Typography>

        <Chip
          label={alive ? "Alive" : "Fallen"}
          color={alive ? "success" : "error"}
          size="small"
          variant={alive ? "filled" : "outlined"}
        />
      </Stack>

      {/* METRICS */}
      <Stack spacing={1}>
        <MetricRow label="Mean X" value={meanX.toFixed(2)} />
        <MetricRow label="Δx" value={formatSigned(deltaX, 3)} />

        <Box>
          <Chip
            label={motion.label}
            color={motion.color}
            size="small"
            variant="outlined"
          />
        </Box>
      </Stack>
    </Paper>
  );
}

/* ==================================================
   SMALL COMPONENTS
   ================================================== */

function MetricRow({ label, value }) {
  return (
    <Stack direction="row" justifyContent="space-between" alignItems="baseline">
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography sx={{ fontWeight: 800 }}>
        {value}
      </Typography>
    </Stack>
  );
}

/* ==================================================
   LOGIC HELPERS (PHYSICAL ONLY)
   ================================================== */

function getMotion(deltaX) {
  if (deltaX > 0.03) return { label: "Moving forward", color: "success" };
  if (deltaX < -0.02) return { label: "Moving backward", color: "error" };
  return { label: "Standing", color: "warning" };
}

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function formatSigned(x, digits = 3) {
  const n = safeNum(x, 0);
  const s = n >= 0 ? "+" : "";
  return `${s}${n.toFixed(digits)}`;
}
