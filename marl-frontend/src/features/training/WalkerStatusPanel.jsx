import {
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  Stack,
  Divider,
} from "@mui/material";

/**
 * WalkerStatusPanel
 * Explains cooperative status of walkers
 */
export default function WalkerStatusPanel({ multiwalker }) {
  if (!multiwalker) return null;

  const {
    health,
    progress,
  } = multiwalker;

  const total = health.alive_agents + health.fallen_agents;

  return (
    <Card elevation={3} sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          🤖 Team & Walker Status
        </Typography>

        {/* ================= TEAM STATUS ================= */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Stat
              label="Alive Agents"
              value={health.alive_agents}
              color="success"
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <Stat
              label="Fallen Agents"
              value={health.fallen_agents}
              color={
                health.fallen_agents > 0
                  ? "error"
                  : "default"
              }
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <Stat
              label="Team Mean X"
              value={progress.mean_x.toFixed(3)}
              color="info"
            />
          </Grid>

          <Grid item xs={12}>
            <Stack
              direction="row"
              spacing={1}
              alignItems="center"
            >
              <Typography variant="body2">
                Team Trend:
              </Typography>
              <Chip
                label={trendLabel(progress)}
                color={trendColor(progress)}
                size="small"
              />
            </Stack>
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        {/* ================= WALKERS ================= */}
        <Typography
          variant="subtitle1"
          gutterBottom
        >
          Walkers
        </Typography>

        <Grid container spacing={2}>
          {Array.from({ length: total }).map((_, idx) => {
            const alive = idx < health.alive_agents;
            return (
              <Grid
                item
                xs={12}
                sm={4}
                key={idx}
              >
                <WalkerChip
                  index={idx}
                  alive={alive}
                />
              </Grid>
            );
          })}
        </Grid>
      </CardContent>
    </Card>
  );
}

/* ---------- helpers ---------- */

function Stat({ label, value, color }) {
  return (
    <>
      <Typography
        variant="caption"
        color="text.secondary"
      >
        {label}
      </Typography>
      <Typography
        variant="h6"
        color={color + ".main"}
      >
        {value}
      </Typography>
    </>
  );
}

function WalkerChip({ index, alive }) {
  return (
    <Stack
      direction="row"
      spacing={1}
      alignItems="center"
    >
      <Typography>
        Walker {index}
      </Typography>
      <Chip
        label={alive ? "Alive" : "Fallen"}
        color={alive ? "success" : "error"}
        size="small"
      />
    </Stack>
  );
}

function trendLabel(progress) {
  if (progress.delta_x > 0.001) return "Advancing";
  if (progress.delta_x < -0.001) return "Regressing";
  return "Stalled";
}

function trendColor(progress) {
  if (progress.delta_x > 0.001) return "success";
  if (progress.delta_x < -0.001) return "error";
  return "warning";
}
