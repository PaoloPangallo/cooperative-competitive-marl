import {
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  Box,
  Paper,
  Grow,
  Stack,
  Tooltip,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import TrendingDownIcon from "@mui/icons-material/TrendingDown";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import CardGiftcardIcon from "@mui/icons-material/CardGiftcard";

/**
 * TrainingOverview - Clean ML Platform Design
 * High-level status + learning phase diagnostics
 */
export default function TrainingOverview({ status, summary, metrics }) {
  const { state, iter, max_iter, progress, message, multiwalker } = status;

  const pct = max_iter ? Math.round((progress || 0) * 100) : 0;

  /* ===============================
     SIGNALS (truth priority)
     =============================== */
  const alive = safeNum(
    multiwalker?.health?.alive_agents,
    metrics?.alive_agents,
    0
  );
  const fallen = safeNum(
    multiwalker?.health?.fallen_agents,
    metrics?.fallen_agents,
    0
  );
  const total = Math.max(1, alive + fallen);

  const meanX = safeNum(
    multiwalker?.progress?.mean_x,
    metrics?.mean_x,
    0
  );
  const deltaX = safeNum(
    multiwalker?.progress?.delta_x,
    metrics?.delta_x,
    0
  );
  const rewardMean = safeNum(
    multiwalker?.episode?.reward_mean,
    metrics?.reward_mean,
    0
  );
  const entropy = safeNum(metrics?.entropy, NaN);

  /* ===============================
     PHASE INFERENCE
     =============================== */
  const phase = inferTrainingPhase({
    alive,
    total,
    deltaX,
    entropy,
  });

  const phaseConfig = getPhaseConfig(phase);

  const quality = computeQuality({
    alive,
    total,
    meanX,
    deltaX,
    rewardMean,
  });

  /* ===============================
     SUMMARY COMPAT
     =============================== */
  const bestReward =
    summary?.best_reward ??
    summary?.reward_max ??
    summary?.bestReward ??
    null;

  const finalReward =
    summary?.final_reward ??
    summary?.reward_last ??
    summary?.finalReward ??
    null;

  return (
    <Grow in timeout={600}>
      <Card
        elevation={0}
        sx={{
          mb: "var(--sp-lg)",
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-light)",
          borderTop: "3px solid var(--accent-secondary)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-sm)",
          transition: "all 0.2s ease",
          "&:hover": {
            boxShadow: "var(--shadow-md)",
            borderColor: "var(--border-color)",
          },
        }}
      >
        <CardContent sx={{ p: "var(--sp-lg)" }}>
          <Grid container spacing="var(--sp-lg)">
            {/* ================= HEADER ================= */}
            <Grid item xs={12}>
              <Box sx={{ display: "flex", gap: "var(--sp-md)", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap" }}>
                <Box>
                  <Typography
                    variant="h5"
                    sx={{
                      fontWeight: 700,
                      color: "var(--text-primary)",
                      mb: "var(--sp-sm)",
                    }}
                  >
                    Training Overview
                  </Typography>

                  <Stack direction="row" spacing="var(--sp-sm)" sx={{ flexWrap: "wrap" }}>
                    {/* Phase Chip */}
                    <Chip
                      label={`Phase: ${phase}`}
                      size="small"
                      sx={{
                        background: phaseConfig.background,
                        color: phaseConfig.textColor,
                        fontWeight: 600,
                        border: `1px solid ${phaseConfig.borderColor}`,
                      }}
                    />

                    {/* Phase Description */}
                    <Tooltip title={phaseConfig.help}>
                      <Chip
                        icon={<InfoIcon sx={{ fontSize: "0.95rem !important" }} />}
                        label={phaseConfig.label}
                        variant="outlined"
                        size="small"
                        sx={{
                          borderColor: "var(--border-color)",
                          color: "var(--text-secondary)",
                          fontWeight: 500,
                          cursor: "help",
                        }}
                      />
                    </Tooltip>

                    {/* Quality Score */}
                    <Tooltip title="Quality score (0–100): stability + progress + distance + reward">
                      <Chip
                        label={`Quality ${quality}`}
                        size="small"
                        sx={{
                          background: qualityBackground(quality),
                          color: qualityTextColor(quality),
                          fontWeight: 600,
                          border: `1px solid ${qualityBorderColor(quality)}`,
                        }}
                      />
                    </Tooltip>
                  </Stack>
                </Box>
              </Box>
            </Grid>

            {/* ================= KEY METRICS GRID ================= */}
            <Grid item xs={12} sm={6} md={3}>
              <StatusTile label="Status" value={state.toUpperCase()} />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <StatusTile
                label="Iteration"
                value={max_iter ? `${iter} / ${max_iter}` : iter}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <StatusTile label="Progress" value={`${pct}%`} />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <StatusTile
                label="Remaining"
                value={max_iter ? `${max_iter - iter}` : "—"}
              />
            </Grid>

            {/* ================= SIGNAL TILES ================= */}
            <Grid item xs={12}>
              <Grid container spacing="var(--sp-md)">
                {/* Δx - Forward Progress */}
                <Grid item xs={12} md={4}>
                  <SignalTile
                    icon={deltaX > 0 ? <TrendingUpIcon /> : <TrendingDownIcon />}
                    title="Δx"
                    value={formatSigned(deltaX, 4)}
                    hint="Forward progress signal"
                    statusColor={deltaX > 0 ? "success" : "warning"}
                  />
                </Grid>

                {/* Mean X - Team Position */}
                <Grid item xs={12} md={4}>
                  <SignalTile
                    icon={<LocationOnIcon />}
                    title="Mean X"
                    value={formatNum(meanX)}
                    hint="Average team position"
                    statusColor="info"
                  />
                </Grid>

                {/* Reward - Episode Return */}
                <Grid item xs={12} md={4}>
                  <SignalTile
                    icon={<CardGiftcardIcon />}
                    title="Reward"
                    value={formatNum(rewardMean)}
                    hint="Team reward per episode"
                    statusColor={rewardMean >= 0 ? "success" : "error"}
                  />
                </Grid>
              </Grid>
            </Grid>

            {/* ================= STATUS MESSAGE ================= */}
            <Grid item xs={12}>
              <Paper
                elevation={0}
                sx={{
                  p: "var(--sp-md)",
                  borderRadius: "var(--radius-lg)",
                  background: "var(--bg-primary)",
                  border: "1px solid var(--border-light)",
                  borderLeft: "3px solid var(--accent-secondary)",
                }}
              >
                <Box sx={{ display: "flex", gap: "var(--sp-sm)", alignItems: "flex-start" }}>
                  <InfoIcon
                    sx={{
                      fontSize: 18,
                      color: "var(--accent-secondary)",
                      mt: "0.25rem",
                      flexShrink: 0,
                    }}
                  />
                  <Typography
                    variant="body2"
                    sx={{
                      color: "var(--text-secondary)",
                      fontWeight: 500,
                      lineHeight: 1.6,
                    }}
                  >
                    {message || "No status message"}
                  </Typography>
                </Box>
              </Paper>
            </Grid>

            {/* ================= SUMMARY STATS ================= */}
            {summary && (
              <>
                <Grid item xs={12} sm={6}>
                  <SummaryTile
                    label="Best Reward"
                    value={bestReward?.toFixed?.(3) ?? "—"}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <SummaryTile
                    label="Final Reward"
                    value={finalReward?.toFixed?.(3) ?? "—"}
                  />
                </Grid>
              </>
            )}
          </Grid>
        </CardContent>
      </Card>
    </Grow>
  );
}

/* ===============================
   PHASE LOGIC
   =============================== */
function inferTrainingPhase({ alive, total, deltaX, entropy }) {
  if (!Number.isFinite(entropy)) return "BOOTSTRAP";

  if (entropy > 4.5 && Math.abs(deltaX) < 0.005) return "BOOTSTRAP";
  if (alive < total) return "UNSTABLE";
  if (Math.abs(deltaX) < 0.01) return "STABILITY";
  if (deltaX > 0.02) return "LOCOMOTION";
  return "STALLED";
}

function getPhaseConfig(phase) {
  const configs = {
    BOOTSTRAP: {
      label: "Exploration",
      background: "rgba(13, 70, 161, 0.1)",
      textColor: "var(--accent-secondary)",
      borderColor: "rgba(13, 70, 161, 0.3)",
      help: "High entropy, random behavior. Agent is exploring.",
    },
    UNSTABLE: {
      label: "Falling",
      background: "rgba(239, 68, 68, 0.1)",
      textColor: "#ef4444",
      borderColor: "rgba(239, 68, 68, 0.3)",
      help: "Agents cannot stay upright. Poor stability.",
    },
    STABILITY: {
      label: "Balancing",
      background: "rgba(217, 119, 6, 0.1)",
      textColor: "var(--accent-tertiary)",
      borderColor: "rgba(217, 119, 6, 0.3)",
      help: "Standing but not moving. Need to encourage locomotion.",
    },
    LOCOMOTION: {
      label: "Walking",
      background: "rgba(16, 160, 127, 0.1)",
      textColor: "var(--accent-primary)",
      borderColor: "rgba(16, 160, 127, 0.3)",
      help: "Stable forward movement. Good progress!",
    },
    STALLED: {
      label: "Stalled",
      background: "rgba(217, 119, 6, 0.1)",
      textColor: "var(--accent-tertiary)",
      borderColor: "rgba(217, 119, 6, 0.3)",
      help: "No forward progress. Training may be stuck.",
    },
  };

  return configs[phase] || configs.STALLED;
}

/* =======================================
   COMPONENT TILES
   ======================================= */

function StatusTile({ label, value }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: "var(--sp-md)",
        borderRadius: "var(--radius-lg)",
        background: "var(--bg-primary)",
        border: "1px solid var(--border-light)",
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: "var(--border-color)",
          boxShadow: "var(--shadow-sm)",
        },
      }}
    >
      <Typography
        variant="caption"
        sx={{
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "var(--text-tertiary)",
          display: "block",
          mb: "var(--sp-sm)",
        }}
      >
        {label}
      </Typography>
      <Typography
        variant="h6"
        sx={{
          fontWeight: 700,
          color: "var(--text-primary)",
        }}
      >
        {value}
      </Typography>
    </Paper>
  );
}

function SignalTile({ icon, title, value, hint, statusColor }) {
  const statusConfig = {
    success: { bg: "rgba(16, 160, 127, 0.08)", border: "rgba(16, 160, 127, 0.2)", color: "var(--accent-primary)" },
    warning: { bg: "rgba(217, 119, 6, 0.08)", border: "rgba(217, 119, 6, 0.2)", color: "var(--accent-tertiary)" },
    error: { bg: "rgba(239, 68, 68, 0.08)", border: "rgba(239, 68, 68, 0.2)", color: "#ef4444" },
    info: { bg: "rgba(13, 70, 161, 0.08)", border: "rgba(13, 70, 161, 0.2)", color: "var(--accent-secondary)" },
  };

  const status = statusConfig[statusColor] || statusConfig.info;

  return (
    <Paper
      elevation={0}
      sx={{
        p: "var(--sp-md)",
        borderRadius: "var(--radius-lg)",
        background: status.bg,
        border: `1px solid ${status.border}`,
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: status.color,
          boxShadow: "var(--shadow-sm)",
        },
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: "var(--sp-sm)", mb: "var(--sp-sm)" }}>
        <Box sx={{ color: status.color, fontSize: 18 }}>
          {icon}
        </Box>
        <Typography
          variant="caption"
          sx={{
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            color: "var(--text-tertiary)",
          }}
        >
          {title}
        </Typography>
      </Box>
      <Typography
        variant="h5"
        sx={{
          fontWeight: 800,
          color: status.color,
          mb: "var(--sp-sm)",
        }}
      >
        {value}
      </Typography>
      <Typography
        variant="caption"
        sx={{
          color: "var(--text-tertiary)",
          fontWeight: 500,
        }}
      >
        {hint}
      </Typography>
    </Paper>
  );
}

function SummaryTile({ label, value }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: "var(--sp-md)",
        borderRadius: "var(--radius-lg)",
        background: "var(--bg-primary)",
        border: "1px solid var(--border-light)",
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: "var(--border-color)",
          boxShadow: "var(--shadow-sm)",
        },
      }}
    >
      <Typography
        variant="caption"
        sx={{
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "var(--text-tertiary)",
          display: "block",
          mb: "var(--sp-sm)",
        }}
      >
        {label}
      </Typography>
      <Typography
        variant="h5"
        sx={{
          fontWeight: 800,
          color: "var(--accent-primary)",
        }}
      >
        {value}
      </Typography>
    </Paper>
  );
}

/* =======================================
   QUALITY COLOR HELPERS
   ======================================= */

function qualityBackground(q) {
  if (q >= 75) return "rgba(16, 160, 127, 0.1)";
  if (q >= 45) return "rgba(217, 119, 6, 0.1)";
  return "rgba(239, 68, 68, 0.1)";
}

function qualityTextColor(q) {
  if (q >= 75) return "var(--accent-primary)";
  if (q >= 45) return "var(--accent-tertiary)";
  return "#ef4444";
}

function qualityBorderColor(q) {
  if (q >= 75) return "rgba(16, 160, 127, 0.3)";
  if (q >= 45) return "rgba(217, 119, 6, 0.3)";
  return "rgba(239, 68, 68, 0.3)";
}

/* =======================================
   MATH HELPERS
   ======================================= */

function computeQuality({ alive, total, meanX, deltaX, rewardMean }) {
  const stability = clamp01(alive / total);
  const progress = clamp01((deltaX + 0.02) / 0.12);
  const distance = clamp01(meanX / 5.0);
  const reward = clamp01((rewardMean + 1) / 2);
  return Math.round(
    100 * (0.35 * stability + 0.35 * progress + 0.15 * distance + 0.15 * reward)
  );
}

function safeNum(...vals) {
  for (const v of vals) {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return 0;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function formatNum(x, d = 3) {
  return safeNum(x).toFixed(d);
}

function formatSigned(x, d = 4) {
  const n = safeNum(x);
  return `${n >= 0 ? "+" : ""}${n.toFixed(d)}`;
}