import {
  Card,
  CardContent,
  Grid,
  Typography,
  Box,
  Paper,
  Grow,
  Chip,
  Stack,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";


/**
 * MetricsChart - Clean ML Platform Design
 * Global training metrics visualization with diagnostics
 */
export default function MetricsCharts({ metrics, summary }) {
  // Loading state
  if (metrics === null) {
    return (
      <Grow in timeout={800}>
        <Card
          elevation={0}
          sx={{
            borderRadius: "var(--radius-lg)",
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-light)",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <CardContent sx={{ p: "var(--sp-lg)", textAlign: "center" }}>
            <Typography variant="body2" color="var(--text-tertiary)">
              Loading metrics…
            </Typography>
          </CardContent>
        </Card>
      </Grow>
    );
  }

  // No data state
  if (!Array.isArray(metrics) || metrics.length === 0) {
    return (
      <Grow in timeout={800}>
        <Card
          elevation={0}
          sx={{
            borderRadius: "var(--radius-lg)",
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-light)",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <CardContent sx={{ p: "var(--sp-lg)", textAlign: "center" }}>
            <Typography variant="body2" color="var(--text-tertiary)">
              No metrics received yet
            </Typography>
          </CardContent>
        </Card>
      </Grow>
    );
  }

  // Ensure sorted by iter
  const data = [...metrics]
    .filter((m) => m && typeof m.iter === "number")
    .sort((a, b) => a.iter - b.iter);

  // Add derived series (moving averages)
  const windowShort = 20;
  const enriched = addDerivedSeries(data, windowShort);

  const last = enriched[enriched.length - 1];

  // KPIs
  const rewardLatest = safeNum(last.reward_mean, 0);
  const rewardMA = safeNum(last.reward_ma, 0);
  const rewardSlope = calcSlope(enriched, "reward_ma", windowShort);

  const meanXLatest = safeNum(last.mean_x, 0);
  const meanXMA = safeNum(last.mean_x_ma, 0);
  const meanXSlope = calcSlope(enriched, "mean_x_ma", windowShort);

  const deltaXLatest = safeNum(last.delta_x, 0);
  const deltaXMA = safeNum(last.delta_x_ma, 0);
  const deltaXSlope = calcSlope(enriched, "delta_x_ma", windowShort);

  const aliveLatest = safeNum(last.alive_agents, 0);
  const fallenLatest = safeNum(last.fallen_agents, 0);
  const totalAgents = Math.max(1, aliveLatest + fallenLatest);
  const aliveRate = aliveLatest / totalAgents;

  const rewardStd = calcStd(enriched, "reward_mean", windowShort);
  const deltaXStd = calcStd(enriched, "delta_x", windowShort);

  const trainingBadge = trainingVerdict({
    aliveRate,
    deltaXMA,
    rewardSlope,
    meanXSlope,
  });

  return (
    <Grow in timeout={1000}>
      <Card
        elevation={0}
        sx={{
          mb: "var(--sp-lg)",
          borderRadius: "var(--radius-lg)",
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-light)",
          borderTop: "3px solid var(--accent-primary)",
          boxShadow: "var(--shadow-sm)",
          transition: "all 0.2s ease",
          "&:hover": {
            boxShadow: "var(--shadow-md)",
            borderColor: "var(--border-color)",
          },
        }}
      >
        <CardContent sx={{ p: "var(--sp-lg)" }}>
          {/* ================= HEADER ================= */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              mb: "var(--sp-lg)",
              flexWrap: "wrap",
              gap: "var(--sp-md)",
            }}
          >
            <Box>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  mb: "var(--sp-sm)",
                }}
              >
                Training Metrics
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: "var(--text-secondary)",
                  fontWeight: 500,
                }}
              >
                Real-time performance visualization with trends
              </Typography>
            </Box>

            {/* Training Verdict Badge */}
            <Chip
              label={`Status: ${trainingBadge.label}`}
              sx={{
                background: trainingBadge.background,
                color: trainingBadge.textColor,
                fontWeight: 700,
                border: `1px solid ${trainingBadge.borderColor}`,
              }}
            />
          </Box>

          {/* ================= KPI GRID ================= */}
          <Grid container spacing="var(--sp-md)" sx={{ mb: "var(--sp-lg)" }}>
            <Grid item xs={12} sm={6} md={3}>
              <KpiTile
                title="Reward"
                rows={[
                  ["Latest", fmt(rewardLatest)],
                  [`MA${windowShort}`, fmt(rewardMA)],
                  [`Slope${windowShort}`, fmtSigned(rewardSlope, 4)],
                  ["Std Dev", fmt(rewardStd)],
                ]}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <KpiTile
                title="Mean X"
                rows={[
                  ["Latest", fmt(meanXLatest)],
                  [`MA${windowShort}`, fmt(meanXMA)],
                  [`Slope${windowShort}`, fmtSigned(meanXSlope, 4)],
                  ["Position", "—"],
                ]}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <KpiTile
                title="Δx (Progress)"
                rows={[
                  ["Latest", fmtSigned(deltaXLatest, 4)],
                  [`MA${windowShort}`, fmtSigned(deltaXMA, 4)],
                  [`Slope${windowShort}`, fmtSigned(deltaXSlope, 5)],
                  ["Std Dev", fmt(deltaXStd)],
                ]}
              />
            </Grid>

            <Grid item xs={12} sm={6} md={3}>
              <KpiTile
                title="Stability"
                rows={[
                  ["Alive", `${aliveLatest}/${totalAgents}`],
                  ["Rate", `${Math.round(aliveRate * 100)}%`],
                  ["Fallen", `${fallenLatest}`],
                  ["—", "—"],
                ]}
              />
            </Grid>
          </Grid>

          {/* ================= CHARTS GRID ================= */}
          <Grid container spacing="var(--sp-lg)">
            {/* Reward Chart */}
            <Grid item xs={12}>
              <ChartSection
                title="Reward Mean"
                subtitle="Raw signal + moving average (trend analysis)"
              >
                <LineChart data={enriched}>
                  <XAxis
                    dataKey="iter"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <YAxis
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{
                      paddingTop: "1rem",
                      fontSize: "0.875rem",
                      fontWeight: 500,
                    }}
                  />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border-light)"
                    vertical={false}
                  />

                  <Line
                    type="monotone"
                    dataKey="reward_mean"
                    name="Reward (raw)"
                    stroke="var(--accent-primary)"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="reward_ma"
                    name={`MA${windowShort}`}
                    stroke="var(--accent-primary)"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ChartSection>
            </Grid>

            {/* Mean X Chart */}
            <Grid item xs={12} md={6}>
              <ChartSection
                title="Mean X (Position)"
                subtitle="Distance + moving average"
              >
                <LineChart data={enriched}>
                  <XAxis
                    dataKey="iter"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <YAxis
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border-light)"
                    vertical={false}
                  />

                  <Line
                    type="monotone"
                    dataKey="mean_x"
                    name="Mean X (raw)"
                    stroke="var(--accent-secondary)"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="mean_x_ma"
                    name={`MA${windowShort}`}
                    stroke="var(--accent-secondary)"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ChartSection>
            </Grid>

            {/* Δx Chart */}
            <Grid item xs={12} md={6}>
              <ChartSection
                title="Δx (Step Progress)"
                subtitle="Is the team moving forward? KEY SIGNAL"
              >
                <LineChart data={enriched}>
                  <XAxis
                    dataKey="iter"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <YAxis
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{
                      paddingTop: "1rem",
                      fontSize: "0.875rem",
                      fontWeight: 500,
                    }}
                  />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border-light)"
                    vertical={false}
                  />

                  <ReferenceLine
                    y={0}
                    stroke="var(--text-tertiary)"
                    strokeDasharray="2 2"
                    opacity={0.5}
                  />

                  <Line
                    type="monotone"
                    dataKey="delta_x"
                    name="Δx (raw)"
                    stroke="var(--accent-tertiary)"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="delta_x_ma"
                    name={`MA${windowShort}`}
                    stroke="var(--accent-tertiary)"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ChartSection>
            </Grid>

            {/* Learning Diagnostics */}
            <Grid item xs={12}>
              <ChartSection
                title="Learning Diagnostics (MAPPO)"
                subtitle="Actor / Critic losses & policy entropy"
              >
                <LineChart data={enriched}>
                  <XAxis
                    dataKey="iter"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <YAxis
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{
                      paddingTop: "1rem",
                      fontSize: "0.875rem",
                      fontWeight: 500,
                    }}
                  />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border-light)"
                    vertical={false}
                  />

                  <Line
                    type="monotone"
                    dataKey="actor_loss"
                    name="Actor Loss"
                    stroke="#ef4444"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="critic_loss"
                    name="Critic Loss"
                    stroke="var(--accent-secondary)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="entropy"
                    name="Entropy"
                    stroke="var(--accent-tertiary)"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ChartSection>
            </Grid>

            {/* Agent Health */}
            <Grid item xs={12}>
              <ChartSection
                title="Cooperation Health"
                subtitle="Alive vs Fallen agents over time"
              >
                <LineChart data={enriched}>
                  <XAxis
                    dataKey="iter"
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <YAxis
                    allowDecimals={false}
                    stroke="var(--text-tertiary)"
                    style={{ fontSize: "0.75rem", fontWeight: 500 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend
                    wrapperStyle={{
                      paddingTop: "1rem",
                      fontSize: "0.875rem",
                      fontWeight: 500,
                    }}
                  />
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border-light)"
                    vertical={false}
                  />

                  <Line
                    type="stepAfter"
                    dataKey="alive_agents"
                    name="Alive"
                    stroke="var(--accent-primary)"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="stepAfter"
                    dataKey="fallen_agents"
                    name="Fallen"
                    stroke="#ef4444"
                    strokeWidth={3}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ChartSection>
            </Grid>
          </Grid>

          {/* ================= FOOTER SUMMARY ================= */}
          {summary && (
            <Box sx={{ mt: "var(--sp-lg)" }}>
              <Paper
                elevation={0}
                sx={{
                  p: "var(--sp-md)",
                  borderRadius: "var(--radius-lg)",
                  background: "var(--bg-primary)",
                  border: "1px solid var(--border-light)",
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    color: "var(--text-secondary)",
                    fontWeight: 500,
                    display: "block",
                    lineHeight: 1.6,
                  }}
                >
                  <strong>Summary:</strong> best_reward=
                  {fmt(summary.best_reward ?? summary.reward_max)} | final_reward=
                  {fmt(summary.final_reward ?? summary.reward_last)} | iterations=
                  {summary.iters ?? "—"}
                </Typography>
              </Paper>
            </Box>
          )}
        </CardContent>
      </Card>
    </Grow>
  );
}

/* =======================================
   CUSTOM COMPONENTS
   ======================================= */

function ChartSection({ title, subtitle, children }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: "var(--sp-lg)",
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
      {/* Header */}
      <Box sx={{ mb: "var(--sp-md)" }}>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            color: "var(--text-primary)",
            mb: "var(--sp-sm)",
          }}
        >
          {title}
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: "var(--text-tertiary)",
            fontWeight: 500,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          {subtitle}
        </Typography>
      </Box>

      {/* Chart */}
      <Box sx={{ width: "100%", height: 260 }}>
        <ResponsiveContainer>{children}</ResponsiveContainer>
      </Box>
    </Paper>
  );
}

function KpiTile({ title, rows }) {
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
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "var(--text-tertiary)",
          display: "block",
          mb: "var(--sp-md)",
        }}
      >
        {title}
      </Typography>

      <Stack spacing="0.5rem">
        {rows.map(([k, v]) => (
          <Stack
            key={k}
            direction="row"
            justifyContent="space-between"
            alignItems="baseline"
          >
            <Typography variant="caption" sx={{ color: "var(--text-tertiary)" }}>
              {k}
            </Typography>
            <Typography
              sx={{
                fontWeight: 700,
                color: "var(--accent-primary)",
                fontSize: "0.875rem",
              }}
            >
              {v}
            </Typography>
          </Stack>
        ))}
      </Stack>
    </Paper>
  );
}

function CustomTooltip({ active, payload }) {
  if (active && payload && payload.length) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: "var(--sp-md)",
          borderRadius: "var(--radius-md)",
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-light)",
          boxShadow: "var(--shadow-md)",
        }}
      >
        {payload.map((entry, idx) => (
          <Typography
            key={idx}
            variant="caption"
            sx={{
              display: "block",
              color: entry.color,
              fontWeight: 600,
              mb: idx < payload.length - 1 ? "var(--sp-sm)" : 0,
            }}
          >
            {entry.name}: {Number(entry.value).toFixed(4)}
          </Typography>
        ))}
      </Paper>
    );
  }
  return null;
}

/* =======================================
   HELPERS & MATH
   ======================================= */

function addDerivedSeries(arr, w) {
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const cur = arr[i];
    const slice = arr.slice(Math.max(0, i - w + 1), i + 1);

    const reward_ma = mean(slice.map((x) => x.reward_mean));
    const mean_x_ma = mean(slice.map((x) => x.mean_x));
    const delta_x_ma = mean(slice.map((x) => x.delta_x));

    const actor_loss_ma = mean(slice.map((x) => x.actor_loss));
    const critic_loss_ma = mean(slice.map((x) => x.critic_loss));
    const entropy_ma = mean(slice.map((x) => x.entropy));

    out.push({
      ...cur,
      reward_ma,
      mean_x_ma,
      delta_x_ma,
      actor_loss: safeNum(cur.actor_loss),
      critic_loss: safeNum(cur.critic_loss),
      entropy: safeNum(cur.entropy),
      actor_loss_ma,
      critic_loss_ma,
      entropy_ma,
      alive_agents: safeNum(cur.alive_agents, 0),
      fallen_agents: safeNum(cur.fallen_agents, 0),
    });
  }
  return out;
}

function mean(vals) {
  const v = vals.filter((x) => Number.isFinite(x));
  if (v.length === 0) return 0;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

function calcSlope(arr, key, w) {
  const n = Math.min(w, arr.length);
  if (n < 2) return 0;

  const slice = arr.slice(arr.length - n);
  const ys = slice.map((d) => safeNum(d[key], 0));
  const xs = slice.map((_, i) => i);

  const xMean = mean(xs);
  const yMean = mean(ys);

  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - xMean;
    num += dx * (ys[i] - yMean);
    den += dx * dx;
  }
  return den === 0 ? 0 : num / den;
}

function calcStd(arr, key, w) {
  const n = Math.min(w, arr.length);
  if (n < 2) return 0;
  const slice = arr.slice(arr.length - n);
  const vals = slice.map((d) => safeNum(d[key], 0));
  const m = mean(vals);
  const v = vals.reduce((acc, x) => acc + (x - m) * (x - m), 0) / (vals.length - 1);
  return Math.sqrt(Math.max(0, v));
}

function trainingVerdict({ aliveRate, deltaXMA, rewardSlope, meanXSlope }) {
  if (aliveRate < 0.67)
    return {
      label: "Unstable",
      background: "rgba(239, 68, 68, 0.1)",
      textColor: "#ef4444",
      borderColor: "rgba(239, 68, 68, 0.3)",
    };
  if (deltaXMA > 0.03 && (meanXSlope > 0 || rewardSlope > 0))
    return {
      label: "Improving",
      background: "rgba(16, 160, 127, 0.1)",
      textColor: "var(--accent-primary)",
      borderColor: "rgba(16, 160, 127, 0.3)",
    };
  if (deltaXMA < -0.005)
    return {
      label: "Regressing",
      background: "rgba(239, 68, 68, 0.1)",
      textColor: "#ef4444",
      borderColor: "rgba(239, 68, 68, 0.3)",
    };
  return {
    label: "Stalled",
    background: "rgba(217, 119, 6, 0.1)",
    textColor: "var(--accent-tertiary)",
    borderColor: "rgba(217, 119, 6, 0.3)",
  };
}

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function fmt(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function fmtSigned(x, digits = 3) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  const s = n >= 0 ? "+" : "";
  return `${s}${n.toFixed(digits)}`;
}