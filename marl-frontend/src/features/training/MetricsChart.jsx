import {
  Card,
  CardContent,
  Grid,
  Typography,
  Box,
  Paper,
  Grow,
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
} from "recharts";

/**
 * MetricsCharts
 * Global training metrics visualization
 */
export default function MetricsCharts({ metrics }) {
  // loading vero
  if (metrics === null) {
    return (
      <Grow in timeout={800}>
        <Card>
          <CardContent sx={{ p: 4, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              Loading metrics…
            </Typography>
          </CardContent>
        </Card>
      </Grow>
    );
  }

  // nessun dato valido (caso raro, ma gestito)
  if (metrics.length === 0) {
    return (
      <Grow in timeout={800}>
        <Card>
          <CardContent sx={{ p: 4, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              No metrics received yet
            </Typography>
          </CardContent>
        </Card>
      </Grow>
    );
  }


  return (
    <Grow in timeout={1000}>
      <Card
        elevation={0}
        sx={{
          mb: 4,
          borderRadius: 4,
          background: "rgba(255, 255, 255, 0.95)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.2)",
          boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)",
          overflow: "visible",
          position: "relative",
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": {
            transform: "translateY(-8px)",
            boxShadow: "0 28px 80px rgba(102, 126, 234, 0.35)",
          },
          "&::before": {
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "4px",
            background: "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
            borderRadius: "4px 4px 0 0",
          },
        }}
      >
        <CardContent sx={{ p: 4 }}>
          {/* Header */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 4 }}>
            <Box
              sx={{
                width: 48,
                height: 48,
                borderRadius: 3,
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "1.5rem",
                boxShadow: "0 8px 24px rgba(102, 126, 234, 0.4)",
              }}
            >
              📈
            </Box>
            <Box>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  letterSpacing: "-0.02em",
                }}
              >
                Training Metrics
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ fontWeight: 500 }}
              >
                Real-time performance visualization
              </Typography>
            </Box>
          </Box>

          <Grid container spacing={3}>
            {/* ================= Reward ================= */}
            <Grid item xs={12}>
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(46, 204, 113, 0.05) 0%, rgba(39, 174, 96, 0.05) 100%)",
                  border: "2px solid rgba(46, 204, 113, 0.2)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    borderColor: "rgba(46, 204, 113, 0.4)",
                    boxShadow: "0 8px 24px rgba(46, 204, 113, 0.15)",
                  },
                }}
              >
                <SectionTitle
                  icon="🏆"
                  title="Reward Mean"
                  subtitle="Is the policy improving?"
                  color="#2ecc71"
                />
                <Chart height={280}>
                  <LineChart data={metrics}>
                    <defs>
                      <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#2ecc71" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#2ecc71" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="iter"
                      stroke="#718096"
                      style={{ fontSize: "0.875rem", fontWeight: 600 }}
                    />
                    <YAxis
                      stroke="#718096"
                      style={{ fontSize: "0.875rem", fontWeight: 600 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(255, 255, 255, 0.95)",
                        backdropFilter: "blur(10px)",
                        border: "1px solid rgba(0, 0, 0, 0.1)",
                        borderRadius: "12px",
                        boxShadow: "0 8px 24px rgba(0, 0, 0, 0.15)",
                        fontWeight: 600,
                      }}
                    />
                    <Legend
                      wrapperStyle={{
                        fontWeight: 600,
                        fontSize: "0.875rem",
                      }}
                    />
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.1)" />
                    <Line
                      type="monotone"
                      dataKey="reward_mean"
                      name="Reward Mean"
                      stroke="#2ecc71"
                      strokeWidth={3}
                      dot={{ fill: "#2ecc71", r: 4 }}
                      activeDot={{ r: 6, fill: "#27ae60" }}
                      fill="url(#colorReward)"
                    />
                  </LineChart>
                </Chart>
              </Paper>
            </Grid>

            {/* ================= Mean X ================= */}
            <Grid item xs={12} md={6}>
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(52, 152, 219, 0.05) 0%, rgba(41, 128, 185, 0.05) 100%)",
                  border: "2px solid rgba(52, 152, 219, 0.2)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    borderColor: "rgba(52, 152, 219, 0.4)",
                    boxShadow: "0 8px 24px rgba(52, 152, 219, 0.15)",
                  },
                }}
              >
                <SectionTitle
                  icon="🎯"
                  title="Mean X"
                  subtitle="Team forward progress"
                  color="#3498db"
                />
                <Chart height={240}>
                  <LineChart data={metrics}>
                    <defs>
                      <linearGradient id="colorMeanX" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3498db" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3498db" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="iter"
                      stroke="#718096"
                      style={{ fontSize: "0.875rem", fontWeight: 600 }}
                    />
                    <YAxis
                      stroke="#718096"
                      style={{ fontSize: "0.875rem", fontWeight: 600 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(255, 255, 255, 0.95)",
                        backdropFilter: "blur(10px)",
                        border: "1px solid rgba(0, 0, 0, 0.1)",
                        borderRadius: "12px",
                        boxShadow: "0 8px 24px rgba(0, 0, 0, 0.15)",
                        fontWeight: 600,
                      }}
                    />
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.1)" />
                    <Line
                      type="monotone"
                      dataKey="mean_x"
                      name="Mean X"
                      stroke="#3498db"
                      strokeWidth={3}
                      dot={{ fill: "#3498db", r: 4 }}
                      activeDot={{ r: 6, fill: "#2980b9" }}
                      fill="url(#colorMeanX)"
                    />
                  </LineChart>
                </Chart>
              </Paper>
            </Grid>

            {/* ================= Health ================= */}
            <Grid item xs={12} md={6}>
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)",
                  border: "2px solid rgba(102, 126, 234, 0.2)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    borderColor: "rgba(102, 126, 234, 0.4)",
                    boxShadow: "0 8px 24px rgba(102, 126, 234, 0.15)",
                  },
                }}
              >
                <SectionTitle
                  icon="❤️"
                  title="Cooperation Health"
                  subtitle="Alive vs Fallen agents"
                  color="#667eea"
                />
                <Chart height={240}>
                  <LineChart data={metrics}>
                    <defs>
                      <linearGradient id="colorAlive" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#2ecc71" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#2ecc71" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorFallen" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#e74c3c" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#e74c3c" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="iter"
                      stroke="#718096"
                      style={{ fontSize: "0.875rem", fontWeight: 600 }}
                    />
                    <YAxis
                      allowDecimals={false}
                      stroke="#718096"
                      style={{ fontSize: "0.875rem", fontWeight: 600 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "rgba(255, 255, 255, 0.95)",
                        backdropFilter: "blur(10px)",
                        border: "1px solid rgba(0, 0, 0, 0.1)",
                        borderRadius: "12px",
                        boxShadow: "0 8px 24px rgba(0, 0, 0, 0.15)",
                        fontWeight: 600,
                      }}
                    />
                    <Legend
                      wrapperStyle={{
                        fontWeight: 600,
                        fontSize: "0.875rem",
                      }}
                    />
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 0, 0, 0.1)" />
                    <Line
                      type="stepAfter"
                      dataKey="alive_agents"
                      name="Alive"
                      stroke="#2ecc71"
                      strokeWidth={3}
                      dot={{ fill: "#2ecc71", r: 4 }}
                      activeDot={{ r: 6, fill: "#27ae60" }}
                      fill="url(#colorAlive)"
                    />
                    <Line
                      type="stepAfter"
                      dataKey="fallen_agents"
                      name="Fallen"
                      stroke="#e74c3c"
                      strokeWidth={3}
                      dot={{ fill: "#e74c3c", r: 4 }}
                      activeDot={{ r: 6, fill: "#c0392b" }}
                      fill="url(#colorFallen)"
                    />
                  </LineChart>
                </Chart>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Grow>
  );
}

/* ---------- helpers ---------- */

function Chart({ children, height }) {
  return (
    <Box sx={{ width: "100%", height, mt: 2 }}>
      <ResponsiveContainer>
        {children}
      </ResponsiveContainer>
    </Box>
  );
}

function SectionTitle({ icon, title, subtitle, color }) {
  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 0.5 }}>
        <Box sx={{ fontSize: "1.5rem" }}>{icon}</Box>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            color: color,
          }}
        >
          {title}
        </Typography>
      </Box>
      <Typography
        variant="caption"
        sx={{
          color: "text.secondary",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {subtitle}
      </Typography>
    </Box>
  );
}