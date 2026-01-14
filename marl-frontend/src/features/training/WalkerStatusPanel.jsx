import {
  Card,
  CardContent,
  Grid,
  Typography,
  Box,
  Paper,
  Grow,
  Stack,
  LinearProgress,
  Chip,
} from "@mui/material";
import FavoriteBorderIcon from "@mui/icons-material/FavoriteBorder";
import HealthAndSafetyIcon from "@mui/icons-material/HealthAndSafety";
import PersonIcon from "@mui/icons-material/Person";
import InfoIcon from "@mui/icons-material/Info";

/**
 * WalkerStatusPanel - Clean ML Platform Design
 * Multi-agent health and individual walker status display
 */
export default function WalkerStatusPanel({ multiwalker }) {
  if (!multiwalker) {
    return (
      <Grow in timeout={600}>
        <Card
          elevation={0}
          sx={{
            mb: "var(--sp-lg)",
            borderRadius: "var(--radius-lg)",
            background: "var(--bg-secondary)",
            border: "1px solid var(--border-light)",
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <CardContent sx={{ p: "var(--sp-lg)", textAlign: "center" }}>
            <Typography variant="body2" color="var(--text-tertiary)">
              No walker data available
            </Typography>
          </CardContent>
        </Card>
      </Grow>
    );
  }

  const { health, walkers } = multiwalker;

  // Extract health metrics
  const aliveAgents = safeNum(health?.alive_agents, 0);
  const fallenAgents = safeNum(health?.fallen_agents, 0);
  const totalAgents = Math.max(1, aliveAgents + fallenAgents);
  const aliveRate = totalAgents > 0 ? (aliveAgents / totalAgents) * 100 : 0;

  // Parse walker data if available
  const walkerList = Array.isArray(walkers) ? walkers : [];
  const hasWalkerDetails = walkerList.length > 0;

  return (
    <Grow in timeout={700}>
      <Card
        elevation={0}
        sx={{
          mb: "var(--sp-lg)",
          borderRadius: "var(--radius-lg)",
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-light)",
          borderTop: "3px solid var(--accent-secondary)",
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
              <Box>
                <Typography
                  variant="h5"
                  sx={{
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    mb: "var(--sp-sm)",
                  }}
                >
                  Walker Status
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: "var(--text-secondary)",
                    fontWeight: 500,
                  }}
                >
                  Multi-agent health and individual agent performance
                </Typography>
              </Box>
            </Grid>

            {/* ================= HEALTH OVERVIEW ================= */}
            <Grid item xs={12}>
              <Grid container spacing="var(--sp-md)">
                {/* Total Agents */}
                <Grid item xs={12} sm={6} md={3}>
                  <HealthTile
                    label="Total Agents"
                    value={totalAgents}
                    icon={<PersonIcon />}
                  />
                </Grid>

                {/* Alive Agents */}
                <Grid item xs={12} sm={6} md={3}>
                  <HealthTile
                    label="Alive"
                    value={aliveAgents}
                    subtitle="Active agents"
                    icon={<FavoriteBorderIcon />}
                    color="success"
                  />
                </Grid>

                {/* Fallen Agents */}
                <Grid item xs={12} sm={6} md={3}>
                  <HealthTile
                    label="Fallen"
                    value={fallenAgents}
                    subtitle="Inactive agents"
                    icon={<HealthAndSafetyIcon />}
                    color="error"
                  />
                </Grid>

                {/* Alive Rate */}
                <Grid item xs={12} sm={6} md={3}>
                  <HealthTile
                    label="Alive Rate"
                    value={`${Math.round(aliveRate)}%`}
                    subtitle="Overall stability"
                    icon={<FavoriteBorderIcon />}
                    color={aliveRate >= 75 ? "success" : aliveRate >= 50 ? "warning" : "error"}
                  />
                </Grid>
              </Grid>
            </Grid>

            {/* ================= ALIVE RATE PROGRESS ================= */}
            <Grid item xs={12}>
              <Box>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: "var(--sp-sm)" }}>
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 600,
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    Agent Stability
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      fontWeight: 700,
                      color: "var(--text-primary)",
                    }}
                  >
                    {aliveAgents} / {totalAgents} alive
                  </Typography>
                </Stack>
                <LinearProgress
                  variant="determinate"
                  value={aliveRate}
                  sx={{
                    height: 6,
                    borderRadius: "var(--radius-md)",
                    background: "var(--bg-tertiary)",
                    "& .MuiLinearProgress-bar": {
                      background: aliveRate >= 75
                        ? "var(--accent-primary)"
                        : aliveRate >= 50
                        ? "var(--accent-tertiary)"
                        : "#ef4444",
                      borderRadius: "var(--radius-md)",
                    },
                  }}
                />
              </Box>
            </Grid>

            {/* ================= INDIVIDUAL WALKER DETAILS ================= */}
            {hasWalkerDetails && (
              <Grid item xs={12}>
                <Box sx={{ mb: "var(--sp-md)" }}>
                  <Typography
                    variant="subtitle2"
                    sx={{
                      fontWeight: 700,
                      mb: "var(--sp-md)",
                      textTransform: "uppercase",
                      fontSize: "0.875rem",
                      letterSpacing: "0.05em",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    Agent Details
                  </Typography>

                  <Grid container spacing="var(--sp-md)">
                    {walkerList.slice(0, 8).map((walker, idx) => (
                      <Grid item xs={12} sm={6} md={3} key={idx}>
                        <WalkerTile walker={walker} index={idx} />
                      </Grid>
                    ))}
                  </Grid>

                  {walkerList.length > 8 && (
                    <Box sx={{ mt: "var(--sp-md)" }}>
                      <Typography
                        variant="caption"
                        sx={{
                          color: "var(--text-tertiary)",
                          fontWeight: 500,
                        }}
                      >
                        +{walkerList.length - 8} more agents
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Grid>
            )}

            {/* ================= INFO BOX ================= */}
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
                    Agent health is critical for training stability. A high alive rate indicates the policy is learning effective locomotion and balance strategies.
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Grow>
  );
}

/* =======================================
   COMPONENT TILES
   ======================================= */

function HealthTile({ label, value, subtitle, icon, color = "default" }) {
  const colorConfig = {
    success: { bg: "rgba(16, 160, 127, 0.08)", textColor: "var(--accent-primary)", borderColor: "rgba(16, 160, 127, 0.2)" },
    warning: { bg: "rgba(217, 119, 6, 0.08)", textColor: "var(--accent-tertiary)", borderColor: "rgba(217, 119, 6, 0.2)" },
    error: { bg: "rgba(239, 68, 68, 0.08)", textColor: "#ef4444", borderColor: "rgba(239, 68, 68, 0.2)" },
    default: { bg: "var(--bg-primary)", textColor: "var(--text-primary)", borderColor: "var(--border-light)" },
  };

  const config = colorConfig[color] || colorConfig.default;

  return (
    <Paper
      elevation={0}
      sx={{
        p: "var(--sp-md)",
        borderRadius: "var(--radius-lg)",
        background: config.bg,
        border: `1px solid ${config.borderColor}`,
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: config.textColor,
          boxShadow: "var(--shadow-sm)",
        },
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: "var(--sp-sm)", mb: "var(--sp-sm)" }}>
        <Box sx={{ color: config.textColor, fontSize: 18 }}>
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
          {label}
        </Typography>
      </Box>

      <Typography
        variant="h5"
        sx={{
          fontWeight: 800,
          color: config.textColor,
          mb: subtitle ? "var(--sp-sm)" : 0,
        }}
      >
        {value}
      </Typography>

      {subtitle && (
        <Typography
          variant="caption"
          sx={{
            color: "var(--text-tertiary)",
            fontWeight: 500,
          }}
        >
          {subtitle}
        </Typography>
      )}
    </Paper>
  );
}

function WalkerTile({ walker, index }) {
  const isAlive = safeNum(walker?.alive, 1) === 1;
  const reward = safeNum(walker?.reward_episode, 0);
  const pos = safeNum(walker?.pos_x, 0);

  return (
    <Paper
      elevation={0}
      sx={{
        p: "var(--sp-md)",
        borderRadius: "var(--radius-lg)",
        background: "var(--bg-primary)",
        border: `2px solid ${isAlive ? "rgba(16, 160, 127, 0.2)" : "rgba(239, 68, 68, 0.2)"}`,
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: isAlive ? "var(--accent-primary)" : "#ef4444",
          boxShadow: "var(--shadow-sm)",
        },
        opacity: isAlive ? 1 : 0.6,
      }}
    >
      <Box sx={{ mb: "var(--sp-md)" }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: "var(--sp-sm)" }}>
          <Typography
            variant="caption"
            sx={{
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: "var(--text-tertiary)",
            }}
          >
            Agent {index + 1}
          </Typography>
          <Chip
            size="small"
            label={isAlive ? "Alive" : "Fallen"}
            sx={{
              background: isAlive ? "rgba(16, 160, 127, 0.1)" : "rgba(239, 68, 68, 0.1)",
              color: isAlive ? "var(--accent-primary)" : "#ef4444",
              fontWeight: 700,
              fontSize: "0.65rem",
              border: `1px solid ${isAlive ? "rgba(16, 160, 127, 0.3)" : "rgba(239, 68, 68, 0.3)"}`,
              height: 22,
            }}
          />
        </Box>
      </Box>

      <Stack spacing="var(--sp-sm)">
        <Box>
          <Typography
            variant="caption"
            sx={{
              color: "var(--text-tertiary)",
              fontWeight: 500,
              display: "block",
              mb: "0.25rem",
            }}
          >
            Reward
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 700,
              color: "var(--accent-primary)",
            }}
          >
            {reward.toFixed(2)}
          </Typography>
        </Box>

        <Box>
          <Typography
            variant="caption"
            sx={{
              color: "var(--text-tertiary)",
              fontWeight: 500,
              display: "block",
              mb: "0.25rem",
            }}
          >
            Position X
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 700,
              color: "var(--accent-secondary)",
            }}
          >
            {pos.toFixed(2)}
          </Typography>
        </Box>
      </Stack>
    </Paper>
  );
}

/* =======================================
   HELPERS
   ======================================= */

function safeNum(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}