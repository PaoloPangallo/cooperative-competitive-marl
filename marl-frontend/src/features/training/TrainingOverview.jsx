import {
  Card,
  CardContent,
  Grid,
  Typography,
  Chip,
  LinearProgress,
  Box,
  Paper,
  Grow,
} from "@mui/material";

/**
 * TrainingOverview
 * High-level status of the training process
 */
export default function TrainingOverview({ status, summary }) {
  const { state, iter, max_iter, progress, message } = status;

  const pct = max_iter ? Math.round(progress * 100) : 0;
  const stateConfig = getStateConfig(state);

  return (
    <Grow in timeout={600}>
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
          <Grid container spacing={3}>
            {/* Title */}
            <Grid item xs={12}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
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
                  🧠
                </Box>
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
                  Training Overview
                </Typography>
              </Box>
            </Grid>

            {/* State */}
            <Grid item xs={12} sm={6} md={3}>
              <Paper
                elevation={0}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: stateConfig.bgGradient,
                  border: `2px solid ${stateConfig.borderColor}`,
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "scale(1.05)",
                    boxShadow: stateConfig.shadow,
                  },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    color: "text.secondary",
                    mb: 1.5,
                    display: "block",
                  }}
                >
                  Status
                </Typography>
                <Chip
                  label={state.toUpperCase()}
                  sx={{
                    background: stateConfig.gradient,
                    color: "white",
                    fontWeight: 800,
                    fontSize: "0.75rem",
                    letterSpacing: "0.05em",
                    height: 32,
                    boxShadow: stateConfig.shadow,
                    "& .MuiChip-label": {
                      px: 2,
                    },
                  }}
                />
              </Paper>
            </Grid>

            {/* Iteration */}
            <Grid item xs={12} sm={6} md={3}>
              <Paper
                elevation={0}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%)",
                  border: "2px solid rgba(102, 126, 234, 0.2)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "scale(1.05)",
                    borderColor: "rgba(102, 126, 234, 0.4)",
                  },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    color: "text.secondary",
                    mb: 1.5,
                    display: "block",
                  }}
                >
                  Iteration
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 800,
                    color: "#667eea",
                    fontSize: "1.5rem",
                  }}
                >
                  {max_iter ? `${iter} / ${max_iter}` : iter}
                </Typography>
              </Paper>
            </Grid>

            {/* Progress */}
            <Grid item xs={12} sm={6} md={3}>
              <Paper
                elevation={0}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%)",
                  border: "2px solid rgba(102, 126, 234, 0.2)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "scale(1.05)",
                    borderColor: "rgba(102, 126, 234, 0.4)",
                  },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    color: "text.secondary",
                    mb: 1.5,
                    display: "block",
                  }}
                >
                  Progress
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 800,
                    color: "#667eea",
                    fontSize: "1.5rem",
                  }}
                >
                  {pct}%
                </Typography>
              </Paper>
            </Grid>

            {/* Estimated Time */}
            <Grid item xs={12} sm={6} md={3}>
              <Paper
                elevation={0}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%)",
                  border: "2px solid rgba(102, 126, 234, 0.2)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    transform: "scale(1.05)",
                    borderColor: "rgba(102, 126, 234, 0.4)",
                  },
                }}
              >
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.1em",
                    color: "text.secondary",
                    mb: 1.5,
                    display: "block",
                  }}
                >
                  Est. Remaining
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 800,
                    color: "#667eea",
                    fontSize: "1.5rem",
                  }}
                >
                  {max_iter ? `${Math.max(0, max_iter - iter)} iter` : "—"}
                </Typography>
              </Paper>
            </Grid>

            {/* Progress bar */}
            <Grid item xs={12}>
              <Box sx={{ mt: 2, mb: 1 }}>
                <Box
                  sx={{
                    position: "relative",
                    height: 16,
                    borderRadius: 3,
                    background: "rgba(102, 126, 234, 0.1)",
                    overflow: "hidden",
                  }}
                >
                  <LinearProgress
                    variant="determinate"
                    value={pct}
                    sx={{
                      height: "100%",
                      borderRadius: 3,
                      background: "transparent",
                      "& .MuiLinearProgress-bar": {
                        borderRadius: 3,
                        background: "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
                        boxShadow: "0 0 20px rgba(102, 126, 234, 0.6)",
                        position: "relative",
                        "&::after": {
                          content: '""',
                          position: "absolute",
                          top: 0,
                          left: 0,
                          bottom: 0,
                          right: 0,
                          background: "linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)",
                          animation: "shimmer 2s infinite",
                        },
                      },
                    }}
                  />
                </Box>
              </Box>
            </Grid>

            {/* Message */}
            <Grid item xs={12}>
              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  borderRadius: 2,
                  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)",
                  borderLeft: "4px solid #667eea",
                }}
              >
                <Typography
                  variant="body2"
                  sx={{
                    color: "text.secondary",
                    fontStyle: "italic",
                    fontWeight: 500,
                  }}
                >
                  💬 {message || "No status message"}
                </Typography>
              </Paper>
            </Grid>

            {/* Optional summary */}
            {summary && (
              <>
                <Grid item xs={12}>
                  <Box sx={{ mt: 2, mb: 1 }}>
                    <Typography
                      variant="subtitle2"
                      sx={{
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.1em",
                        color: "text.secondary",
                      }}
                    >
                      Performance Metrics
                    </Typography>
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 3,
                      borderRadius: 3,
                      background: "linear-gradient(135deg, rgba(52, 199, 89, 0.1) 0%, rgba(48, 209, 88, 0.1) 100%)",
                      border: "2px solid rgba(52, 199, 89, 0.3)",
                      transition: "all 0.3s ease",
                      "&:hover": {
                        transform: "translateY(-4px)",
                        boxShadow: "0 12px 32px rgba(52, 199, 89, 0.25)",
                      },
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
                      <Box
                        sx={{
                          fontSize: "1.5rem",
                        }}
                      >
                        🏆
                      </Box>
                      <Typography
                        variant="caption"
                        sx={{
                          fontWeight: 700,
                          textTransform: "uppercase",
                          letterSpacing: "0.1em",
                          color: "text.secondary",
                        }}
                      >
                        Best Reward
                      </Typography>
                    </Box>
                    <Typography
                      variant="h4"
                      sx={{
                        fontWeight: 800,
                        color: "#34c759",
                        fontSize: "2.5rem",
                      }}
                    >
                      {summary.reward_max?.toFixed(3) ?? "—"}
                    </Typography>
                  </Paper>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 3,
                      borderRadius: 3,
                      background: "linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)",
                      border: "2px solid rgba(102, 126, 234, 0.3)",
                      transition: "all 0.3s ease",
                      "&:hover": {
                        transform: "translateY(-4px)",
                        boxShadow: "0 12px 32px rgba(102, 126, 234, 0.25)",
                      },
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1 }}>
                      <Box
                        sx={{
                          fontSize: "1.5rem",
                        }}
                      >
                        📊
                      </Box>
                      <Typography
                        variant="caption"
                        sx={{
                          fontWeight: 700,
                          textTransform: "uppercase",
                          letterSpacing: "0.1em",
                          color: "text.secondary",
                        }}
                      >
                        Last Reward
                      </Typography>
                    </Box>
                    <Typography
                      variant="h4"
                      sx={{
                        fontWeight: 800,
                        color: "#667eea",
                        fontSize: "2.5rem",
                      }}
                    >
                      {summary.reward_last?.toFixed(3) ?? "—"}
                    </Typography>
                  </Paper>
                </Grid>
              </>
            )}
          </Grid>
        </CardContent>
      </Card>
    </Grow>
  );
}

/* ---------- helpers ---------- */

function getStateConfig(state) {
  const configs = {
    running: {
      gradient: "linear-gradient(135deg, #34c759 0%, #30d158 100%)",
      bgGradient: "linear-gradient(135deg, rgba(52, 199, 89, 0.1) 0%, rgba(48, 209, 88, 0.1) 100%)",
      borderColor: "rgba(52, 199, 89, 0.3)",
      shadow: "0 8px 24px rgba(52, 199, 89, 0.3)",
    },
    finished: {
      gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      bgGradient: "linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)",
      borderColor: "rgba(102, 126, 234, 0.3)",
      shadow: "0 8px 24px rgba(102, 126, 234, 0.3)",
    },
    error: {
      gradient: "linear-gradient(135deg, #ff3b30 0%, #ff453a 100%)",
      bgGradient: "linear-gradient(135deg, rgba(255, 59, 48, 0.1) 0%, rgba(255, 69, 58, 0.1) 100%)",
      borderColor: "rgba(255, 59, 48, 0.3)",
      shadow: "0 8px 24px rgba(255, 59, 48, 0.3)",
    },
    stopping: {
      gradient: "linear-gradient(135deg, #ff9500 0%, #ff9f0a 100%)",
      bgGradient: "linear-gradient(135deg, rgba(255, 149, 0, 0.1) 0%, rgba(255, 159, 10, 0.1) 100%)",
      borderColor: "rgba(255, 149, 0, 0.3)",
      shadow: "0 8px 24px rgba(255, 149, 0, 0.3)",
    },
  };
  return configs[state] || configs.running;
}