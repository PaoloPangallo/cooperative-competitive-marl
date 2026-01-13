import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Chip,
  Stack,
  Box,
  Paper,
  Fade,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";

import { startTraining, stopTraining } from "../../api/train.api";

/**
 * TrainingHeader
 * Top command bar for training control
 */
export default function TrainingHeader({ status }) {
  const { state, iter, max_iter } = status;

  const isRunning = state === "running";
  const isIdle = state === "idle" || state === "finished";
  const stateConfig = getStateConfig(state);

  return (
    <Fade in timeout={600}>
      <Paper
        elevation={0}
        sx={{
          mb: 4,
          borderRadius: 4,
          background: "rgba(255, 255, 255, 0.95)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.2)",
          boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)",
          overflow: "hidden",
          position: "relative",
          "&::before": {
            content: '""',
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "4px",
            background: "linear-gradient(90deg, #667eea 0%, #764ba2 100%)",
          },
        }}
      >
        <Box sx={{ p: 3 }}>
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={3}
            alignItems={{ xs: "stretch", md: "center" }}
          >
            {/* Left: title */}
            <Box sx={{ flexGrow: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Box
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: 3,
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "2rem",
                    boxShadow: "0 8px 24px rgba(102, 126, 234, 0.4)",
                  }}
                >
                  🤖
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
                      lineHeight: 1.2,
                    }}
                  >
                    MARL Training
                  </Typography>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ fontWeight: 600, mt: 0.5 }}
                  >
                    Multi-Agent Reinforcement Learning
                  </Typography>
                </Box>
              </Box>
            </Box>

            {/* Center: Stats */}
            <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
              {/* Iteration Box */}
              <Paper
                elevation={0}
                sx={{
                  px: 3,
                  py: 2,
                  borderRadius: 3,
                  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%)",
                  border: "2px solid rgba(102, 126, 234, 0.2)",
                  minWidth: 140,
                  textAlign: "center",
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
                    display: "block",
                    mb: 0.5,
                  }}
                >
                  Iteration
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 800,
                    color: "#667eea",
                  }}
                >
                  {max_iter ? `${iter} / ${max_iter}` : iter}
                </Typography>
              </Paper>

              {/* Status chip */}
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  px: 2,
                }}
              >
                <Chip
                  label={state.toUpperCase()}
                  icon={stateConfig.icon}
                  sx={{
                    background: stateConfig.gradient,
                    color: "white",
                    fontWeight: 800,
                    fontSize: "0.8rem",
                    letterSpacing: "0.05em",
                    height: 40,
                    px: 1,
                    boxShadow: stateConfig.shadow,
                    "& .MuiChip-label": {
                      px: 2,
                    },
                    "& .MuiChip-icon": {
                      color: "white",
                      fontSize: "1.2rem",
                    },
                  }}
                />
              </Box>
            </Stack>

            {/* Controls */}
            <Stack direction="row" spacing={1.5}>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={startTraining}
                disabled={!isIdle}
                sx={{
                  background: isIdle
                    ? "linear-gradient(135deg, #34c759 0%, #30d158 100%)"
                    : "rgba(0, 0, 0, 0.12)",
                  color: "white",
                  fontWeight: 700,
                  px: 3,
                  py: 1.5,
                  borderRadius: 3,
                  textTransform: "none",
                  fontSize: "1rem",
                  boxShadow: isIdle ? "0 8px 24px rgba(52, 199, 89, 0.4)" : "none",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    background: isIdle
                      ? "linear-gradient(135deg, #30d158 0%, #34c759 100%)"
                      : "rgba(0, 0, 0, 0.12)",
                    transform: isIdle ? "translateY(-2px)" : "none",
                    boxShadow: isIdle ? "0 12px 32px rgba(52, 199, 89, 0.5)" : "none",
                  },
                  "&:disabled": {
                    color: "rgba(0, 0, 0, 0.26)",
                  },
                }}
              >
                Start
              </Button>

              <Button
                variant="contained"
                startIcon={<StopIcon />}
                onClick={stopTraining}
                disabled={!isRunning}
                sx={{
                  background: isRunning
                    ? "linear-gradient(135deg, #ff3b30 0%, #ff453a 100%)"
                    : "rgba(0, 0, 0, 0.12)",
                  color: "white",
                  fontWeight: 700,
                  px: 3,
                  py: 1.5,
                  borderRadius: 3,
                  textTransform: "none",
                  fontSize: "1rem",
                  boxShadow: isRunning ? "0 8px 24px rgba(255, 59, 48, 0.4)" : "none",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    background: isRunning
                      ? "linear-gradient(135deg, #ff453a 0%, #ff3b30 100%)"
                      : "rgba(0, 0, 0, 0.12)",
                    transform: isRunning ? "translateY(-2px)" : "none",
                    boxShadow: isRunning ? "0 12px 32px rgba(255, 59, 48, 0.5)" : "none",
                  },
                  "&:disabled": {
                    color: "rgba(0, 0, 0, 0.26)",
                  },
                }}
              >
                Stop
              </Button>
            </Stack>
          </Stack>
        </Box>
      </Paper>
    </Fade>
  );
}

/* ---------- helpers ---------- */

function getStateConfig(state) {
  const configs = {
    running: {
      gradient: "linear-gradient(135deg, #34c759 0%, #30d158 100%)",
      shadow: "0 4px 12px rgba(52, 199, 89, 0.4)",
      icon: <Box component="span">▶️</Box>,
    },
    finished: {
      gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      shadow: "0 4px 12px rgba(102, 126, 234, 0.4)",
      icon: <Box component="span">✅</Box>,
    },
    error: {
      gradient: "linear-gradient(135deg, #ff3b30 0%, #ff453a 100%)",
      shadow: "0 4px 12px rgba(255, 59, 48, 0.4)",
      icon: <Box component="span">❌</Box>,
    },
    idle: {
      gradient: "linear-gradient(135deg, #8e8e93 0%, #636366 100%)",
      shadow: "0 4px 12px rgba(142, 142, 147, 0.4)",
      icon: <Box component="span">⏸️</Box>,
    },
    stopping: {
      gradient: "linear-gradient(135deg, #ff9500 0%, #ff9f0a 100%)",
      shadow: "0 4px 12px rgba(255, 149, 0, 0.4)",
      icon: <Box component="span">⏹️</Box>,
    },
  };
  return configs[state] || configs.idle;
}