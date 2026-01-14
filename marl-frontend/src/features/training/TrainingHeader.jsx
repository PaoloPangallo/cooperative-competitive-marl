import {
  Typography,
  Button,
  Chip,
  Stack,
  Box,
  Paper,
  Fade,
  LinearProgress,
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import PauseIcon from "@mui/icons-material/Pause";
import AutorenewIcon from "@mui/icons-material/Autorenew";

import { startTraining, stopTraining } from "../../api/train.api";

/**
 * TrainingHeader - Clean ML Platform Design
 * Global command bar for MARL training
 */
export default function TrainingHeader({ status, phase }) {
  if (!status) return null;

  const { state, iter, max_iter, progress } = status;

  const isRunning = state === "running";
  const isIdle = state === "idle" || state === "finished";
  const stateConfig = getStateConfig(state);
  const pct = max_iter ? Math.round((progress || 0) * 100) : 0;

  return (
    <Fade in timeout={600}>
      <Paper
        elevation={0}
        className="card"
        sx={{
          mb: "var(--sp-lg)",
          borderTop: "3px solid var(--accent-primary)",
        }}
      >
        <Box sx={{ p: "var(--sp-lg)" }}>
          <Stack spacing="var(--sp-lg)">
            {/* ================= TITLE + STATUS ================= */}
            <Stack
              direction={{ xs: "column", md: "row" }}
              spacing="var(--sp-lg)"
              alignItems={{ xs: "stretch", md: "center" }}
              sx={{ justifyContent: "space-between" }}
            >
              {/* Left: Title */}
              <Box>
                <Typography
                  variant="h5"
                  sx={{
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    mb: "var(--sp-sm)",
                  }}
                >
                  Training Session
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    color: "var(--text-secondary)",
                    fontWeight: 500,
                  }}
                >
                  Multi-Agent Reinforcement Learning
                </Typography>
              </Box>

              {/* Right: State Badge */}
              <Chip
                icon={stateConfig.icon}
                label={stateConfig.label}
                sx={{
                  background: stateConfig.background,
                  color: stateConfig.textColor,
                  fontWeight: 700,
                  fontSize: "0.875rem",
                  height: 40,
                  px: "var(--sp-md)",
                  border: `1px solid ${stateConfig.borderColor}`,
                  "& .MuiChip-icon": {
                    color: "inherit",
                    fontSize: "1.2rem",
                  },
                  transition: "all 0.2s ease",
                  "&:hover": {
                    boxShadow: "var(--shadow-md)",
                  },
                }}
              />
            </Stack>

            {/* ================= PROGRESS BAR ================= */}
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
                  Progress
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    fontWeight: 700,
                    color: "var(--text-primary)",
                  }}
                >
                  {max_iter ? `${iter} / ${max_iter}` : iter} iterations
                </Typography>
              </Stack>
              <LinearProgress
                variant="determinate"
                value={pct}
                sx={{
                  height: 6,
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-tertiary)",
                  "& .MuiLinearProgress-bar": {
                    background: "var(--accent-primary)",
                    borderRadius: "var(--radius-md)",
                  },
                }}
              />
              <Typography
                variant="caption"
                sx={{
                  color: "var(--text-tertiary)",
                  mt: "var(--sp-sm)",
                  display: "block",
                  fontWeight: 500,
                }}
              >
                {pct}% complete {max_iter && `· ${max_iter - iter} remaining`}
              </Typography>
            </Box>

            {/* ================= CONTROLS ================= */}
            <Stack direction="row" spacing="var(--sp-md)" sx={{ pt: "var(--sp-sm)" }}>
              <Button
                variant="contained"
                startIcon={<PlayArrowIcon />}
                onClick={startTraining}
                disabled={!isIdle}
                sx={{
                  background: isIdle ? "var(--accent-primary)" : "var(--text-tertiary)",
                  color: "white",
                  fontWeight: 600,
                  px: "var(--sp-lg)",
                  py: "0.625rem",
                  borderRadius: "var(--radius-md)",
                  textTransform: "none",
                  fontSize: "0.95rem",
                  border: "none",
                  cursor: isIdle ? "pointer" : "not-allowed",
                  transition: "all 0.2s ease",
                  "&:hover": isIdle
                    ? {
                        background: "#0d8c6f",
                        boxShadow: "var(--shadow-md)",
                      }
                    : {},
                }}
              >
                Start Training
              </Button>

              <Button
                variant="contained"
                startIcon={<StopIcon />}
                onClick={stopTraining}
                disabled={!isRunning}
                sx={{
                  background: isRunning ? "#ef4444" : "var(--text-tertiary)",
                  color: "white",
                  fontWeight: 600,
                  px: "var(--sp-lg)",
                  py: "0.625rem",
                  borderRadius: "var(--radius-md)",
                  textTransform: "none",
                  fontSize: "0.95rem",
                  border: "none",
                  cursor: isRunning ? "pointer" : "not-allowed",
                  transition: "all 0.2s ease",
                  "&:hover": isRunning
                    ? {
                        background: "#dc2626",
                        boxShadow: "var(--shadow-md)",
                      }
                    : {},
                }}
              />

              {phase && (
                <Chip
                  label={`Phase: ${phase}`}
                  sx={{
                    background: "var(--bg-tertiary)",
                    color: "var(--text-primary)",
                    fontWeight: 600,
                    border: "1px solid var(--border-color)",
                    ml: "auto",
                  }}
                />
              )}
            </Stack>
          </Stack>
        </Box>
      </Paper>
    </Fade>
  );
}

/* ==================================================
   STATE CONFIG - Icons + Colors
   ================================================== */

function getStateConfig(state) {
  const configs = {
    running: {
      label: "RUNNING",
      icon: <AutorenewIcon className="animate-spin" />,
      background: "rgba(16, 160, 127, 0.1)",
      textColor: "var(--accent-primary)",
      borderColor: "rgba(16, 160, 127, 0.3)",
    },
    finished: {
      label: "FINISHED",
      icon: <CheckCircleIcon />,
      background: "rgba(13, 70, 161, 0.1)",
      textColor: "var(--accent-secondary)",
      borderColor: "rgba(13, 70, 161, 0.3)",
    },
    error: {
      label: "ERROR",
      icon: <ErrorIcon />,
      background: "rgba(239, 68, 68, 0.1)",
      textColor: "#ef4444",
      borderColor: "rgba(239, 68, 68, 0.3)",
    },
    idle: {
      label: "IDLE",
      icon: <PauseIcon />,
      background: "rgba(149, 157, 165, 0.1)",
      textColor: "var(--text-secondary)",
      borderColor: "rgba(149, 157, 165, 0.3)",
    },
    stopping: {
      label: "STOPPING",
      icon: <AutorenewIcon className="animate-spin" />,
      background: "rgba(217, 119, 6, 0.1)",
      textColor: "var(--accent-tertiary)",
      borderColor: "rgba(217, 119, 6, 0.3)",
    },
  };

  return configs[state] || configs.idle;
}