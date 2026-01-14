import {
  Container,
  Box,
  Paper,
  Typography,
  Fade,
  CircularProgress,
} from "@mui/material";

import { useTrainStatus } from "../../hooks/useTrainStatus";
import { useTrainMetrics } from "../../hooks/useTrainMetrics";

import TrainingHeader from "./TrainingHeader";
import TrainingOverview from "./TrainingOverview";
import WalkerStatusPanel from "./WalkerStatusPanel";
import MetricsCharts from "./MetricsChart";

export default function TrainingPage() {
  const { status, error: statusError } = useTrainStatus();
  const {
    metrics,
    summary,
    error: metricsError,
  } = useTrainMetrics();

  /* ---------- ERRORI ---------- */

  if (statusError || metricsError) {
    const message = statusError || metricsError;

    return (
      <Container maxWidth={false} sx={{ py: 4 }}>
        <Fade in timeout={600}>
          <Paper
            elevation={0}
            sx={{
              p: 4,
              borderRadius: 4,
              background:
                "linear-gradient(135deg, rgba(255, 59, 48, 0.95) 0%, rgba(255, 69, 58, 0.95) 100%)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              boxShadow: "0 20px 60px rgba(255, 59, 48, 0.4)",
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Typography variant="h4" sx={{ fontSize: "3rem" }}>
                ❌
              </Typography>
              <Box>
                <Typography
                  variant="h6"
                  sx={{ color: "white", fontWeight: 700, mb: 0.5 }}
                >
                  Training Error
                </Typography>
                <Typography
                  variant="body1"
                  sx={{ color: "rgba(255, 255, 255, 0.9)" }}
                >
                  {message}
                </Typography>
              </Box>
            </Box>
          </Paper>
        </Fade>
      </Container>
    );
  }

  /* ---------- LOADING STATUS ---------- */

  if (!status) {
    return (
      <Container maxWidth="lg" sx={{ pt: 8 }}>
        <Fade in timeout={600}>
          <Paper
            elevation={0}
            sx={{
              p: 6,
              borderRadius: 4,
              background: "rgba(255, 255, 255, 0.95)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)",
              textAlign: "center",
            }}
          >
            <CircularProgress
              size={60}
              thickness={4}
              sx={{ color: "#667eea", mb: 3 }}
            />
            <Typography
              variant="h6"
              sx={{ color: "#667eea", fontWeight: 600 }}
            >
              Loading training data…
            </Typography>
          </Paper>
        </Fade>
      </Container>
    );
  }

  /* ---------- MAIN ---------- */

 return (
  <Box
    sx={{
      maxWidth: 1600,
      mx: "auto",      // 🔥 CENTRA ORIZZONTALMENTE
      px: 3,           // padding laterale
      py: 4,
    }}
  >
    <TrainingHeader status={status} />
    <TrainingOverview status={status} summary={summary} metrics={metrics} />
    <WalkerStatusPanel multiwalker={status.multiwalker} />
    <MetricsCharts metrics={metrics} summary={summary} />
  </Box>
);

}
