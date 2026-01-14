import { Box, Container, Stack, Typography } from "@mui/material";
import TimelineIcon from "@mui/icons-material/Timeline";

/**
 * Navbar - Professional ML Platform Style
 * Simple, clean header with branding
 */
export default function Navbar() {
  return (
    <Box
      component="nav"
      sx={{
        background: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border-light)",
        boxShadow: "var(--shadow-sm)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "var(--sp-lg) 0",
            minHeight: "64px",
          }}
        >
          {/* Logo / Brand */}
          <Stack
            direction="row"
            spacing="var(--sp-md)"
            alignItems="center"
            sx={{ cursor: "pointer" }}
          >
            <Box
              sx={{
                width: 40,
                height: 40,
                borderRadius: "var(--radius-md)",
                background: "var(--accent-primary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s ease",
                "&:hover": {
                  transform: "scale(1.05)",
                  boxShadow: "var(--shadow-md)",
                },
              }}
            >
              <TimelineIcon sx={{ color: "white", fontSize: 20 }} />
            </Box>

            <Box>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  fontSize: "1.125rem",
                  lineHeight: 1.2,
                }}
              >
                MARL Training
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: "var(--text-tertiary)",
                  fontWeight: 500,
                  display: "block",
                }}
              >
                Multi-Agent Reinforcement Learning
              </Typography>
            </Box>
          </Stack>

          {/* Right section - Status indicator (optional) */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: "var(--sp-md)",
            }}
          >
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "var(--accent-primary)",
                animation: "pulse 2s ease-in-out infinite",
              }}
            />
            <Typography
              variant="caption"
              sx={{
                color: "var(--text-secondary)",
                fontWeight: 500,
              }}
            >
              Ready
            </Typography>
          </Box>
        </Box>
      </Container>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </Box>
  );
}