import { Box, Container } from "@mui/material";
import TrainingPage from "./features/training/TrainingPage";
import "./App.css";
import Navbar from "./component/layout/navbar.jsx";

export default function App() {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "var(--bg-primary)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flex: 1,
          background: "var(--bg-primary)",
          paddingTop: "var(--sp-xl)",
          paddingBottom: "var(--sp-xl)",
        }}
      >
        <Container maxWidth={false} disableGutters>
          <TrainingPage />
        </Container>
      </Box>

      {/* Footer (optional) */}
      <Box
        component="footer"
        sx={{
          background: "var(--bg-secondary)",
          borderTop: "1px solid var(--border-light)",
          padding: "var(--sp-lg)",
          textAlign: "center",
        }}
      >
        <Box
          sx={{
            maxWidth: "1280px",
            margin: "0 auto",
            fontSize: "0.875rem",
            color: "var(--text-tertiary)",
          }}
        >
          Multi-Agent RL Training Platform © 2024
        </Box>
      </Box>
    </Box>
  );
}