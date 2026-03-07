import { useState, useMemo } from 'react'
import { createTheme, ThemeProvider, CssBaseline } from "@mui/material";
import { Box, Typography } from "@mui/material";
import './App.css'
import AlignerContentBlock from '../components/AlignerContentBlock.jsx'
import Navbar from '../components/NavigationBar.jsx'
import AboutPage from '../components/AboutPage.jsx';


export default function App() {
  const [currentPageContent, setCurrentPage] = useState("aligner")
  const [mode, setMode] = useState("dark");

  const theme = useMemo(() => createTheme({
    palette: {
      mode,
      primary:   { main: '#5C6BC0' },  // indigo
      secondary: { main: '#26C6DA' },  // cyan
    },
    typography: {
      fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
      h1: { fontWeight: 800, letterSpacing: '-0.03em' },
      h2: { fontWeight: 700, letterSpacing: '-0.02em' },
      h3: { fontWeight: 700, letterSpacing: '-0.02em' },
      h4: { fontWeight: 600 },
      h5: { fontWeight: 600 },
      h6: { fontWeight: 600 },
      // Disable all-caps on buttons
      button: { textTransform: 'none', fontWeight: 500 },
    },
    shape: { borderRadius: 10 },
    components: {
      MuiButton:     { defaultProps:   { disableElevation: true } },
      MuiPaper:      { styleOverrides: { root: { backgroundImage: 'none' } } },
      MuiTextField:  { defaultProps:   { size: 'small' } },
      MuiSelect:     { defaultProps:   { size: 'small' } },
      MuiInputLabel: { defaultProps:   { size: 'small' } },
    },
  }), [mode]);

  function switchThemes(event) {
    setMode(event.target.checked ? "dark" : "light");
  }

  const renderPage = () => {
    switch (currentPageContent) {
      case "aligner": return <AlignerContentBlock />;
      case "about":   return <AboutPage />;
      case "contact": return (
        <Box sx={{ textAlign: 'center', py: 10 }}>
          <Typography variant="h5" gutterBottom>Contact</Typography>
          <Typography color="text.secondary">Coming soon.</Typography>
        </Box>
      );
      default: return null;
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div className="app-wrapper">

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <Box component="header" sx={{ textAlign: 'center', pt: 5, pb: 3 }}>
          <Typography
            variant="h3"
            component="h1"
            color="primary"
            sx={{ fontWeight: 800, letterSpacing: '-0.04em' }}
          >
            ALMA
          </Typography>
          <Typography
            variant="overline"
            color="text.secondary"
            sx={{ display: 'block', mt: 0.5, letterSpacing: '0.14em' }}
          >
            Automated Alignment for Manuscript Analysis
          </Typography>
        </Box>

        {/* ── Navigation ─────────────────────────────────────────────────── */}
        <Navbar
          setCurrentPage={setCurrentPage}
          currentPage={currentPageContent}
          mode={mode}
          switchThemes={switchThemes}
        />

        {/* ── Page content ───────────────────────────────────────────────── */}
        <main>
          {renderPage()}
        </main>

      </div>
    </ThemeProvider>
  )
}
