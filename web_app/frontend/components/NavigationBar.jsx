import { Box, Tabs, Tab, FormControlLabel, Switch, Typography } from '@mui/material';

const PAGES = [
  { key: "aligner", label: "Aligner" },
  { key: "about",   label: "About"   },
  { key: "contact", label: "Contact" },
];

/**
 * React navigation bar component.
 * @param {function} setCurrentPage The function used to set the application page.
 * @param {string} currentPage The currently active page key.
 * @param {string} mode The current colour-scheme mode ("light" | "dark").
 * @param {function} switchThemes Toggle handler for the dark/light switch.
 */
export default function Navbar({ setCurrentPage, currentPage, mode, switchThemes }) {
  return (
    <Box
      sx={{
        borderBottom: 1,
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'space-between',
        mb: 4,
      }}
    >
      <Tabs
        value={currentPage}
        onChange={(_, v) => setCurrentPage(v)}
        textColor="primary"
        indicatorColor="primary"
      >
        {PAGES.map(({ key, label }) => (
          <Tab key={key} label={label} value={key} disableRipple />
        ))}
      </Tabs>

      <Box sx={{ pb: 1 }}>
        <FormControlLabel
          control={
            <Switch onChange={switchThemes} checked={mode === "dark"} size="small" />
          }
          label={
            <Typography variant="body2" color="text.secondary">
              {mode === 'dark' ? 'Dark' : 'Light'}
            </Typography>
          }
        />
      </Box>
    </Box>
  );
}
