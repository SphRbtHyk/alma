import { Box, Typography, Paper, Link, Chip } from '@mui/material';

const ALGORITHMS = [
  {
    name:  'Collatex',
    label: 'Edit-distance',
    desc:  'Classical collation based on the Levenshtein edit distance.',
    href:  'https://github.com/interedition/collatex',
  },
  {
    name:  'POS',
    label: 'Part-of-speech',
    desc:  'Part-of-speech–based word alignment for monolingual text pairs.',
    href:  'https://github.com/ma-sultan/monolingual-word-aligner',
  },
  {
    name:  'SimAlign',
    label: 'Embedding-based',
    desc:  'Multilingual word alignment using contextual embedding similarity.',
    href:  'https://github.com/cisnlp/simalign',
  },
  {
    name:  'ALMA',
    label: 'Semantic + linguistic',
    desc:  'A novel algorithm combining semantic embeddings and syntactic dependency analysis for ancient text alignment.',
    href:  'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
  },
];


const AboutPage = () => (
  <Box sx={{ maxWidth: 780, mx: 'auto', py: 2 }}>

    {/* ── Description ──────────────────────────────────────────────────────── */}
    <Typography variant="body1" sx={{ lineHeight: 1.85, mb: 4 }}>
      <strong>ALMA</strong> (Automated Alignment for Manuscript Analysis) is a
      Python package for the automatic alignment and collation of ancient texts.
      It exposes a unified API over several collation algorithms, ranging from
      classical edit-distance methods to a novel approach based on semantic and
      linguistic analysis.
    </Typography>

    {/* ── Algorithms ───────────────────────────────────────────────────────── */}
    <Typography variant="h6" sx={{ mb: 2 }}>Algorithms</Typography>
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {ALGORITHMS.map(({ name, label, desc, href }) => (
        <Paper key={name} variant="outlined" sx={{ p: 2.5, display: 'flex', gap: 2, alignItems: 'flex-start' }}>
          <Box sx={{ flexGrow: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Typography variant="subtitle1" fontWeight={600} color="primary.main">
                {name}
              </Typography>
              <Chip label={label} size="small" variant="outlined" sx={{ fontSize: '0.7rem', height: 20 }} />
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.65 }}>
              {desc}
            </Typography>
          </Box>
          <Link href={href} target="_blank" rel="noreferrer" sx={{ whiteSpace: 'nowrap', fontSize: '0.8125rem', mt: 0.25 }}>
            GitHub ↗
          </Link>
        </Paper>
      ))}
    </Box>

  </Box>
);

export default AboutPage;
