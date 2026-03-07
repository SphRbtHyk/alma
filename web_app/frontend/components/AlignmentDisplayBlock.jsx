import { useState, useEffect } from 'react';
import ContentPasteIcon from '@mui/icons-material/ContentPaste';
import DownloadIcon from '@mui/icons-material/Download';
import { Paper, Box, Typography, IconButton, Divider, Tooltip, Button } from '@mui/material';
import generatePalette from './colorPaletGen.js';
import AnalysisResultTab from './AnalysisResultTable.jsx';
import './AlignmentDisplayBlock.css';

// ── Cell colour palette ─────────────────────────────────────────────────────
// Material Design 100-level pastels: light enough to read black text on them,
// subtle enough not to hurt the eyes.
const UNMATCH_COLOR             = "#FFCDD2";  // Red 100
const MATCH_COLOR               = "#C8E6C9";  // Green 100
const TRANSPOSITION_BASE_PALETTE = [
  "#FFF9C4",  // Yellow 100
  "#B2EBF2",  // Cyan 100
  "#FFE0B2",  // Orange 100
  "#E1BEE7",  // Purple 100
  "#B2DFDB",  // Teal 100
  "#FFCCBC",  // Deep Orange 100
  "#DCEDC8",  // Light Green 100
];


// ── CopyButton ──────────────────────────────────────────────────────────────
function CopyButton({ targetId }) {
  const [showPopup, setShowPopup] = useState(false);

  const handleCopy = () => {
    const textToCopy = document.getElementById(targetId).innerText;
    navigator.clipboard.writeText(textToCopy).then(() => {
      setShowPopup(true);
      setTimeout(() => setShowPopup(false), 1500);
    });
  };

  return (
    <Box sx={{ position: "relative", display: "inline-block", ml: 1 }}>
      <Tooltip title="Copy raw index pairs">
        <IconButton size="small" onClick={handleCopy} className="clipboardBtn">
          <ContentPasteIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      {showPopup && <div className="copyPopup">Copied!</div>}
    </Box>
  );
}

// ── CopyValueButton ──────────────────────────────────────────────────────────
function CopyValueButton({ getText, tooltip = "Copy" }) {
  const [showPopup, setShowPopup] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(getText()).then(() => {
      setShowPopup(true);
      setTimeout(() => setShowPopup(false), 1500);
    });
  };

  return (
    <Box sx={{ position: "relative", display: "inline-block", ml: 1 }}>
      <Tooltip title={tooltip}>
        <IconButton size="small" onClick={handleCopy} className="clipboardBtn">
          <ContentPasteIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      {showPopup && <div className="copyPopup">Copied!</div>}
    </Box>
  );
}

// ── exportCsv ────────────────────────────────────────────────────────────────
function exportCsv(refText, compText, result) {
  const refWords  = refText.trim().split(/\s+/);
  const compWords = compText.trim().split(/\s+/);

  const rows = [['Type', 'Reference Index', 'Compared Index', 'Reference Token', 'Compared Token']];

  result.matches.forEach(([ri, ci]) => {
    const isTransposition = Number.parseInt(ri) !== Number.parseInt(ci);
    rows.push([
      isTransposition ? 'TRANSPOSITION' : 'MATCH',
      ri, ci,
      refWords[ri]  ?? '',
      compWords[ci] ?? '',
    ]);
  });

  result.unmatches.forEach(([r, c]) => {
    if (Number.isInteger(r) && Number.isInteger(c)) {
      rows.push(['SUBSTITUTION', r, c, refWords[r] ?? '', compWords[c] ?? '']);
    } else if (Number.isInteger(r)) {
      rows.push(['ADDITION OR DELETION', r, '', refWords[r] ?? '', '']);
    } else {
      rows.push(['ADDITION OR DELETION', '', c, '', compWords[c] ?? '']);
    }
  });

  const csv  = rows.map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'alignment_results.csv';
  a.click();
  URL.revokeObjectURL(url);
}


// ── AlignmentDisplayBlock ───────────────────────────────────────────────────
const AlignmentDisplayBlock = ({ refText, compText, result }) => {
  const unmatchColor             = UNMATCH_COLOR;
  const matchColor               = MATCH_COLOR;
  const transpositionBasePalette = TRANSPOSITION_BASE_PALETTE;

  const [localRefAlignedArray2,  setLocalRefAlignedArray2]  = useState([]);
  const [localCompAlignedArray2, setLocalCompAlignedArray2] = useState([]);
  const [refColorDict,   setRefColorDict]   = useState({});
  const [compColorDict,  setCompColorDict]  = useState({});
  const [sameColorDict,  setSameColorDict]  = useState({});

  const alignWordsFromResult = (refText, compText, result) => {
    let matchIdx = 0;
    let unmatchRefIdx = 0;
    let unmatchCompIdx = 0;
    let refArray = refText.trim(" ").split(" ");
    let compArray = compText.trim(" ").split(" ");
    let outputRefArray = [];
    let outputCompArray = [];
    let outputColors = [];
    let sortedUnmatchesRef = [];
    let sortedUnmatchesComp = [];
    result.unmatches.map((e) => {
      if (Number.isInteger(e[0])) sortedUnmatchesRef.push(parseFloat(e[0]));
      else                        sortedUnmatchesComp.push(parseFloat(e[1]));
    });
    sortedUnmatchesRef.sort((a, b) => (a > b ? 0 : -1));
    sortedUnmatchesComp.sort((a, b) => (a > b ? 0 : -1));
    let sortedMatches = result.matches.sort((a, b) => (a[0] > b[0] ? 0 : -1));
    for (let i = 0; i < sortedMatches.length + sortedUnmatchesRef.length + sortedUnmatchesComp.length; i++) {
      if (matchIdx < sortedMatches.length) {
        if (sortedUnmatchesRef[unmatchRefIdx] < sortedMatches[matchIdx][0]) {
          outputRefArray.push(refArray[sortedUnmatchesRef[unmatchRefIdx]]);
          outputCompArray.push("-");
          outputColors.push(unmatchColor);
          unmatchRefIdx++;
        } else if (sortedUnmatchesComp[unmatchCompIdx] < sortedMatches[matchIdx][1]) {
          outputCompArray.push(compArray[sortedUnmatchesComp[unmatchCompIdx]]);
          outputRefArray.push("-");
          outputColors.push(unmatchColor);
          unmatchCompIdx++;
        } else {
          outputRefArray.push(refArray[sortedMatches[matchIdx][0]]);
          outputCompArray.push(compArray[sortedMatches[matchIdx][1]]);
          outputColors.push(matchColor);
          matchIdx++;
        }
      } else {
        if (unmatchCompIdx >= sortedUnmatchesComp.length || sortedUnmatchesRef[unmatchRefIdx] <= sortedUnmatchesComp[unmatchCompIdx]) {
          outputRefArray.push(refArray[sortedUnmatchesRef[unmatchRefIdx]]);
          outputCompArray.push("-");
          outputColors.push(unmatchColor);
          unmatchRefIdx++;
        } else {
          outputCompArray.push(compArray[sortedUnmatchesComp[unmatchCompIdx]]);
          outputRefArray.push("-");
          outputColors.push(unmatchColor);
          unmatchCompIdx++;
        }
      }
    }
    return [outputRefArray, outputCompArray, outputColors];
  };

  const buildRefDicts = (matches, colorPalet) => {
    let refOut = new Map();
    let compOut = new Map();
    let sameOut = new Map();
    let paletteIdx = 0;
    for (let i = 0; i < [...matches].length; i++) {
      if (Number.parseInt(matches[i][0]) !== Number.parseInt(matches[i][1])) {
        refOut[Number.parseInt(matches[i][0])]  = colorPalet[paletteIdx];
        compOut[Number.parseInt(matches[i][1])] = colorPalet[paletteIdx];
        paletteIdx++;
      } else {
        sameOut[Number.parseInt(matches[i][0])] = matchColor;
      }
    }
    return [refOut, compOut, sameOut];
  };

  function shiftIndexes(result, refAlignedArray, compAlignedArray, originalCompTextArray, originalRefTextArray) {
    let newMatches = new Array([...result["matches"]].length);
    for (let i = 0; i < newMatches.length; i++) newMatches[i] = new Array(2);
    let newCompAlignedArray = new Array(compAlignedArray.length);
    let newRefAlignedArray  = new Array(refAlignedArray.length);
    let refShiftNb  = 0;
    let compShiftNb = 0;
    let unmatchesRefDict  = {};
    let unmatchesCompDict = {};
    for (let i = 0; i < [...result["unmatches"]].length + [...result["matches"]].length; i++) {
      if (i < [...result["unmatches"]].length && result["unmatches"][i][0] !== "-") {
        unmatchesRefDict[Number.parseInt(result["unmatches"][i][0])]  = unmatchColor;
      } else if (i < [...result["unmatches"]].length) {
        unmatchesCompDict[Number.parseInt(result["unmatches"][i][1])] = unmatchColor;
      }
    }
    for (let i = 0; i < refAlignedArray.length; i++) {
      newCompAlignedArray[i] = originalCompTextArray[i - compShiftNb];
      newRefAlignedArray[i]  = originalRefTextArray[i - refShiftNb];
      if (i - refShiftNb in unmatchesRefDict) {
        newCompAlignedArray[i] = "-";
        compShiftNb++;
      } else if (i - compShiftNb in unmatchesCompDict) {
        newRefAlignedArray[i] = "-";
        refShiftNb++;
      } else if (i - refShiftNb - compShiftNb < [...result["matches"]].length) {
        let idx = i - refShiftNb - compShiftNb;
        newMatches[idx][0] = Number.parseInt(result["matches"][idx][0]) + refShiftNb;
        newMatches[idx][1] = Number.parseInt(result["matches"][idx][1]) + compShiftNb;
      }
    }
    return [newMatches, newCompAlignedArray, newRefAlignedArray];
  }

  function contTranspositions(matches) {
    let count = 0;
    for (let i = 0; i < matches.length; i++) {
      if (matches[i][0] !== matches[i][1]) count++;
    }
    return count;
  }

  useEffect(() => {
    if (!result) return;
    const [refAlignedArray, compAlignedArray] = alignWordsFromResult(refText, compText, result);
    const compTextArray = compText.split(" ");
    const refTextArray  = refText.split(" ");
    const [newMatches, compAlignedArray2, refAlignedArray2] = shiftIndexes(
      result, refAlignedArray, compAlignedArray, compTextArray, refTextArray,
    );
    const colorPalette = generatePalette(transpositionBasePalette, contTranspositions(newMatches));
    setLocalCompAlignedArray2(compAlignedArray2);
    setLocalRefAlignedArray2(refAlignedArray2);
    const [refDict, compDict, sameDict] = buildRefDicts([...newMatches], colorPalette);
    setRefColorDict(refDict);
    setCompColorDict(compDict);
    setSameColorDict(sameDict);
  }, [result]);

  function pickColor(refDict, sameDict, idx) {
    if (refDict[idx])  return refDict[idx];
    if (sameDict[idx]) return matchColor;
    return unmatchColor;
  }

  // ── Empty state ────────────────────────────────────────────────────────────
  if (!result) {
    return (
      <Box className="noResultsPlaceholder">
        <Typography variant="body2" color="text.disabled" fontStyle="italic">
          Submit the form above to see the alignment result.
        </Typography>
      </Box>
    );
  }

  // ── Results ────────────────────────────────────────────────────────────────
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>

      {/* Raw index pairs */}
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Alignment Result</Typography>

        <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Matches
        </Typography>
        <Box className="matchPairDiv" id="matchPairsDiv" sx={{ mb: 2 }}>
          <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
            {result.matches.map((e, idx) => (
              <span key={idx} className="matchNbPaire">
                {`[${e[0]}, ${e[1]}]${idx < result.matches.length - 1 ? ', ' : ''}`}
              </span>
            ))}
          </Box>
          <CopyButton targetId="matchPairsDiv" />
        </Box>

        <Divider sx={{ mb: 1.5 }} />

        <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
          Unmatches
        </Typography>
        <Box className="matchPairDiv" id="unmatchPairsDiv">
          <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
            {result.unmatches.map((e, idx) => (
              <span key={idx} className="unmatchNbPaire">
                {`[${e[0]}, ${e[1]}]${idx < result.unmatches.length - 1 ? ', ' : ''}`}
              </span>
            ))}
          </Box>
          <CopyButton targetId="unmatchPairsDiv" />
        </Box>
      </Paper>

      {/* Matched pairs */}
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6">Matched Pairs</Typography>
          <CopyValueButton
            tooltip="Copy matched pairs with word values"
            getText={() => {
              const refWords  = refText.trim().split(/\s+/);
              const compWords = compText.trim().split(/\s+/);
              return result.matches
                .map(([ri, ci]) => `${refWords[ri] ?? '?'} ↔ ${compWords[ci] ?? '?'}`)
                .join(', ');
            }}
          />
        </Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {result.matches.map(([refIdx, compIdx], i) => {
            const refWord  = refText.trim().split(/\s+/)[refIdx]  ?? '?';
            const compWord = compText.trim().split(/\s+/)[compIdx] ?? '?';
            return (
              <Box
                key={i}
                sx={{
                  display: 'flex', alignItems: 'center', gap: 0.75,
                  border: '1px solid', borderColor: 'divider', borderRadius: 2,
                  px: 1.5, py: 0.5, bgcolor: 'background.paper',
                }}
              >
                <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>
                  {refWord}
                </Typography>
                <Typography variant="body2" color="text.disabled">↔</Typography>
                <Typography variant="body2" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                  {compWord}
                </Typography>
              </Box>
            );
          })}
          {result.matches.length === 0 && (
            <Typography variant="body2" color="text.disabled" fontStyle="italic">No matches found.</Typography>
          )}
        </Box>
      </Paper>

      {/* Unmatched pairs */}
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6">Unmatched Words</Typography>
          <CopyValueButton
            tooltip="Copy unmatched words"
            getText={() => {
              const refWords  = refText.trim().split(/\s+/);
              const compWords = compText.trim().split(/\s+/);
              return result.unmatches
                .map(([r, c]) => {
                  if (Number.isInteger(r) && Number.isInteger(c))
                    return `${refWords[r] ?? '?'} → ${compWords[c] ?? '?'}`;
                  return Number.isInteger(r)
                    ? `${refWords[r] ?? '?'} (ref)`
                    : `${compWords[c] ?? '?'} (comp)`;
                })
                .join(', ');
            }}
          />
        </Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {result.unmatches.map((e, i) => {
            const isSubstitution = Number.isInteger(e[0]) && Number.isInteger(e[1]);
            if (isSubstitution) {
              const refWord  = refText.trim().split(/\s+/)[e[0]] ?? '?';
              const compWord = compText.trim().split(/\s+/)[e[1]] ?? '?';
              return (
                <Box
                  key={i}
                  sx={{
                    display: 'flex', alignItems: 'center', gap: 0.75,
                    border: '1px solid', borderColor: 'error.light', borderRadius: 2,
                    px: 1.5, py: 0.5, bgcolor: 'background.paper',
                  }}
                >
                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>{refWord}</Typography>
                  <Typography variant="body2" color="text.disabled">→</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'secondary.main' }}>{compWord}</Typography>
                </Box>
              );
            }
            const isRefUnmatch = Number.isInteger(e[0]);
            const word = isRefUnmatch
              ? refText.trim().split(/\s+/)[e[0]]  ?? '?'
              : compText.trim().split(/\s+/)[e[1]] ?? '?';
            return (
              <Box
                key={i}
                sx={{
                  display: 'flex', alignItems: 'center', gap: 0.75,
                  border: '1px solid', borderColor: 'error.light', borderRadius: 2,
                  px: 1.5, py: 0.5, bgcolor: 'background.paper',
                }}
              >
                <Typography
                  variant="body2"
                  sx={{ fontWeight: 600, color: isRefUnmatch ? 'primary.main' : 'secondary.main' }}
                >
                  {word}
                </Typography>
                <Typography variant="caption" color="text.disabled" sx={{ ml: 0.25 }}>
                  {isRefUnmatch ? 'ref' : 'comp'}
                </Typography>
              </Box>
            );
          })}
          {result.unmatches.length === 0 && (
            <Typography variant="body2" color="text.disabled" fontStyle="italic">All words matched.</Typography>
          )}
        </Box>
      </Paper>

      {/* CSV Export */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="outlined"
          size="small"
          startIcon={<DownloadIcon />}
          onClick={() => exportCsv(refText, compText, result)}
        >
          Export CSV
        </Button>
      </Box>

      {/* Alignment table */}
      {/* <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>Alignment Table</Typography>
        <div className="tableDiv2">
          <table className="alignmentTable" id="alignmentTable" key={result}>
            <thead>
              <tr>
                <th></th>
                {localRefAlignedArray2.map((_, idx) => (
                  <th key={idx} scope="col">{idx}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="rowLabel">Ref</td>
                {localRefAlignedArray2.map((word, idx) => (
                  <td key={idx} className="wordCell" style={{ backgroundColor: pickColor(refColorDict, sameColorDict, idx) }}>{word}</td>
                ))}
              </tr>
              <tr>
                <td className="rowLabel">Comp</td>
                {localCompAlignedArray2.map((word, idx) => (
                  <td key={idx} className="wordCell" style={{ backgroundColor: pickColor(compColorDict, sameColorDict, idx) }}>{word}</td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </Paper> */}

      {/* Summary table */}
      {/* <Paper variant="outlined" sx={{ p: 3 }}>
        <AnalysisResultTab
          results={result}
          refTextArray={refText.split(" ")}
          compTextArray={compText.split(" ")}
        />
      </Paper> */}

    </Box>
  );
};

export default AlignmentDisplayBlock;
