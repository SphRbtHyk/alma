import React, { useState } from 'react';
import AlgoParametersMenu from './AlgoParameters.jsx';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import { Select, InputLabel, MenuItem, FormControl, Paper, Box, Alert } from '@mui/material';


const AlignmentForm = ({ setRefText, setCompText, setAlgo, setAlignmentResult }) => {
    const defaultAlgorithm = "alma";
    const [localRef,  setLocalRef]  = useState('');
    const [localComp, setLocalComp] = useState('');
    const [localAlgo, setLocalAlgo] = useState(defaultAlgorithm);
    const [params, setParams] = useState({});
    const [refTextInputError,  setRefTextInputError]  = useState(false);
    const [refHelperText,      setRefHelperText]      = useState('');
    const [compTextInputError, setCompTextInputError] = useState(false);
    const [compHelperText,     setCompHelperText]     = useState('');
    const [waitingForServerResponse, setWaitingForServerResponse] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setRefText(localRef);
        setCompText(localComp);
        setAlgo(localAlgo);

        if (localRef === '') {
          setRefTextInputError(true);
          setRefHelperText('A reference text must be provided');
        } else {
          setRefTextInputError(false);
          setRefHelperText('');
        }
        if (localComp === '') {
          setCompTextInputError(true);
          setCompHelperText('A compared text must be provided');
        } else {
          setCompTextInputError(false);
          setCompHelperText('');
        }
        if (localComp === '' || localRef === '') return null;
        if (localAlgo === 'alma' && !params.no_parse && !params.spacy_model) {
          setErrorMessage('No spaCy model selected. Please wait for the model list to load or enable "No parser" mode.');
          return null;
        }

        const payload = {
          ref_text:  localRef,
          comp_text: localComp,
          algo:      localAlgo,
          params:    params,
        };
        setErrorMessage('');
        setWaitingForServerResponse(true);
        const abortController = new AbortController();
        const timeoutMs = ['simalign', 'alma'].includes(localAlgo) ? 300_000 : 30_000;
        const timeoutID = setTimeout(() => abortController.abort(), timeoutMs);
        try {
          const response = await fetch("/alignment_request", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(payload),
            signal:  abortController.signal,
          });
          clearTimeout(timeoutID);
          if (!response.ok) {
            const body = await response.text().catch(() => '');
            const isModelError = /model|OSError|not found|load/i.test(body);
            setErrorMessage(
              isModelError
                ? "Uh oh, we couldn't find your model 😕 — check the model ID and try again."
                : `Server error (${response.status}). Please try again.`
            );
            setWaitingForServerResponse(false);
            return;
          }
          const data = await response.json();
          setWaitingForServerResponse(false);
          setAlignmentResult(data);
        } catch (error) {
          clearTimeout(timeoutID);
          setWaitingForServerResponse(false);
          setErrorMessage(
            error.name === "AbortError"
              ? "The server took too long to respond. Please try again."
              : "An error occurred while contacting the server."
          );
        }
    };

    return (
      <Paper variant="outlined" sx={{ p: { xs: 2, sm: 3 } }}>
        <form id="textInputForm" onSubmit={handleSubmit}>

          {/* ── Text inputs: side-by-side on medium+ screens ─────────────── */}
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
              gap: 2,
              mb: 2,
            }}
          >
            <TextField
              fullWidth
              variant="outlined"
              id="refTextInput"
              label="Reference text"
              placeholder="Enter reference text here …"
              multiline
              minRows={3}
              maxRows={6}
              error={refTextInputError}
              helperText={refHelperText}
              onChange={(e) => {
                setLocalRef(e.target.value.trim());
                if (localRef !== '') setRefTextInputError(false);
              }}
            />
            <TextField
              fullWidth
              variant="outlined"
              id="compTextInput"
              label="Compared text"
              placeholder="Enter compared text here …"
              multiline
              minRows={3}
              maxRows={6}
              error={compTextInputError}
              helperText={compHelperText}
              onChange={(e) => {
                setLocalComp(e.target.value.trim());
                if (localComp !== '') setCompTextInputError(false);
              }}
            />
          </Box>

          {/* ── Algorithm selector ───────────────────────────────────────── */}
          <FormControl sx={{ minWidth: 160 }}>
            <InputLabel>Algorithm</InputLabel>
            <Select
              label="Algorithm"
              value={localAlgo}
              onChange={(e) => setLocalAlgo(e.target.value)}
            >
              <MenuItem value="alma">ALMA</MenuItem>
              <MenuItem value="collatex">Collatex</MenuItem>
              <MenuItem value="pos">POS</MenuItem>
              <MenuItem value="simalign">SimAlign</MenuItem>
            </Select>
          </FormControl>

          {/* ── Algorithm-specific parameters ───────────────────────────── */}
          <AlgoParametersMenu algo={localAlgo} setParams={setParams} />

          {/* ── Error message ────────────────────────────────────────────── */}
          {errorMessage && (
            <Alert severity="error" sx={{ mt: 2 }} onClose={() => setErrorMessage('')}>
              {errorMessage}
            </Alert>
          )}

          {/* ── Submit ───────────────────────────────────────────────────── */}
          <Box sx={{ mt: 2 }}>
            <Button
              variant="contained"
              type="submit"
              size="large"
              loading={waitingForServerResponse}
              loadingPosition="end"
            >
              Get alignment
            </Button>
          </Box>

        </form>
      </Paper>
    );
};

export default AlignmentForm;
