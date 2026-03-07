import { useState, useEffect } from 'react';
import { NumericFormat } from 'react-number-format';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import {
  Select,
  InputLabel,
  MenuItem,
  FormControl,
  Box,
  Checkbox,
  FormControlLabel,
  TextField,
  Typography,
  Divider,
  Tooltip,
} from '@mui/material';

// ── Single source of truth for embedding model dropdowns ─────────────────────
const EMBEDDING_MODELS = [
  { value: 'FacebookAI/xlm-roberta-base',                label: 'XLM-RoBERTa' },
  { value: 'google-bert/bert-base-multilingual-cased',   label: 'mBERT'       },
];

const DEFAULT_MODEL = EMBEDDING_MODELS[0].value;
const OTHER_VALUE   = '__other__';

// ── Small info icon that shows a tooltip on hover ────────────────────────────
function InfoIcon({ title }) {
  return (
    <Tooltip title={title} placement="top" arrow>
      <InfoOutlinedIcon sx={{ fontSize: 14, color: 'text.disabled', cursor: 'help', flexShrink: 0 }} />
    </Tooltip>
  );
}

// ── Checkbox label with inline info icon ─────────────────────────────────────
function CheckboxLabel({ label, tooltip }) {
  return (
    <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
      {label}
      <InfoIcon title={tooltip} />
    </Box>
  );
}

// ── Reusable embedding-model selector with free-text fallback ────────────────
function EmbeddingModelSelect({ label, paramKey, defaultVal, onUpdate, tooltip }) {
  const [selected, setSelected] = useState(defaultVal);
  const [custom,   setCustom]   = useState('');

  const handleSelect = (v) => {
    setSelected(v);
    if (v !== OTHER_VALUE) onUpdate(paramKey, v);
  };

  const handleCustom = (v) => {
    setCustom(v);
    onUpdate(paramKey, v);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel>{label}</InputLabel>
          <Select label={label} value={selected} onChange={(e) => handleSelect(e.target.value)}>
            {EMBEDDING_MODELS.map(({ value, label: lbl }) => (
              <MenuItem key={value} value={value}>{lbl}</MenuItem>
            ))}
            <MenuItem value={OTHER_VALUE} sx={{ fontStyle: 'italic' }}>Other…</MenuItem>
          </Select>
        </FormControl>
        {tooltip && <InfoIcon title={tooltip} />}
      </Box>
      {selected === OTHER_VALUE && (
        <TextField
          size="small"
          label="HuggingFace model ID"
          placeholder="e.g. org/model-name"
          value={custom}
          onChange={(e) => handleCustom(e.target.value)}
          sx={{ minWidth: 240 }}
        />
      )}
    </Box>
  );
}

// ── Main parameter menu ───────────────────────────────────────────────────────
const AlgoParametersMenu = ({ algo, setParams }) => {
  const defaultParameterValues = {
    "alma": {
      "no_parse":             false,
      "spacy_model":          '',
      "embedding_model":      DEFAULT_MODEL,
      "similarity_threshold": 1,
      "select_maximum":       true,
      "strip_diacritics":     false,
    },
    "simalign": {
      "matching_methods": "i",
      "simalign_model":   DEFAULT_MODEL,
      "token_type":       "bpe",
      "strip_diacritics": false,
    },
    "pos":      { "model_language": "latin" },
    "collatex": { "near_match": false },
  };

  const refAlgoPrettyNameDict = {
    "alma":     "ALMA",
    "simalign": "SimAlign",
    "pos":      "POS",
    "collatex": "Collatex",
  };

  const updateParam = (key, value) => {
    setLocalParams((prev) => ({ ...prev, [key]: value }));
  };

  const [localParams, setLocalParams] = useState(defaultParameterValues[algo]);
  const [spacyModels, setSpacyModels] = useState([]);
  const [spacyModelsLoading, setSpacyModelsLoading] = useState(true);

  // Fetch installed spaCy models from the backend once on mount.
  // Also set spacy_model in the same batch so the Select never renders blank.
  useEffect(() => {
    fetch('/spacy_models')
      .then(r => r.json())
      .then(models => {
        setSpacyModels(models);
        setSpacyModelsLoading(false);
        if (models.length > 0) {
          setLocalParams(prev => ({
            ...prev,
            spacy_model: prev.spacy_model || models[0],
          }));
        }
      })
      .catch(() => setSpacyModelsLoading(false));
  }, []);

  useEffect(() => {
    setParams(localParams);
  }, [localParams, setParams]);

  // Reset params when the selected algorithm changes.
  // Must be declared BEFORE the auto-select effect so the reset runs first.
  useEffect(() => { setLocalParams(defaultParameterValues[algo]); }, [algo]);

  // Re-select the first spaCy model after an algo reset (spacyModels is
  // already populated at this point since the fetch runs once on mount).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (algo === 'alma' && spacyModels.length > 0) {
      setLocalParams(prev => ({
        ...prev,
        spacy_model: prev.spacy_model || spacyModels[0],
      }));
    }
  }, [algo]);

  const renderFields = () => {
    switch (algo) {

      case "alma":
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Selects row */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'flex-start' }}>

              {!localParams.no_parse && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <FormControl sx={{ minWidth: 210 }}>
                    <InputLabel>spaCy model</InputLabel>
                    <Select
                      label="spaCy model"
                      value={localParams.spacy_model || ''}
                      disabled={spacyModelsLoading}
                      onChange={(e) => updateParam("spacy_model", e.target.value)}
                    >
                      {spacyModelsLoading
                        ? <MenuItem value="">Loading…</MenuItem>
                        : spacyModels.map(m => (
                            <MenuItem key={m} value={m}>{m}</MenuItem>
                          ))
                      }
                    </Select>
                  </FormControl>
                  <InfoIcon title="NLP model used for lemmatization and dependency parsing. Must match the language of your input text." />
                </Box>
              )}

              <EmbeddingModelSelect
                key={`${algo}-embed`}
                label="Language model"
                paramKey="embedding_model"
                defaultVal={DEFAULT_MODEL}
                onUpdate={updateParam}
                tooltip="HuggingFace model used to compute semantic similarity between words."
              />

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <NumericFormat
                  onChange={(e) => updateParam("similarity_threshold", parseFloat(e.target.value))}
                  allowNegative={false}
                  allowLeadingZeros={false}
                  customInput={TextField}
                  label="Similarity threshold"
                  placeholder="1"
                  defaultValue="1"
                  sx={{ width: 160 }}
                />
                <InfoIcon title="Threshold for strictness of semantic matching — higher values require stronger similarity to create a match." />
              </Box>

            </Box>

            {/* Checkboxes row */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={localParams.no_parse}
                    onChange={(e) => updateParam("no_parse", e.target.checked)}
                  />
                }
                label={
                  <CheckboxLabel
                    label="No parser"
                    tooltip="Skip spaCy/Stanza entirely. Only semantic similarity matching runs — useful when no NLP model is available for your language."
                  />
                }
              />
              <FormControlLabel
                control={
                  <Checkbox onChange={(e) => updateParam("strip_diacritics", e.target.checked)} />
                }
                label={
                  <CheckboxLabel
                    label="Strip diacritics"
                    tooltip="Remove diacritical marks before computing similarity if your text has inconsistent accent notation."
                  />
                }
              />
            </Box>
          </Box>
        );

      case "collatex":
        return (
          <FormControlLabel
            control={
              <Checkbox onChange={(e) => updateParam("near_match", e.target.checked)} />
            }
            label={
              <CheckboxLabel
                label="Use Levenshtein distance"
                tooltip="Enable near-match mode: tokens are matched even if they differ by a small edit distance, which helps with spelling variants."
              />
            }
          />
        );

      case "pos":
        return (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <FormControl sx={{ minWidth: 140 }}>
              <InputLabel>Language</InputLabel>
              <Select
                label="Language"
                defaultValue="latin"
                onChange={(e) => updateParam("model_language", e.target.value)}
              >
                <MenuItem value="latin">Latin</MenuItem>
                <MenuItem value="grc">Greek</MenuItem>
                <MenuItem value="en">English</MenuItem>
              </Select>
            </FormControl>
            <InfoIcon title="Language of the input texts — selects the appropriate part-of-speech tagger." />
          </Box>
        );

      case "simalign":
        return (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Selects row */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'flex-start' }}>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <FormControl sx={{ minWidth: 150 }}>
                  <InputLabel>Matching method</InputLabel>
                  <Select
                    label="Matching method"
                    defaultValue="i"
                    onChange={(e) => updateParam("matching_methods", e.target.value)}
                  >
                    <MenuItem value="i">itermax</MenuItem>
                    <MenuItem value="a">inter</MenuItem>
                    <MenuItem value="m">mwmf</MenuItem>
                    <MenuItem value="f">fwd</MenuItem>
                    <MenuItem value="r">rev</MenuItem>
                  </Select>
                </FormControl>
                <InfoIcon title="Algorithm used to extract word alignments from the similarity matrix. itermax is the default and generally most accurate." />
              </Box>

              <EmbeddingModelSelect
                key={`${algo}-embed`}
                label="Language model"
                paramKey="simalign_model"
                defaultVal={DEFAULT_MODEL}
                onUpdate={updateParam}
                tooltip="HuggingFace multilingual model used to compute cross-lingual word similarity."
              />

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <FormControl sx={{ minWidth: 120 }}>
                  <InputLabel>Token type</InputLabel>
                  <Select
                    label="Token type"
                    defaultValue="bpe"
                    onChange={(e) => updateParam("token_type", e.target.value)}
                  >
                    <MenuItem value="bpe">BPE</MenuItem>
                    <MenuItem value="word">Word</MenuItem>
                  </Select>
                </FormControl>
                <InfoIcon title="Tokenization strategy: BPE (subword) aligns by model tokens, word aligns by whitespace-separated words." />
              </Box>

            </Box>

            {/* Checkbox row */}
            <Box>
              <FormControlLabel
                control={
                  <Checkbox onChange={(e) => updateParam("strip_diacritics", e.target.checked)} />
                }
                label={
                  <CheckboxLabel
                    label="Strip diacritics"
                    tooltip="Remove diacritical marks before computing similarity — useful for ancient texts with inconsistent accent notation."
                  />
                }
              />
            </Box>
          </Box>
        );

      default:
        return (
          <Typography variant="body2" color="text.secondary">
            No parameters for {refAlgoPrettyNameDict[algo]}.
          </Typography>
        );
    }
  };

  return (
    <Box sx={{ mt: 2 }}>
      <Divider sx={{ mb: 2 }}>
        <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: '0.12em' }}>
          {refAlgoPrettyNameDict[algo]} parameters
        </Typography>
      </Divider>
      {renderFields()}
    </Box>
  );
};

export default AlgoParametersMenu;
