# ALMA: Automated Alignment of Ancient Texts using Linguistic and Semantic Analysis

**ALMA** (**A**lignment and **L**earning for **M**anuscript **A**nalysis) is a Python package for the automatic alignment of ancient texts.
It provides a unified API for multiple collation algorithms, including a novel approach based on semantic and linguistic analysis.

---

## Repository Structure

```
├── src/         # Python package source code
├── web_app/     # Web interface (FastAPI backend + React/Vite frontend)
├── paper/       # Research datasets, benchmarking scripts, and results
│   ├── data/         # Ground-truth alignment datasets
│   ├── benchmarking/ # Benchmarking and ablation scripts
│   └── results/      # Pre-computed benchmark results (CSV)
└── tests/       # Unit tests
```

---

## 🛠️ Installation

ALMA can be installed through regular Python installation through `pip`:

`pip install .`

---

## 🚀 Usage

### 1️⃣ Python API

The main entry point is the `Collation` class, which supports **four collation models**:

* **collatex** – classical edit-distance collation, see [here](https://github.com/interedition/collatex).
* **POS** – part-of-speech–based alignment, more details [here](https://github.com/ma-sultan/monolingual-word-aligner).
* **SimAlign** – multilingual embedding-based alignment, more details [here](https://github.com/cisnlp/simalign).
* **ALMA** – semantic + linguistic alignment, the novel algorithm introduced in our paper (currently under review).

You can retrieve for each of the tested algorithms:

* ✅ `matches` – words aligned between source and target
* ❌ `unmatches` – words that could not be matched

#### 🧑‍💻 Example

```python
from alma import Collation

collator = Collation(collation_model="collatex")

collator = Collation(collation_model="pos",
                     model_language="latin")

collator = Collation(collation_model="simalign",
                     matching_methods="a", simalign_model="bert-base-multilingual-cased")

collator = Collation(collation_model="alma",
                     spacy_model="la_core_web_lg",
                     similarity_threshold=1,
                     embedding_model="google-bert/bert-base-multilingual-cased")

# Perform the collation
collation_matches = collator.collate(
    "A",
    "extersit pedes eius capillis suis cuius frater lazarus infirmabatur",
    "B",
    "exterserat capillis suis pedes eius cuius frater lazarus infirmabatur")

# Print the results of the collation
print(collation_matches)
```

#### 📝 Specificities of ALMA's Implementation

ALMA is a multi-step alignment pipeline that progressively matches tokens between two witnesses using linguistic and semantic signals. Each step targets tokens left unmatched by the previous one.

> **Warning:** The SpaCy or Stanza model for your target language must be installed separately **before** running ALMA with linguistic analysis (`tagger="spacy"` or `tagger="stanza"`). Refer to the [SpaCy models hub](https://spacy.io/models) or the [Stanza documentation](https://stanfordnlp.github.io/stanza/available_models.html) for installation instructions specific to your language. If no suitable model exists for your language, use `no_parse=True` to rely solely on the embedding model.

**Pipeline overview:**

| Step | Signal | Controlled by |
|------|--------|---------------|
| 1 | Exact lemma match | `lemma_match` |
| 2 | Semantic similarity (embedding + z-score) | `semantic_match`, `similarity_threshold` |
| 3 | Part-of-speech match | `pos_match` |
| 4 | Syntactic dependency match | `dependency_match` |
| — | Unique-alignment enforcement (after steps 3 & 4) | `unique_alignment` |
| — | Positional pruning (optional) | `pruning`, `sentence_length_ratio` |

**Full parameter reference:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | — | HuggingFace model ID or local path used to compute token embeddings (e.g. `"google-bert/bert-base-multilingual-cased"`). |
| `tagger` | `str` | `"spacy"` | NLP backend for lemmatization and syntactic analysis. Either `"spacy"` or `"stanza"`. |
| `spacy_model` | `str` | `None` | SpaCy model to load (required when `tagger="spacy"`, e.g. `"la_core_web_lg"`). |
| `stanza_config` | `dict` | `None` | Configuration dict forwarded to Stanza (required when `tagger="stanza"`). |
| `no_parse` | `bool` | `False` | When `True`, skips the NLP parser entirely and disables lemma, POS and dependency steps. Only semantic similarity is used. Useful when no SpaCy/Stanza model exists for the target language. |
| `similarity_threshold` | `int` | `2` | Z-score cut-off for the semantic matching step. A token pair is matched when its z-score exceeds this value. Lower values (e.g. `1`) yield more matches; higher values (e.g. `3`) are stricter. |
| `lemmatized` | `bool` | `False` | When `True`, computes embeddings on the lemmatized form of each token instead of the surface form. |
| `strip_diacritics` | `bool` | `False` | Removes diacritical marks before computing the similarity matrix (surface form only; does not affect lemma/POS/dependency steps). |
| `select_maximum` | `bool` | `False` | In the semantic step, only retain the single highest-scoring match per token instead of all matches above the threshold. |
| `lemma_match` | `bool` | `True` | Enable/disable the lemma-matching step. Automatically set to `False` when `no_parse=True`. |
| `semantic_match` | `bool` | `True` | Enable/disable the semantic-similarity step. |
| `pos_match` | `bool` | `True` | Enable/disable the POS-matching step. Automatically set to `False` when `no_parse=True`. Note: `AUX` and `VERB` tags are treated as equivalent. |
| `dependency_match` | `bool` | `True` | Enable/disable the dependency-matching step (matches on shared dependency relation and governing head lemma). Automatically set to `False` when `no_parse=True`. |
| `unique_alignment` | `bool` | `True` | After POS and dependency steps, resolves conflicts so that each token is aligned to at most one counterpart (ties broken by semantic similarity score). |
| `pruning` | `bool` | `False` | Removes matches whose relative positional distortion exceeds `sentence_length_ratio`. Useful to enforce roughly monotone alignments. |
| `sentence_length_ratio` | `float` | `0.1` | Maximum allowed positional distortion `|i/N − j/M|` when `pruning=True`. |
| `verbose` | `bool` | `False` | Logs intermediate alignment matrices at each step via `loguru`. |

**With a SpaCy model:**

```python
collator = Collation(
    collation_model="alma",
    tagger="spacy",
    spacy_model="la_core_web_lg",          # SpaCy model for the target language
    embedding_model="google-bert/bert-base-multilingual-cased",
    similarity_threshold=1,                # z-score threshold (lower = more matches)
)
```

**With a Stanza model:**

```python
collator = Collation(
    collation_model="alma",
    tagger="stanza",
    stanza_config={"lang": "la", "processors": "tokenize,pos,lemma,depparse"},
    embedding_model="google-bert/bert-base-multilingual-cased",
    similarity_threshold=1,
)
```

**Without any NLP parser (`no_parse=True`):**

When no SpaCy or Stanza model is available for the target language, set `no_parse=True`. ALMA will rely solely on the embedding model to make alignment decisions.

```python
collator = Collation(
    collation_model="alma",
    no_parse=True,                         # Disable lemma, POS, dependency steps
    embedding_model="google-bert/bert-base-multilingual-cased",
    similarity_threshold=1,
)
```

**Fine-grained control over pipeline steps:**

Individual steps can be turned on or off independently. For example, to run only the semantic step:

```python
collator = Collation(
    collation_model="alma",
    spacy_model="la_core_web_lg",
    embedding_model="google-bert/bert-base-multilingual-cased",
    similarity_threshold=2,
    lemma_match=False,
    pos_match=False,
    dependency_match=False,
    semantic_match=True,
)
```

**Inspecting results:**

```python
result = collator.collate("A", "text of witness A", "B", "text of witness B")

result["matches"]    # list of (i, j) index pairs of aligned tokens
result["unmatches"]  # list of (i, j) pairs for unaligned tokens

collator.display()          # binary alignment matrix as a pandas DataFrame
collator.display_tuples()   # matches/unmatches as (token_A, token_B) string pairs
```

---

### 2️⃣ Web Interface

ALMA includes a **web interface** for interactively exploring collation results.

#### 🐋 Deploying through docker
The easiest usage is through docker, that you need to have installed on your system.

Then, you can simply run in the main repository:
`docker-compose build`
and:
`docker-compose up`.

#### 🖥️ Running locally

Additionally, you can run it locally by launching each service separately.

First, build the frontend:

```bash
cd web_app/frontend
npm install
npm run build
```

Then launch the backend (which also serves the built frontend):

```bash
cd web_app/backend
python main.py
```

🌐 Open your browser at `http://localhost:8000` to visualize alignments using different models.

---

### 3️⃣ Training a Model

📈 To achieve the best results, the package provides a command for fine-tuning a base model on your own corpus before using ALMA.

The training data must be a JSON file with a three-level hierarchy of `manuscript → chapter → verse`:

```json
{
    "manuscript_id": {
        "1": {
            "1": "Your text here"
        }
    }
}
```

Run the training command:

```bash
train \
    --base_model google-bert/bert-base-multilingual-cased \
    --data_path my_data.json \
    --output_path my_model
```

💾 This saves your fine-tuned model to `my_model/`.
Run `train --help` for all available options.

---

## 📂 Data & Reproducibility

### Datasets

The [`paper/data/`](paper/data/) directory contains the ground-truth alignment datasets used to evaluate ALMA.
Each dataset provides raw manuscript texts alongside manually annotated word-level alignments.

| Dataset | Language | Description |
|---|---|---|
| `vetus_latina` | Latin | Vetus Latina manuscripts of the Gospel of John |
| `john_greek` | Ancient Greek | Greek New Testament manuscripts of the Gospel of John |

Each dataset follows the same layout:

```
paper/data/<dataset>/
├── raw/             # Raw manuscript texts (JSON)
└── ground_truth/    # Word-level alignment annotations (XML [UGARIT standard] + aligned CSV)
```

### Reproducing Paper Results

The benchmarking scripts and pre-computed results are available in [`paper/benchmarking/`](paper/benchmarking/) and [`paper/results/`](paper/results/), respectively, in order to reproduce the results reported in the paper.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
