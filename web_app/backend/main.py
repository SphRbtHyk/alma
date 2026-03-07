import logging
import os
from pathlib import Path
from typing import Any, Literal, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import spacy
import uvicorn
from pydantic import BaseModel

from alma.collation_algorithms.collation import Collation

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Frontend location
# ---------------------------------------------------------------------------
# Override FRONTEND_DIST_DIR to point at your compiled Vite output, e.g.:
#   FRONTEND_DIST_DIR=/app/web_app/frontend/dist
# Defaults to the sibling frontend/dist directory (works for local runs and
# for the production Docker image that copies dist/ into place).
_default_dist = Path(__file__).parent.parent / "frontend" / "dist"
FRONTEND_DIST_DIR = Path(os.environ.get("FRONTEND_DIST_DIR", _default_dist))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI()

# CORS – wildcard is fine for a tool served on localhost; tighten by setting
# CORS_ORIGINS to a comma-separated list of allowed origins in production.
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",") + ["5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve compiled JS/CSS assets when the frontend has been built.
_assets_dir = FRONTEND_DIST_DIR / "assets"
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")
    logger.info("Serving frontend static assets from %s", _assets_dir)
else:
    logger.info(
        "Frontend dist not found at %s – static assets will not be served. "
        "Set FRONTEND_DIST_DIR or run 'npm run build' in web_app/frontend/.",
        FRONTEND_DIST_DIR,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AlignRequest(BaseModel):
    """Alignment request received by the server."""

    ref_text: str
    comp_text: str
    algo: Literal["collatex", "pos", "simalign", "alma"]
    params: Any


class AlignmentResponse(BaseModel):
    """Response format returned by the server."""

    matches: list[tuple[int, int]]
    unmatches: list[Union[tuple[int, int], tuple[int, str], tuple[str, int]]]


# ---------------------------------------------------------------------------
# API routes  (must be registered before the SPA catch-all below)
# ---------------------------------------------------------------------------
@app.get("/spacy_models")
def get_spacy_models() -> list[str]:
    """Return the list of spaCy models currently installed on the server."""
    return spacy.util.get_installed_models()


@app.post("/alignment_request", response_model=AlignmentResponse)
def alignment_request(align_request: AlignRequest):
    """Run the requested collation algorithm and return the result."""
    collator = Collation(align_request.algo, **align_request.params)
    return collator.collate(
        "Reference text",
        align_request.ref_text,
        "Compared text",
        align_request.comp_text,
    )


# ---------------------------------------------------------------------------
# SPA catch-all  (must be last)
# ---------------------------------------------------------------------------
@app.get("/{full_path:path}")
def serve_spa():
    """Return index.html for every unmatched GET path (client-side routing)."""
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {
        "error": (
            "Frontend not available. "
            "Build it with 'npm run build' in web_app/frontend/ "
            "or set FRONTEND_DIST_DIR to an existing dist directory."
        )
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
