from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it"
# Teacher, tokenizer and weight source for the E4B diffusion student.
DEFAULT_DIFFUSION_REFERENCE = DEFAULT_BASE_MODEL
DEFAULT_STUDENT_MODEL = "artifacts/transplanted"
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CANVAS_LENGTH = 256
DEFAULT_PREFIX_LENGTH = 512

def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
