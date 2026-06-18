from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it"
DEFAULT_DIFFUSION_REFERENCE = "google/diffusiongemma-26B-A4B-it"
DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CANVAS_LENGTH = 256
DEFAULT_PREFIX_LENGTH = 512

FORBIDDEN_TEST_ONLY_PROMPTS = [
    "Explain TCP vs UDP in a concise technical answer.",
    "Write a Python function that validates a Sudoku board.",
    "Translate this sentence to Traditional Chinese: The model should preserve its interface.",
    "Summarize the trade-offs of diffusion language models.",
    "Give a short example of a JSON tool call schema.",
]

FORBIDDEN_PROMPT_PATH_PARTS = {
    "readme",
    "docs",
    "examples",
    "tests",
    "seed_prompts",
    "prompts",
    "prompt_bank",
}


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
