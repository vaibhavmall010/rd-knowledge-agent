"""Central configuration, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("RDAGENT_DATA_DIR", PROJECT_ROOT / "data")))
    eval_file: Path = field(
        default_factory=lambda: Path(os.getenv("RDAGENT_EVAL_FILE", PROJECT_ROOT / "eval" / "questions.yaml"))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RDAGENT_REPORTS_DIR", PROJECT_ROOT / "reports"))
    )
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("RDAGENT_MODEL", "llama3.1"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("RDAGENT_MAX_STEPS", "6")))


def get_settings() -> Settings:
    return Settings()
