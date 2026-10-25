from pathlib import Path

import pytest

from rdagent.tools import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_dir(ROOT / "data")


@pytest.fixture(scope="session")
def eval_file() -> Path:
    return ROOT / "eval" / "questions.yaml"
