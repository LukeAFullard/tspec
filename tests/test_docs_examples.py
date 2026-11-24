"""Smoke-test every Python code block inside docs/tutorials/ and docs/guide/concepts.md."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
TUTORIAL_FILES = [
    DOCS_DIR / "index.md",
    DOCS_DIR / "guide" / "concepts.md",
    DOCS_DIR / "tutorials" / "water-quality-censored.md",
    DOCS_DIR / "tutorials" / "vegetation-ndvi-wildfire.md",
    DOCS_DIR / "tutorials" / "subsidence-turning-point.md",
    DOCS_DIR / "tutorials" / "combined-workflow.md",
]


@pytest.mark.parametrize("md_path", TUTORIAL_FILES, ids=lambda p: p.name)
def test_doc_python_blocks_execute_cleanly(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 1
    ns: dict[str, object] = {}
    for block in blocks:
        exec(compile(block, f"{md_path.name}", "exec"), ns)
