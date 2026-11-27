"""Smoke-test every Python code block in README, tutorials, guide pages, and API reference."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import tspec

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
TUTORIAL_FILES = [
    ROOT_DIR / "README.md",
    DOCS_DIR / "index.md",
    DOCS_DIR / "guide" / "concepts.md",
    DOCS_DIR / "guide" / "interpreting-results.md",
    DOCS_DIR / "guide" / "censored-data.md",
    DOCS_DIR / "tutorials" / "water-quality-censored.md",
    DOCS_DIR / "tutorials" / "vegetation-ndvi-wildfire.md",
    DOCS_DIR / "tutorials" / "subsidence-turning-point.md",
    DOCS_DIR / "tutorials" / "near-real-time-monitoring.md",
    DOCS_DIR / "tutorials" / "combined-workflow.md",
    DOCS_DIR / "tutorials" / "robust-outliers-and-red-noise.md",
    DOCS_DIR / "tutorials" / "spectral-analysis-lssa-allssa.md",
]


@pytest.mark.parametrize("md_path", TUTORIAL_FILES, ids=lambda p: p.name)
def test_doc_python_blocks_execute_cleanly(md_path: Path) -> None:
    if not md_path.exists():
        pytest.skip(f"Documentation file {md_path.name} not bundled in source distribution.")
    text = md_path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 1
    ns: dict[str, object] = {}
    for block in blocks:
        exec(compile(block, f"{md_path.name}", "exec"), ns)

    # Flagship examples must recover their advertised jump, turning-point, and spectral targets
    if md_path.name == "README.md":
        res = ns["result"]
        assert isinstance(res, tspec.PipelineResult)
        assert res.jumps is not None and len(res.jumps.jumps) == 1
        assert abs(res.jumps.jumps[0].time_year - 2018.5) <= 0.4
        assert res.turning_points is not None and len(res.turning_points.turning_points) == 1
        assert abs(res.turning_points.turning_points[0].time_year - 2021.8) <= 0.6
    elif md_path.name == "index.md":
        res = ns["result"]
        assert isinstance(res, tspec.PipelineResult)
        assert res.jumps is not None and len(res.jumps.jumps) == 1
        assert abs(res.jumps.jumps[0].time_year - 2019.5) <= 0.4
        assert res.turning_points is not None and len(res.turning_points.turning_points) == 1
        assert abs(res.turning_points.turning_points[0].time_year - 2022.0) <= 0.6
    elif md_path.name == "combined-workflow.md":
        res = ns["res"]
        assert isinstance(res, tspec.PipelineResult)
        assert res.jumps is not None and len(res.jumps.jumps) == 1
        assert abs(res.jumps.jumps[0].time_year - 2017.5) <= 0.4
        assert res.turning_points is not None and len(res.turning_points.turning_points) == 1
        assert abs(res.turning_points.turning_points[0].time_year - 2021.0) <= 0.6
    elif md_path.name == "robust-outliers-and-red-noise.md":
        res_rob = ns["res_rob"]
        assert isinstance(res_rob, tspec.PipelineResult)
        assert res_rob.jumps is not None and len(res_rob.jumps.jumps) == 1
        assert abs(res_rob.jumps.jumps[0].time_year - 2019.0) <= 0.4
        assert res_rob.turning_points is not None and len(res_rob.turning_points.turning_points) == 1
        assert abs(res_rob.turning_points.turning_points[0].time_year - 2022.0) <= 0.6
        tp_w = ns["tp_weighted"]
        assert isinstance(tp_w, tspec.StpdResult)
        assert len(tp_w.turning_points) == 1
        assert abs(tp_w.turning_points[0].time_year - 2019.0) <= 0.5
    elif md_path.name == "spectral-analysis-lssa-allssa.md":
        import numpy as np

        res_a = ns["res_allssa"]
        assert isinstance(res_a, tspec.AllssaResult)
        np.testing.assert_allclose(np.sort(res_a.frequencies), [0.73, 1.85, 3.25], atol=0.05)
        res_j = ns["res_joint"]
        assert isinstance(res_j, tspec.AllssaResult)
        np.testing.assert_allclose(np.sort(res_j.frequencies), [0.73, 1.85, 3.25], atol=0.05)
        res_n = ns["allssa_null"]
        assert isinstance(res_n, tspec.AllssaResult)
        assert res_n.frequencies.size == 0


def test_all_public_symbols_documented_in_api_md() -> None:
    """MIN-03: every public symbol in tspec.__all__ (except __version__) is listed in api.md."""
    api_md = DOCS_DIR / "reference" / "api.md"
    if not api_md.exists():
        pytest.skip("docs/reference/api.md not bundled in source distribution.")
    text = api_md.read_text(encoding="utf-8")
    missing = [
        sym
        for sym in tspec.__all__
        if sym != "__version__" and f"::: tspec.{sym}" not in text
    ]
    assert not missing, f"Public symbols missing from docs/reference/api.md: {missing}"
