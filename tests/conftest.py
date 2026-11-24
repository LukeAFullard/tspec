"""Shared deterministic fixtures and synthetic time-series generators for pytest."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic NumPy Generator."""
    return np.random.default_rng(np.random.SeedSequence(20260401))


def make_irregular_times(
    n_base: int,
    t_start: float,
    t_end: float,
    drop_frac: float,
    rng: np.random.Generator,
    blackout: tuple[float, float] | None = None,
) -> np.ndarray:
    """Create sorted irregular timestamps with random dropout and optional blackout gap."""
    t_grid = np.linspace(t_start, t_end, n_base)
    jitter = rng.uniform(-0.15, 0.15, size=n_base) * ((t_end - t_start) / n_base)
    t_all = np.sort(t_grid + jitter)
    t_all[0] = t_start
    t_all[-1] = t_end

    keep = rng.uniform(size=n_base) >= drop_frac
    keep[0] = True
    keep[-1] = True
    if blackout is not None:
        lo, hi = blackout
        keep &= ~((t_all > lo) & (t_all < hi))

    t_out = np.unique(t_all[keep])
    return t_out
