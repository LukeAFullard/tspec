"""Shared deterministic fixtures for pytest."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic NumPy Generator."""
    return np.random.default_rng(np.random.SeedSequence(20260401))
