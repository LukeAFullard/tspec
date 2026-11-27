"""Tests for numerically stable whitened QR/SVD solver and vectorized Schur LSS."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from tspec._linalg import (
    compute_lss_vectorized,
    prepare_weights,
    solve_weighted_ols,
)
from tspec.exceptions import RankDeficiencyError


def test_weighted_ols_matches_exact_normal_equations_for_1d_and_2d_weights(
    rng: np.random.Generator,
) -> None:
    n = 60
    t = np.linspace(0.0, 5.0, n)
    D = np.column_stack([np.ones(n), t, np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    beta_true = np.array([2.5, -0.4, 1.2, -0.7])
    y = D @ beta_true + rng.normal(0.0, 0.2, size=n)

    # 1D diagonal weights
    w1d = rng.uniform(0.5, 5.0, size=n)
    pw1 = prepare_weights(w1d, n)
    res1 = solve_weighted_ols(D, y, pw1)
    P_diag = np.diag(w1d)
    ref_c = np.linalg.solve(D.T @ P_diag @ D, D.T @ P_diag @ y)
    assert np.allclose(res1.coeff, ref_c, rtol=1e-11, atol=1e-12)

    # 2D correlated SPD matrix (AR(1) precision matrix)
    phi = 0.5
    idx = np.arange(n)
    cov_mat = phi ** np.abs(idx[:, None] - idx[None, :])
    P2d = np.linalg.inv(cov_mat)
    pw2 = prepare_weights(P2d, n)
    res2 = solve_weighted_ols(D, y, pw2)
    ref_c2 = np.linalg.solve(D.T @ P2d @ D, D.T @ P2d @ y)
    ref_r2 = y - D @ ref_c2
    ref_norm2 = float(ref_r2.T @ P2d @ ref_r2)
    ref_cov2 = (ref_norm2 / (n - D.shape[1])) * np.linalg.inv(D.T @ P2d @ D)
    assert np.allclose(res2.coeff, ref_c2, rtol=1e-10, atol=1e-11)
    assert np.isclose(res2.norm_res, ref_norm2, rtol=1e-10)
    assert np.allclose(res2.cov, ref_cov2, rtol=1e-10, atol=1e-11)


def test_vectorized_lss_matches_explicit_schur_complement(rng: np.random.Generator) -> None:
    n = 50
    t = np.sort(rng.uniform(0.0, 4.0, size=n))
    y = 1.0 + 0.3 * t + 0.8 * np.cos(2 * np.pi * 1.4 * t) + rng.normal(0.0, 0.1, size=n)
    D = np.column_stack([np.ones(n), t])
    w = rng.uniform(0.5, 3.0, size=n)
    pw = prepare_weights(w, n)
    ols = solve_weighted_ols(D, y, pw)

    omega = np.linspace(0.3, 3.5, 25)
    spec, cs = compute_lss_vectorized(t, omega, ols, pw)

    # Explicit dense reference
    P = np.diag(w)
    N11_inv = np.linalg.inv(D.T @ P @ D)
    rP = P @ ols.residuals
    for k, om in enumerate(omega):
        Phi = np.column_stack([np.cos(2 * np.pi * om * t), np.sin(2 * np.pi * om * t)])
        N22 = Phi.T @ P @ Phi
        N12 = D.T @ P @ Phi
        S = N22 - N12.T @ N11_inv @ N12
        chat = np.linalg.solve(S, Phi.T @ rP)
        s_ref = float((Phi.T @ rP) @ chat) / ols.norm_res
        assert np.isclose(spec[k], s_ref, rtol=1e-10, atol=1e-12)
        assert np.allclose(cs[k], chat, rtol=1e-10, atol=1e-11)


def test_rank_deficient_fallback_and_strict_mode() -> None:
    n = 20
    t = np.linspace(0.0, 2.0, n)
    # Duplicate column makes design rank-deficient
    D = np.column_stack([np.ones(n), t, t])
    y = 1.0 + 2.0 * t
    pw = prepare_weights(None, n)

    res = solve_weighted_ols(D, y, pw, allow_rank_deficient=True)
    assert res.rank_deficient is True
    assert res.rank == 2
    assert res.norm_res < 1e-24

    with pytest.raises(RankDeficiencyError):
        solve_weighted_ols(D, y, pw, allow_rank_deficient=False)


def test_invalid_weights_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_weights(-1.0, 10)
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_weights(np.array([1.0, 0.0, 2.0]), 3)
    with pytest.raises(RankDeficiencyError, match="symmetric"):
        prepare_weights(np.array([[2.0, 1.0], [0.0, 2.0]]), 2)
    with pytest.raises(RankDeficiencyError, match="SPD"):
        prepare_weights(np.array([[1.0, 2.0], [2.0, 1.0]]), 2)


def test_1d_weights_memory_is_linear_in_n() -> None:
    n = 10_000
    t = np.linspace(0.0, 100.0, n)
    y = np.sin(2 * np.pi * t)
    w = np.ones(n, dtype=np.float64) * 2.0
    D = np.column_stack([np.ones(n), t])

    tracemalloc.start()
    pw = prepare_weights(w, n)
    ols = solve_weighted_ols(D, y, pw)
    spec, _ = compute_lss_vectorized(t, np.linspace(0.5, 2.5, 30), ols, pw)
    _curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert spec.size == 30
    # Dense 10k x 10k matrix would exceed 800 MB; O(n) path stays well below 50 MB
    assert peak < 50 * 1024 * 1024


def test_kahan_matrix_exact_svd_conditioning() -> None:
    """MAJ-01: solve_weighted_ols uses exact svdvals(R), not diagonal ratio of R."""
    p = 15
    c, s = 0.6, 0.8
    R_k = np.eye(p)
    for i in range(p):
        R_k[i, i:] = s**i * np.r_[1.0, np.full(p - i - 1, -c)]
    D = np.vstack([R_k, np.zeros((15, p))])
    y = np.arange(30, dtype=np.float64)
    pw = prepare_weights(None, 30)
    ols = solve_weighted_ols(D, y, pw)
    true_cond = float(np.linalg.cond(D))
    assert np.isclose(ols.condition_number, true_cond, rtol=1e-6)
    assert ols.condition_number > 1e4

