"""Numerically stable weighted linear algebra and vectorized Schur-complement LSS.

All routines avoid forming normal equations ``D.T @ P @ D`` explicitly by
whitening the design matrix and observation vector via scalar scaling, O(n)
1D square-root weights, or Cholesky factorization of 2D SPD weight matrices,
followed by thin QR decomposition (with SVD fallback on rank deficiency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import linalg

from ._types import BoolArray, FloatArray, IntArray, WeightLike
from .exceptions import RankDeficiencyError


@dataclass(frozen=True)
class PreparedWeights:
    """Validated weight representation supporting O(n) 1D vectors and 2D SPD matrices."""

    kind: Literal["none", "scalar", "vector", "matrix"]
    scalar: float = 1.0
    vector: FloatArray | None = None
    sqrt_vector: FloatArray | None = None
    matrix: FloatArray | None = None
    cholesky_lower: FloatArray | None = None

    def slice_range(self, start: int, stop: int) -> PreparedWeights:
        """Extract a contiguous window ``[start:stop]`` without unnecessary copies."""
        if self.kind in ("none", "scalar"):
            return self
        if self.kind == "vector":
            assert self.vector is not None and self.sqrt_vector is not None
            return PreparedWeights(
                kind="vector",
                vector=self.vector[start:stop],
                sqrt_vector=self.sqrt_vector[start:stop],
            )
        assert self.matrix is not None
        sub = np.ascontiguousarray(self.matrix[start:stop, start:stop])
        chol = _factor_spd(sub)
        return PreparedWeights(kind="matrix", matrix=sub, cholesky_lower=chol)

    def subset(self, mask: BoolArray | IntArray) -> PreparedWeights:
        """Subset weights by a boolean mask or integer index array."""
        if self.kind in ("none", "scalar"):
            return self
        if self.kind == "vector":
            assert self.vector is not None and self.sqrt_vector is not None
            return PreparedWeights(
                kind="vector",
                vector=self.vector[mask],
                sqrt_vector=self.sqrt_vector[mask],
            )
        assert self.matrix is not None
        idx = np.flatnonzero(mask) if mask.dtype == np.bool_ else mask
        sub = np.ascontiguousarray(self.matrix[np.ix_(idx, idx)])
        chol = _factor_spd(sub)
        return PreparedWeights(kind="matrix", matrix=sub, cholesky_lower=chol)

    def scale_entries(self, entry_factors: FloatArray) -> PreparedWeights:
        """Multiply observation weights by per-entry positive factors (e.g. imputed down-weighting)."""
        if np.allclose(entry_factors, 1.0):
            return self
        if np.any(~np.isfinite(entry_factors)) or np.any(entry_factors <= 0):
            raise ValueError("Per-entry weight scaling factors must be strictly positive.")
        if self.kind == "none":
            return PreparedWeights(
                kind="vector",
                vector=entry_factors.astype(np.float64, copy=True),
                sqrt_vector=np.sqrt(entry_factors),
            )
        if self.kind == "scalar":
            vec = self.scalar * entry_factors
            return PreparedWeights(kind="vector", vector=vec, sqrt_vector=np.sqrt(vec))
        if self.kind == "vector":
            assert self.vector is not None
            vec = self.vector * entry_factors
            return PreparedWeights(kind="vector", vector=vec, sqrt_vector=np.sqrt(vec))
        assert self.matrix is not None
        s = np.sqrt(entry_factors)
        scaled = (s[:, None] * self.matrix) * s[None, :]
        chol = _factor_spd(scaled)
        return PreparedWeights(kind="matrix", matrix=scaled, cholesky_lower=chol)


def _factor_spd(mat: FloatArray) -> FloatArray:
    """Validate symmetry and positive definiteness via Cholesky factorization P = L L^T."""
    if not np.all(np.isfinite(mat)):
        raise ValueError("2D weight matrix `weights` contains NaN or infinite values.")
    max_abs = float(np.max(np.abs(mat))) if mat.size else 0.0
    sym_err = float(np.max(np.abs(mat - mat.T))) if mat.size else 0.0
    if sym_err > 1e-10 * max(1.0, max_abs):
        raise RankDeficiencyError(
            f"2D weight matrix `weights` must be symmetric (max asymmetry {sym_err:.2e})."
        )
    sym = 0.5 * (mat + mat.T)
    try:
        return np.asarray(linalg.cholesky(sym, lower=True), dtype=np.float64)
    except linalg.LinAlgError as exc:
        raise RankDeficiencyError(
            "2D weight matrix `weights` must be symmetric positive definite (SPD)."
        ) from exc


def prepare_weights(weights: WeightLike | None, n: int) -> PreparedWeights:
    """Validate and wrap observation weights (None, scalar, 1D vector, or 2D SPD matrix)."""
    if weights is None:
        return PreparedWeights(kind="none")

    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 0:
        val = float(arr.item())
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"Scalar `weights` must be finite and strictly positive; got {val}.")
        if val == 1.0:
            return PreparedWeights(kind="none")
        return PreparedWeights(kind="scalar", scalar=val)

    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(
                f"1D `weights` length ({arr.size}) must match series length ({n})."
            )
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError("1D `weights` must contain only finite, strictly positive values.")
        return PreparedWeights(
            kind="vector",
            vector=arr.copy(),
            sqrt_vector=np.sqrt(arr),
        )

    if arr.ndim == 2:
        if arr.shape != (n, n):
            raise ValueError(
                f"2D `weights` must have shape ({n}, {n}); got {arr.shape}."
            )
        chol = _factor_spd(arr)
        return PreparedWeights(
            kind="matrix",
            matrix=np.ascontiguousarray(0.5 * (arr + arr.T)),
            cholesky_lower=chol,
        )

    raise ValueError(f"`weights` must be a scalar, 1D vector, or 2D matrix; got {arr.ndim}D.")


def whiten_vector(v: FloatArray, pw: PreparedWeights) -> FloatArray:
    """Apply whitening operator W to a 1D vector (W^T W = P)."""
    if pw.kind == "none":
        return v
    if pw.kind == "scalar":
        return float(np.sqrt(pw.scalar)) * v
    if pw.kind == "vector":
        assert pw.sqrt_vector is not None
        return pw.sqrt_vector * v
    assert pw.cholesky_lower is not None
    # Since P = L L^T, W = L^T satisfies W^T W = L L^T = P.
    return np.asarray(pw.cholesky_lower.T @ v, dtype=np.float64)


def whiten_matrix(M: FloatArray, pw: PreparedWeights) -> FloatArray:
    """Apply whitening operator W to a 2D matrix (W^T W = P)."""
    if pw.kind == "none":
        return M
    if pw.kind == "scalar":
        return float(np.sqrt(pw.scalar)) * M
    if pw.kind == "vector":
        assert pw.sqrt_vector is not None
        return pw.sqrt_vector[:, None] * M
    assert pw.cholesky_lower is not None
    return np.asarray(pw.cholesky_lower.T @ M, dtype=np.float64)


@dataclass(frozen=True)
class WeightedOLSResult:
    """Result of a whitened QR/SVD weighted least-squares fit."""

    coeff: FloatArray
    residuals: FloatArray
    whitened_residuals: FloatArray
    norm_res: float
    y_w_norm_sq: float
    cov: FloatArray
    Q: FloatArray
    condition_number: float
    rank: int
    rank_deficient: bool


def _detect_intercept_partition_cols(D: FloatArray) -> int:
    """Return k >= 1 if D[:, :k] is a 0/1 partition of unity (constant/piecewise intercepts), else 0."""
    nc = D.shape[1]
    if nc == 0:
        return 0
    # Fast path: single constant column D[:, 0] == 1
    if np.all(D[:, 0] == 1.0):
        return 1
    # Piecewise constant intercepts: binary indicator columns summing to 1 on every row
    row_sum = np.zeros(D.shape[0], dtype=np.float64)
    for k in range(nc):
        col = D[:, k]
        if not np.all((col == 0.0) | (col == 1.0)):
            return 0
        row_sum += col
        if np.all(row_sum == 1.0):
            return k + 1
        if np.any(row_sum > 1.0):
            return 0
    return 0


def solve_weighted_ols(
    D: FloatArray,
    y: FloatArray,
    pw: PreparedWeights,
    *,
    col_scales: FloatArray | None = None,
    rcond: float = 1e-12,
    allow_rank_deficient: bool = True,
    compute_cov: bool = True,
) -> WeightedOLSResult:
    """Solve weighted least-squares min ||W(y - D c)||_2^2 via QR with SVD fallback.

    Parameters
    ----------
    D:
        Design matrix of shape ``(n, nc)``, optionally column-scaled.
    y:
        Observation vector of shape ``(n,)``.
    pw:
        Prepared weight structure.
    col_scales:
        Optional positive scaling vector ``s`` such that ``D_scaled = D_orig * s``;
        returned ``coeff`` and ``cov`` are analytically unscaled back to original
        coordinates (``c_orig = c_scaled * s``, ``Cov_orig = diag(s) Cov_scaled diag(s)``).
    rcond:
        Relative singular-value cutoff for rank-deficiency detection.
    allow_rank_deficient:
        If ``False``, raises :class:`RankDeficiencyError` on rank deficiency.
    compute_cov:
        If ``False``, skips inverting ``R`` to form ``cov`` (faster inside iterative
        ALLSSA/JUST inner loops where only residuals and ``Q`` are needed).
    """
    n, nc = D.shape

    # Exact shift-invariant mean-centering when D spans the constant vector 1_n
    # (prevents catastrophic cancellation on large baselines such as y + 1e13).
    k_int = _detect_intercept_partition_cols(D)
    if k_int > 0 and (col_scales is None or np.all(col_scales[:k_int] == 1.0)):
        y_shift = float(np.mean(y))
        y_work = y - y_shift
    else:
        k_int = 0
        y_shift = 0.0
        y_work = y

    y_w = whiten_vector(y_work, pw)
    y_w_norm_sq = float(np.dot(y_w, y_w))

    if nc == 0:
        norm_res = y_w_norm_sq
        return WeightedOLSResult(
            coeff=np.empty(0, dtype=np.float64),
            residuals=y.copy(),
            whitened_residuals=y_w.copy(),
            norm_res=norm_res,
            y_w_norm_sq=y_w_norm_sq,
            cov=np.empty((0, 0), dtype=np.float64),
            Q=np.empty((n, 0), dtype=np.float64),
            condition_number=1.0,
            rank=0,
            rank_deficient=False,
        )

    D_w = whiten_matrix(D, pw)
    Q, R = np.linalg.qr(D_w, mode="reduced")

    # Exact 2-norm conditioning from SVD of the small (nc x nc) upper-triangular factor R.
    # Diagonal ratios alone can severely underestimate cond(R) on Kahan-type matrices.
    s_vals = linalg.svdvals(R, check_finite=False)
    s_max = float(s_vals[0]) if s_vals.size else 0.0
    s_min = float(s_vals[-1]) if s_vals.size else 0.0
    cond = (s_max / s_min) if s_min > 0.0 else np.inf
    rank_deficient = (s_max == 0.0) or (s_min <= rcond * s_max)

    if not rank_deficient:
        qty = Q.T @ y_w
        coeff_scaled = linalg.solve_triangular(R, qty, lower=False, check_finite=False)
        r_w = y_w - Q @ qty
        norm_res = float(np.dot(r_w, r_w))
        dof = n - nc
        if compute_cov and dof > 0:
            r_inv = linalg.solve_triangular(
                R, np.eye(nc, dtype=np.float64), lower=False, check_finite=False
            )
            cov_scaled = (norm_res / dof) * (r_inv @ r_inv.T)
        elif compute_cov:
            cov_scaled = np.full((nc, nc), np.nan, dtype=np.float64)
        else:
            cov_scaled = np.empty((0, 0), dtype=np.float64)
        Q_basis = Q
        rank = nc
    else:
        if not allow_rank_deficient:
            raise RankDeficiencyError(
                f"Design matrix is rank-deficient (condition number {cond:.2e})."
            )
        U, s, Vt = linalg.svd(D_w, full_matrices=False, check_finite=False)
        cutoff = rcond * float(s[0]) if s.size and s[0] > 0.0 else 0.0
        keep = s > cutoff
        rank = int(np.sum(keep))
        if rank == 0:
            coeff_scaled = np.zeros(nc, dtype=np.float64)
            r_w = y_w.copy()
            norm_res = float(np.dot(r_w, r_w))
            cov_scaled = (
                np.full((nc, nc), np.nan, dtype=np.float64)
                if compute_cov
                else np.empty((0, 0), dtype=np.float64)
            )
            Q_basis = np.empty((n, 0), dtype=np.float64)
        else:
            Uk = U[:, keep]
            sk = s[keep]
            Vtk = Vt[keep, :]
            uky = Uk.T @ y_w
            coeff_scaled = Vtk.T @ (uky / sk)
            r_w = y_w - Uk @ uky
            norm_res = float(np.dot(r_w, r_w))
            dof = n - rank
            if compute_cov and dof > 0:
                scaled_v = Vtk.T / sk[None, :]
                cov_scaled = (norm_res / dof) * (scaled_v @ scaled_v.T)
            elif compute_cov:
                cov_scaled = np.full((nc, nc), np.nan, dtype=np.float64)
            else:
                cov_scaled = np.empty((0, 0), dtype=np.float64)
            Q_basis = Uk

    residuals = y_work - D @ coeff_scaled
    if k_int > 0 and y_shift != 0.0:
        coeff_scaled = coeff_scaled.copy()
        coeff_scaled[:k_int] += y_shift

    if col_scales is not None:
        coeff = coeff_scaled * col_scales
        cov = (
            (col_scales[:, None] * cov_scaled) * col_scales[None, :]
            if cov_scaled.size > 0
            else cov_scaled
        )
    else:
        coeff = coeff_scaled
        cov = cov_scaled

    return WeightedOLSResult(
        coeff=np.asarray(coeff, dtype=np.float64),
        residuals=np.asarray(residuals, dtype=np.float64),
        whitened_residuals=np.asarray(r_w, dtype=np.float64),
        norm_res=norm_res,
        y_w_norm_sq=y_w_norm_sq,
        cov=np.asarray(cov, dtype=np.float64),
        Q=np.asarray(Q_basis, dtype=np.float64),
        condition_number=float(cond),
        rank=rank,
        rank_deficient=rank_deficient,
    )


def compute_lss_vectorized(
    t0: FloatArray,
    omega: FloatArray,
    ols: WeightedOLSResult,
    pw: PreparedWeights,
    *,
    known_freqs: FloatArray | None = None,
    freq_tol: float = 1e-12,
    chunk_size: int = 512,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate normalized LSS and out-of-context (cos, sin) coefficients across ``omega``.

    Uses exact QR orthogonal projection ``Phi_perp = (I - Q Q^T) W Phi`` and
    analytical 2x2 Schur-complement inversion, vectorized in frequency chunks
    for O(n) weight memory, numerical stability near collinear subspaces, and
    high throughput.
    """
    k_total = int(omega.size)
    spectrum = np.zeros(k_total, dtype=np.float64)
    cs_coeffs = np.zeros((k_total, 2), dtype=np.float64)

    if k_total == 0 or ols.norm_res <= 0.0 or ols.norm_res <= 1e-28 * ols.y_w_norm_sq:
        return spectrum, cs_coeffs

    known = known_freqs if known_freqs is not None else np.empty(0, dtype=np.float64)
    Q = ols.Q
    has_q = Q.shape[1] > 0
    r_w = ols.whitened_residuals
    norm_res = ols.norm_res
    two_pi_t = (2.0 * np.pi) * t0

    for start in range(0, k_total, chunk_size):
        stop = min(start + chunk_size, k_total)
        om_chunk = omega[start:stop]

        # Build phase matrix (n, m)
        phases = np.outer(two_pi_t, om_chunk)
        C = np.cos(phases)
        S = np.sin(phases)

        Cw = whiten_matrix(C, pw)
        Sw = whiten_matrix(S, pw)

        a = np.sum(Cw * Cw, axis=0)
        d = np.sum(Sw * Sw, axis=0)
        b = np.sum(Cw * Sw, axis=0)
        g1 = Cw.T @ r_w
        g2 = Sw.T @ r_w

        if has_q:
            cw_norm2 = a
            sw_norm2 = d
            qtc = Q.T @ Cw
            qts = Q.T @ Sw
            a = cw_norm2 - np.sum(qtc * qtc, axis=0)
            d = sw_norm2 - np.sum(qts * qts, axis=0)
            b = b - np.sum(qtc * qts, axis=0)

            # Fallback to explicit orthogonal projection whenever >=3 digits of
            # cancellation occur (e.g., near known frequencies or low-freq trends).
            cancel = (
                (a < 1e-3 * cw_norm2)
                | (d < 1e-3 * sw_norm2)
                | ((a * d - b * b) < 1e-6 * cw_norm2 * sw_norm2)
            )
            if np.any(cancel):
                Cw_p = Cw[:, cancel] - Q @ qtc[:, cancel]
                Sw_p = Sw[:, cancel] - Q @ qts[:, cancel]
                a[cancel] = np.sum(Cw_p * Cw_p, axis=0)
                d[cancel] = np.sum(Sw_p * Sw_p, axis=0)
                b[cancel] = np.sum(Cw_p * Sw_p, axis=0)
                g1[cancel] = Cw_p.T @ r_w
                g2[cancel] = Sw_p.T @ r_w

        det = a * d - b * b
        scale_sq = (a + d) ** 2
        good = (scale_sq > 0.0) & (det > 1e-28 * scale_sq) & (np.abs(om_chunk) > 0.0)

        if known.size > 0:
            # Exclude frequencies already included in the known-constituents matrix D
            dist_known = np.min(np.abs(om_chunk[:, None] - known[None, :]), axis=1)
            good &= dist_known > freq_tol

        if np.any(good):
            det_g = det[good]
            c1 = (d[good] * g1[good] - b[good] * g2[good]) / det_g
            c2 = (-b[good] * g1[good] + a[good] * g2[good]) / det_g
            spec = (g1[good] * c1 + g2[good] * c2) / norm_res
            spec = np.clip(spec, 0.0, 1.0)

            chunk_spec = np.zeros(stop - start, dtype=np.float64)
            chunk_cs = np.zeros((stop - start, 2), dtype=np.float64)
            chunk_spec[good] = spec
            chunk_cs[good, 0] = c1
            chunk_cs[good, 1] = c2
            spectrum[start:stop] = chunk_spec
            cs_coeffs[start:stop] = chunk_cs

    return spectrum, cs_coeffs


def orthonormal_design_basis(
    D: FloatArray,
    pw: PreparedWeights,
    *,
    rcond: float = 1e-12,
) -> FloatArray:
    """Return an orthonormal basis of the whitened column space ``col(W D)``."""
    if D.shape[1] == 0:
        return np.empty((D.shape[0], 0), dtype=np.float64)
    U, s, _ = linalg.svd(whiten_matrix(D, pw), full_matrices=False, check_finite=False)
    keep = s > (rcond * float(s[0]) if s.size and s[0] > 0.0 else np.inf)
    return np.asarray(U[:, keep], dtype=np.float64)


def max_lss_null_draws(
    t0: FloatArray,
    omega: FloatArray,
    q_basis: FloatArray,
    pw: PreparedWeights,
    *,
    known_freqs: FloatArray | None = None,
    freq_tol: float = 1e-12,
    n_mc: int = 999,
    seed: int | tuple[int, ...] = 0,
    chunk_size: int = 256,
) -> FloatArray:
    """Monte Carlo draws of ``max_omega s(omega)`` under the Gaussian white-noise null.

    The normalized LSS is invariant to the error scale and to the known-constituent
    coefficients, so standard-normal errors in whitened coordinates (projected off
    ``q_basis``) give the exact finite-sample null distribution of the spectral
    maximum over the searched grid ``omega`` for the given design and weights.
    """
    n = int(t0.size)
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, int(n_mc)))
    return max_lss_batch(
        t0, omega, q_basis, pw, e, known_freqs=known_freqs, freq_tol=freq_tol, chunk_size=chunk_size
    )


def max_lss_batch(
    t0: FloatArray,
    omega: FloatArray,
    q_basis: FloatArray,
    pw: PreparedWeights,
    e: FloatArray,
    *,
    known_freqs: FloatArray | None = None,
    freq_tol: float = 1e-12,
    chunk_size: int = 256,
) -> FloatArray:
    """Maximum normalized LSS over ``omega`` for each column of whitened residuals ``e``.

    ``e`` holds whitened series (``W r``) as columns; they are projected off the
    orthonormal whitened design basis ``q_basis`` before the spectrum is computed.
    """
    e = np.asarray(e, dtype=np.float64)
    if q_basis.shape[1] > 0:
        e = e - q_basis @ (q_basis.T @ e)
    rr = np.sum(e * e, axis=0)
    best = np.zeros(int(e.shape[1]), dtype=np.float64)
    known = known_freqs if known_freqs is not None else np.empty(0, dtype=np.float64)
    two_pi_t = (2.0 * np.pi) * t0
    for start in range(0, int(omega.size), chunk_size):
        om = omega[start : start + chunk_size]
        if known.size > 0:
            om = om[np.min(np.abs(om[:, None] - known[None, :]), axis=1) > freq_tol]
        om = om[np.abs(om) > 0.0]
        if om.size == 0:
            continue
        ph = np.outer(two_pi_t, om)
        cw = whiten_matrix(np.cos(ph), pw)
        sw = whiten_matrix(np.sin(ph), pw)
        if q_basis.shape[1] > 0:
            cw = cw - q_basis @ (q_basis.T @ cw)
            sw = sw - q_basis @ (q_basis.T @ sw)
        a = np.sum(cw * cw, axis=0)
        d = np.sum(sw * sw, axis=0)
        b = np.sum(cw * sw, axis=0)
        det = a * d - b * b
        good = (det > 1e-12 * (a + d) ** 2) & ((a + d) > 0.0)
        if not np.any(good):
            continue
        cw, sw, a, d, b, det = cw[:, good], sw[:, good], a[good], d[good], b[good], det[good]
        g1 = cw.T @ e
        g2 = sw.T @ e
        spec = (d[:, None] * g1 * g1 - 2.0 * b[:, None] * g1 * g2 + a[:, None] * g2 * g2) / det[:, None]
        best = np.maximum(best, np.max(spec, axis=0) / rr)
    return best
