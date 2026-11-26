"""Bootstrap bands for detect-then-fit piecewise trends (audit CRT-04).

Analytical bands for a piecewise-linear trend treat the detected turning points as
known. They omit the uncertainty of the knot locations and therefore under-cover after
detection. This module re-estimates the knot locations in every bootstrap replicate:

1. Data are simulated from the fitted structure, ``y* = f + g + e * v``. Here ``f`` is
   the fitted connected trend, ``g`` holds the other fitted components (jump offsets,
   seasonal terms), ``e`` are the residuals inflated by ``sqrt(n / (n - p))``, and ``v``
   are Rademacher signs (wild bootstrap, so heteroscedasticity and the weights are kept).
2. Each knot is re-located by profile least squares over the indices between its
   neighbours (other knots held at their fitted positions), using rank-one
   Schur-complement updates vectorised over candidates and replicates.
3. The connected model ``[1, u, extra columns, hinges at the re-estimated knots]`` is
   refitted. Its trend part (intercept, slope, hinges and any columns flagged as part
   of the trend, such as jump steps) forms the replicate curve.

Pointwise bands are basic-bootstrap intervals,
``[f_rep - q_{1-a/2}(d), f_rep - q_{a/2}(d)]`` with ``d = curve* - f``, so the
estimator's selection-induced variability and bias are both propagated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._types import FloatArray


@dataclass(frozen=True)
class BootstrapBand:
    """Replicate curves summarised as pointwise deviations from the bootstrap truth."""

    dev_lo: FloatArray  # quantile q_{a/2} of (curve* - truth), per observation
    dev_hi: FloatArray  # quantile q_{1-a/2}
    knots: np.ndarray  # (n_boot, K) re-estimated knot indices
    coef_u: FloatArray  # (n_boot, K + 1) replicate segment slopes in time units
    n_boot: int


def _proj(q: FloatArray, a: FloatArray) -> FloatArray:
    return a - q @ (q.T @ a)


def bootstrap_piecewise_band(
    t: FloatArray,
    trend_fit: FloatArray,
    other_fit: FloatArray,
    y: FloatArray,
    knots: list[int],
    *,
    weights: FloatArray | None = None,
    extra_cols: FloatArray | None = None,
    trend_extra: np.ndarray | None = None,
    n_params: int | None = None,
    margin: int = 3,
    n_boot: int = 499,
    ci_level: float = 0.95,
    seed: int | tuple[int, ...] = 20261126,
) -> BootstrapBand | None:
    """Wild-bootstrap band for a connected piecewise trend with re-estimated knots.

    ``trend_fit`` is the reported connected trend ``f`` and ``other_fit`` the remaining
    fitted structure ``g`` (zeros if none). ``extra_cols`` are additional regressors
    re-estimated in every replicate (jump steps, harmonics). ``trend_extra`` flags the
    extra columns that belong to the reported trend curve. Returns ``None`` when the
    problem is degenerate (no knots or too few observations).
    """
    n = int(t.size)
    k_n = len(knots)
    if k_n == 0 or n < 10:
        return None
    sw = np.sqrt(weights) if weights is not None else np.ones(n)
    span = float(t[-1] - t[0]) if t[-1] > t[0] else 1.0
    u = (t - t[0]) / span
    ex = extra_cols if extra_cols is not None else np.empty((n, 0))
    tex = trend_extra if trend_extra is not None else np.zeros(ex.shape[1], dtype=bool)

    resid = (y - trend_fit - other_fit) * sw
    p = n_params if n_params is not None else 2 + k_n + ex.shape[1]
    resid = resid * np.sqrt(n / max(1, n - p))
    rng = np.random.default_rng(seed)
    v = rng.choice(np.array([-1.0, 1.0]), size=(n, int(n_boot)))
    ystar_w = ((trend_fit + other_fit) * sw)[:, None] + resid[:, None] * v  # whitened

    def hinge(k: int) -> FloatArray:
        return np.maximum(0.0, u - u[k])

    knots_sorted = sorted(int(k) for k in knots)
    cur = np.tile(np.array(knots_sorted, dtype=np.intp), (int(n_boot), 1))
    base = np.column_stack([np.ones(n), u, ex]) if ex.shape[1] else np.column_stack([np.ones(n), u])
    for j in range(k_n):
        lo = (knots_sorted[j - 1] + margin) if j > 0 else margin
        hi = (knots_sorted[j + 1] - margin) if j + 1 < k_n else n - margin
        cands = np.arange(max(margin, lo), min(n - margin, hi) + 1)
        if cands.size < 2:
            continue
        others = [hinge(k) for i, k in enumerate(knots_sorted) if i != j]
        x0 = np.column_stack([base, *others]) if others else base
        q0, _ = np.linalg.qr(x0 * sw[:, None], mode="reduced")
        hc = np.maximum(0.0, u[:, None] - u[cands][None, :]) * sw[:, None]
        hp = _proj(q0, hc)
        hh = np.sum(hp * hp, axis=0)
        ok = hh > 1e-12 * max(float(np.max(hh)), 1e-300)
        r0 = _proj(q0, ystar_w)
        g = hp.T @ r0  # (C, B)
        score = np.where(ok[:, None], g * g / np.where(ok, hh, 1.0)[:, None], -np.inf)
        cur[:, j] = cands[np.argmax(score, axis=0)]

    curves = np.empty((n, int(n_boot)), dtype=np.float64)
    slopes = np.empty((int(n_boot), k_n + 1), dtype=np.float64)
    for b in range(int(n_boot)):
        kb = np.sort(cur[b])
        hcols = np.column_stack([hinge(k) for k in kb])
        xb = np.column_stack([base, hcols])
        coef, *_ = np.linalg.lstsq(xb * sw[:, None], ystar_w[:, b], rcond=None)
        n_ex = ex.shape[1]
        curve = coef[0] + coef[1] * u + hcols @ coef[2 + n_ex :]
        if n_ex and np.any(tex):
            curve = curve + ex[:, tex] @ coef[2 : 2 + n_ex][tex]
        curves[:, b] = curve
        slopes[b] = np.cumsum(np.concatenate([[coef[1]], coef[2 + n_ex :]])) / span
    a = 1.0 - ci_level
    dev = curves - trend_fit[:, None]
    return BootstrapBand(
        dev_lo=np.quantile(dev, 0.5 * a, axis=1),
        dev_hi=np.quantile(dev, 1.0 - 0.5 * a, axis=1),
        knots=cur,
        coef_u=slopes,
        n_boot=int(n_boot),
    )
