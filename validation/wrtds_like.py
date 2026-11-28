"""WRTDS-style decomposition without discharge (comparator for the decomposition validations).

Weighted regression on time and season (Hirsch, Moyer & Archfield, 2010): for each target time
tau, fit y ~ 1 + (t - tau) + sin(2 pi t) + cos(2 pi t) by weighted least squares with tricube
weights in time distance (half-window 7 years) and in seasonal distance (half-window 0.5 year),
widening both windows by 10% until at least 100 observations carry weight (EGRET defaults
windowY = 7, windowS = 0.5, minNumObs = 100). The discharge term of WRTDS is omitted.

Trend: 12-month centred mean of the fitted surface on a monthly grid (the annual mean, as in
EGRET); seasonal = fitted value - trend; remainder = y - fitted value.
"""
from __future__ import annotations

import numpy as np


def _tricube(d):
    d = np.clip(np.abs(d), 0.0, 1.0)
    return (1.0 - d ** 3) ** 3


def wrtds_surface(t_obs, y_obs, targets, h_time=7.0, h_season=0.5, min_obs=100):
    t_obs = np.asarray(t_obs, float)
    y_obs = np.asarray(y_obs, float)
    out = np.empty(targets.size)
    s_obs, c_obs = np.sin(2 * np.pi * t_obs), np.cos(2 * np.pi * t_obs)
    for k0 in range(0, targets.size, 256):
        tau = targets[k0:k0 + 256, None]
        dt = t_obs[None, :] - tau
        ds = np.abs(((dt + 0.5) % 1.0) - 0.5)
        ht = np.full((tau.shape[0], 1), h_time)
        hs = np.full((tau.shape[0], 1), h_season)
        for _ in range(30):
            w = _tricube(dt / ht) * _tricube(ds / hs)
            short = (w > 0).sum(axis=1) < min(min_obs, t_obs.size)
            if not short.any():
                break
            ht[short] *= 1.1
            hs[short] = np.minimum(hs[short] * 1.1, 0.5)
        X = np.stack([np.ones_like(dt), dt, np.broadcast_to(s_obs, dt.shape), np.broadcast_to(c_obs, dt.shape)], axis=-1)
        XtW = X * w[..., None]
        A = np.einsum("kni,knj->kij", XtW, X) + 1e-10 * np.eye(4)
        b = np.einsum("kni,kn->ki", XtW, np.broadcast_to(y_obs, dt.shape))
        beta = np.linalg.solve(A, b[..., None])[..., 0]
        tc = targets[k0:k0 + 256]
        out[k0:k0 + 256] = beta[:, 0] + beta[:, 2] * np.sin(2 * np.pi * tc) + beta[:, 3] * np.cos(2 * np.pi * tc)
    return out


def run_wrtds(t, y):
    """Decompose; returns dict like the other comparators (trend/seasonal at t, monthly grid)."""
    t = np.asarray(t, float)
    k0, k1 = int(np.floor((t[0] - 2000) * 12)) - 6, int(np.floor((t[-1] - 2000) * 12)) + 7
    grid = 2000.0 + (np.arange(k0, k1 + 1) + 0.5) / 12
    f_grid = wrtds_surface(t, y, grid)
    trend_grid = np.convolve(f_grid, np.r_[0.5, np.ones(11), 0.5] / 12.0, mode="same")
    valid = slice(6, grid.size - 6)
    grid, f_grid, trend_grid = grid[valid], f_grid[valid], trend_grid[valid]
    f_obs = wrtds_surface(t, y, t)
    trend = np.interp(t, grid, trend_grid)
    seas_grid = f_grid - trend_grid
    return dict(trend=trend, seasonal=f_obs - trend, grid=grid, trend_grid=trend_grid, seas_grid=seas_grid)
