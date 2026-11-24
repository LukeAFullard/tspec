"""First-class censored time-series preprocessing via ``ndimpute``.

Provides statistically honest handling of left-censored (``<LOD``),
right-censored (``>UL``), mixed-censored, and interval-censored (``[L, R]``)
observations with explicit time-series safeguards:

1. **Temporal-order safety:** Prevents deterministic trend injection from
   stable-sorted ROS quantile positions on tied detection limits by defaulting
   to ``impute_type='stochastic'`` (seeded) and permuting tied detection-limit
   groups whenever ``impute_type='quantile'`` is requested.
2. **Non-stationarity scopes:** Supports ``'global'``, ``'windowed'``, and
   ``'seasonal_stratified'`` imputation scopes plus automatic non-stationarity
   diagnostics.
3. **Detection-limit shift defense:** Detects step changes in active detection
   limits over time, flags breakpoints coinciding with lab method upgrades, and
   optionally verifies whether a detected jump vanishes under common-limit
   recensoring (Helsel, 2012).
4. **Uncertainty & stability:** Supports optional imputed-point down-weighting
   and multi-draw sensitivity checks (fixed-parameter stochastic draws or
   bootstrap-refit draws that propagate parameter uncertainty).
5. **Tiered censoring guardrails:** Emits structured diagnostics above 20% and
   50% censoring and blocks >80% censoring unless explicitly opted in.

``ndimpute`` and ``pandas`` are imported strictly lazily so ``tspec`` retains
full lite-environment compatibility when censored data are not used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._diagnostics import WarningItem
from ._time import TimeSeriesInput, coerce_times, estimate_sampling_rate_ratio
from ._types import ArrayLike, BoolArray, FloatArray, IntArray, TimeLike, ValueLike
from .exceptions import CensoringError, InsufficientDataError

_MISSING_STRINGS = frozenset({"", "na", "nan", "n/a", "none", "null"})
_STATUS_STRINGS: dict[str, int] = {
    "lt": -1,
    "<": -1,
    "left": -1,
    "gt": 1,
    ">": 1,
    "right": 1,
    "not": 0,
    "ok": 0,
    "obs": 0,
    "observed": 0,
    "": 0,
}


def _require_ndimpute() -> Any:
    try:
        import ndimpute
    except ImportError as exc:
        raise ImportError(
            "ndimpute is required for censored-data workflows. Install it with "
            "`pip install ndimpute` (or reinstall `pip install tspec`)."
        ) from exc
    return ndimpute


@dataclass(frozen=True)
class CensoredConfig:
    """Configuration for ``ndimpute``-backed censored time-series analysis."""

    method: Literal["ros", "parametric", "substitution"] = "ros"
    censoring_type: Literal["left", "right", "mixed", "interval"] | None = None
    dist: Literal["lognormal", "normal", "weibull", "auto"] = "auto"
    impute_type: Literal["stochastic", "mean", "quantile"] = "stochastic"
    scope: Literal["global", "windowed", "seasonal_stratified"] = "global"
    random_state: int | np.random.Generator | None = 0
    clamp: bool = True
    imputed_weight_factor: float = 1.0
    n_stability_draws: int = 0
    stability_mode: Literal["stochastic", "bootstrap"] = "stochastic"
    detect_dl_shifts: bool = True
    dl_shift_rel_threshold: float = 0.20
    recensor_check: bool = False
    warn_censor_fraction: float = 0.20
    max_censor_fraction: float = 0.80
    allow_heavy_censoring: bool = False
    left_marker: str = "<"
    right_marker: str = ">"
    ndimpute_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.imputed_weight_factor <= 1.0):
            raise ValueError(
                f"`imputed_weight_factor` must lie in (0, 1]; got {self.imputed_weight_factor}."
            )
        if self.n_stability_draws < 0:
            raise ValueError("`n_stability_draws` must be >= 0.")
        if not (0.0 < self.warn_censor_fraction < 1.0):
            raise ValueError("`warn_censor_fraction` must lie in (0, 1).")
        if not (0.0 < self.max_censor_fraction <= 1.0):
            raise ValueError("`max_censor_fraction` must lie in (0, 1].")
        if self.left_marker == self.right_marker:
            raise ValueError("`left_marker` and `right_marker` must differ.")


@dataclass(frozen=True)
class DlShiftEvent:
    """Detected change in the active laboratory censoring / detection limit."""

    index: int
    valid_index: int
    time: Any
    time_year: float
    old_limit: float
    new_limit: float
    relative_change: float


@dataclass(frozen=True)
class CensoringProvenance:
    """Full provenance of censored inputs, imputation outputs, and diagnostics."""

    method: str
    censoring_type: str
    dist: str
    impute_type: str
    scope: str
    censor_fraction: float
    n_censored: int
    n_observed: int
    is_censored: BoolArray
    status_codes: IntArray  # -1 left, 0 observed, 1 right, 2 interval
    original_values: FloatArray
    original_left: FloatArray | None
    original_right: FloatArray | None
    imputed_values: FloatArray
    dl_shifts: tuple[DlShiftEvent, ...]
    ndimpute_attrs: dict[str, Any]
    imputed_weight_factor: float
    n_stability_draws: int
    stability_mode: str
    window_fallback_count: int = 0


@dataclass(frozen=True)
class PreparedCensoredSeries:
    """Bundle returned by :func:`prepare_series_with_censoring`."""

    series: TimeSeriesInput
    provenance: CensoringProvenance | None
    warnings: tuple[WarningItem, ...]
    weight_factors: FloatArray | None
    config: CensoredConfig | None


def _to_obj_array(obj: Any) -> np.ndarray[Any, Any]:
    if type(obj).__module__.startswith("pandas") and hasattr(obj, "to_numpy"):
        return np.asarray(obj.to_numpy())
    return np.asarray(obj)


def has_censored_input(
    y: ValueLike,
    censor_status: ArrayLike | None,
    censoring: CensoredConfig | None,
) -> bool:
    """Return ``True`` if input values require the censored-data pipeline."""
    if censor_status is not None or censoring is not None:
        return True
    arr = _to_obj_array(y)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return True
    if arr.dtype.kind in ("U", "S", "O"):
        for item in arr.ravel():
            if isinstance(item, str) and any(ch in item for ch in ("<", ">")):
                return True
    return False


def _parse_censored_values(
    y: ValueLike,
    censor_status: ArrayLike | None,
    config: CensoredConfig,
) -> tuple[
    FloatArray,
    IntArray,
    FloatArray | None,
    FloatArray | None,
    str,
    BoolArray,
]:
    """Parse 1D mixed strings/numeric or (N, 2) interval bounds into canonical arrays.

    Returns ``(vals, status_codes, left_bounds, right_bounds, ctype, missing_mask)``
    where ``status_codes`` uses ``-1`` (left), ``0`` (observed), ``1`` (right),
    ``2`` (interval).
    """
    raw = _to_obj_array(y)

    # Interval censoring: (N, 2) bounds
    if config.censoring_type == "interval" or (raw.ndim == 2 and raw.shape[1] == 2):
        bounds = np.asarray(raw, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise CensoringError(
                "For censoring_type='interval', `y` must be an (N, 2) array of (lower, upper) bounds."
            )
        lo = bounds[:, 0]
        hi = bounds[:, 1]
        missing = np.isnan(lo) | np.isnan(hi)
        if np.any(np.isinf(lo) & np.isinf(hi)):
            raise CensoringError("Interval bounds cannot both be infinite.")
        valid = ~missing
        if np.any(lo[valid] > hi[valid]):
            raise CensoringError("Interval lower bounds must not exceed upper bounds.")
        is_interval = valid & (lo != hi)
        codes = np.where(is_interval, 2, 0).astype(np.intp)
        mid = np.where(valid, 0.5 * (lo + hi), np.nan)
        return mid, codes, lo, hi, "interval", missing

    if raw.ndim != 1:
        raise CensoringError(f"Expected 1-D values (or (N, 2) intervals); got shape {raw.shape}.")

    l_esc = re.escape(config.left_marker)
    r_esc = re.escape(config.right_marker)
    num_pat = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(rf"^\s*({l_esc}|{r_esc})?\s*({num_pat})\s*$")

    items = list(np.asarray(raw, dtype=object).ravel())
    n = len(items)
    vals = np.empty(n, dtype=np.float64)
    marker_codes = np.zeros(n, dtype=np.intp)

    for i, item in enumerate(items):
        if item is None:
            vals[i] = np.nan
            continue
        if isinstance(item, (bool, np.bool_)):
            raise CensoringError(
                f"Boolean entry {item!r} found in `y`; pass flags via `censor_status`."
            )
        if isinstance(item, (int, float, np.integer, np.floating)):
            vals[i] = float(item)
            continue
        s_item = str(item).strip()
        if s_item.lower() in _MISSING_STRINGS:
            vals[i] = np.nan
            continue
        m = pattern.match(s_item)
        if not m:
            raise CensoringError(
                f"Could not parse censored entry {item!r}. Expected numeric value or "
                f"prefixed limit like '{config.left_marker}0.05' or '{config.right_marker}10'."
            )
        marker, num_str = m.group(1), m.group(2)
        vals[i] = float(num_str)
        if marker == config.left_marker:
            marker_codes[i] = -1
        elif marker == config.right_marker:
            marker_codes[i] = 1

    if np.any(np.isinf(vals)):
        raise CensoringError("Value vector `y` contains infinite entries.")

    missing = np.isnan(vals)
    has_left = bool(np.any(marker_codes == -1))
    has_right = bool(np.any(marker_codes == 1))

    if censor_status is None:
        codes = marker_codes
        inferred = "mixed" if (has_left and has_right) else ("right" if has_right else "left")
        ctype = config.censoring_type or inferred
    else:
        st_arr = _to_obj_array(censor_status)
        if st_arr.ndim != 1 or st_arr.size != n:
            raise CensoringError(
                f"`censor_status` must be 1-D with length {n}; got shape {st_arr.shape}."
            )
        if st_arr.dtype == np.bool_:
            flags = st_arr.astype(np.bool_)
            if config.censoring_type == "right":
                codes = np.where(flags, 1, 0).astype(np.intp)
                ctype = "right"
            else:
                codes = np.where(flags, -1, 0).astype(np.intp)
                ctype = config.censoring_type or "left"
        elif st_arr.dtype.kind in ("O", "U", "S"):
            parsed_codes = np.empty(n, dtype=np.intp)
            for i, s_val in enumerate(st_arr):
                key = str(s_val).strip().lower()
                if key not in _STATUS_STRINGS:
                    raise CensoringError(
                        f"Unrecognized censor_status string {s_val!r}. Use 'lt', 'gt', or 'not'."
                    )
                parsed_codes[i] = _STATUS_STRINGS[key]
            codes = parsed_codes
            inferred = (
                "mixed"
                if (np.any(codes == -1) and np.any(codes == 1))
                else ("right" if np.any(codes == 1) else "left")
            )
            ctype = config.censoring_type or inferred
        else:
            f_arr = np.asarray(st_arr, dtype=np.float64)
            if not np.all(np.isfinite(f_arr)) or not np.all(f_arr == np.round(f_arr)):
                raise CensoringError("Numeric `censor_status` must contain only integers (-1, 0, 1).")
            int_codes = f_arr.astype(np.intp)
            uniq = set(np.unique(int_codes).tolist())
            if config.censoring_type in ("left", "right"):
                if not uniq <= {0, 1}:
                    raise CensoringError(
                        f"For censoring_type={config.censoring_type!r}, `censor_status` must be "
                        f"boolean or 0/1; got {sorted(uniq)}."
                    )
                sign = -1 if config.censoring_type == "left" else 1
                codes = int_codes * sign
                ctype = config.censoring_type
            else:
                if not uniq <= {-1, 0, 1}:
                    raise CensoringError(
                        f"`censor_status` must contain only -1, 0, 1; got {sorted(uniq)}."
                    )
                if config.censoring_type == "mixed" or -1 in uniq:
                    codes = int_codes
                    ctype = "mixed"
                else:
                    codes = -int_codes
                    ctype = "left"

    if ctype == "left" and np.any(codes == 1):
        raise CensoringError(
            "censoring_type='left' requested, but data contain right-censored entries."
        )
    if ctype == "right" and np.any(codes == -1):
        raise CensoringError(
            "censoring_type='right' requested, but data contain left-censored entries."
        )

    codes = np.where(missing, 0, codes)
    return vals, codes, None, None, ctype, missing


def _call_ndimpute_single(
    vals: FloatArray,
    codes: IntArray,
    left_b: FloatArray | None,
    right_b: FloatArray | None,
    ctype: str,
    config: CensoredConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, dict[str, Any]]:
    """Invoke ``ndimpute.impute`` safely with temporal tie-shuffling for quantile ROS."""
    nd = _require_ndimpute()
    n = int(vals.size)

    if not np.any(codes != 0):
        return vals.copy(), {
            "method": config.method,
            "censoring_type": ctype,
            "dist": config.dist,
            "impute_type": config.impute_type,
        }

    # Resolve method & impute_type compatibility cleanly
    method = config.method
    impute_type = config.impute_type
    if ctype == "interval" and method == "substitution":
        method = "parametric"
    if method == "parametric" and impute_type == "quantile":
        impute_type = "mean"

    kw: dict[str, Any] = dict(config.ndimpute_kwargs)
    if method != "substitution":
        kw.setdefault("dist", config.dist)
        kw.setdefault("impute_type", impute_type)
    if method == "ros":
        kw.setdefault("clamp", config.clamp)
    if impute_type == "stochastic" or (method == "ros" and ctype == "mixed"):
        kw["random_state"] = rng

    # CRITICAL SAFEGUARD:
    # ndimpute's ROS quantile mode uses stable sort on values, which maps tied
    # detection limits in chronological order to strictly increasing plotting
    # positions (injecting an artificial upward trend). Permuting the input order
    # before calling ndimpute and inverting afterwards eliminates temporal bias
    # while preserving exact NADA marginal quantiles.
    shuffle_needed = method == "ros" and impute_type == "quantile" and ctype != "interval"

    if ctype == "interval":
        assert left_b is not None and right_b is not None
        bounds = np.column_stack([left_b, right_b])
        if method == "ros" and kw.get("impute_type") == "quantile":
            kw["impute_type"] = "stochastic"
            kw["random_state"] = rng
        try:
            df = nd.impute(bounds, method=method, censoring_type="interval", **kw)
        except (ValueError, RuntimeError) as exc:
            raise CensoringError(f"ndimpute interval imputation failed: {exc}") from exc
        return np.asarray(df["imputed_value"].to_numpy(), dtype=np.float64), dict(df.attrs)

    if shuffle_needed:
        perm = rng.permutation(n)
        inv_perm = np.empty(n, dtype=np.intp)
        inv_perm[perm] = np.arange(n, dtype=np.intp)
        v_in = vals[perm]
        c_in = codes[perm]
    else:
        v_in = vals
        c_in = codes
        inv_perm = np.arange(n, dtype=np.intp)

    # Map canonical (-1, 0, 1) codes to ndimpute's status convention
    if ctype == "left":
        status_in: Any = c_in == -1
    elif ctype == "right":
        status_in = c_in == 1
    else:
        status_in = c_in

    try:
        df = nd.impute(v_in, status=status_in, method=method, censoring_type=ctype, **kw)
    except (ValueError, RuntimeError) as exc:
        raise CensoringError(f"ndimpute imputation failed: {exc}") from exc

    imp = np.asarray(df["imputed_value"].to_numpy(), dtype=np.float64)
    if shuffle_needed:
        imp = imp[inv_perm]
    return imp, dict(df.attrs)


def detect_dl_shifts(
    times_numeric: FloatArray,
    times_raw: np.ndarray[Any, Any],
    valid_to_orig: IntArray,
    original_values: FloatArray,
    status_codes: IntArray,
    *,
    rel_threshold: float = 0.20,
    min_run: int = 2,
) -> tuple[DlShiftEvent, ...]:
    """Detect step shifts in active censoring limits along the time axis.

    Detects two real-world laboratory detection-limit change patterns:
    1. **Explicit limit transition:** The sequence of censoring limits shifts
       by at least ``rel_threshold`` (even if only 1–2 observations fall below
       a newly lowered detection limit).
    2. **Implicit DL drop (censoring disappearance):** Censoring at an old
       limit ``L_old`` ceases and multiple uncensored detections appear
       strictly below ``(1 - rel_threshold) * L_old``.
    """
    cen_pos = np.flatnonzero((status_codes == -1) | (status_codes == 1))
    if cen_pos.size < min_run:
        return ()

    lims = original_values[cen_pos]
    m = int(lims.size)
    h = min(5, max(min_run, m // 8))

    candidates: list[tuple[int, float, float, float]] = []
    for k in range(1, m):
        left_slice = lims[max(0, k - h) : k]
        right_slice = lims[k : min(m, k + h)]
        if left_slice.size < min_run and right_slice.size < min_run:
            continue
        left_med = float(np.median(left_slice))
        right_med = float(np.median(right_slice))
        denom = max(abs(left_med), abs(right_med), 1e-12)
        rel = abs(right_med - left_med) / denom
        step_at_k = abs(lims[k] - lims[k - 1]) / denom
        if rel >= rel_threshold and step_at_k >= rel_threshold * 0.5:
            pos_idx = int(cen_pos[k])
            # If a left-DL dropped, the new lower limit became active as soon as
            # uncensored detections strictly below the old limit began appearing.
            if status_codes[cen_pos[k - 1]] == -1 and right_med < left_med:
                prev_pos = int(cen_pos[k - 1])
                inter = np.arange(prev_pos + 1, pos_idx)
                if inter.size > 0:
                    inter_det_below = inter[
                        (status_codes[inter] == 0)
                        & (original_values[inter] < left_med * (1.0 - rel_threshold))
                    ]
                    if inter_det_below.size > 0:
                        pos_idx = int(inter_det_below[0])
            candidates.append((pos_idx, left_med, right_med, rel))

    # Implicit left-DL drop: censoring at L_old stops and >=3 uncensored detections
    # subsequently appear well below L_old (e.g., lab upgraded from <2.0 to <0.2 so
    # almost nothing is censored afterward, yet values 0.4, 0.7, 1.1 are now detected).
    left_pos = np.flatnonzero(status_codes == -1)
    if left_pos.size >= min_run:
        last_left = int(left_pos[-1])
        old_dl = float(original_values[last_left])
        after_mask = (np.arange(original_values.size) > last_left) & (status_codes == 0)
        after_vals = original_values[after_mask]
        below_old = after_vals[after_vals < old_dl * (1.0 - rel_threshold)]
        if below_old.size >= 3:
            first_below_pos = int(
                np.flatnonzero(after_mask & (original_values < old_dl * (1.0 - rel_threshold)))[0]
            )
            inferred_new = float(np.min(below_old))
            rel = abs(old_dl - inferred_new) / max(abs(old_dl), 1e-12)
            # Only add if not already covered by an explicit transition nearby
            if not any(abs(c[0] - first_below_pos) <= max(5, h) for c in candidates):
                candidates.append((first_below_pos, old_dl, inferred_new, rel))

    if not candidates:
        return ()

    candidates.sort(key=lambda item: item[0])
    # Merge adjacent transitions within 5 valid observations, keeping largest relative jump
    merged: list[tuple[int, float, float, float]] = [candidates[0]]
    for cand in candidates[1:]:
        if cand[0] - merged[-1][0] <= max(5, h):
            if cand[3] > merged[-1][3]:
                merged[-1] = cand
        else:
            merged.append(cand)

    events: list[DlShiftEvent] = []
    for v_idx, old_l, new_l, rel in merged:
        events.append(
            DlShiftEvent(
                index=int(valid_to_orig[v_idx]),
                valid_index=v_idx,
                time=times_raw[v_idx],
                time_year=float(times_numeric[v_idx]),
                old_limit=old_l,
                new_limit=new_l,
                relative_change=rel,
            )
        )
    return tuple(events)


def check_dl_shift_coincidence(
    event_time_year: float,
    dl_shifts: tuple[DlShiftEvent, ...],
    times_numeric: FloatArray,
) -> bool:
    """Return ``True`` if a detected breakpoint falls close to a detected DL shift."""
    if not dl_shifts:
        return False
    dt_med = float(np.median(np.diff(times_numeric))) if times_numeric.size > 1 else 0.1
    prox = max(0.5, 3.0 * dt_med)
    return any(abs(event_time_year - shift.time_year) <= prox for shift in dl_shifts)


def _impute_seasonal_stratified(
    t_valid: FloatArray,
    vals: FloatArray,
    codes: IntArray,
    left_b: FloatArray | None,
    right_b: FloatArray | None,
    ctype: str,
    config: CensoredConfig,
    rng: np.random.Generator,
    global_imp: FloatArray,
) -> tuple[FloatArray, int]:
    """Stratify imputation by quarterly phase ``floor((t % 1.0) * 4)`` with global fallback."""
    quarters = np.floor((t_valid % 1.0) * 4.0).astype(np.intp) % 4
    out = global_imp.copy()
    fallbacks = 0
    for q in range(4):
        mask = quarters == q
        n_q = int(np.sum(mask))
        if n_q == 0:
            continue
        c_q = codes[mask]
        n_cen_q = int(np.sum(c_q != 0))
        if n_cen_q == 0:
            continue
        n_det_q = n_q - n_cen_q
        det_vals = vals[mask][c_q == 0]
        if n_det_q < 5 or (n_cen_q / n_q) > 0.80 or np.unique(det_vals).size < 2:
            fallbacks += 1
            continue
        lb_q = left_b[mask] if left_b is not None else None
        rb_q = right_b[mask] if right_b is not None else None
        try:
            imp_q, _ = _call_ndimpute_single(vals[mask], c_q, lb_q, rb_q, ctype, config, rng)
            out[mask] = imp_q
        except CensoringError:
            fallbacks += 1
    return out, fallbacks


def impute_window_local(
    provenance: CensoringProvenance,
    config: CensoredConfig,
    start: int,
    stop: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, bool]:
    """Impute a local sliding window ``[start:stop]``, falling back to global if too censored."""
    codes_w = provenance.status_codes[start:stop]
    n_w = int(codes_w.size)
    n_cen_w = int(np.sum(codes_w != 0))
    if n_cen_w == 0:
        return provenance.imputed_values[start:stop].copy(), False

    vals_w = provenance.original_values[start:stop]
    n_det_w = n_w - n_cen_w
    det_vals = vals_w[codes_w == 0]
    if n_det_w < 5 or (n_cen_w / n_w) > 0.80 or np.unique(det_vals).size < 2:
        return provenance.imputed_values[start:stop].copy(), True

    lb_w = provenance.original_left[start:stop] if provenance.original_left is not None else None
    rb_w = provenance.original_right[start:stop] if provenance.original_right is not None else None
    try:
        imp_w, _ = _call_ndimpute_single(
            vals_w, codes_w, lb_w, rb_w, provenance.censoring_type, config, rng
        )
        return imp_w, False
    except CensoringError:
        return provenance.imputed_values[start:stop].copy(), True


def recensor_to_common_limit(
    provenance: CensoringProvenance,
    config: CensoredConfig,
    start: int,
    stop: int,
    rng: np.random.Generator,
) -> FloatArray | None:
    """Temporarily recensor all observations below max active left-DL in ``[start:stop]``.

    Returns the re-imputed window series (or ``None`` if recensoring leaves too
    few detects or there is only one unique left detection limit).
    """
    codes_w = provenance.status_codes[start:stop].copy()
    vals_w = provenance.original_values[start:stop].copy()
    left_mask = codes_w == -1
    if not np.any(left_mask):
        return None
    unique_dls = np.unique(vals_w[left_mask])
    if unique_dls.size < 2:
        return None

    l_max = float(np.max(unique_dls))
    recen_mask = (codes_w == -1) | ((codes_w == 0) & (vals_w < l_max))
    vals_recen = np.where(recen_mask, l_max, vals_w)
    codes_recen = np.where(recen_mask, -1, codes_w)

    n_det = int(np.sum(codes_recen == 0))
    if n_det < 5 or np.unique(vals_recen[codes_recen == 0]).size < 2:
        return None

    try:
        imp_recen, _ = _call_ndimpute_single(
            vals_recen, codes_recen, None, None, provenance.censoring_type, config, rng
        )
        return imp_recen
    except CensoringError:
        return None


def _bootstrap_refit_draw(
    provenance: CensoringProvenance,
    config: CensoredConfig,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate one proper bootstrap-refit imputation draw propagating parameter uncertainty.

    Resamples ``(values, status)`` with replacement to fit perturbed parameters,
    then imputes the *original* series' censored observations conditioned on the
    bootstrap-refitted distribution.
    """
    n = int(provenance.original_values.size)
    vals = provenance.original_values
    codes = provenance.status_codes
    ctype = provenance.censoring_type
    cen_mask = codes != 0

    if not np.any(cen_mask):
        return np.asarray(vals.copy(), dtype=np.float64)

    # Draw bootstrap resample with at least 2 distinct detects
    boot_idx: IntArray | None = None
    for _ in range(15):
        cand = rng.integers(0, n, size=n)
        c_b = codes[cand]
        v_b = vals[cand]
        if int(np.sum(c_b == 0)) >= 3 and np.unique(v_b[c_b == 0]).size >= 2:
            boot_idx = cand
            break
    if boot_idx is None:
        # Fallback to fixed-theta stochastic draw
        cfg_stoch = CensoredConfig(
            method=config.method,
            censoring_type=config.censoring_type,
            dist=config.dist,
            impute_type="stochastic",
            clamp=config.clamp,
            random_state=rng,
            ndimpute_kwargs=config.ndimpute_kwargs,
        )
        imp, _ = _call_ndimpute_single(
            vals,
            codes,
            provenance.original_left,
            provenance.original_right,
            ctype,
            cfg_stoch,
            rng,
        )
        return np.asarray(imp, dtype=np.float64)

    # Fit parametric or ROS model on bootstrap sample + evaluate truncated draw on original censored limits
    # Combine bootstrap detects & original censored entries cleanly via parametric/ROS fit on bootstrap sample:
    from scipy.stats import CensoredData, lognorm, norm, weibull_min

    v_b = vals[boot_idx]
    c_b = codes[boot_idx]
    dist_name = provenance.dist
    if dist_name not in ("lognormal", "normal", "weibull"):
        dist_name = "lognormal" if np.all(vals > 0) else "normal"

    # Fit MLE on bootstrap sample for coherent parameter draw across left/right/mixed/interval
    try:
        if ctype == "interval":
            assert provenance.original_left is not None and provenance.original_right is not None
            lo_b = provenance.original_left[boot_idx]
            hi_b = provenance.original_right[boot_idx]
            exact_b = lo_b == hi_b
            lo_orig = provenance.original_left
            hi_orig = provenance.original_right
        else:
            lower_open = 0.0 if dist_name in ("lognormal", "weibull") else -np.inf
            lo_b = np.where(c_b == -1, lower_open, v_b)
            hi_b = np.where(c_b == 1, np.inf, v_b)
            exact_b = c_b == 0
            lo_orig = np.where(codes == -1, lower_open, vals)
            hi_orig = np.where(codes == 1, np.inf, vals)

        def _tx(x: FloatArray) -> FloatArray:
            return np.asarray(np.log(x) if dist_name == "lognormal" else x, dtype=np.float64)

        if dist_name == "lognormal" and (np.any(hi_b <= 0) or np.any(lo_b[exact_b] <= 0)):
            raise ValueError("Non-positive values for lognormal.")

        left_b_mask = (~exact_b) & (
            (lo_b <= 0) if dist_name in ("lognormal", "weibull") else np.isneginf(lo_b)
        )
        right_b_mask = (~exact_b) & np.isposinf(hi_b)
        int_b_mask = (~exact_b) & (~left_b_mask) & (~right_b_mask)

        cd = CensoredData(
            uncensored=_tx(lo_b[exact_b]),
            left=_tx(hi_b[left_b_mask]) if np.any(left_b_mask) else None,
            right=_tx(lo_b[right_b_mask]) if np.any(right_b_mask) else None,
            interval=np.column_stack([_tx(lo_b[int_b_mask]), _tx(hi_b[int_b_mask])])
            if np.any(int_b_mask)
            else None,
        )

        out = vals.copy()
        u = rng.uniform(size=int(np.sum(cen_mask)))
        lo_c = lo_orig[cen_mask]
        hi_c = hi_orig[cen_mask]

        if dist_name in ("normal", "lognormal"):
            mu, sigma = norm.fit(cd)
            if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
                raise ValueError("Invalid bootstrap fit.")
            a_t = np.where(lo_c <= 0, -np.inf, np.log(lo_c)) if dist_name == "lognormal" else lo_c
            b_t = np.where(np.isposinf(hi_c), np.inf, np.log(hi_c)) if dist_name == "lognormal" else hi_c
            cdf_lo = norm.cdf(a_t, loc=mu, scale=sigma)
            cdf_hi = norm.cdf(b_t, loc=mu, scale=sigma)
            p_draw = np.clip(cdf_lo + u * np.maximum(cdf_hi - cdf_lo, 1e-15), 1e-15, 1.0 - 1e-15)
            z_draw = norm.ppf(p_draw, loc=mu, scale=sigma)
            draws = np.exp(z_draw) if dist_name == "lognormal" else z_draw
            if config.clamp:
                draws = np.clip(draws, lo_c, hi_c)
            out[cen_mask] = draws
        else:
            shape, _, scale = weibull_min.fit(cd, floc=0)
            if not (np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0):
                raise ValueError("Invalid Weibull bootstrap fit.")
            dist_obj = weibull_min(shape, scale=scale)
            cdf_lo = dist_obj.cdf(np.maximum(lo_c, 0.0))
            cdf_hi = np.where(np.isposinf(hi_c), 1.0, dist_obj.cdf(hi_c))
            p_draw = np.clip(cdf_lo + u * np.maximum(cdf_hi - cdf_lo, 1e-15), 1e-15, 1.0 - 1e-15)
            draws = dist_obj.ppf(p_draw)
            if config.clamp:
                draws = np.clip(draws, lo_c, hi_c)
            out[cen_mask] = draws

        _ = lognorm
        return np.asarray(out, dtype=np.float64)
    except Exception:
        cfg_stoch = CensoredConfig(
            method=config.method,
            censoring_type=config.censoring_type,
            dist=config.dist,
            impute_type="stochastic",
            clamp=config.clamp,
            random_state=rng,
            ndimpute_kwargs=config.ndimpute_kwargs,
        )
        imp, _ = _call_ndimpute_single(
            vals,
            codes,
            provenance.original_left,
            provenance.original_right,
            ctype,
            cfg_stoch,
            rng,
        )
        return np.asarray(imp, dtype=np.float64)


def generate_stability_draws(
    provenance: CensoringProvenance,
    config: CensoredConfig,
    t_valid: FloatArray,
) -> list[FloatArray]:
    """Generate ``config.n_stability_draws`` alternative imputed series for stability checks."""
    n_draws = int(config.n_stability_draws)
    if n_draws <= 0 or provenance.n_censored == 0:
        return []

    base_seed = config.random_state if isinstance(config.random_state, (int, np.integer)) else 0
    ss = np.random.SeedSequence(int(base_seed) + 10_007)
    child_rngs = [np.random.default_rng(s) for s in ss.spawn(n_draws)]

    draws: list[FloatArray] = []
    for rng in child_rngs:
        if config.stability_mode == "bootstrap":
            draws.append(_bootstrap_refit_draw(provenance, config, rng))
        else:
            cfg_draw = CensoredConfig(
                method=config.method,
                censoring_type=config.censoring_type,
                dist=config.dist,
                impute_type="stochastic",
                scope=config.scope,
                random_state=rng,
                clamp=config.clamp,
                ndimpute_kwargs=config.ndimpute_kwargs,
            )
            imp, _ = _call_ndimpute_single(
                provenance.original_values,
                provenance.status_codes,
                provenance.original_left,
                provenance.original_right,
                provenance.censoring_type,
                cfg_draw,
                rng,
            )
            if config.scope == "seasonal_stratified":
                imp, _ = _impute_seasonal_stratified(
                    t_valid,
                    provenance.original_values,
                    provenance.status_codes,
                    provenance.original_left,
                    provenance.original_right,
                    provenance.censoring_type,
                    cfg_draw,
                    rng,
                    imp,
                )
            draws.append(imp)
    return draws


def prepare_series_with_censoring(
    t: TimeLike,
    y: ValueLike,
    *,
    censor_status: ArrayLike | None = None,
    censoring: CensoredConfig | None = None,
    allow_duplicates: bool = False,
) -> PreparedCensoredSeries:
    """Unified entry point preparing numeric OR censored time series for JUST/STPD.

    When inputs are purely numeric and neither ``censor_status`` nor ``censoring``
    is provided, bypasses ``ndimpute`` completely (zero pandas/ndimpute imports).
    """
    if not has_censored_input(y, censor_status, censoring):
        from ._time import prepare_numeric_series

        series = prepare_numeric_series(t, y, allow_duplicates=allow_duplicates)
        return PreparedCensoredSeries(
            series=series,
            provenance=None,
            warnings=(),
            weight_factors=None,
            config=None,
        )

    cfg = censoring if censoring is not None else CensoredConfig()
    t_num, t_raw, is_dt = coerce_times(t, allow_duplicates=allow_duplicates)
    vals_all, codes_all, left_all, right_all, ctype, missing = _parse_censored_values(
        y, censor_status, cfg
    )

    if vals_all.size != t_num.size:
        raise ValueError(
            f"`t` and `y` must have the same length; got {t_num.size} and {vals_all.size}."
        )

    valid_mask = ~missing
    n_input = int(vals_all.size)
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        raise InsufficientDataError("All observations in `y` are missing (NaN).")

    valid_to_orig = np.flatnonzero(valid_mask)
    t_valid = t_num[valid_mask]
    t_raw_valid = t_raw[valid_mask]
    vals_valid = vals_all[valid_mask]
    codes_valid = codes_all[valid_mask]
    left_valid = left_all[valid_mask] if left_all is not None else None
    right_valid = right_all[valid_mask] if right_all is not None else None

    is_cen = codes_valid != 0
    n_cen = int(np.sum(is_cen))
    n_obs = n_valid - n_cen
    cen_frac = n_cen / n_valid

    warnings_list: list[WarningItem] = []

    # Tiered guardrails BEFORE calling ndimpute
    if n_cen > 0:
        n_distinct_det = int(np.unique(vals_valid[~is_cen]).size) if n_obs > 0 else 0
        if n_obs < 2 or n_distinct_det < 2:
            raise CensoringError(
                f"Too few uncensored observations to fit imputation model "
                f"({n_obs} detected, {n_distinct_det} distinct; need at least 2 distinct detects)."
            )
        if cen_frac > cfg.max_censor_fraction and not cfg.allow_heavy_censoring:
            raise CensoringError(
                f"{cen_frac:.1%} of observations are censored (exceeds max_censor_fraction="
                f"{cfg.max_censor_fraction:.0%}). Refusing to fit change-point model on mostly "
                "imputed data; pass `CensoredConfig(allow_heavy_censoring=True)` to override."
            )
        if n_obs < 5 and not cfg.allow_heavy_censoring:
            raise CensoringError(
                f"Only {n_obs} uncensored detections available; imputation model is poorly "
                "determined. Pass `CensoredConfig(allow_heavy_censoring=True)` to override."
            )
        if cen_frac > 0.50:
            warnings_list.append(
                WarningItem(
                    code="HEAVY_CENSORING",
                    severity="warning",
                    message=(
                        f"{cen_frac:.1%} of observations ({n_cen}/{n_valid}) are censored (>50%). "
                        "Change-point estimates depend strongly on distributional assumptions."
                    ),
                    suggestion=(
                        "Enable `n_stability_draws=20` (or `stability_mode='bootstrap'`) and "
                        "interpret results cautiously alongside non-parametric checks."
                    ),
                )
            )
        elif cen_frac > cfg.warn_censor_fraction:
            warnings_list.append(
                WarningItem(
                    code="MODERATE_CENSORING",
                    severity="caution",
                    message=(
                        f"{cen_frac:.1%} of observations ({n_cen}/{n_valid}) are censored "
                        f"(>{cfg.warn_censor_fraction:.0%})."
                    ),
                    suggestion=(
                        "Consider enabling `n_stability_draws=20` to check whether detected "
                        "breakpoints are stable across imputation draws."
                    ),
                )
            )
        else:
            warnings_list.append(
                WarningItem(
                    code="CENSORED_IMPUTATION_APPLIED",
                    severity="info",
                    message=(
                        f"Imputed {n_cen} censored observation(s) ({cen_frac:.1%}) via "
                        f"ndimpute (method={cfg.method!r}, impute_type={cfg.impute_type!r}, "
                        f"scope={cfg.scope!r})."
                    ),
                )
            )

        if cfg.impute_type == "mean":
            warnings_list.append(
                WarningItem(
                    code="CONDITIONAL_MEAN_VARIANCE_SHRINKAGE",
                    severity="info",
                    message=(
                        "Conditional-mean imputation assigns identical values to tied limits, "
                        "shrinking residual variance and potentially making p-values optimistic."
                    ),
                    suggestion="Prefer `impute_type='stochastic'` with `n_stability_draws >= 20`.",
                )
            )

        if cfg.imputed_weight_factor < 0.5:
            warnings_list.append(
                WarningItem(
                    code="EXTREME_IMPUTED_DOWNWEIGHTING",
                    severity="caution",
                    message=(
                        f"`imputed_weight_factor={cfg.imputed_weight_factor}` heavily down-weights "
                        "censored observations, which can mimic dropping non-detects and introduce bias."
                    ),
                    suggestion="Keep `imputed_weight_factor` in [0.5, 1.0].",
                )
            )

        # Non-stationarity check across first vs second half censoring rates
        if cfg.scope == "global" and n_valid >= 20:
            half = n_valid // 2
            f1 = float(np.mean(is_cen[:half]))
            f2 = float(np.mean(is_cen[half:]))
            if abs(f1 - f2) > 0.30:
                warnings_list.append(
                    WarningItem(
                        code="NONSTATIONARY_CENSORING_PATTERN",
                        severity="caution",
                        message=(
                            f"Censoring rate shifts sharply between the first half ({f1:.0%}) "
                            f"and second half ({f2:.0%}) of the record."
                        ),
                        suggestion=(
                            "Global imputation assumes a stationary distribution; consider "
                            "`CensoredConfig(scope='windowed')` so post-jump regimes do not "
                            "bias pre-jump imputations."
                        ),
                    )
                )

    rng = (
        cfg.random_state
        if isinstance(cfg.random_state, np.random.Generator)
        else np.random.default_rng(cfg.random_state)
    )

    imp_global, attrs = _call_ndimpute_single(
        vals_valid, codes_valid, left_valid, right_valid, ctype, cfg, rng
    )
    fallback_count = 0
    if cfg.scope == "seasonal_stratified" and n_cen > 0:
        imp_values, fallback_count = _impute_seasonal_stratified(
            t_valid,
            vals_valid,
            codes_valid,
            left_valid,
            right_valid,
            ctype,
            cfg,
            rng,
            imp_global,
        )
    else:
        imp_values = imp_global

    dl_shifts: tuple[DlShiftEvent, ...] = ()
    if cfg.detect_dl_shifts and n_cen > 0 and ctype != "interval":
        dl_shifts = detect_dl_shifts(
            t_valid,
            t_raw_valid,
            valid_to_orig,
            vals_valid,
            codes_valid,
            rel_threshold=cfg.dl_shift_rel_threshold,
        )

    provenance = CensoringProvenance(
        method=str(attrs.get("method", cfg.method)),
        censoring_type=str(attrs.get("censoring_type", ctype)),
        dist=str(attrs.get("dist", cfg.dist)),
        impute_type=str(attrs.get("impute_type", cfg.impute_type)),
        scope=cfg.scope,
        censor_fraction=float(cen_frac),
        n_censored=n_cen,
        n_observed=n_obs,
        is_censored=is_cen,
        status_codes=codes_valid,
        original_values=vals_valid,
        original_left=left_valid,
        original_right=right_valid,
        imputed_values=imp_values,
        dl_shifts=dl_shifts,
        ndimpute_attrs=attrs,
        imputed_weight_factor=cfg.imputed_weight_factor,
        n_stability_draws=cfg.n_stability_draws,
        stability_mode=cfg.stability_mode,
        window_fallback_count=fallback_count,
    )

    rate_ratio = estimate_sampling_rate_ratio(t_valid)
    series = TimeSeriesInput(
        times_numeric=t_valid,
        times_raw=t_raw_valid,
        is_datetime=is_dt,
        values=imp_values,
        valid_mask=valid_mask,
        valid_to_orig=valid_to_orig,
        n_input=n_input,
        n_valid=n_valid,
        n_dropped_nan=n_input - n_valid,
        sampling_rate_ratio=rate_ratio,
    )

    weight_factors: FloatArray | None = None
    if n_cen > 0 and cfg.imputed_weight_factor < 1.0:
        weight_factors = np.where(is_cen, cfg.imputed_weight_factor, 1.0).astype(np.float64)

    return PreparedCensoredSeries(
        series=series,
        provenance=provenance,
        warnings=tuple(warnings_list),
        weight_factors=weight_factors,
        config=cfg,
    )
