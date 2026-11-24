"""Shared static typing aliases for tspec."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]

ArrayLike: TypeAlias = Sequence[float | int] | NDArray[Any] | Any
TimeLike: TypeAlias = (
    Sequence[float | int | datetime | date | np.datetime64] | NDArray[Any] | Any
)
ValueLike: TypeAlias = Sequence[float | int | str | None] | NDArray[Any] | Any
WeightLike: TypeAlias = float | int | Sequence[float | int] | NDArray[Any]
