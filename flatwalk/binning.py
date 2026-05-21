"""Bin schemes for Wang-Landau sampling.

The driver indexes ``g`` and ``H`` through a `BinScheme` so the same loop works
for 1D order parameters now and ≥2D extensions later. Concrete subclasses must
map an order-parameter value to a flat integer index, report dimensionality,
and expose bin edges/centers.

Only ``Bin1D`` is implemented in M1. A future ``BinND`` slots in by
implementing the same ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

QValue = float | int | np.floating | np.integer | np.ndarray


class BinScheme(ABC):
    """Abstract mapping between order-parameter values and flat bin indices."""

    @abstractmethod
    def value_to_index(self, q: QValue) -> int:
        """Return the flat bin index for ``q``. Raises ``IndexError`` if out of range."""

    @abstractmethod
    def index_to_center(self, idx: int) -> float | np.ndarray:
        """Return the bin center for flat index ``idx``."""

    @abstractmethod
    def in_range(self, q: QValue) -> bool:
        """Return True iff ``q`` lies in the binned domain (inclusive)."""

    @abstractmethod
    def value_to_index_batched(self, q: np.ndarray) -> np.ndarray:
        """Vectorized `value_to_index` for a stack of N order-parameter values.

        Returns an integer index array of length N. Out-of-range entries get
        the sentinel index ``-1`` (rather than raising) so the batched trial
        step can mask them — the inverse of the scalar contract, which raises.
        Callers must not index ``g``/``H`` with a ``-1`` entry; gate on
        `in_range_batched` first.
        """

    @abstractmethod
    def in_range_batched(self, q: np.ndarray) -> np.ndarray:
        """Vectorized `in_range`: boolean mask of length N (inclusive domain)."""

    @property
    @abstractmethod
    def n_bins(self) -> int:
        """Total number of flat bins."""

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """Dimension of the order-parameter space (1 for 1D, 2 for 2D, ...)."""

    @property
    @abstractmethod
    def edges(self) -> np.ndarray:
        """Bin edges. For 1D: shape ``(n_bins + 1,)``."""

    @property
    @abstractmethod
    def centers(self) -> np.ndarray:
        """Bin centers. For 1D: shape ``(n_bins,)``."""


def _to_scalar(q: QValue) -> float:
    """Coerce a scalar-like input to ``float``. Rejects multi-element arrays."""
    if isinstance(q, np.ndarray):
        if q.size != 1:
            raise ValueError(f"Bin1D received array of size {q.size}; expected a scalar.")
        return float(q.reshape(()).item())
    return float(q)


class Bin1D(BinScheme):
    """Uniform 1D binning of the closed interval ``[low, high]``.

    Bin edges are ``np.linspace(low, high, n_bins + 1)`` so the first and last
    edges land exactly on ``low`` and ``high``. ``value_to_index(high)`` returns
    ``n_bins - 1`` (the top edge is treated as inside the top bin).
    """

    def __init__(self, low: float, high: float, n_bins: int) -> None:
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError(f"low and high must be finite, got {low}, {high}")
        if high <= low:
            raise ValueError(f"high ({high}) must exceed low ({low})")
        if n_bins < 1:
            raise ValueError(f"n_bins must be ≥ 1, got {n_bins}")
        self._low = float(low)
        self._high = float(high)
        self._n_bins = int(n_bins)
        self._width = (self._high - self._low) / self._n_bins
        self._edges = np.linspace(self._low, self._high, self._n_bins + 1, dtype=np.float64)
        self._centers = 0.5 * (self._edges[:-1] + self._edges[1:])

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    @property
    def width(self) -> float:
        return self._width

    @property
    def n_bins(self) -> int:
        return self._n_bins

    @property
    def dimensionality(self) -> int:
        return 1

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    @property
    def centers(self) -> np.ndarray:
        return self._centers

    def in_range(self, q: QValue) -> bool:
        x = _to_scalar(q)
        return self._low <= x <= self._high

    def value_to_index(self, q: QValue) -> int:
        x = _to_scalar(q)
        if not (self._low <= x <= self._high):
            raise IndexError(
                f"value {x} outside Bin1D domain [{self._low}, {self._high}]; "
                "callers must check in_range() first."
            )
        idx = int((x - self._low) / self._width)
        if idx == self._n_bins:
            idx = self._n_bins - 1
        return idx

    def in_range_batched(self, q: np.ndarray) -> np.ndarray:
        x = np.asarray(q, dtype=np.float64)
        return (x >= self._low) & (x <= self._high)

    def value_to_index_batched(self, q: np.ndarray) -> np.ndarray:
        x = np.asarray(q, dtype=np.float64)
        idx = ((x - self._low) / self._width).astype(np.int64)
        # Exact top edge (and float-rounding past it) folds into the top bin,
        # matching the scalar `value_to_index(high) == n_bins - 1` convention.
        idx[idx == self._n_bins] = self._n_bins - 1
        # Out-of-range entries get the sentinel; do this last so it overrides
        # the truncation above (e.g. x just above `high` rounds to n_bins).
        idx[(x < self._low) | (x > self._high)] = -1
        return idx

    def index_to_center(self, idx: int) -> float:
        if not (0 <= idx < self._n_bins):
            raise IndexError(f"bin index {idx} out of [0, {self._n_bins})")
        return float(self._centers[idx])

    def __repr__(self) -> str:
        return (
            f"Bin1D(low={self._low}, high={self._high}, n_bins={self._n_bins}, "
            f"width={self._width})"
        )
