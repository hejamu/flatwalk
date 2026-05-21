"""Walker — the per-replica state container.

The driver's per-trial logic operates on a `Walker`, not on attributes of the
driver itself. This keeps the door open for shared-`g`/multi-walker runs and
for replica exchange (which swaps walker states) without rewriting the loop.

A single-walker WL run still uses exactly one ``Walker``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Walker:
    """State owned by one Markov-chain walker.

    Attributes
    ----------
    state:
        Whatever the user wants to represent a configuration. The driver
        passes it through to callbacks without inspection.
    bin_current:
        Current flat bin index. ``-1`` means "not yet placed" (the driver
        sets it on the first call to `place`).
    energy:
        Cached energy of ``state`` (avoids re-calling ``energy_fn`` after every
        reject; updated on every accepted move).
    rng:
        Per-walker random source. Multi-walker runs give each walker an
        independent stream so a fixed master seed reproduces exactly.
    n_accepted:
        Trials accepted since the last counter reset (used for trace
        acceptance-rate columns; reset by the driver every check interval).
    n_attempted:
        Trials attempted since the last counter reset.
    """

    state: Any
    bin_current: int = -1
    energy: float = np.nan
    rng: np.random.Generator | None = None
    n_accepted: int = 0
    n_attempted: int = 0
    extra: dict = field(default_factory=dict)

    def reset_counters(self) -> None:
        self.n_accepted = 0
        self.n_attempted = 0

    def acceptance_rate(self) -> float:
        if self.n_attempted == 0:
            return float("nan")
        return self.n_accepted / self.n_attempted


@dataclass
class WalkerBatch:
    """State for N walkers carried in flat arrays (the N-walker `Walker`).

    The design rule (docs §4): for ≥2 walkers the per-tick primitives act on
    all N states at once. There is deliberately no Python-side list of
    `Walker` to iterate — one `WalkerBatch` holds N walkers' worth of state in
    parallel arrays, and a single shared ``rng`` provides vectorized draws.

    Attributes
    ----------
    state:
        Opaque batched configuration, passed through to the batched callbacks
        without inspection. The driver applies accepted moves with
        boolean-mask assignment (``state[accept] = new_state[accept]``), so it
        must support that — a stacked ``ndarray[N, ...]`` or ``torch.Tensor``
        does; a plain Python list of N objects does not.
    bin_current:
        ``ndarray[N]`` int — each walker's current flat bin index.
    energy:
        ``ndarray[N]`` float — cached energy per walker. Left untouched when
        ``beta == 0`` (the energy term drops out of acceptance and the batched
        energy call is skipped, per docs §4).
    rng:
        One shared `numpy.random.Generator`. A fixed master seed reproduces a
        batched run exactly.
    n_attempted, n_accepted:
        ``ndarray[N]`` int per-walker counters since the last reset (the driver
        resets them every check interval).
    """

    state: Any
    bin_current: np.ndarray
    energy: np.ndarray
    rng: np.random.Generator | None = None
    n_attempted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_accepted: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    extra: dict = field(default_factory=dict)

    @property
    def n_walkers(self) -> int:
        return int(self.bin_current.shape[0])

    def reset_counters(self) -> None:
        self.n_attempted[:] = 0
        self.n_accepted[:] = 0

    def acceptance_rate(self) -> float:
        """Aggregate accept fraction across all walkers since the last reset.

        The trace carries a single acceptance-rate column, so we pool the
        per-walker counters rather than reporting N separate rates.
        """
        total_attempted = int(self.n_attempted.sum())
        if total_attempted == 0:
            return float("nan")
        return int(self.n_accepted.sum()) / total_attempted
