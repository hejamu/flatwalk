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
