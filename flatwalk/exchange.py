"""Exchange handler interface — empty hook point for replica-exchange WL (REWL).

The single-walker driver never instantiates anything from this module. It
exists so that a shared-``g`` REWL variant can plug into the core loop
*without* touching the loop: the driver only knows how to call
``handler.maybe_exchange(...)`` every ``N_exchange`` trials.

The shipped REWL path (per-window ``g``) does not use this hook — see
:class:`flatwalk.RewlDriver`. This module is kept for the shared-``g`` REWL
variant when it is wired up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .walker import Walker


@dataclass
class ExchangeResult:
    """Outcome of an exchange attempt.

    ``new_bin`` is the (possibly-swapped) walker's flat bin index after the
    exchange. ``g_delta`` is an optional sparse update applied to the shared
    log-DoS array (REWL variants that share ``g`` across windows may use this;
    others can leave it ``None``).
    """

    swapped: bool
    new_bin: int
    g_delta: np.ndarray | None = None


class ExchangeHandler(ABC):
    """Abstract REWL exchange handler. Not wired up in the current release."""

    @property
    @abstractmethod
    def n_exchange(self) -> int:
        """Driver calls ``maybe_exchange`` every ``n_exchange`` trials."""

    @abstractmethod
    def maybe_exchange(
        self,
        walker: Walker,
        g: np.ndarray,
    ) -> ExchangeResult | None:
        """Attempt an exchange. Return ``None`` for no-op.

        Implementations may mutate ``walker.state`` and ``walker.bin_current``
        in place when swapping; the driver re-syncs from the return value.
        """
