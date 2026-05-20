"""Wang-Landau driver — config, result, and main driver class.

M1 ships the public API surface (``WLConfig``, ``WLResult``, ``WLDriver``)
and the dataclass machinery; the per-trial loop, f-stage schedule, and
1/t-WL transition land in M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from .binning import BinScheme
from .exchange import ExchangeHandler

EnergyFn = Callable[[Any], float]
OrderParamFn = Callable[[Any], Union[float, np.ndarray]]
ProposeMoveFn = Callable[[Any, np.random.Generator], Tuple[Any, float]]


@dataclass
class WLConfig:
    """Configuration for one WL run.

    Notes
    -----
    - ``bin_scheme`` carries Q_min, Q_max, n_bins, dimensionality. The
      legacy fields in the spec (Q_min/Q_max/n_bins on WLConfig) are
      derivable from ``bin_scheme`` and intentionally not duplicated here.
    - ``beta = 1/(k_B T)``. Wang-Landau samples ``g(Q)`` independently of
      temperature, so ``beta`` only enters the acceptance criterion when
      the energy axis is *not* the order parameter (e.g. binning on a
      magnetization while sampling at temperature T). For canonical
      "WL on E" runs, set ``beta = 0``; the energy term in the
      acceptance criterion then drops out by construction.
    """

    bin_scheme: BinScheme
    beta: float = 0.0
    flatness_threshold: float = 0.8
    n_check: int = 10_000
    ln_f_initial: float = 1.0
    ln_f_final: float = 1e-8
    checkpoint_path: Optional[Path] = None
    checkpoint_every_t: int = 1_000_000
    trace_path: Optional[Path] = None
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not (0.0 < self.flatness_threshold < 1.0):
            raise ValueError(
                f"flatness_threshold must be in (0, 1); got {self.flatness_threshold}"
            )
        if self.n_check < 1:
            raise ValueError(f"n_check must be ≥ 1; got {self.n_check}")
        if self.ln_f_final <= 0:
            raise ValueError(f"ln_f_final must be > 0; got {self.ln_f_final}")
        if self.ln_f_initial <= self.ln_f_final:
            raise ValueError(
                f"ln_f_initial ({self.ln_f_initial}) must exceed "
                f"ln_f_final ({self.ln_f_final})"
            )
        if self.checkpoint_every_t < 1:
            raise ValueError("checkpoint_every_t must be ≥ 1")


@dataclass
class WLResult:
    """End-of-run summary. Mirrors the on-disk checkpoint contents."""

    g: np.ndarray
    H: np.ndarray
    visited: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    t_total: int
    n_f_stages: int
    ln_f_final: float
    converged: bool
    final_state: Any
    rng_state: Optional[dict] = None
    extra: dict = field(default_factory=dict)


class WLDriver:
    """Order-parameter-agnostic Wang-Landau sampler.

    The driver is a thin orchestrator: it owns ``g``, ``H``, ``ln_f``, ``t``,
    and the f-stage schedule. Everything *physics-specific* — what a state
    looks like, what its energy is, what its order parameter is, how to
    propose a move — lives in user callbacks passed to ``run``.

    Out-of-range proposals (Q outside ``bin_scheme``'s domain) are rejected
    and the current bin is updated anyway (reflecting boundaries; spec §1.3).
    """

    def __init__(self, config: WLConfig) -> None:
        self.config = config
        self.bin_scheme = config.bin_scheme

    def run(
        self,
        initial_state: Any,
        energy_fn: EnergyFn,
        order_parameter_fn: OrderParamFn,
        propose_move_fn: ProposeMoveFn,
        max_trials: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        exchange_handler: Optional[ExchangeHandler] = None,
        resume_from: Optional[Path] = None,
    ) -> WLResult:
        """Run a Wang-Landau simulation. Returns the final ``WLResult``.

        ``propose_move_fn`` returns ``(new_state, log_proposal_ratio)``.
        The acceptance criterion is::

            Δ = -β · (E_new - E_old) + g[bin_old] - g[bin_new] + log_proposal_ratio
            accept ⇔ U < exp(min(0, Δ))

        Implementation lands in M2.
        """
        raise NotImplementedError(
            "WLDriver.run is scheduled for M2 — see flatwalk/core.py."
        )
