"""Wang-Landau driver — config, result, and the main driver class.

Algorithm reference: Wang & Landau PRL 86, 2050 (2001) for the standard
flat-histogram scheme; Belardinelli & Pereyra PRE 75, 046701 (2007) for the
1/t refinement that the driver switches to when the standard halving
would put ``ln_f`` below ``1/t``.

Architectural notes
-------------------
- The per-trial logic operates on a `Walker` (not on driver attributes),
  so a future multi-walker / REWL extension is additive, not a rewrite.
- All bin arithmetic goes through ``self.bin_scheme``; no array indexing
  presumes 1D.
- An optional ``exchange_handler`` slot is present in the loop so REWL
  drops in without touching this file.
- Out-of-range proposals (spec §1.3): rejected. ``g`` and ``H`` are still
  updated at the *current* bin (reflecting-boundary convention).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .binning import BinScheme
from .diagnostics import TraceRow, TraceWriter
from .exchange import ExchangeHandler
from .walker import Walker, WalkerBatch

logger = logging.getLogger(__name__)

EnergyFn = Callable[[Any], float]
OrderParamFn = Callable[[Any], float | np.ndarray]
ProposeMoveFn = Callable[[Any, np.random.Generator], tuple[Any, float]]

# Batched callbacks for ≥2 walkers (docs §4). Each takes one opaque
# ``state_batch`` carrying N walkers and operates on all N at once — one
# stacked call per tick, never a Python loop over walkers. ``state_batch`` is
# opaque to the driver exactly as scalar ``state`` is; the driver only ever
# applies accepted moves to it via boolean-mask assignment, so it must support
# that (a stacked ``ndarray[N, ...]`` or ``torch.Tensor`` does).
BatchedEnergyFn = Callable[[Any], np.ndarray]  # state_batch -> E[N]
BatchedOrderParamFn = Callable[[Any], np.ndarray]  # state_batch -> Q[N] (or Q[N, D])
BatchedProposeMoveFn = Callable[
    [Any, np.random.Generator], tuple[Any, np.ndarray]
]  # (state_batch, rng) -> (new_state_batch, log_proposal_ratio[N])


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------


@dataclass
class WLConfig:
    """Configuration for one WL run.

    Notes
    -----
    - ``bin_scheme`` carries the order-parameter domain and dimensionality.
    - ``beta = 1/(k_B T)``. Wang-Landau samples ``g(Q)`` independently of
      temperature; ``beta`` only enters the acceptance criterion when the
      energy axis is *not* the order parameter (e.g. binning on
      magnetization while sampling at temperature T). For canonical "WL on
      E" runs, set ``beta = 0``; the energy term drops out by construction.
    """

    bin_scheme: BinScheme
    beta: float = 0.0
    flatness_threshold: float = 0.8
    n_check: int = 10_000
    ln_f_initial: float = 1.0
    ln_f_final: float = 1e-8
    checkpoint_path: Path | None = None
    checkpoint_every_t: int = 1_000_000
    trace_path: Path | None = None
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
    in_1overt: bool = False
    bin_current: int = -1
    walker_energy: float = float("nan")
    rng_state: dict | None = None
    # Populated by the batched path; left at the scalar defaults otherwise.
    n_walkers: int = 1
    walker_bins: np.ndarray | None = None
    walker_energies: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers (testable in isolation)
# ---------------------------------------------------------------------------


def compute_flatness(H: np.ndarray, visited: np.ndarray) -> float:
    """Return ``min(H[visited]) / mean(H[visited])``, or 0.0 if degenerate.

    "Visited" is the cumulative mask (any bin ever entered), per spec §1.5.
    The ratio is over **visited bins only**, which matters during early
    exploration when many bins still have ``H = 0``.
    """
    if not visited.any():
        return 0.0
    hv = H[visited]
    mean = float(hv.mean())
    if mean <= 0.0:
        return 0.0
    return float(hv.min()) / mean


def attempt_halve(ln_f: float, t: int, in_1overt: bool) -> tuple[float, bool]:
    """Apply one f-stage transition.

    In the standard regime, halve ``ln_f``. If the halved value would be
    below ``1/t``, switch to the 1/t-WL regime instead, with ``ln_f = 1/t``.

    Returns ``(new_ln_f, new_in_1overt)``. Caller is responsible for
    resetting ``H`` (standard regime only) and updating ``n_f_stages``.

    Parameters
    ----------
    ln_f
        Current modification factor.
    t
        Total trials attempted so far (must be > 0 to switch).
    in_1overt
        Should not normally be called with ``True`` (the 1/t regime
        updates every trial via a different path). Defensive: if True,
        return ``(1/t, True)`` unchanged.
    """
    if in_1overt:
        return (1.0 / t, True) if t > 0 else (ln_f, True)
    halved = ln_f / 2.0
    if t > 0 and halved < 1.0 / t:
        return (1.0 / t, True)
    return (halved, False)


def build_trace_row(
    *,
    t: int,
    ln_f: float,
    flatness: float,
    acceptance_rate: float,
    H: np.ndarray,
    visited: np.ndarray,
    in_1overt: bool,
    stage_index: int,
) -> TraceRow:
    """Assemble one diagnostic row from the live driver state.

    Factored out so the scalar and batched loops emit identical trace columns
    without copying the min/mean/max-over-visited reduction.
    """
    if visited.any():
        hv = H[visited]
        min_H = int(hv.min())
        max_H = int(hv.max())
        mean_H = float(hv.mean())
    else:
        min_H = max_H = 0
        mean_H = 0.0
    return TraceRow(
        t=t,
        ln_f=ln_f,
        flatness=flatness,
        acceptance_rate=acceptance_rate,
        min_H_visited=min_H,
        max_H_visited=max_H,
        mean_H_visited=mean_H,
        n_visited=int(visited.sum()),
        in_1overt=in_1overt,
        stage_index=stage_index,
    )


# ---------------------------------------------------------------------------
# Batched trial step (shared by the shared-g and replica-exchange drivers)
# ---------------------------------------------------------------------------


def _grouped_trial_step(
    bin_scheme: BinScheme,
    wb: WalkerBatch,
    g: np.ndarray,
    H: np.ndarray,
    visited: np.ndarray,
    group: np.ndarray,
    b_lo,
    b_hi,
    ln_f: float,
    energy_fn: BatchedEnergyFn,
    order_parameter_fn: BatchedOrderParamFn,
    propose_move_fn: BatchedProposeMoveFn,
    beta: float,
) -> np.ndarray:
    """One batched WL trial for all walkers at once. Returns the accept mask.

    The single batched step both batched drivers call (design:
    ``design-unified-batched-step``). It is parameterised over two arrays so
    the shared-``g`` and per-window cases are the *same* code with different
    index maps:

    - ``g``, ``H``, ``visited`` have shape ``(G, B)``; ``group`` is an
      ``int[N]`` mapping each of the ``N`` walkers to its row.
    - ``b_lo``, ``b_hi`` are inclusive bin bounds (scalars or ``int[N]``)
      confining each walker; a proposal landing outside ``[b_lo, b_hi]`` is
      rejected. Full-grid bounds ``0 … B-1`` reproduce the plain in-range mask,
      because :meth:`BinScheme.value_to_index_batched` returns ``-1`` off-grid.

    Updates are scattered with ``np.add.at`` keyed on ``(group, bin)``, so
    several walkers sharing a ``(group, bin)`` in one tick all count — this is
    what makes ≥2 walkers per group correct. Mutates ``wb``, ``g``, ``H``, and
    ``visited`` in place.
    """
    N = wb.n_walkers
    new_state, log_proposal_ratio = propose_move_fn(wb.state, wb.rng)
    q_new = np.asarray(order_parameter_fn(new_state))
    bin_new = bin_scheme.value_to_index_batched(q_new)  # -1 where off-grid
    in_bounds = (bin_new >= b_lo) & (bin_new <= b_hi)  # also rejects the -1 sentinel

    # β = 0 ⇒ the energy term drops out, so skip the (expensive, batched) call.
    e_new = np.asarray(energy_fn(new_state), dtype=np.float64) if beta != 0.0 else wb.energy

    # Never index g with the -1 sentinel: out-of-bounds walkers compare the
    # current bin against itself (Δ contribution 0) and are masked out below.
    safe_bin_new = np.where(in_bounds, bin_new, wb.bin_current)
    delta = (
        -beta * (e_new - wb.energy)
        + g[group, wb.bin_current]
        - g[group, safe_bin_new]
        + np.asarray(log_proposal_ratio, dtype=np.float64)
    )
    u = wb.rng.random(N)
    accept = in_bounds & ((delta >= 0.0) | (u < np.exp(np.minimum(delta, 0.0))))

    if accept.any():
        wb.state[accept] = new_state[accept]
    wb.bin_current = np.where(accept, bin_new, wb.bin_current)
    if beta != 0.0:
        wb.energy = np.where(accept, e_new, wb.energy)

    # np.add.at is the safe scatter for the repeated (group, bin) indices that
    # arise when several walkers share a group and land on the same bin.
    np.add.at(g, (group, wb.bin_current), ln_f)
    np.add.at(H, (group, wb.bin_current), 1)
    visited[group, wb.bin_current] = True
    wb.n_attempted += 1
    wb.n_accepted += accept.astype(np.int64)
    return accept


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class WLDriver:
    """Order-parameter-agnostic Wang-Landau sampler."""

    def __init__(self, config: WLConfig) -> None:
        self.config = config
        self.bin_scheme = config.bin_scheme

    # ---- per-trial logic (kept small so a multi-walker loop can call it) ----

    def _trial_step(
        self,
        walker: Walker,
        g: np.ndarray,
        H: np.ndarray,
        visited: np.ndarray,
        ln_f: float,
        energy_fn: EnergyFn,
        order_parameter_fn: OrderParamFn,
        propose_move_fn: ProposeMoveFn,
        beta: float,
    ) -> bool:
        """Execute one trial. Returns True iff the move was accepted.

        Mutates ``walker``, ``g``, ``H``, and ``visited`` in place.
        """
        new_state, log_proposal_ratio = propose_move_fn(walker.state, walker.rng)
        q_new = order_parameter_fn(new_state)
        accepted = False
        if self.bin_scheme.in_range(q_new):
            bin_new = self.bin_scheme.value_to_index(q_new)
            e_new = energy_fn(new_state)
            delta = (
                -beta * (e_new - walker.energy)
                + g[walker.bin_current]
                - g[bin_new]
                + log_proposal_ratio
            )
            if delta >= 0.0 or walker.rng.random() < math.exp(delta):
                walker.state = new_state
                walker.bin_current = bin_new
                walker.energy = e_new
                accepted = True
        # Out-of-range proposals fall through unchanged (spec §1.3).
        # g/H/visited update happens at the *current* bin, whether accepted
        # or not, after the decision (spec §1.1 step 5).
        g[walker.bin_current] += ln_f
        H[walker.bin_current] += 1
        visited[walker.bin_current] = True
        walker.n_attempted += 1
        if accepted:
            walker.n_accepted += 1
        return accepted

    # ---- batched per-trial logic (N walkers, one tick, no Python loop) ------

    def _trial_step_batched(
        self,
        wb: WalkerBatch,
        g: np.ndarray,
        H: np.ndarray,
        visited: np.ndarray,
        ln_f: float,
        energy_fn: BatchedEnergyFn,
        order_parameter_fn: BatchedOrderParamFn,
        propose_move_fn: BatchedProposeMoveFn,
        beta: float,
    ) -> np.ndarray:
        """Execute one trial for all N walkers at once. Returns the accept mask.

        Mutates ``wb``, ``g``, ``H``, and ``visited`` in place. A thin adapter
        over :func:`_grouped_trial_step`: the shared-``g`` case is a single
        group with full-grid bounds, and the driver's 1D ``g``/``H``/``visited``
        are bridged to the primitive's ``(G, B)`` contract by a ``[None]`` view
        (so the scatter writes straight back into the 1D buffers, and the result
        and checkpoint stay 1D). Mirrors `_trial_step`'s acceptance and
        reflecting-boundary conventions per walker.
        """
        return _grouped_trial_step(
            self.bin_scheme,
            wb,
            g[None],
            H[None],
            visited[None],
            np.zeros(wb.n_walkers, dtype=np.intp),
            0,
            self.bin_scheme.n_bins - 1,
            ln_f,
            energy_fn,
            order_parameter_fn,
            propose_move_fn,
            beta,
        )

    # ---- main loop ---------------------------------------------------------

    def run(
        self,
        initial_state: Any,
        energy_fn: EnergyFn,
        order_parameter_fn: OrderParamFn,
        propose_move_fn: ProposeMoveFn,
        max_trials: int | None = None,
        rng: np.random.Generator | None = None,
        exchange_handler: ExchangeHandler | None = None,
        resume_from: Path | None = None,
    ) -> WLResult:
        """Run a Wang-Landau simulation. Returns the final ``WLResult``.

        ``propose_move_fn`` returns ``(new_state, log_proposal_ratio)``.
        The acceptance criterion is::

            Δ = -β · (E_new - E_old) + g[bin_old] - g[bin_new] + log_proposal_ratio
            accept ⇔ U < exp(min(0, Δ))

        Out-of-range proposals are rejected; g/H/visited are still updated
        at the current bin (reflecting-boundary convention).
        """
        from .io import load_checkpoint, save_checkpoint  # local to avoid cycle

        cfg = self.config
        n = self.bin_scheme.n_bins

        # ---------- initialize from checkpoint or fresh ----------
        if resume_from is not None:
            cp = load_checkpoint(Path(resume_from))
            if cp["n_bins"] != n:
                raise ValueError(
                    f"checkpoint n_bins ({cp['n_bins']}) ≠ driver n_bins ({n})"
                )
            g = cp["g"].astype(np.float64, copy=True)
            H = cp["H"].astype(np.int64, copy=True)
            visited = cp["visited"].astype(bool, copy=True)
            t = int(cp["t_total"])
            n_f_stages = int(cp["n_f_stages"])
            ln_f = float(cp["ln_f"])
            in_1overt = bool(cp["in_1overt"])
            walker_state = cp["walker_state"]
            bin_current = int(cp["bin_current"])
            walker_energy = float(cp["walker_energy"])
            rng = _restore_rng(cp["rng_state"])
        else:
            if rng is None:
                rng = np.random.default_rng()
            g = np.zeros(n, dtype=np.float64)
            H = np.zeros(n, dtype=np.int64)
            visited = np.zeros(n, dtype=bool)
            t = 0
            n_f_stages = 0
            ln_f = cfg.ln_f_initial
            in_1overt = False
            walker_state = initial_state
            walker_energy = float(energy_fn(initial_state))
            q_initial = order_parameter_fn(initial_state)
            if not self.bin_scheme.in_range(q_initial):
                raise ValueError(
                    f"initial state order parameter {q_initial} outside bin domain"
                )
            bin_current = self.bin_scheme.value_to_index(q_initial)

        walker = Walker(
            state=walker_state,
            bin_current=bin_current,
            energy=walker_energy,
            rng=rng,
        )
        # Make sure the starting bin counts as visited so flatness math is sane.
        visited[walker.bin_current] = True

        trace_writer = TraceWriter(cfg.trace_path)
        converged = False
        interrupted = False
        wall_stage_start = time.perf_counter()

        logger.info(
            "WL run start: n_bins=%d, ln_f=%.3g → %.3g, n_check=%d, t0=%d, in_1overt=%s",
            n,
            ln_f,
            cfg.ln_f_final,
            cfg.n_check,
            t,
            in_1overt,
        )

        with trace_writer:
            try:
                while True:
                    # ---- stop checks (evaluated before the next trial) ----
                    if ln_f < cfg.ln_f_final:
                        converged = True
                        break
                    if max_trials is not None and (t - 0) >= max_trials:
                        break

                    # ---- one trial ----
                    self._trial_step(
                        walker,
                        g,
                        H,
                        visited,
                        ln_f,
                        energy_fn,
                        order_parameter_fn,
                        propose_move_fn,
                        cfg.beta,
                    )
                    t += 1

                    # ---- 1/t regime: continuously update ln_f ----
                    if in_1overt:
                        ln_f = 1.0 / t

                    # ---- periodic check ----
                    if t % cfg.n_check == 0:
                        flatness = compute_flatness(H, visited)
                        wrote_stage_transition = False

                        if not in_1overt:
                            # Standard regime: halve on flat
                            if flatness >= cfg.flatness_threshold:
                                new_ln_f, new_in_1overt = attempt_halve(ln_f, t, False)
                                wall_stage = time.perf_counter() - wall_stage_start
                                logger.info(
                                    "f-stage %d→%d at t=%d: ln_f %.6g → %.6g, "
                                    "flatness=%.3f, n_visited=%d, dt_stage=%.2fs",
                                    n_f_stages,
                                    n_f_stages + 1,
                                    t,
                                    ln_f,
                                    new_ln_f,
                                    flatness,
                                    int(visited.sum()),
                                    wall_stage,
                                )
                                if new_in_1overt and not in_1overt:
                                    logger.info(
                                        "Entering 1/t-WL regime at t=%d, ln_f=%.6g",
                                        t,
                                        new_ln_f,
                                    )
                                ln_f = new_ln_f
                                in_1overt = new_in_1overt
                                # Reset H only in standard regime — once we're
                                # in 1/t we keep accumulating.
                                if not in_1overt:
                                    H[:] = 0
                                # Numerical hygiene: keep g bounded.
                                if visited.any():
                                    g -= float(g[visited].min())
                                n_f_stages += 1
                                wall_stage_start = time.perf_counter()
                                wrote_stage_transition = True

                        # ---- emit one trace row per check (both regimes) ----
                        if visited.any():
                            hv = H[visited]
                            min_H = int(hv.min())
                            max_H = int(hv.max())
                            mean_H = float(hv.mean())
                        else:
                            min_H = max_H = 0
                            mean_H = 0.0
                        trace_writer.write(
                            TraceRow(
                                t=t,
                                ln_f=ln_f,
                                flatness=flatness,
                                acceptance_rate=walker.acceptance_rate(),
                                min_H_visited=min_H,
                                max_H_visited=max_H,
                                mean_H_visited=mean_H,
                                n_visited=int(visited.sum()),
                                in_1overt=in_1overt,
                                stage_index=n_f_stages,
                            )
                        )
                        if not wrote_stage_transition:
                            logger.debug(
                                "check at t=%d: ln_f=%.3g flatness=%.3f "
                                "min/mean/max H = %d/%.1f/%d, accept=%.3f",
                                t,
                                ln_f,
                                flatness,
                                min_H,
                                mean_H,
                                max_H,
                                walker.acceptance_rate(),
                            )

                        walker.reset_counters()

                    # ---- periodic checkpoint ----
                    if cfg.checkpoint_path is not None and t % cfg.checkpoint_every_t == 0:
                        save_checkpoint(
                            Path(cfg.checkpoint_path),
                            g=g,
                            H=H,
                            visited=visited,
                            bin_edges=self.bin_scheme.edges,
                            bin_centers=self.bin_scheme.centers,
                            n_bins=n,
                            t_total=t,
                            n_f_stages=n_f_stages,
                            ln_f=ln_f,
                            in_1overt=in_1overt,
                            walker_state=walker.state,
                            bin_current=walker.bin_current,
                            walker_energy=walker.energy,
                            rng_state=_capture_rng(walker.rng),
                        )

                    # ---- REWL hook (single-walker driver: never fires) ----
                    if exchange_handler is not None:
                        if t % exchange_handler.n_exchange == 0:
                            result = exchange_handler.maybe_exchange(walker, g)
                            if result is not None:
                                walker.bin_current = result.new_bin
                                if result.g_delta is not None:
                                    g += result.g_delta

            except KeyboardInterrupt:
                interrupted = True
                logger.info("Run interrupted at t=%d, ln_f=%.6g", t, ln_f)

        # ---- finalize ----
        if cfg.checkpoint_path is not None:
            save_checkpoint(
                Path(cfg.checkpoint_path),
                g=g,
                H=H,
                visited=visited,
                bin_edges=self.bin_scheme.edges,
                bin_centers=self.bin_scheme.centers,
                n_bins=n,
                t_total=t,
                n_f_stages=n_f_stages,
                ln_f=ln_f,
                in_1overt=in_1overt,
                walker_state=walker.state,
                bin_current=walker.bin_current,
                walker_energy=walker.energy,
                rng_state=_capture_rng(walker.rng),
            )

        logger.info(
            "WL run end: converged=%s interrupted=%s t=%d ln_f=%.6g n_f_stages=%d "
            "n_visited=%d/%d",
            converged,
            interrupted,
            t,
            ln_f,
            n_f_stages,
            int(visited.sum()),
            n,
        )

        return WLResult(
            g=g,
            H=H,
            visited=visited,
            bin_edges=self.bin_scheme.edges,
            bin_centers=self.bin_scheme.centers,
            t_total=t,
            n_f_stages=n_f_stages,
            ln_f_final=ln_f,
            converged=converged,
            final_state=walker.state,
            in_1overt=in_1overt,
            bin_current=walker.bin_current,
            walker_energy=walker.energy,
            rng_state=_capture_rng(walker.rng),
            extra={"interrupted": interrupted},
        )

    # ---- batched main loop -------------------------------------------------

    def run_batched(
        self,
        initial_state: Any,
        energy_fn: BatchedEnergyFn,
        order_parameter_fn: BatchedOrderParamFn,
        propose_move_fn: BatchedProposeMoveFn,
        n_walkers: int,
        *,
        max_trials: int | None = None,
        rng: np.random.Generator | None = None,
        resume_from: Path | None = None,
        exchange_handler: ExchangeHandler | None = None,
    ) -> WLResult:
        """Run N walkers through a shared ``g`` as one batch (docs §4).

        The N walkers advance together: each tick is one stacked call to each
        batched callback, never a Python loop over walkers. All N contribute to
        the *same* ``g``/``H`` via a scatter add, so a single window converges
        faster than one walker would.

        ``t_total`` counts individual moves (``n_walkers`` per tick), so the
        1/t-WL schedule, ``n_check``, ``checkpoint_every_t``, and ``max_trials``
        all use the same units as the scalar `run`; with ``n_walkers == 1`` the
        bookkeeping reduces to the scalar schedule exactly.

        Replica exchange (per-window ``g``) is docs §5 and not wired here yet;
        passing ``exchange_handler`` raises ``NotImplementedError``.
        """
        from .io import (  # local to avoid cycle
            load_checkpoint_batched,
            save_checkpoint_batched,
        )

        if exchange_handler is not None:
            raise NotImplementedError(
                "batched replica exchange is not implemented yet (docs §5); "
                "run_batched currently supports a single shared-g window."
            )
        if n_walkers < 1:
            raise ValueError(f"n_walkers must be ≥ 1; got {n_walkers}")

        cfg = self.config
        n = self.bin_scheme.n_bins

        # ---------- initialize from checkpoint or fresh ----------
        if resume_from is not None:
            cp = load_checkpoint_batched(Path(resume_from))
            if cp["n_bins"] != n:
                raise ValueError(
                    f"checkpoint n_bins ({cp['n_bins']}) ≠ driver n_bins ({n})"
                )
            if cp["n_walkers"] != n_walkers:
                raise ValueError(
                    f"checkpoint n_walkers ({cp['n_walkers']}) ≠ "
                    f"requested n_walkers ({n_walkers})"
                )
            g = cp["g"].astype(np.float64, copy=True)
            H = cp["H"].astype(np.int64, copy=True)
            visited = cp["visited"].astype(bool, copy=True)
            t = int(cp["t_total"])
            n_f_stages = int(cp["n_f_stages"])
            ln_f = float(cp["ln_f"])
            in_1overt = bool(cp["in_1overt"])
            batch_state = cp["walker_state"]
            bin_current = cp["bin_current"].astype(np.int64, copy=True)
            energy = cp["energy"].astype(np.float64, copy=True)
            n_attempted = cp["n_attempted"].astype(np.int64, copy=True)
            n_accepted = cp["n_accepted"].astype(np.int64, copy=True)
            rng = _restore_rng(cp["rng_state"])
        else:
            if rng is None:
                rng = np.random.default_rng()
            g = np.zeros(n, dtype=np.float64)
            H = np.zeros(n, dtype=np.int64)
            visited = np.zeros(n, dtype=bool)
            t = 0
            n_f_stages = 0
            ln_f = cfg.ln_f_initial
            in_1overt = False
            # Accepted moves are applied in place (boolean-mask assignment), so
            # copy first — the driver must not mutate the caller's input.
            batch_state = copy.deepcopy(initial_state)
            q_initial = np.asarray(order_parameter_fn(batch_state))
            if q_initial.shape[0] != n_walkers:
                raise ValueError(
                    f"order_parameter_fn returned {q_initial.shape[0]} values; "
                    f"expected n_walkers={n_walkers}"
                )
            if not self.bin_scheme.in_range_batched(q_initial).all():
                raise ValueError("one or more initial states are outside the bin domain")
            bin_current = self.bin_scheme.value_to_index_batched(q_initial).astype(np.int64)
            if cfg.beta != 0.0:
                energy = np.asarray(energy_fn(batch_state), dtype=np.float64).copy()
            else:
                # Energy term drops out; per-walker energy stays zero (unused).
                energy = np.zeros(n_walkers, dtype=np.float64)
            n_attempted = np.zeros(n_walkers, dtype=np.int64)
            n_accepted = np.zeros(n_walkers, dtype=np.int64)

        wb = WalkerBatch(
            state=batch_state,
            bin_current=bin_current,
            energy=energy,
            rng=rng,
            n_attempted=n_attempted,
            n_accepted=n_accepted,
        )
        # Starting bins count as visited so the flatness math is sane.
        visited[wb.bin_current] = True

        trace_writer = TraceWriter(cfg.trace_path)
        converged = False
        interrupted = False
        wall_stage_start = time.perf_counter()

        logger.info(
            "WL batched run start: n_walkers=%d, n_bins=%d, ln_f=%.3g → %.3g, "
            "n_check=%d, t0=%d, in_1overt=%s",
            n_walkers,
            n,
            ln_f,
            cfg.ln_f_final,
            cfg.n_check,
            t,
            in_1overt,
        )

        def _crossed(boundary: int, lo: int, hi: int) -> bool:
            # A multiple of ``boundary`` lies in (lo, hi]. t jumps by n_walkers
            # per tick, so exact ``t % boundary == 0`` is unreliable; this fires
            # the schedule once per window and is a deterministic function of t.
            return (hi // boundary) != (lo // boundary)

        def _save() -> None:
            save_checkpoint_batched(
                Path(cfg.checkpoint_path),
                g=g,
                H=H,
                visited=visited,
                bin_edges=self.bin_scheme.edges,
                bin_centers=self.bin_scheme.centers,
                bin_current=wb.bin_current,
                energy=wb.energy,
                n_attempted=wb.n_attempted,
                n_accepted=wb.n_accepted,
                n_bins=n,
                t_total=t,
                n_f_stages=n_f_stages,
                ln_f=ln_f,
                in_1overt=in_1overt,
                n_walkers=n_walkers,
                walker_state=wb.state,
                rng_state=_capture_rng(wb.rng),
            )

        with trace_writer:
            try:
                while True:
                    # ---- stop checks (evaluated before the next tick) ----
                    if ln_f < cfg.ln_f_final:
                        converged = True
                        break
                    if max_trials is not None and t >= max_trials:
                        break

                    # ---- one batched tick (n_walkers moves) ----
                    t_before = t
                    self._trial_step_batched(
                        wb,
                        g,
                        H,
                        visited,
                        ln_f,
                        energy_fn,
                        order_parameter_fn,
                        propose_move_fn,
                        cfg.beta,
                    )
                    t += n_walkers

                    # ---- 1/t regime: continuously update ln_f ----
                    if in_1overt:
                        ln_f = 1.0 / t

                    # ---- periodic check ----
                    if _crossed(cfg.n_check, t_before, t):
                        flatness = compute_flatness(H, visited)
                        wrote_stage_transition = False

                        if not in_1overt and flatness >= cfg.flatness_threshold:
                            new_ln_f, new_in_1overt = attempt_halve(ln_f, t, False)
                            wall_stage = time.perf_counter() - wall_stage_start
                            logger.info(
                                "f-stage %d→%d at t=%d: ln_f %.6g → %.6g, "
                                "flatness=%.3f, n_visited=%d, dt_stage=%.2fs",
                                n_f_stages,
                                n_f_stages + 1,
                                t,
                                ln_f,
                                new_ln_f,
                                flatness,
                                int(visited.sum()),
                                wall_stage,
                            )
                            if new_in_1overt:
                                logger.info(
                                    "Entering 1/t-WL regime at t=%d, ln_f=%.6g",
                                    t,
                                    new_ln_f,
                                )
                            ln_f = new_ln_f
                            in_1overt = new_in_1overt
                            if not in_1overt:
                                H[:] = 0
                            if visited.any():
                                g -= float(g[visited].min())
                            n_f_stages += 1
                            wall_stage_start = time.perf_counter()
                            wrote_stage_transition = True

                        trace_writer.write(
                            build_trace_row(
                                t=t,
                                ln_f=ln_f,
                                flatness=flatness,
                                acceptance_rate=wb.acceptance_rate(),
                                H=H,
                                visited=visited,
                                in_1overt=in_1overt,
                                stage_index=n_f_stages,
                            )
                        )
                        if not wrote_stage_transition:
                            logger.debug(
                                "check at t=%d: ln_f=%.3g flatness=%.3f accept=%.3f",
                                t,
                                ln_f,
                                flatness,
                                wb.acceptance_rate(),
                            )
                        wb.reset_counters()

                    # ---- periodic checkpoint ----
                    if cfg.checkpoint_path is not None and _crossed(
                        cfg.checkpoint_every_t, t_before, t
                    ):
                        _save()

            except KeyboardInterrupt:
                interrupted = True
                logger.info("Batched run interrupted at t=%d, ln_f=%.6g", t, ln_f)

        # ---- finalize ----
        if cfg.checkpoint_path is not None:
            _save()

        logger.info(
            "WL batched run end: converged=%s interrupted=%s t=%d ln_f=%.6g "
            "n_f_stages=%d n_visited=%d/%d",
            converged,
            interrupted,
            t,
            ln_f,
            n_f_stages,
            int(visited.sum()),
            n,
        )

        return WLResult(
            g=g,
            H=H,
            visited=visited,
            bin_edges=self.bin_scheme.edges,
            bin_centers=self.bin_scheme.centers,
            t_total=t,
            n_f_stages=n_f_stages,
            ln_f_final=ln_f,
            converged=converged,
            final_state=wb.state,
            in_1overt=in_1overt,
            n_walkers=n_walkers,
            walker_bins=wb.bin_current,
            walker_energies=wb.energy,
            rng_state=_capture_rng(wb.rng),
            extra={"interrupted": interrupted},
        )


# ---------------------------------------------------------------------------
# RNG state helpers (used by both core and io)
# ---------------------------------------------------------------------------


def _capture_rng(rng: np.random.Generator) -> dict:
    """Snapshot a Generator's bit-generator state for serialization."""
    return rng.bit_generator.state  # plain dict of ints + name string


def _restore_rng(state: dict) -> np.random.Generator:
    """Rehydrate a Generator from a state previously captured by `_capture_rng`."""
    bg_name = state["bit_generator"]
    bg_cls = getattr(np.random, bg_name)
    bg = bg_cls()
    bg.state = state
    return np.random.Generator(bg)
