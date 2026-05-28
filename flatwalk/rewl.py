"""Replica-exchange Wang-Landau (REWL) on top of the batched-walker layer.

This module is entirely additive: it imports the schedule helpers already
factored out of :mod:`flatwalk.core` and reuses :class:`flatwalk.walker.WalkerBatch`.
The scalar single-walker driver, the shared-``g`` :meth:`WLDriver.run_batched`,
and every existing example stay exactly as they are — REWL lives here.

The picture
-----------
``W`` windows tile the order-parameter range with overlap; each window has one
walker confined to its sub-range and its own log-density estimate. All windows
share the *same global bin grid* (``config.bin_scheme``) — a window is just a
contiguous bin-index interval ``[b_lo, b_hi]`` of that grid. That single-grid
choice makes two things trivial:

- the per-walker bin index is the global :meth:`Bin1D.value_to_index_batched`,
  so the trial step is the batched one with a 2-D ``g_windows[W, B]`` gathered
  per walker, and
- joining the windows afterwards aligns bin-for-bin in the overlap regions.

Every per-tick primitive is vectorised over the ``W`` walkers — there is no
``for w in walkers`` in the inner loop, just as in the batched single-window
driver. The only Python loops over windows are in setup (`make_windows`), the
periodic schedule check, and post-processing (`join_g`), none of which touch
the energy backend.

REWL exchange uses the standard entropy-based (temperature-independent)
criterion::

    Δ = g_i(E_j) − g_i(E_i) + g_j(E_i) − g_j(E_j)
    accept ⇔ U < exp(min(0, Δ))

evaluated batched over all adjacent pairs, alternating even/odd parity per
sweep for detailed balance. A swap is only valid when each config falls inside
the *other* window's range (the overlap condition).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .binning import BinScheme
from .core import (
    BatchedEnergyFn,
    BatchedOrderParamFn,
    BatchedProposeMoveFn,
    WLConfig,
    _capture_rng,
    _grouped_trial_step,
    attempt_halve,
    build_trace_row,
)
from .diagnostics import TraceWriter
from .walker import WalkerBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def make_windows(
    bin_scheme: BinScheme, n_windows: int, overlap: int
) -> list[tuple[float, float]]:
    """Tile the grid with ``n_windows`` equal-width overlapping windows.

    Returns a list of ``(low, high)`` value bounds. Every window has the same
    bin width ``Lw`` and adjacent windows share ``overlap`` bins, so the
    windows are balanced (no artificially narrow end window — important when
    the density of states is steep in the tails). Bounds are bin *centers*, so
    they round-trip exactly through :meth:`value_to_index`.

    With ``Lw = ceil((B + (W-1)·overlap) / W)`` and stride ``Lw - overlap``,
    window ``k`` is ``[k·stride, k·stride + Lw - 1]``; the last window is
    clamped to end on the top bin.
    """
    B = bin_scheme.n_bins
    if n_windows < 1:
        raise ValueError(f"n_windows must be ≥ 1; got {n_windows}")
    if n_windows > B:
        raise ValueError(f"n_windows ({n_windows}) cannot exceed n_bins ({B})")
    centers = bin_scheme.centers
    if n_windows == 1:
        return [(float(centers[0]), float(centers[B - 1]))]
    if overlap < 1:
        raise ValueError(
            f"overlap must be ≥ 1 to keep adjacent windows joinable; got {overlap}"
        )

    width = min(-(-(B + (n_windows - 1) * overlap) // n_windows), B)  # ceil, capped at B
    stride = width - overlap
    if stride < 1:
        raise ValueError(
            f"overlap ({overlap}) too large for n_bins={B}, n_windows={n_windows}"
        )

    bounds: list[tuple[int, int]] = []
    for k in range(n_windows):
        b_lo = k * stride
        b_hi = b_lo + width - 1
        if b_hi >= B - 1 or k == n_windows - 1:
            b_hi = B - 1
            b_lo = min(b_lo, b_hi)
        bounds.append((b_lo, b_hi))

    # Guard the invariants downstream code relies on: full coverage and a
    # non-empty overlap between every adjacent pair.
    for k in range(1, n_windows):
        if bounds[k][0] > bounds[k - 1][1]:
            raise ValueError(
                f"windows {k - 1} and {k} do not overlap (overlap={overlap} too small "
                f"for n_bins={B}, n_windows={n_windows})"
            )
    if bounds[0][0] != 0 or bounds[-1][1] != B - 1:
        raise ValueError("windows do not cover the full grid; check overlap/n_windows")
    return [(float(centers[b_lo]), float(centers[b_hi])) for b_lo, b_hi in bounds]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RewlResult:
    """End-of-run REWL summary. Per-window log-densities plus diagnostics.

    Call :func:`join_g` on ``g_windows``/``visited_windows`` to stitch the
    windows into a single ``g`` over the full grid.
    """

    g_windows: np.ndarray  # (W, B) log-density per window
    H_windows: np.ndarray  # (W, B) histogram per window
    visited_windows: np.ndarray  # (W, B) bool
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    windows: list[tuple[float, float]]  # (low, high) value bounds as given
    window_bins: np.ndarray  # (W, 2) int bin bounds [b_lo, b_hi]
    t_total: int  # ticks (one synchronized move per window per tick)
    n_f_stages: int
    ln_f_final: float
    converged: bool
    final_state: Any
    walker_bins: np.ndarray  # (W,) final global bin per walker
    walker_energies: np.ndarray  # (W,)
    n_windows: int
    exchange_attempts: np.ndarray  # (W-1,) per adjacent boundary
    exchange_accepts: np.ndarray  # (W-1,)
    rng_state: dict | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vectorised primitives
# ---------------------------------------------------------------------------


def _flatness_per_window(H_windows: np.ndarray, visited_windows: np.ndarray) -> np.ndarray:
    """Per-window ``min(H_visited) / mean(H_visited)`` (0 where degenerate).

    Vectorised equivalent of calling :func:`flatwalk.core.compute_flatness` on
    each row, without a Python loop and without NaN warnings.
    """
    H = H_windows.astype(np.float64)
    v = visited_windows
    counts = v.sum(axis=1)
    sums = (H * v).sum(axis=1)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    mins = np.where(v, H, np.inf).min(axis=1)
    mins = np.where(counts > 0, mins, 0.0)
    ok = (counts > 0) & (means > 0)
    return np.where(ok, mins / np.where(means > 0, means, 1.0), 0.0)


def _normalize_g(g_windows: np.ndarray, visited_windows: np.ndarray) -> None:
    """Shift each window's ``g`` so its minimum over visited bins is 0 (in place)."""
    row_min = np.where(visited_windows, g_windows, np.inf).min(axis=1)
    has_v = visited_windows.any(axis=1)
    g_windows[has_v] -= row_min[has_v, None]


def _rewl_trial_step(
    bin_scheme: BinScheme,
    wb: WalkerBatch,
    g_windows: np.ndarray,
    H_windows: np.ndarray,
    visited_windows: np.ndarray,
    b_lo: np.ndarray,
    b_hi: np.ndarray,
    ln_f: float,
    energy_fn: BatchedEnergyFn,
    order_parameter_fn: BatchedOrderParamFn,
    propose_move_fn: BatchedProposeMoveFn,
    beta: float,
) -> np.ndarray:
    """One WL trial for every window's walker at once. Returns the accept mask.

    A thin adapter over :func:`flatwalk.core._grouped_trial_step`: with one
    walker per window, the walker→window map is ``group = arange(W)`` and each
    walker is confined to its window's ``[b_lo, b_hi]``. Same acceptance and
    reflecting-boundary conventions as the shared-``g`` batched step, but each
    walker reads/writes *its own* window's row of ``g_windows`` (a proposal
    leaving the window is rejected, like an out-of-range proposal).
    """
    group = np.arange(wb.n_walkers)
    return _grouped_trial_step(
        bin_scheme,
        wb,
        g_windows,
        H_windows,
        visited_windows,
        group,
        b_lo,
        b_hi,
        ln_f,
        energy_fn,
        order_parameter_fn,
        propose_move_fn,
        beta,
    )


# ---------------------------------------------------------------------------
# Exchange handler
# ---------------------------------------------------------------------------


class ReplicaExchangeHandler:
    """Batched adjacent-window replica exchange.

    Holds the windows' bin bounds and the exchange period; one call swaps
    configurations between adjacent windows for a given parity. All adjacent
    pairs of that parity are evaluated in three numpy ops — no walker loop.

    The configuration (``state``, ``bin_current``, ``energy``) travels with an
    accepted swap; the per-window WL counters stay with their slot.
    """

    def __init__(self, b_lo: np.ndarray, b_hi: np.ndarray, n_exchange: int | None) -> None:
        self.b_lo = np.asarray(b_lo, dtype=np.int64)
        self.b_hi = np.asarray(b_hi, dtype=np.int64)
        self.n_windows = int(self.b_lo.shape[0])
        self.n_exchange = n_exchange
        n_boundaries = max(self.n_windows - 1, 0)
        self.attempts = np.zeros(n_boundaries, dtype=np.int64)
        self.accepts = np.zeros(n_boundaries, dtype=np.int64)

    def exchange(self, wb: WalkerBatch, g_windows: np.ndarray, parity: int) -> None:
        """Attempt swaps across adjacent windows ``(i, i+1)`` of the given parity."""
        W = self.n_windows
        i = np.arange(parity, W - 1, 2)
        if i.size == 0:
            return  # nothing to do (and, crucially, no RNG drawn)
        j = i + 1
        bin_i = wb.bin_current[i]
        bin_j = wb.bin_current[j]

        # Overlap condition: each config must fall inside the other window.
        fits_i_in_j = (bin_i >= self.b_lo[j]) & (bin_i <= self.b_hi[j])
        fits_j_in_i = (bin_j >= self.b_lo[i]) & (bin_j <= self.b_hi[i])
        valid = fits_i_in_j & fits_j_in_i

        delta = (
            g_windows[i, bin_j]
            - g_windows[i, bin_i]
            + g_windows[j, bin_i]
            - g_windows[j, bin_j]
        )
        u = wb.rng.random(i.size)
        accept = valid & ((delta >= 0.0) | (u < np.exp(np.minimum(delta, 0.0))))

        self.attempts[i] += 1
        self.accepts[i] += accept.astype(np.int64)

        ia = i[accept]
        ja = j[accept]
        if ia.size:
            perm = np.arange(W)
            perm[ia] = ja
            perm[ja] = ia
            wb.state = wb.state[perm]
            wb.bin_current = wb.bin_current[perm]
            wb.energy = wb.energy[perm]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class RewlDriver:
    """Replica-exchange Wang-Landau driver over a shared global bin grid.

    Construct with a :class:`WLConfig` (whose ``bin_scheme`` is the *global*
    grid) and a list of ``(low, high)`` order-parameter windows — typically
    from :func:`make_windows`. One walker per window.

    Units: ``t``, ``max_trials``, ``n_check`` (from the config) and
    ``n_exchange`` all count **ticks**, where one tick is a single synchronized
    move per window (``n_windows`` moves total). With ``n_windows == 1`` and a
    full-range window this reduces, tick for tick, to
    :meth:`WLDriver.run_batched` with one walker.

    The f-stage schedule is *synchronized*: ``ln_f`` halves only when every
    window's histogram is flat, so each window is individually flat at every
    reduction. Checkpoint/resume is not implemented here yet; setting
    ``config.checkpoint_path`` raises.
    """

    def __init__(self, config: WLConfig, windows: list[tuple[float, float]]) -> None:
        self.config = config
        self.bin_scheme = config.bin_scheme
        if len(windows) < 1:
            raise ValueError("need at least one window")
        b_lo: list[int] = []
        b_hi: list[int] = []
        for lo, hi in windows:
            if not self.bin_scheme.in_range(lo) or not self.bin_scheme.in_range(hi):
                raise ValueError(f"window bound ({lo}, {hi}) outside the grid domain")
            blo = self.bin_scheme.value_to_index(lo)
            bhi = self.bin_scheme.value_to_index(hi)
            if bhi < blo:
                raise ValueError(f"window ({lo}, {hi}) maps to empty bin range")
            b_lo.append(blo)
            b_hi.append(bhi)
        self.windows = list(windows)
        self.b_lo = np.array(b_lo, dtype=np.int64)
        self.b_hi = np.array(b_hi, dtype=np.int64)
        self.n_windows = len(windows)

    def run(
        self,
        initial_state: Any,
        energy_fn: BatchedEnergyFn,
        order_parameter_fn: BatchedOrderParamFn,
        propose_move_fn: BatchedProposeMoveFn,
        *,
        n_exchange: int | None = None,
        max_trials: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> RewlResult:
        """Run REWL. ``initial_state`` carries one config per window, each with
        its order parameter inside that window's range."""
        cfg = self.config
        if cfg.checkpoint_path is not None:
            raise NotImplementedError(
                "REWL checkpoint/resume is not implemented yet; unset "
                "config.checkpoint_path."
            )
        B = self.bin_scheme.n_bins
        W = self.n_windows
        if rng is None:
            rng = np.random.default_rng()

        # Accepted moves and swaps mutate the batch in place / reorder it, so
        # copy first — never mutate the caller's input.
        state = copy.deepcopy(initial_state)
        q0 = np.asarray(order_parameter_fn(state))
        if q0.shape[0] != W:
            raise ValueError(
                f"order_parameter_fn returned {q0.shape[0]} values; expected one "
                f"per window (n_windows={W})"
            )
        bin0 = self.bin_scheme.value_to_index_batched(q0)
        in_win0 = (bin0 >= self.b_lo) & (bin0 <= self.b_hi)
        if not in_win0.all():
            bad = np.where(~in_win0)[0].tolist()
            raise ValueError(f"initial configs for windows {bad} are outside their range")

        g_windows = np.zeros((W, B), dtype=np.float64)
        H_windows = np.zeros((W, B), dtype=np.int64)
        visited_windows = np.zeros((W, B), dtype=bool)
        if cfg.beta != 0.0:
            energy = np.asarray(energy_fn(state), dtype=np.float64).copy()
        else:
            energy = np.zeros(W, dtype=np.float64)

        wb = WalkerBatch(
            state=state,
            bin_current=bin0.astype(np.int64),
            energy=energy,
            rng=rng,
            n_attempted=np.zeros(W, dtype=np.int64),
            n_accepted=np.zeros(W, dtype=np.int64),
        )
        widx = np.arange(W)
        visited_windows[widx, wb.bin_current] = True

        handler = ReplicaExchangeHandler(self.b_lo, self.b_hi, n_exchange)
        t = 0
        ln_f = cfg.ln_f_initial
        in_1overt = False
        n_f_stages = 0
        parity = 0
        converged = False
        trace_writer = TraceWriter(cfg.trace_path)

        logger.info(
            "REWL run start: n_windows=%d, n_bins=%d, ln_f=%.3g → %.3g, n_exchange=%s",
            W,
            B,
            ln_f,
            cfg.ln_f_final,
            n_exchange,
        )

        with trace_writer:
            while True:
                if ln_f < cfg.ln_f_final:
                    converged = True
                    break
                if max_trials is not None and t >= max_trials:
                    break

                _rewl_trial_step(
                    self.bin_scheme,
                    wb,
                    g_windows,
                    H_windows,
                    visited_windows,
                    self.b_lo,
                    self.b_hi,
                    ln_f,
                    energy_fn,
                    order_parameter_fn,
                    propose_move_fn,
                    cfg.beta,
                )
                t += 1
                if in_1overt:
                    ln_f = 1.0 / t

                if n_exchange is not None and t % n_exchange == 0:
                    handler.exchange(wb, g_windows, parity)
                    parity ^= 1

                if t % cfg.n_check == 0:
                    flat = _flatness_per_window(H_windows, visited_windows)
                    min_flat = float(flat.min())
                    if not in_1overt and min_flat >= cfg.flatness_threshold:
                        new_ln_f, new_in_1overt = attempt_halve(ln_f, t, False)
                        logger.info(
                            "f-stage %d→%d at t=%d: ln_f %.6g → %.6g, min flatness=%.3f",
                            n_f_stages,
                            n_f_stages + 1,
                            t,
                            ln_f,
                            new_ln_f,
                            min_flat,
                        )
                        ln_f = new_ln_f
                        in_1overt = new_in_1overt
                        if not in_1overt:
                            H_windows[:] = 0
                        _normalize_g(g_windows, visited_windows)
                        n_f_stages += 1
                    trace_writer.write(
                        build_trace_row(
                            t=t,
                            ln_f=ln_f,
                            flatness=min_flat,
                            acceptance_rate=wb.acceptance_rate(),
                            H=H_windows.ravel(),
                            visited=visited_windows.ravel(),
                            in_1overt=in_1overt,
                            stage_index=n_f_stages,
                        )
                    )
                    wb.reset_counters()

        logger.info(
            "REWL run end: converged=%s t=%d ln_f=%.6g n_f_stages=%d",
            converged,
            t,
            ln_f,
            n_f_stages,
        )

        return RewlResult(
            g_windows=g_windows,
            H_windows=H_windows,
            visited_windows=visited_windows,
            bin_edges=self.bin_scheme.edges,
            bin_centers=self.bin_scheme.centers,
            windows=self.windows,
            window_bins=np.stack([self.b_lo, self.b_hi], axis=1),
            t_total=t,
            n_f_stages=n_f_stages,
            ln_f_final=ln_f,
            converged=converged,
            final_state=wb.state,
            walker_bins=wb.bin_current,
            walker_energies=wb.energy,
            n_windows=W,
            exchange_attempts=handler.attempts,
            exchange_accepts=handler.accepts,
            rng_state=_capture_rng(wb.rng),
        )


# ---------------------------------------------------------------------------
# Post-processing: join per-window g into one curve
# ---------------------------------------------------------------------------


def join_g(
    g_windows: np.ndarray, visited_windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch per-window log-``g`` into a single curve over the global grid.

    Each window's ``g`` is defined up to an additive constant. Sweeping left to
    right, window ``w`` is shifted by the mean log-difference from the already
    placed window ``w-1`` over their shared visited bins (the least-squares
    constant offset). Bins covered by several windows are averaged.

    Returns ``(g_joined, visited_joined)``; bins visited by no window are
    ``-inf`` in ``g_joined`` and ``False`` in ``visited_joined``.
    """
    W, B = g_windows.shape
    shifted = np.array(g_windows, dtype=np.float64, copy=True)
    for w in range(1, W):
        overlap = visited_windows[w - 1] & visited_windows[w]
        if not overlap.any():
            raise ValueError(
                f"windows {w - 1} and {w} share no visited bins; cannot join "
                "(increase overlap or run longer)"
            )
        c = float(np.mean(shifted[w - 1][overlap] - g_windows[w][overlap]))
        shifted[w] = g_windows[w] + c

    accum = np.zeros(B, dtype=np.float64)
    counts = np.zeros(B, dtype=np.int64)
    for w in range(W):
        v = visited_windows[w]
        accum[v] += shifted[w][v]
        counts[v] += 1
    visited_joined = counts > 0
    g_joined = np.full(B, -np.inf, dtype=np.float64)
    g_joined[visited_joined] = accum[visited_joined] / counts[visited_joined]
    return g_joined, visited_joined
