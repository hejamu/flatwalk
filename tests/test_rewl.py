"""Tests for replica-exchange Wang-Landau (`flatwalk.rewl`).

Layers:

1. Window construction (`make_windows`) and the log-`g` join (`join_g`).
2. Vectorised primitives: per-window flatness, the window-confined trial step,
   and the adjacent-window exchange (favorable/non-overlap/no-op cases).
3. End-to-end `RewlDriver`: flat-`g` recovery on a uniform-DOS walk, a
   bit-for-bit reduction to `run_batched` at one full-range window, plus
   reproducibility and an Ising L=4 smoke against Beale's exact n(E).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from flatwalk import (
    Bin1D,
    ReplicaExchangeHandler,
    RewlDriver,
    WalkerBatch,
    WLConfig,
    WLDriver,
    join_g,
    make_windows,
)
from flatwalk.core import compute_flatness
from flatwalk.rewl import _flatness_per_window, _rewl_trial_step

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


# ---------------------------------------------------------------------------
# Synthetic batched bounded random walk (state value == bin index)
# ---------------------------------------------------------------------------


def _walk_callbacks():
    def energy_fn(batch):
        return np.zeros(len(batch))

    def order_parameter_fn(batch):
        return batch.astype(np.float64)

    def propose_move_fn(batch, rng):
        steps = np.where(rng.random(len(batch)) < 0.5, 1, -1)
        return batch + steps, np.zeros(len(batch))

    return energy_fn, order_parameter_fn, propose_move_fn


# ---------------------------------------------------------------------------
# 1. Window construction and join
# ---------------------------------------------------------------------------


class TestMakeWindows:
    @pytest.mark.parametrize(
        "n_bins, n_windows, overlap",
        [(65, 4, 8), (30, 3, 3), (17, 3, 4), (100, 5, 2), (20, 2, 5)],
    )
    def test_coverage_and_overlap(self, n_bins, n_windows, overlap):
        scheme = Bin1D(-0.5, n_bins - 0.5, n_bins)
        windows = make_windows(scheme, n_windows, overlap)
        assert len(windows) == n_windows
        bins = [
            (scheme.value_to_index(lo), scheme.value_to_index(hi)) for lo, hi in windows
        ]
        # First starts at bin 0, last ends at the top bin → full coverage.
        assert bins[0][0] == 0
        assert bins[-1][1] == n_bins - 1
        # Every adjacent pair shares at least one bin.
        for k in range(1, n_windows):
            assert bins[k][0] <= bins[k - 1][1]
        # Union covers every bin.
        covered = np.zeros(n_bins, dtype=bool)
        for lo, hi in bins:
            covered[lo : hi + 1] = True
        assert covered.all()
        # Windows are balanced: no end window is left far narrower than the
        # others (a narrow tail window badly under-resolves a steep DOS there).
        widths = [hi - lo + 1 for lo, hi in bins]
        assert min(widths) >= 0.5 * max(widths)

    def test_single_window_is_full_range(self):
        scheme = Bin1D(0.0, 10.0, 20)
        (w,) = make_windows(scheme, 1, overlap=1)
        assert scheme.value_to_index(w[0]) == 0
        assert scheme.value_to_index(w[1]) == 19

    def test_zero_overlap_rejected(self):
        scheme = Bin1D(0.0, 1.0, 20)
        with pytest.raises(ValueError, match="overlap"):
            make_windows(scheme, 3, overlap=0)


def test_join_g_recovers_known_g_up_to_constant():
    B = 30
    rng = np.random.default_rng(0)
    g_true = rng.normal(size=B).cumsum()
    bounds = [(0, 13), (10, 22), (19, 29)]
    W = len(bounds)
    g_windows = np.zeros((W, B))
    visited = np.zeros((W, B), dtype=bool)
    offsets = [5.0, -3.0, 10.0]  # each window known only up to its own constant
    for w, (lo, hi) in enumerate(bounds):
        sl = slice(lo, hi + 1)
        g_windows[w, sl] = g_true[sl] + offsets[w]
        visited[w, sl] = True
    joined, vj = join_g(g_windows, visited)
    assert vj.all()
    diff = joined[vj] - g_true[vj]
    assert diff.std() < 1e-9  # the windows differ from the truth by a pure constant


def test_join_g_raises_without_overlap():
    B = 10
    g_windows = np.zeros((2, B))
    visited = np.zeros((2, B), dtype=bool)
    visited[0, 0:4] = True
    visited[1, 6:10] = True  # disjoint → no shared visited bin
    with pytest.raises(ValueError, match="no visited"):
        join_g(g_windows, visited)


# ---------------------------------------------------------------------------
# 2. Vectorised primitives
# ---------------------------------------------------------------------------


def test_flatness_per_window_matches_scalar():
    rng = np.random.default_rng(1)
    W, B = 5, 12
    H = rng.integers(0, 50, size=(W, B)).astype(np.int64)
    visited = rng.random((W, B)) < 0.7
    # Force an all-unvisited row and an all-zero visited row (degenerate cases).
    visited[0] = False
    H[1] = 0
    visited[1] = True
    flat = _flatness_per_window(H, visited)
    expected = np.array([compute_flatness(H[w], visited[w]) for w in range(W)])
    np.testing.assert_allclose(flat, expected)


def test_rewl_trial_step_confines_walker_to_its_window():
    scheme = Bin1D(-0.5, 9.5, 10)
    b_lo = np.array([2])
    b_hi = np.array([5])
    g = np.zeros((1, 10))
    H = np.zeros((1, 10), dtype=np.int64)
    v = np.zeros((1, 10), dtype=bool)
    wb = WalkerBatch(
        state=np.array([5]),
        bin_current=np.array([5]),
        energy=np.zeros(1),
        rng=np.random.default_rng(0),
        n_attempted=np.zeros(1, dtype=np.int64),
        n_accepted=np.zeros(1, dtype=np.int64),
    )
    v[0, 5] = True

    def propose(batch, rng):  # 5 → 6, just outside the window [2, 5]
        return batch + 1, np.zeros(len(batch))

    accept = _rewl_trial_step(
        scheme,
        wb,
        g,
        H,
        v,
        b_lo,
        b_hi,
        0.5,
        energy_fn=lambda b: np.zeros(len(b)),
        order_parameter_fn=lambda b: b.astype(np.float64),
        propose_move_fn=propose,
        beta=0.0,
    )
    assert not accept[0]
    assert wb.bin_current[0] == 5
    assert g[0, 5] == pytest.approx(0.5)
    assert H[0, 5] == 1


class TestExchange:
    def _two_window_setup(self):
        # B=8; window 0 covers bins [0,4], window 1 covers [3,7] (overlap 3,4).
        return ReplicaExchangeHandler(np.array([0, 3]), np.array([4, 7]), n_exchange=1)

    def test_favorable_overlapping_swap_is_accepted(self):
        handler = self._two_window_setup()
        g = np.zeros((2, 8))
        # walker0 at bin 4 (also inside window1), walker1 at bin 3 (also in window0).
        # Δ = g0[3]-g0[4] + g1[4]-g1[3]; make it strongly positive → always accept.
        g[0, 3] = 10.0
        g[1, 4] = 10.0
        wb = WalkerBatch(
            state=np.array([100, 200]),
            bin_current=np.array([4, 3]),
            energy=np.array([1.0, 2.0]),
            rng=np.random.default_rng(0),
            n_attempted=np.array([7, 8]),
            n_accepted=np.array([1, 2]),
        )
        handler.exchange(wb, g, parity=0)
        # Config (state/bin/energy) travels with the swap; counters stay put.
        np.testing.assert_array_equal(wb.state, [200, 100])
        np.testing.assert_array_equal(wb.bin_current, [3, 4])
        np.testing.assert_array_equal(wb.energy, [2.0, 1.0])
        np.testing.assert_array_equal(wb.n_attempted, [7, 8])
        assert handler.accepts[0] == 1
        assert handler.attempts[0] == 1

    def test_non_overlapping_pair_is_rejected(self):
        handler = self._two_window_setup()
        g = np.zeros((2, 8))
        g[0, 7] = 10.0  # would-be favorable, but configs don't fit the other window
        g[1, 0] = 10.0
        wb = WalkerBatch(
            state=np.array([100, 200]),
            bin_current=np.array([0, 7]),  # 0 ∉ window1[3,7]; 7 ∉ window0[0,4]
            energy=np.array([1.0, 2.0]),
            rng=np.random.default_rng(0),
            n_attempted=np.array([0, 0]),
            n_accepted=np.array([0, 0]),
        )
        handler.exchange(wb, g, parity=0)
        np.testing.assert_array_equal(wb.state, [100, 200])  # unchanged
        np.testing.assert_array_equal(wb.bin_current, [0, 7])
        assert handler.accepts[0] == 0
        assert handler.attempts[0] == 1

    def test_single_window_is_noop_and_draws_no_rng(self):
        handler = ReplicaExchangeHandler(np.array([0]), np.array([7]), n_exchange=1)
        rng = np.random.default_rng(0)
        state_before = rng.bit_generator.state
        wb = WalkerBatch(
            state=np.array([1]),
            bin_current=np.array([0]),
            energy=np.zeros(1),
            rng=rng,
            n_attempted=np.zeros(1, dtype=np.int64),
            n_accepted=np.zeros(1, dtype=np.int64),
        )
        handler.exchange(wb, np.zeros((1, 8)), parity=0)
        np.testing.assert_array_equal(wb.state, [1])
        assert rng.bit_generator.state == state_before  # no draw consumed


# ---------------------------------------------------------------------------
# 3. End-to-end RewlDriver
# ---------------------------------------------------------------------------


class TestRunRewl:
    def _windows_and_init(self, scheme, n_windows, overlap):
        windows = make_windows(scheme, n_windows, overlap)
        driver = RewlDriver(WLConfig(bin_scheme=scheme), windows)
        mids = ((driver.b_lo + driver.b_hi) // 2).astype(np.int64)
        return windows, mids

    def test_flat_g_after_join_on_uniform_dos(self):
        n = 12
        scheme = Bin1D(-0.5, n - 0.5, n)
        windows, mids = self._windows_and_init(scheme, n_windows=3, overlap=3)
        cfg = WLConfig(
            bin_scheme=scheme,
            n_check=1000,
            ln_f_initial=1.0,
            ln_f_final=1e-4,
            flatness_threshold=0.8,
        )
        energy_fn, op_fn, propose = _walk_callbacks()
        result = RewlDriver(cfg, windows).run(
            initial_state=mids.copy(),
            energy_fn=energy_fn,
            order_parameter_fn=op_fn,
            propose_move_fn=propose,
            n_exchange=100,
            rng=np.random.default_rng(0),
            max_trials=2_000_000,
        )
        assert result.converged
        assert result.exchange_accepts.sum() > 0  # exchange actually fired
        joined, vj = join_g(result.g_windows, result.visited_windows)
        assert vj.all()  # every bin covered by some window
        assert joined[vj].max() - joined[vj].min() < 1.0  # flat log-DOS

    def test_reduces_to_run_batched_at_one_full_window(self):
        """REWL with a single full-range window and N=1 must match
        run_batched(n_walkers=1) bit-for-bit: same RNG order, same math, and
        the exchange step is a no-op that draws nothing."""
        n = 10
        scheme = Bin1D(-0.5, n - 0.5, n)
        common = dict(
            n_check=200, ln_f_initial=1.0, ln_f_final=1e-30, flatness_threshold=0.01
        )
        energy_fn, op_fn, propose = _walk_callbacks()
        start = np.array([5], dtype=np.int64)
        T = 5000
        seed = 12345

        batched = WLDriver(WLConfig(bin_scheme=scheme, **common)).run_batched(
            initial_state=start.copy(),
            energy_fn=energy_fn,
            order_parameter_fn=op_fn,
            propose_move_fn=propose,
            n_walkers=1,
            rng=np.random.default_rng(seed),
            max_trials=T,
        )

        windows = make_windows(scheme, 1, overlap=1)
        rewl = RewlDriver(WLConfig(bin_scheme=scheme, **common), windows).run(
            initial_state=start.copy(),
            energy_fn=energy_fn,
            order_parameter_fn=op_fn,
            propose_move_fn=propose,
            n_exchange=50,  # set, but one window has no pair → no-op, no draw
            rng=np.random.default_rng(seed),
            max_trials=T,
        )

        np.testing.assert_array_equal(rewl.g_windows[0], batched.g)
        np.testing.assert_array_equal(rewl.H_windows[0], batched.H)
        np.testing.assert_array_equal(rewl.walker_bins, batched.walker_bins)
        np.testing.assert_array_equal(rewl.final_state, batched.final_state)
        assert rewl.t_total == batched.t_total
        assert rewl.n_f_stages == batched.n_f_stages
        assert rewl.ln_f_final == batched.ln_f_final

    def test_reproducible_and_does_not_mutate_initial_state(self):
        n = 12
        scheme = Bin1D(-0.5, n - 0.5, n)
        windows, mids = self._windows_and_init(scheme, n_windows=3, overlap=3)
        cfg = WLConfig(bin_scheme=scheme, n_check=500, ln_f_final=1e-30)
        energy_fn, op_fn, propose = _walk_callbacks()
        kw = dict(
            energy_fn=energy_fn,
            order_parameter_fn=op_fn,
            propose_move_fn=propose,
            n_exchange=50,
            max_trials=20_000,
        )
        before = mids.copy()
        r1 = RewlDriver(cfg, windows).run(
            initial_state=mids, rng=np.random.default_rng(7), **kw
        )
        np.testing.assert_array_equal(mids, before)  # input untouched
        r2 = RewlDriver(cfg, windows).run(
            initial_state=mids, rng=np.random.default_rng(7), **kw
        )
        np.testing.assert_array_equal(r1.g_windows, r2.g_windows)
        np.testing.assert_array_equal(r1.H_windows, r2.H_windows)
        np.testing.assert_array_equal(r1.exchange_accepts, r2.exchange_accepts)

    def test_checkpoint_path_raises_not_implemented(self, tmp_path):
        n = 8
        scheme = Bin1D(-0.5, n - 0.5, n)
        windows, mids = self._windows_and_init(scheme, n_windows=2, overlap=2)
        cfg = WLConfig(bin_scheme=scheme, checkpoint_path=tmp_path / "cp.npz")
        energy_fn, op_fn, propose = _walk_callbacks()
        with pytest.raises(NotImplementedError, match="checkpoint"):
            RewlDriver(cfg, windows).run(
                initial_state=mids,
                energy_fn=energy_fn,
                order_parameter_fn=op_fn,
                propose_move_fn=propose,
                n_exchange=10,
                max_trials=10,
            )

    def test_initial_config_outside_window_raises(self):
        n = 10
        scheme = Bin1D(-0.5, n - 0.5, n)
        windows = make_windows(scheme, 2, overlap=2)
        driver = RewlDriver(WLConfig(bin_scheme=scheme), windows)
        energy_fn, op_fn, propose = _walk_callbacks()
        # Put window-1's walker at bin 0, which is outside its range.
        bad = np.array([driver.b_lo[0], 0], dtype=np.int64)
        with pytest.raises(ValueError, match="outside their range"):
            driver.run(
                initial_state=bad,
                energy_fn=energy_fn,
                order_parameter_fn=op_fn,
                propose_move_fn=propose,
                n_exchange=10,
                max_trials=10,
            )


def test_rewl_ising_l4_smoke():
    """End-to-end REWL on L=4 Ising: build windows, run short, join, and check
    the joined n(E) is in the right ballpark vs Beale's exact result. Loose by
    design — the strict L=8 criteria live in examples/ising_rewl_validation.py."""
    import beale  # noqa: E402  (examples on sys.path)
    import ising  # noqa: E402
    import ising_batched  # noqa: E402

    L = 4
    low, high, n_bins = ising.ising_energy_bins(L)
    scheme = Bin1D(low, high, n_bins)
    windows = make_windows(scheme, n_windows=3, overlap=4)

    cb = ising_batched.make_batched_ising_callbacks(L)
    rng = np.random.default_rng(0)
    init = ising_batched.initial_states_for_windows(L, windows, rng)

    cfg = WLConfig(
        bin_scheme=scheme,
        beta=0.0,
        flatness_threshold=0.8,
        n_check=2000,
        ln_f_initial=1.0,
        ln_f_final=1e-3,
    )
    result = RewlDriver(cfg, windows).run(
        initial_state=init,
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        n_exchange=100,
        rng=rng,
        max_trials=5_000_000,
    )
    joined, vj = join_g(result.g_windows, result.visited_windows)

    g_exact = beale.beale_g_E(L)
    log_g_exact = beale.log_g_E_array(L, g_exact, scheme.centers)
    n_exact = np.exp(log_g_exact)
    valid = vj & np.isfinite(log_g_exact) & (n_exact > 0)
    assert valid.sum() >= 5

    shifted = joined[valid] - joined[valid].max()
    n_wl = np.exp(shifted)
    n_wl *= n_exact[valid].sum() / n_wl.sum()
    eps = np.abs(n_wl - n_exact[valid]) / n_exact[valid]
    central = np.ones_like(eps, dtype=bool)
    central[0] = False
    central[-1] = False
    assert np.isfinite(eps).all()
    assert eps[central].mean() < 0.6  # loose smoke threshold
