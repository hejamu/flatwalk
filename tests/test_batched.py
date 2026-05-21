"""Tests for the batched (≥2-walker) path.

Layers mirror `test_core.py`:

1. `WalkerBatch` container mechanics.
2. `WLDriver._trial_step_batched` verified bit-for-bit against the scalar
   `_trial_step` on identical inputs (the spec's "batched primitives are
   themselves verified" check, docs §5).
3. End-to-end `run_batched`: flat-`g` recovery on a uniform-DOS walk,
   out-of-range convention, reproducibility, and bit-identical resume.

End-to-end N=1 batched is deliberately *not* asserted equal to the scalar
`run`: the scalar acceptance short-circuits the RNG draw when Δ ≥ 0 (it never
calls `rng.random()`), whereas the vectorised path always draws one per
walker, so the two streams diverge by construction. The primitive-level test
below pins the equivalence where it is actually well-defined.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flatwalk import Bin1D, Walker, WalkerBatch, WLConfig, WLDriver
from flatwalk.io import load_checkpoint_batched, save_checkpoint_batched

# ---------------------------------------------------------------------------
# A reusable batched bounded random walk: integer state per walker, propose
# ±1, zero energy, order parameter = state. Reflecting boundaries (out-of-range
# proposals are rejected), so the true DOS is uniform → converged g is flat.
# ---------------------------------------------------------------------------


def _batched_walk_system(n_states: int, n_walkers: int, start: int | None = None):
    scheme = Bin1D(-0.5, n_states - 0.5, n_states)
    if start is None:
        start = n_states // 2

    def energy_fn(batch):
        return np.zeros(len(batch))

    def order_parameter_fn(batch):
        return batch.astype(np.float64)

    def propose_move_fn(batch, rng):
        steps = np.where(rng.random(len(batch)) < 0.5, 1, -1)
        return batch + steps, np.zeros(len(batch))

    return {
        "scheme": scheme,
        "initial_state": np.full(n_walkers, start, dtype=np.int64),
        "energy_fn": energy_fn,
        "order_parameter_fn": order_parameter_fn,
        "propose_move_fn": propose_move_fn,
        "n_walkers": n_walkers,
    }


# ---------------------------------------------------------------------------
# 1. WalkerBatch mechanics
# ---------------------------------------------------------------------------


class TestWalkerBatch:
    def _make(self, n=4):
        return WalkerBatch(
            state=np.arange(n),
            bin_current=np.zeros(n, dtype=np.int64),
            energy=np.zeros(n),
            rng=np.random.default_rng(0),
            n_attempted=np.zeros(n, dtype=np.int64),
            n_accepted=np.zeros(n, dtype=np.int64),
        )

    def test_n_walkers(self):
        assert self._make(7).n_walkers == 7

    def test_reset_counters_zeroes_both(self):
        wb = self._make(3)
        wb.n_attempted[:] = [5, 6, 7]
        wb.n_accepted[:] = [1, 2, 3]
        wb.reset_counters()
        np.testing.assert_array_equal(wb.n_attempted, 0)
        np.testing.assert_array_equal(wb.n_accepted, 0)

    def test_acceptance_rate_is_pooled(self):
        wb = self._make(3)
        wb.n_attempted[:] = [10, 10, 10]
        wb.n_accepted[:] = [1, 2, 3]
        assert wb.acceptance_rate() == pytest.approx(6 / 30)

    def test_acceptance_rate_nan_when_no_attempts(self):
        wb = self._make(3)
        assert np.isnan(wb.acceptance_rate())


# ---------------------------------------------------------------------------
# 2. Batched trial step == scalar trial step (bit-for-bit, N=1)
# ---------------------------------------------------------------------------


def test_batched_trial_step_matches_scalar_n1():
    """One batched walker must make the exact same accept decisions and g/H
    updates as the scalar path, given identical inputs and RNG.

    The system is rigged so Δ ≈ -0.7 < 0 on every step: the scalar path then
    *does* draw an acceptance random (no short-circuit), so both paths consume
    exactly one float per step and the shared seed keeps them in lockstep. The
    geometry (increasing g, deterministic +1 move) yields a mix of accepts and
    rejects, exercising both branches.
    """
    n = 200
    scheme = Bin1D(-0.5, n - 0.5, n)
    driver = WLDriver(WLConfig(bin_scheme=scheme))
    g_init = 0.7 * np.arange(n)
    ln_f = 1e-3
    seed = 4242
    K = 40

    # scalar replica
    gs = g_init.copy()
    Hs = np.zeros(n, dtype=np.int64)
    vs = np.zeros(n, dtype=bool)
    ws = Walker(state=0, bin_current=0, energy=0.0, rng=np.random.default_rng(seed))
    vs[0] = True

    def scalar_propose(state, rng):
        return state + 1, 0.0

    # batched replica (N=1)
    gb = g_init.copy()
    Hb = np.zeros(n, dtype=np.int64)
    vb = np.zeros(n, dtype=bool)
    wb = WalkerBatch(
        state=np.array([0], dtype=np.int64),
        bin_current=np.array([0], dtype=np.int64),
        energy=np.zeros(1),
        rng=np.random.default_rng(seed),
        n_attempted=np.zeros(1, dtype=np.int64),
        n_accepted=np.zeros(1, dtype=np.int64),
    )
    vb[0] = True

    def batched_propose(batch, rng):
        return batch + 1, np.zeros(len(batch))

    n_accept = 0
    for _ in range(K):
        a_s = driver._trial_step(
            ws,
            gs,
            Hs,
            vs,
            ln_f,
            energy_fn=lambda s: 0.0,
            order_parameter_fn=lambda s: float(s),
            propose_move_fn=scalar_propose,
            beta=0.0,
        )
        a_b = driver._trial_step_batched(
            wb,
            gb,
            Hb,
            vb,
            ln_f,
            energy_fn=lambda b: np.zeros(len(b)),
            order_parameter_fn=lambda b: b.astype(np.float64),
            propose_move_fn=batched_propose,
            beta=0.0,
        )
        assert bool(a_b[0]) == a_s
        assert int(wb.bin_current[0]) == ws.bin_current
        np.testing.assert_array_equal(gb, gs)
        np.testing.assert_array_equal(Hb, Hs)
        np.testing.assert_array_equal(vb, vs)
        n_accept += int(a_s)

    assert 0 < n_accept < K  # both branches exercised


def test_batched_trial_step_out_of_range_rejects_and_updates_current_bin():
    """A boundary walker proposing out of range stays put; g/H update at the
    current bin (the reflecting-boundary convention, per walker)."""
    n = 5
    scheme = Bin1D(-0.5, n - 0.5, n)
    driver = WLDriver(WLConfig(bin_scheme=scheme))
    g = np.zeros(n)
    H = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    wb = WalkerBatch(
        state=np.array([0], dtype=np.int64),
        bin_current=np.array([0], dtype=np.int64),
        energy=np.zeros(1),
        rng=np.random.default_rng(0),
        n_attempted=np.zeros(1, dtype=np.int64),
        n_accepted=np.zeros(1, dtype=np.int64),
    )
    visited[0] = True

    def propose_down(batch, rng):  # state 0 → -1, out of range
        return batch - 1, np.zeros(len(batch))

    accept = driver._trial_step_batched(
        wb,
        g,
        H,
        visited,
        0.5,
        energy_fn=lambda b: np.zeros(len(b)),
        order_parameter_fn=lambda b: b.astype(np.float64),
        propose_move_fn=propose_down,
        beta=0.0,
    )
    assert not accept[0]
    assert int(wb.bin_current[0]) == 0
    assert int(wb.state[0]) == 0
    assert g[0] == pytest.approx(0.5)
    assert H[0] == 1


def test_batched_shared_g_scatter_accumulates_all_walkers():
    """When several walkers share a bin, the scatter add must count each one
    (np.add.at semantics), so a tick deposits exactly n_walkers of ln_f/H."""
    n = 5
    scheme = Bin1D(-0.5, n - 0.5, n)
    driver = WLDriver(WLConfig(bin_scheme=scheme))
    g = np.zeros(n)
    H = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=bool)
    n_walkers = 6
    wb = WalkerBatch(
        state=np.full(n_walkers, 2, dtype=np.int64),
        bin_current=np.full(n_walkers, 2, dtype=np.int64),
        energy=np.zeros(n_walkers),
        rng=np.random.default_rng(0),
        n_attempted=np.zeros(n_walkers, dtype=np.int64),
        n_accepted=np.zeros(n_walkers, dtype=np.int64),
    )
    visited[2] = True

    def stay(batch, rng):  # propose out of range so all stay in bin 2
        return batch + 100, np.zeros(len(batch))

    driver._trial_step_batched(
        wb,
        g,
        H,
        visited,
        0.5,
        energy_fn=lambda b: np.zeros(len(b)),
        order_parameter_fn=lambda b: b.astype(np.float64),
        propose_move_fn=stay,
        beta=0.0,
    )
    assert H[2] == n_walkers
    assert g[2] == pytest.approx(n_walkers * 0.5)
    assert H.sum() == n_walkers


# ---------------------------------------------------------------------------
# 3. End-to-end run_batched
# ---------------------------------------------------------------------------


class TestRunBatched:
    def test_flat_g_on_uniform_dos(self):
        """A reflecting random walk has uniform DOS; converged g is flat."""
        sys = _batched_walk_system(n_states=10, n_walkers=8)
        cfg = WLConfig(
            bin_scheme=sys["scheme"],
            n_check=2000,
            ln_f_initial=1.0,
            ln_f_final=1e-5,
            flatness_threshold=0.8,
        )
        result = WLDriver(cfg).run_batched(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            n_walkers=sys["n_walkers"],
            rng=np.random.default_rng(0),
            max_trials=5_000_000,
        )
        assert result.converged
        assert result.visited.all()
        g = result.g[result.visited]
        # log-DOS should be flat to well within an order unity over 10 bins.
        assert g.max() - g.min() < 1.0

    def test_reproducible(self):
        sys = _batched_walk_system(n_states=8, n_walkers=4)
        kw = dict(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            n_walkers=sys["n_walkers"],
            max_trials=20_000,
        )
        cfg = WLConfig(bin_scheme=sys["scheme"], n_check=500, ln_f_final=1e-30)
        before = sys["initial_state"].copy()
        r1 = WLDriver(cfg).run_batched(rng=np.random.default_rng(7), **kw)
        # The driver must not mutate the caller's initial_state in place.
        np.testing.assert_array_equal(sys["initial_state"], before)
        r2 = WLDriver(cfg).run_batched(rng=np.random.default_rng(7), **kw)
        np.testing.assert_array_equal(r1.g, r2.g)
        np.testing.assert_array_equal(r1.H, r2.H)
        np.testing.assert_array_equal(r1.final_state, r2.final_state)
        assert r1.t_total == r2.t_total
        assert r1.n_f_stages == r2.n_f_stages

    def test_t_total_counts_moves(self):
        sys = _batched_walk_system(n_states=8, n_walkers=4)
        cfg = WLConfig(bin_scheme=sys["scheme"], n_check=500, ln_f_final=1e-30)
        result = WLDriver(cfg).run_batched(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            n_walkers=4,
            rng=np.random.default_rng(0),
            max_trials=4_000,
        )
        # t advances n_walkers per tick and stops on reaching max_trials.
        assert result.t_total == 4_000
        assert result.n_walkers == 4
        assert result.walker_bins.shape == (4,)

    def test_initial_states_out_of_range_raise(self):
        sys = _batched_walk_system(n_states=5, n_walkers=3)
        cfg = WLConfig(bin_scheme=sys["scheme"])
        with pytest.raises(ValueError, match="outside the bin domain"):
            WLDriver(cfg).run_batched(
                initial_state=np.array([2, 999, 1]),  # second walker out of range
                energy_fn=sys["energy_fn"],
                order_parameter_fn=sys["order_parameter_fn"],
                propose_move_fn=sys["propose_move_fn"],
                n_walkers=3,
                max_trials=10,
            )

    def test_exchange_handler_not_implemented(self):
        sys = _batched_walk_system(n_states=5, n_walkers=2)
        cfg = WLConfig(bin_scheme=sys["scheme"])

        class _Dummy:
            n_exchange = 10

            def maybe_exchange(self, walker, g):  # pragma: no cover - never called
                return None

        with pytest.raises(NotImplementedError, match="replica exchange"):
            WLDriver(cfg).run_batched(
                initial_state=sys["initial_state"],
                energy_fn=sys["energy_fn"],
                order_parameter_fn=sys["order_parameter_fn"],
                propose_move_fn=sys["propose_move_fn"],
                n_walkers=2,
                exchange_handler=_Dummy(),
            )


# ---------------------------------------------------------------------------
# 4. Batched checkpoint I/O
# ---------------------------------------------------------------------------


def test_batched_checkpoint_round_trip(tmp_path: Path):
    path = tmp_path / "cp.npz"
    g = np.linspace(0.0, 1.0, 6)
    H = np.arange(6, dtype=np.int64)
    visited = H > 0
    edges = np.linspace(-0.5, 5.5, 7)
    centers = 0.5 * (edges[:-1] + edges[1:])
    state = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)  # 3 walkers
    rng = np.random.default_rng(11)
    rng.standard_normal(5)
    rng_state = rng.bit_generator.state

    save_checkpoint_batched(
        path,
        g=g,
        H=H,
        visited=visited,
        bin_edges=edges,
        bin_centers=centers,
        bin_current=np.array([0, 2, 4], dtype=np.int64),
        energy=np.array([-1.0, -2.0, -3.0]),
        n_attempted=np.array([10, 11, 12], dtype=np.int64),
        n_accepted=np.array([1, 2, 3], dtype=np.int64),
        n_bins=6,
        t_total=999,
        n_f_stages=2,
        ln_f=0.125,
        in_1overt=True,
        n_walkers=3,
        walker_state=state,
        rng_state=rng_state,
    )

    loaded = load_checkpoint_batched(path)
    np.testing.assert_array_equal(loaded["g"], g)
    np.testing.assert_array_equal(loaded["bin_current"], [0, 2, 4])
    np.testing.assert_array_equal(loaded["energy"], [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(loaded["n_attempted"], [10, 11, 12])
    assert loaded["n_walkers"] == 3
    assert loaded["t_total"] == 999
    assert loaded["in_1overt"] is True
    np.testing.assert_array_equal(loaded["walker_state"], state)
    assert loaded["walker_state"].dtype == state.dtype
    assert loaded["rng_state"] == rng_state


def test_batched_resume_is_bit_identical(tmp_path: Path):
    """Run A uninterrupted; run B checkpoints at the halfway move count, then
    resumes. Final g, H, visited, and walker states must match A exactly."""
    n_walkers = 4
    N = 4000
    N_half = N // 2  # multiple of n_walkers → checkpoint lands on a tick
    seed = 9999

    def fresh_system():
        return _batched_walk_system(n_states=6, n_walkers=n_walkers)

    common = dict(
        n_check=200,
        ln_f_initial=1.0,
        ln_f_final=1e-30,  # never converge by ln_f
        flatness_threshold=0.01,
    )

    # Reference run.
    sys_a = fresh_system()
    result_a = WLDriver(WLConfig(bin_scheme=sys_a["scheme"], **common)).run_batched(
        initial_state=sys_a["initial_state"],
        energy_fn=sys_a["energy_fn"],
        order_parameter_fn=sys_a["order_parameter_fn"],
        propose_move_fn=sys_a["propose_move_fn"],
        n_walkers=n_walkers,
        rng=np.random.default_rng(seed),
        max_trials=N,
    )
    assert result_a.t_total == N

    # Run B part 1: checkpoint at N_half moves.
    cp_path = tmp_path / "cp.npz"
    sys_b = fresh_system()
    result_b1 = WLDriver(
        WLConfig(
            bin_scheme=sys_b["scheme"],
            checkpoint_path=cp_path,
            checkpoint_every_t=N_half,
            **common,
        )
    ).run_batched(
        initial_state=sys_b["initial_state"],
        energy_fn=sys_b["energy_fn"],
        order_parameter_fn=sys_b["order_parameter_fn"],
        propose_move_fn=sys_b["propose_move_fn"],
        n_walkers=n_walkers,
        rng=np.random.default_rng(seed),
        max_trials=N_half,
    )
    assert result_b1.t_total == N_half
    assert cp_path.exists()

    # Run B part 2: resume.
    sys_b2 = fresh_system()
    result_b2 = WLDriver(WLConfig(bin_scheme=sys_b2["scheme"], **common)).run_batched(
        initial_state=None,  # ignored on resume
        energy_fn=sys_b2["energy_fn"],
        order_parameter_fn=sys_b2["order_parameter_fn"],
        propose_move_fn=sys_b2["propose_move_fn"],
        n_walkers=n_walkers,
        resume_from=cp_path,
        max_trials=N,
    )
    assert result_b2.t_total == N

    np.testing.assert_array_equal(result_a.g, result_b2.g)
    np.testing.assert_array_equal(result_a.H, result_b2.H)
    np.testing.assert_array_equal(result_a.visited, result_b2.visited)
    np.testing.assert_array_equal(result_a.final_state, result_b2.final_state)
    np.testing.assert_array_equal(result_a.walker_bins, result_b2.walker_bins)
    assert result_a.ln_f_final == result_b2.ln_f_final
    assert result_a.n_f_stages == result_b2.n_f_stages
    assert result_a.in_1overt == result_b2.in_1overt


def test_batched_resume_rejects_mismatched_n_walkers(tmp_path: Path):
    sys = _batched_walk_system(n_states=5, n_walkers=3)
    cp_path = tmp_path / "cp.npz"
    cfg = WLConfig(bin_scheme=sys["scheme"], checkpoint_path=cp_path, checkpoint_every_t=30)
    WLDriver(cfg).run_batched(
        initial_state=sys["initial_state"],
        energy_fn=sys["energy_fn"],
        order_parameter_fn=sys["order_parameter_fn"],
        propose_move_fn=sys["propose_move_fn"],
        n_walkers=3,
        rng=np.random.default_rng(0),
        max_trials=30,
    )
    with pytest.raises(ValueError, match="n_walkers"):
        WLDriver(WLConfig(bin_scheme=sys["scheme"])).run_batched(
            initial_state=None,
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            n_walkers=5,  # mismatch
            resume_from=cp_path,
            max_trials=60,
        )
