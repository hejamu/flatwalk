"""Tests for the Wang-Landau core (`flatwalk.core`).

Three layers of test:

1. Pure helpers (`compute_flatness`, `attempt_halve`) — math in isolation.
2. Per-trial step (`WLDriver._trial_step`) — out-of-range convention,
   g/H update on current bin, accept/reject mechanics.
3. End-to-end `WLDriver.run` — f-stage halving, 1/t-regime entry,
   reproducibility, convergence/stopping, max_trials, visited persistence.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from flatwalk import Bin1D, Walker, WLConfig, WLDriver
from flatwalk.core import attempt_halve, compute_flatness
from flatwalk.diagnostics import read_trace


# ---------------------------------------------------------------------------
# Reusable "tiny random walk" system: integer state ∈ {0..n-1}, propose ±1,
# zero energy, order parameter = state. Out-of-range proposals exercise the
# spec §1.3 reflecting-boundary convention.
# ---------------------------------------------------------------------------

def _tiny_system(n_states: int = 10):
    scheme = Bin1D(-0.5, n_states - 0.5, n_states)

    def propose(state, rng):
        step = 1 if rng.random() < 0.5 else -1
        return state + step, 0.0

    return {
        "scheme": scheme,
        "initial_state": n_states // 2,
        "energy_fn": lambda s: 0.0,
        "order_parameter_fn": lambda s: float(s),
        "propose_move_fn": propose,
    }


# ---------------------------------------------------------------------------
# 1. Pure helpers
# ---------------------------------------------------------------------------

class TestComputeFlatness:
    def test_no_visited_returns_zero(self):
        H = np.zeros(5, dtype=np.int64)
        v = np.zeros(5, dtype=bool)
        assert compute_flatness(H, v) == 0.0

    def test_uniform_histogram(self):
        H = np.full(5, 100, dtype=np.int64)
        v = np.ones(5, dtype=bool)
        assert compute_flatness(H, v) == pytest.approx(1.0)

    def test_visited_subset(self):
        # only bins 1,2,3 visited; bin 4 has H=0 but unvisited → excluded
        H = np.array([0, 90, 100, 110, 0], dtype=np.int64)
        v = np.array([False, True, True, True, False])
        # min/mean = 90/100 = 0.9
        assert compute_flatness(H, v) == pytest.approx(0.9)

    def test_visited_bin_with_zero_count_kills_flatness(self):
        # A visited-but-zeroed bin (just reset by halving, never re-entered)
        # should drag the min down to zero.
        H = np.array([0, 100, 100], dtype=np.int64)
        v = np.array([True, True, True])
        assert compute_flatness(H, v) == 0.0


class TestAttemptHalve:
    def test_standard_halve_stays_in_standard(self):
        new_ln_f, in_1overt = attempt_halve(ln_f=1.0, t=10, in_1overt=False)
        assert new_ln_f == pytest.approx(0.5)
        assert in_1overt is False

    def test_halve_below_1_over_t_triggers_switch(self):
        # halved = 0.001 / 2 = 5e-4; 1/t = 1/100 = 0.01; 5e-4 < 0.01 → switch
        new_ln_f, in_1overt = attempt_halve(ln_f=1e-3, t=100, in_1overt=False)
        assert in_1overt is True
        assert new_ln_f == pytest.approx(1.0 / 100)

    def test_exact_transition_boundary(self):
        """At t such that ln_f/2 is *exactly* the threshold below 1/t.

        For t=8, 1/t = 0.125. ln_f=0.4 → halved=0.2, which is > 0.125, no
        switch. ln_f=0.2 → halved=0.1, which is < 0.125, switch.
        """
        no_switch_ln_f, no_switch = attempt_halve(0.4, 8, False)
        assert no_switch is False
        assert no_switch_ln_f == pytest.approx(0.2)

        switch_ln_f, switch = attempt_halve(0.2, 8, False)
        assert switch is True
        assert switch_ln_f == pytest.approx(0.125)

    def test_1overt_passthrough(self):
        ln_f, in_1overt = attempt_halve(ln_f=999.0, t=1000, in_1overt=True)
        assert in_1overt is True
        assert ln_f == pytest.approx(1.0 / 1000)


# ---------------------------------------------------------------------------
# 2. _trial_step mechanics
# ---------------------------------------------------------------------------

class TestTrialStep:
    def _setup(self, propose):
        sys = _tiny_system(5)
        sys["propose_move_fn"] = propose
        scheme = sys["scheme"]
        driver = WLDriver(WLConfig(bin_scheme=scheme))
        g = np.zeros(scheme.n_bins, dtype=np.float64)
        H = np.zeros(scheme.n_bins, dtype=np.int64)
        visited = np.zeros(scheme.n_bins, dtype=bool)
        walker = Walker(state=2, bin_current=2, energy=0.0,
                        rng=np.random.default_rng(7))
        visited[walker.bin_current] = True
        return driver, sys, walker, g, H, visited

    def test_out_of_range_proposal_rejects_and_updates_current_bin(self):
        # Force a downward step from state 0 → state -1, out of range.
        def propose(state, rng):
            return state - 1, 0.0

        driver, sys, walker, g, H, visited = self._setup(propose)
        # Move walker to state 0 (bin 0) first.
        walker.state = 0
        walker.bin_current = 0
        visited[0] = True

        accepted = driver._trial_step(
            walker, g, H, visited, ln_f=0.5,
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=propose,
            beta=0.0,
        )
        assert accepted is False
        # State and bin unchanged
        assert walker.state == 0
        assert walker.bin_current == 0
        # g and H updated at *current* bin (reflecting convention, spec §1.3)
        assert g[0] == pytest.approx(0.5)
        assert H[0] == 1
        assert visited[0] is True or bool(visited[0])
        # Counters
        assert walker.n_attempted == 1
        assert walker.n_accepted == 0

    def test_in_range_accept_when_g_old_geq_g_new(self):
        # Propose state+1 deterministically. g[old]=10, g[new]=0 → Δ=+10, accept.
        def propose(state, rng):
            return state + 1, 0.0

        driver, sys, walker, g, H, visited = self._setup(propose)
        g[walker.bin_current] = 10.0
        g[walker.bin_current + 1] = 0.0

        accepted = driver._trial_step(
            walker, g, H, visited, ln_f=0.1,
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=propose,
            beta=0.0,
        )
        assert accepted is True
        assert walker.state == 3
        assert walker.bin_current == 3
        assert g[3] == pytest.approx(0.0 + 0.1)
        assert H[3] == 1
        assert visited[3]

    def test_in_range_reject_when_delta_very_negative(self):
        # g[old]=0, g[new]=20 → Δ=-20, exp(Δ)~2e-9, essentially never accept.
        def propose(state, rng):
            return state + 1, 0.0

        driver, sys, walker, g, H, visited = self._setup(propose)
        g[walker.bin_current] = 0.0
        g[walker.bin_current + 1] = 20.0

        accepted = driver._trial_step(
            walker, g, H, visited, ln_f=0.1,
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=propose,
            beta=0.0,
        )
        assert accepted is False
        # Update at current bin
        assert g[2] == pytest.approx(0.1)
        assert H[2] == 1

    def test_log_proposal_ratio_enters_acceptance(self):
        """A large positive log_proposal_ratio forces acceptance."""
        def propose(state, rng):
            return state + 1, 50.0  # huge boost

        driver, sys, walker, g, H, visited = self._setup(propose)
        g[walker.bin_current] = 0.0
        g[walker.bin_current + 1] = 30.0  # would normally reject

        accepted = driver._trial_step(
            walker, g, H, visited, ln_f=0.1,
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=propose,
            beta=0.0,
        )
        # Δ = 0 - 30 + 50 = +20 → always accept.
        assert accepted is True


# ---------------------------------------------------------------------------
# 3. End-to-end run mechanics
# ---------------------------------------------------------------------------

class TestRun:
    def test_reproducibility(self):
        sys = _tiny_system(5)
        cfg = WLConfig(bin_scheme=sys["scheme"], n_check=50, ln_f_final=1e-30)
        d1 = WLDriver(cfg)
        d2 = WLDriver(cfg)
        r1 = d1.run(initial_state=sys["initial_state"],
                    energy_fn=sys["energy_fn"],
                    order_parameter_fn=sys["order_parameter_fn"],
                    propose_move_fn=sys["propose_move_fn"],
                    rng=np.random.default_rng(123), max_trials=2000)
        r2 = d2.run(initial_state=sys["initial_state"],
                    energy_fn=sys["energy_fn"],
                    order_parameter_fn=sys["order_parameter_fn"],
                    propose_move_fn=sys["propose_move_fn"],
                    rng=np.random.default_rng(123), max_trials=2000)
        np.testing.assert_array_equal(r1.g, r2.g)
        np.testing.assert_array_equal(r1.H, r2.H)
        assert r1.t_total == r2.t_total
        assert r1.n_f_stages == r2.n_f_stages
        assert r1.final_state == r2.final_state

    def test_max_trials_stops(self):
        sys = _tiny_system(5)
        cfg = WLConfig(bin_scheme=sys["scheme"], n_check=50, ln_f_final=1e-30)
        result = WLDriver(cfg).run(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            rng=np.random.default_rng(0),
            max_trials=300,
        )
        assert result.t_total == 300
        assert result.converged is False
        # Histogram tally invariant: every trial increments H exactly once
        # (either at the new bin if accepted or the current bin if rejected).
        # H can be reset on standard-regime halves, but at 300 trials with
        # ln_f_final=1e-30 and small max we don't expect any reset to swallow
        # the entire count — at minimum sum(H) > 0.
        assert result.H.sum() > 0
        assert result.visited.any()

    def test_convergence_on_ln_f_final(self):
        """With a lax flatness threshold, ln_f should halve until < ln_f_final."""
        sys = _tiny_system(3)
        cfg = WLConfig(
            bin_scheme=sys["scheme"],
            n_check=100,
            ln_f_initial=1.0,
            ln_f_final=0.4,           # converge after 2 halves (1.0 → 0.5 → 0.25)
            flatness_threshold=0.01,  # virtually always satisfied
        )
        result = WLDriver(cfg).run(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            rng=np.random.default_rng(0),
            max_trials=10_000,
        )
        assert result.converged is True
        assert result.ln_f_final < 0.4
        assert result.n_f_stages >= 2
        # We didn't enter 1/t regime: 0.5 > 1/100 = 0.01 and 0.25 > 1/200 = 0.005.
        assert result.in_1overt is False

    def test_1overt_transition_at_known_t(self, tmp_path: Path):
        """Spec §7 explicit test: 1/t regime triggers at the first halve
        that would drop ln_f below 1/t. With ln_f_initial=1e-3 and
        n_check=20, the first check fires at t=20 where halved=5e-4 and
        1/t=0.05, so the switch happens immediately at t=20.
        """
        sys = _tiny_system(3)
        trace = tmp_path / "trace.tsv"
        cfg = WLConfig(
            bin_scheme=sys["scheme"],
            n_check=20,
            ln_f_initial=1e-3,
            ln_f_final=1e-30,
            flatness_threshold=0.01,  # always satisfied
            trace_path=trace,
        )
        result = WLDriver(cfg).run(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            rng=np.random.default_rng(0),
            max_trials=60,
        )
        assert result.in_1overt is True
        # After max_trials=60 in 1/t regime, ln_f should be 1/60.
        assert result.ln_f_final == pytest.approx(1.0 / 60)
        # Trace: first row at t=20 should be in 1/t and stage_index=1.
        rows = read_trace(trace)
        assert rows[0].t == 20
        assert rows[0].in_1overt is True
        assert rows[0].stage_index == 1
        # ln_f in trace at t=20 should be 1/20
        assert rows[0].ln_f == pytest.approx(1.0 / 20)

    def test_visited_persists_across_halves(self):
        """visited is a cumulative mask; it must NOT reset when H resets."""
        sys = _tiny_system(5)
        cfg = WLConfig(
            bin_scheme=sys["scheme"],
            n_check=50,
            ln_f_initial=1.0,
            ln_f_final=0.1,
            flatness_threshold=0.01,
        )
        result = WLDriver(cfg).run(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            rng=np.random.default_rng(0),
            max_trials=2_000,
        )
        assert result.n_f_stages >= 3
        # We started at the central bin so it must be visited; if visited
        # were reset on halve, only bins visited *between* the last halve
        # and end would show — at least the centre bin survives in either
        # case, but inspect cumulative coverage to make sure the mask is
        # not being clobbered.
        assert result.visited.sum() >= 3

    def test_initial_state_out_of_range_raises(self):
        sys = _tiny_system(5)
        cfg = WLConfig(bin_scheme=sys["scheme"])
        with pytest.raises(ValueError):
            WLDriver(cfg).run(
                initial_state=999,  # way out
                energy_fn=sys["energy_fn"],
                order_parameter_fn=sys["order_parameter_fn"],
                propose_move_fn=sys["propose_move_fn"],
                max_trials=10,
            )

    def test_g_normalization_keeps_min_at_zero_after_halve(self):
        sys = _tiny_system(3)
        cfg = WLConfig(
            bin_scheme=sys["scheme"], n_check=100,
            ln_f_initial=1.0, ln_f_final=0.4,
            flatness_threshold=0.01,
        )
        result = WLDriver(cfg).run(
            initial_state=sys["initial_state"],
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            rng=np.random.default_rng(0),
            max_trials=10_000,
        )
        # After the last halve, g[visited].min() was subtracted off. Then
        # the driver kept running and only adds ln_f per trial, monotonically.
        # So min(g[visited]) at the end is at least 0 and at most
        # n_trials_in_final_stage * ln_f_initial — but importantly the
        # final g array should not be hugely negative.
        assert result.g[result.visited].min() >= 0.0
