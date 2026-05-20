"""Checkpoint/restart tests.

The spec demands bit-identical resume: a run interrupted mid-way then
resumed from disk must produce the *exact same* ``g`` and ``H`` arrays as
a run that was never interrupted (same RNG seed). This is the canonical
correctness property of the I/O layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flatwalk import Bin1D, WLConfig, WLDriver
from flatwalk.io import load_checkpoint, save_checkpoint


# ---- save/load round-trip --------------------------------------------------

def test_save_load_round_trip(tmp_path: Path):
    path = tmp_path / "cp.npz"
    g = np.linspace(0.0, 1.0, 10)
    H = np.arange(10, dtype=np.int64)
    visited = (H > 0)
    edges = np.linspace(-0.5, 9.5, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    walker_state = np.array([1, 2, 3, 4], dtype=np.int8)  # tests ndarray boxing
    rng = np.random.default_rng(42)
    # Burn a few draws so the state is non-default.
    rng.standard_normal(7)
    rng_state = rng.bit_generator.state

    save_checkpoint(
        path,
        g=g, H=H, visited=visited,
        bin_edges=edges, bin_centers=centers,
        n_bins=10, t_total=12345, n_f_stages=3,
        ln_f=0.0625, in_1overt=False,
        bin_current=4, walker_energy=-7.5,
        walker_state=walker_state, rng_state=rng_state,
    )

    loaded = load_checkpoint(path)
    np.testing.assert_array_equal(loaded["g"], g)
    np.testing.assert_array_equal(loaded["H"], H)
    np.testing.assert_array_equal(loaded["visited"], visited)
    np.testing.assert_array_equal(loaded["bin_edges"], edges)
    np.testing.assert_array_equal(loaded["bin_centers"], centers)
    assert loaded["n_bins"] == 10
    assert loaded["t_total"] == 12345
    assert loaded["n_f_stages"] == 3
    assert loaded["ln_f"] == 0.0625
    assert loaded["in_1overt"] is False
    assert loaded["bin_current"] == 4
    assert loaded["walker_energy"] == -7.5
    np.testing.assert_array_equal(loaded["walker_state"], walker_state)
    assert loaded["walker_state"].dtype == walker_state.dtype
    assert loaded["rng_state"] == rng_state


def test_save_is_atomic_no_tmp_left(tmp_path: Path):
    path = tmp_path / "cp.npz"
    save_checkpoint(
        path,
        g=np.zeros(3), H=np.zeros(3, dtype=np.int64), visited=np.zeros(3, dtype=bool),
        bin_edges=np.linspace(0, 1, 4), bin_centers=np.array([1/6, 1/2, 5/6]),
        n_bins=3, t_total=0, n_f_stages=0,
        ln_f=1.0, in_1overt=False,
        bin_current=0, walker_energy=0.0,
        walker_state=0, rng_state=np.random.default_rng(0).bit_generator.state,
    )
    # No leftover .tmp sidecar
    assert path.exists()
    siblings = list(tmp_path.iterdir())
    assert all(not p.name.endswith(".tmp") for p in siblings), siblings


def test_save_appends_npz_extension(tmp_path: Path):
    """save_checkpoint should canonicalize ``foo`` → ``foo.npz``."""
    base = tmp_path / "foo"
    save_checkpoint(
        base,
        g=np.zeros(3), H=np.zeros(3, dtype=np.int64), visited=np.zeros(3, dtype=bool),
        bin_edges=np.linspace(0, 1, 4), bin_centers=np.array([1/6, 1/2, 5/6]),
        n_bins=3, t_total=0, n_f_stages=0,
        ln_f=1.0, in_1overt=False,
        bin_current=0, walker_energy=0.0,
        walker_state=0, rng_state=np.random.default_rng(0).bit_generator.state,
    )
    assert (tmp_path / "foo.npz").exists()


# ---- bit-identical resume --------------------------------------------------

def _tiny_system():
    """Tiny 5-state random walk; same as test_core to keep behavior consistent."""
    scheme = Bin1D(-0.5, 4.5, 5)

    def propose(state, rng):
        step = 1 if rng.random() < 0.5 else -1
        return state + step, 0.0

    return {
        "scheme": scheme,
        "initial_state": 2,
        "energy_fn": lambda s: 0.0,
        "order_parameter_fn": lambda s: float(s),
        "propose_move_fn": propose,
    }


def test_resume_is_bit_identical(tmp_path: Path):
    """Run A uninterrupted N trials.  Run B = N/2 trials → checkpoint →
    resume → another N/2. Final g, H, visited, walker state must match
    A bit-exactly."""
    sys = _tiny_system()
    N = 4000
    N_half = N // 2
    seed = 9999

    # ----- Reference run (uninterrupted) -----
    cfg_a = WLConfig(
        bin_scheme=sys["scheme"],
        n_check=200,
        ln_f_initial=1.0,
        ln_f_final=1e-30,        # never converge by ln_f
        flatness_threshold=0.01,  # easy
    )
    result_a = WLDriver(cfg_a).run(
        initial_state=sys["initial_state"],
        energy_fn=sys["energy_fn"],
        order_parameter_fn=sys["order_parameter_fn"],
        propose_move_fn=sys["propose_move_fn"],
        rng=np.random.default_rng(seed),
        max_trials=N,
    )
    assert result_a.t_total == N

    # ----- Run B part 1: checkpoint at t=N_half -----
    cp_path = tmp_path / "cp.npz"
    cfg_b1 = WLConfig(
        bin_scheme=sys["scheme"],
        n_check=200,
        ln_f_initial=1.0,
        ln_f_final=1e-30,
        flatness_threshold=0.01,
        checkpoint_path=cp_path,
        checkpoint_every_t=N_half,  # checkpoint exactly at N_half
    )
    result_b1 = WLDriver(cfg_b1).run(
        initial_state=sys["initial_state"],
        energy_fn=sys["energy_fn"],
        order_parameter_fn=sys["order_parameter_fn"],
        propose_move_fn=sys["propose_move_fn"],
        rng=np.random.default_rng(seed),
        max_trials=N_half,
    )
    assert result_b1.t_total == N_half
    assert cp_path.exists()

    # ----- Run B part 2: resume from checkpoint -----
    cfg_b2 = WLConfig(
        bin_scheme=sys["scheme"],
        n_check=200,
        ln_f_initial=1.0,         # irrelevant after resume
        ln_f_final=1e-30,
        flatness_threshold=0.01,
    )
    result_b2 = WLDriver(cfg_b2).run(
        initial_state=None,        # ignored on resume
        energy_fn=sys["energy_fn"],
        order_parameter_fn=sys["order_parameter_fn"],
        propose_move_fn=sys["propose_move_fn"],
        resume_from=cp_path,
        max_trials=N,
    )
    assert result_b2.t_total == N

    # ----- Bit-identicality -----
    np.testing.assert_array_equal(result_a.g, result_b2.g)
    np.testing.assert_array_equal(result_a.H, result_b2.H)
    np.testing.assert_array_equal(result_a.visited, result_b2.visited)
    assert result_a.final_state == result_b2.final_state
    assert result_a.ln_f_final == result_b2.ln_f_final
    assert result_a.n_f_stages == result_b2.n_f_stages
    assert result_a.in_1overt == result_b2.in_1overt


def test_resume_rejects_mismatched_n_bins(tmp_path: Path):
    sys = _tiny_system()
    cp_path = tmp_path / "cp.npz"
    cfg = WLConfig(
        bin_scheme=sys["scheme"],
        checkpoint_path=cp_path,
        checkpoint_every_t=50,
    )
    WLDriver(cfg).run(
        initial_state=sys["initial_state"],
        energy_fn=sys["energy_fn"],
        order_parameter_fn=sys["order_parameter_fn"],
        propose_move_fn=sys["propose_move_fn"],
        rng=np.random.default_rng(0),
        max_trials=50,
    )
    # Different n_bins → mismatch
    other_scheme = Bin1D(-0.5, 9.5, 10)
    cfg_other = WLConfig(bin_scheme=other_scheme)
    with pytest.raises(ValueError, match="n_bins"):
        WLDriver(cfg_other).run(
            initial_state=None,
            energy_fn=sys["energy_fn"],
            order_parameter_fn=sys["order_parameter_fn"],
            propose_move_fn=sys["propose_move_fn"],
            resume_from=cp_path,
            max_trials=100,
        )
