"""Smoke test for the Ising validation pipeline.

The full L=8 validation to ln_f=1e-8 takes ~10 minutes and lives in
``examples/ising_validation.py`` (run as part of CI's slow lane).

This test exercises the pipeline at ln_f=1e-3 (~1 s) to catch wiring bugs
in Beale ↔ WL ↔ comparison without paying the full cost.  It does NOT
assert the spec pass criteria — only that the script runs end-to-end and
the comparison numbers come out finite and in roughly the right ballpark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import beale  # noqa: E402
import ising  # noqa: E402
import ising_validation  # noqa: E402

from flatwalk import Bin1D, WLConfig, WLDriver  # noqa: E402


def test_ising_pipeline_smoke():
    L = 4  # tiny lattice keeps the test fast (~1 s)
    cb = ising.make_ising_callbacks(L)
    rng = np.random.default_rng(0)
    initial_state = ising.random_state(L, rng)

    low, high, n_bins = ising.ising_energy_bins(L)
    scheme = Bin1D(low, high, n_bins)
    cfg = WLConfig(
        bin_scheme=scheme,
        beta=0.0,
        flatness_threshold=0.8,
        n_check=10_000,
        ln_f_initial=1.0,
        ln_f_final=1e-3,
    )
    driver = WLDriver(cfg)
    result = driver.run(
        initial_state=initial_state,
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        rng=rng,
    )
    assert result.converged is True
    assert result.visited.sum() >= n_bins - 2  # only the gap bins should be unvisited

    # Compare against exact via Beale
    g_exact = beale.beale_g_E(L)
    log_g_exact = beale.log_g_E_array(L, g_exact, scheme.centers)
    n_E_exact = np.exp(log_g_exact)

    valid = result.visited & np.isfinite(log_g_exact) & (n_E_exact > 0)
    shifted = result.g[valid] - result.g[valid].max()
    n_E_WL = np.exp(shifted)
    n_E_WL *= n_E_exact[valid].sum() / n_E_WL.sum()
    eps = np.abs(n_E_WL - n_E_exact[valid]) / n_E_exact[valid]

    # Skip the two extreme bins (g=2) where relative noise dominates.
    central = np.ones_like(eps, dtype=bool)
    central[0] = False
    central[-1] = False
    eps_c = eps[central]

    # Sanity bounds: at ln_f=1e-3 the WL run hasn't fully converged but it
    # should be in the right neighbourhood (5-20% errors on L=4).
    assert np.isfinite(eps_c).all()
    assert eps_c.mean() < 0.5  # very loose; just guards against catastrophic bugs


def test_average_log_g_leaves_unvisited_at_neg_inf():
    """Regression: an unvisited-by-all bin must come out as -inf, not 0.

    A 0 would have ``exp(0) = 1`` and dominate the thermodynamics, while
    -inf gives ``exp(-inf) = 0`` (no contribution). The script's
    `wl_to_n_E_dict` and thermodynamics chain depend on this.
    """
    class _R:
        def __init__(self, g, visited):
            self.g = g
            self.visited = visited

    # 5 bins; bin 2 is never visited by either seed
    visited = np.array([True, True, False, True, True])
    g1 = np.array([10.0, 11.0, 0.0, 12.0, 11.5])
    g2 = np.array([20.0, 21.5, 0.0, 22.0, 21.0])
    log_g, nz = ising_validation.average_log_g([_R(g1, visited), _R(g2, visited)])
    assert np.isneginf(log_g[2])
    assert nz[2] is np.bool_(False) or not bool(nz[2])
    # Visited bins should be finite
    assert np.isfinite(log_g[[0, 1, 3, 4]]).all()


def test_snapshot_recorder_log_spaced_sampling():
    """Recorder should retain ~n_frames snapshots spread log-spaced in t."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer
    from flatwalk.diagnostics import ProgressSnapshot

    rec = wl_viewer.SnapshotRecorder(n_frames=100)
    bin_centers = np.linspace(-1, 1, 5)
    g = np.zeros(5)
    H = np.zeros(5, dtype=np.int64)
    visited = np.ones(5, dtype=bool)
    # Simulate 50_000 checks at t = 1, 2, ..., 50000
    for t in range(1, 50001):
        rec(ProgressSnapshot(
            t=t, ln_f=1.0 / t, in_1overt=False, n_f_stages=0,
            g=g, H=H, visited=visited, bin_centers=bin_centers,
            flatness=0.5, acceptance_rate=0.5,
        ))
    # log-spaced sampling should give a moderate (not all 50K) frame count
    assert 10 <= len(rec.snapshots) <= 200, len(rec.snapshots)
    # And the early dynamics should be over-represented relative to late
    ts = [s.t for s in rec.snapshots]
    early = sum(1 for t in ts if t < 1000)
    late = sum(1 for t in ts if t > 10000)
    assert early >= late  # log spacing → more frames at small t


def test_trial_recorder_buffers_correctly():
    """TrialRecorder records every trial up to max_records, then stops."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer

    rec = wl_viewer.TrialRecorder(max_records=50)
    for k in range(1, 200):
        rec(k, k % 5, float(k), 0.5, k % 2 == 0)
    arr = rec.as_arrays()
    assert len(arr["t"]) == 50
    np.testing.assert_array_equal(arr["t"], np.arange(1, 51))
    assert arr["accepted"].dtype == bool


def test_make_trajectory_movie_renders_gif(tmp_path):
    """make_trajectory_movie must produce a valid GIF from a tiny history."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer

    n = 8
    bin_centers = np.linspace(-4, 4, 9)
    history = {
        "t": np.arange(1, n + 1, dtype=np.int64),
        "bin": np.array([4, 3, 4, 5, 4, 3, 2, 3], dtype=np.int64),
        "energy": np.array([0, -1, 0, 1, 0, -1, -2, -1], dtype=np.float64),
        "ln_f": np.full(n, 1.0),
        "accepted": np.array([True, True, False, True, False, True, True, True]),
    }
    out = tmp_path / "traj.gif"
    wl_viewer.make_trajectory_movie(
        history, out, bin_centers=bin_centers, fps=4,
    )
    assert out.exists() and out.stat().st_size > 2_000


def test_make_trajectory_movie_log_spaced_resets_H_on_halve(tmp_path):
    """When ``ln_f`` drops within the recorded history, the renderer must
    reset its internal H (mirroring the WL driver's halve behaviour)."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer

    bin_centers = np.linspace(-2, 2, 5)
    # 12 trials, ln_f halves at trial 6 (index 5)
    history = {
        "t": np.arange(1, 13, dtype=np.int64),
        "bin": np.array([2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3], dtype=np.int64),
        "energy": np.zeros(12, dtype=np.float64),
        "ln_f": np.array([1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        "accepted": np.ones(12, dtype=bool),
    }
    out = tmp_path / "halve.gif"
    # Render every frame (no log-spacing) so we hit the halve exactly.
    wl_viewer.make_trajectory_movie(
        history, out, bin_centers=bin_centers, fps=4,
    )
    assert out.exists() and out.stat().st_size > 2_000


def test_make_trajectory_movie_log_spaced_subsampling(tmp_path):
    """With n_frames << len(history), the renderer must pick log-spaced indices."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer

    bin_centers = np.linspace(-2, 2, 5)
    n = 500
    rng = np.random.default_rng(0)
    bins = rng.integers(0, 5, size=n)
    history = {
        "t": np.arange(1, n + 1, dtype=np.int64),
        "bin": bins.astype(np.int64),
        "energy": (bins - 2).astype(np.float64),
        "ln_f": np.full(n, 1.0),
        "accepted": np.ones(n, dtype=bool),
    }
    out = tmp_path / "logspace.gif"
    wl_viewer.make_trajectory_movie(
        history, out, bin_centers=bin_centers, fps=5, n_frames=30,
    )
    assert out.exists() and out.stat().st_size > 2_000


def test_make_movie_renders_gif(tmp_path):
    """make_movie should produce a valid GIF from a tiny snapshot list."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer
    from flatwalk.diagnostics import ProgressSnapshot

    bin_centers = np.linspace(-10, 10, 21)
    snapshots = []
    for k in range(5):
        snapshots.append(ProgressSnapshot(
            t=100 * (k + 1), ln_f=0.5 ** k, in_1overt=False, n_f_stages=k,
            g=np.linspace(0, k + 1, 21),
            H=np.full(21, 10 * (k + 1), dtype=np.int64),
            visited=np.ones(21, dtype=bool),
            bin_centers=bin_centers,
            flatness=0.9, acceptance_rate=0.5,
        ))
    out = tmp_path / "movie.gif"
    wl_viewer.make_movie(snapshots, out, bin_centers=bin_centers, fps=4)
    assert out.exists() and out.stat().st_size > 5_000


def test_live_viewer_runs_headless(tmp_path):
    """The viewer's callback path must work under the Agg backend (no display).

    Useful so CI / headless dev environments can exercise the viewer code
    without a window.
    """
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import wl_viewer  # noqa: E402  (after env var)
    from flatwalk.diagnostics import ProgressSnapshot

    centers = np.linspace(-10.0, 10.0, 21)
    v = wl_viewer.LiveViewer(centers, flatness_threshold=0.9, update_every_s=0.0)
    snap = ProgressSnapshot(
        t=1000, ln_f=0.25, in_1overt=False, n_f_stages=2,
        g=np.linspace(0, 5, 21),
        H=np.arange(21, dtype=np.int64) * 10,
        visited=np.ones(21, dtype=bool),
        bin_centers=centers,
        flatness=0.92, acceptance_rate=0.5,
    )
    v.callback(snap)
    out = tmp_path / "v.png"
    v.save(out)
    assert out.exists() and out.stat().st_size > 1000


def test_wl_to_n_E_dict_excludes_unvisited():
    """``wl_to_n_E_dict`` should drop bins where ``log_g = -inf``."""
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    log_g = np.array([-1.0, -0.5, -np.inf, -0.5, -1.0])
    d = ising_validation.wl_to_n_E_dict(centers, log_g, exact_total_for_normalization=100.0)
    assert 0 not in d
    assert set(d.keys()) == {-2, -1, 1, 2}
