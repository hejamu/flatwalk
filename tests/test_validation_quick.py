"""Smoke test for the Ising validation pipeline.

The full L=8 validation to ln_f=1e-8 takes ~10 minutes and lives in
``examples/ising_validation.py`` (run as part of CI's slow lane).

This test exercises the pipeline at ln_f=1e-3 (~1 s) to catch wiring bugs
in Beale ↔ WL ↔ comparison without paying the full cost.  It does NOT
assert the strict pass criteria — only that the script runs end-to-end
and the comparison numbers come out finite and in roughly the right ballpark.
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

    g_exact = beale.beale_g_E(L)
    log_g_exact = beale.log_g_E_array(L, g_exact, scheme.centers)
    n_E_exact = np.exp(log_g_exact)

    valid = result.visited & np.isfinite(log_g_exact) & (n_E_exact > 0)
    shifted = result.g[valid] - result.g[valid].max()
    n_E_WL = np.exp(shifted)
    n_E_WL *= n_E_exact[valid].sum() / n_E_WL.sum()
    eps = np.abs(n_E_WL - n_E_exact[valid]) / n_E_exact[valid]

    central = np.ones_like(eps, dtype=bool)
    central[0] = False
    central[-1] = False
    eps_c = eps[central]

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

    visited = np.array([True, True, False, True, True])
    g1 = np.array([10.0, 11.0, 0.0, 12.0, 11.5])
    g2 = np.array([20.0, 21.5, 0.0, 22.0, 21.0])
    log_g, nz = ising_validation.average_log_g([_R(g1, visited), _R(g2, visited)])
    assert np.isneginf(log_g[2])
    assert nz[2] is np.bool_(False) or not bool(nz[2])
    assert np.isfinite(log_g[[0, 1, 3, 4]]).all()


def test_wl_to_n_E_dict_excludes_unvisited():
    """``wl_to_n_E_dict`` should drop bins where ``log_g = -inf``."""
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    log_g = np.array([-1.0, -0.5, -np.inf, -0.5, -1.0])
    d = ising_validation.wl_to_n_E_dict(centers, log_g, exact_total_for_normalization=100.0)
    assert 0 not in d
    assert set(d.keys()) == {-2, -1, 1, 2}
