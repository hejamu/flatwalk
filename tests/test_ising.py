"""Sanity tests for the Ising callbacks in `examples/ising.py`."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import ising  # noqa: E402


def test_ground_state_energy(L=4):
    """All-up configuration has E = -2 J L² (every bond satisfied)."""
    spins = np.ones((L, L), dtype=np.int8)
    assert ising.total_energy(spins) == -2 * L * L


def test_antiferromagnetic_energy_even_L():
    """Checkerboard on even L has E = +2 J L² (every bond unsatisfied)."""
    L = 4
    spins = np.where(((np.arange(L)[:, None] + np.arange(L)[None, :]) % 2) == 0, 1, -1).astype(np.int8)
    assert ising.total_energy(spins) == +2 * L * L


@pytest.mark.parametrize("L", [4, 6, 8])
def test_dE_matches_full_recompute(L):
    """The ΔE shortcut in propose_move_fn must agree with a full recompute."""
    rng = np.random.default_rng(0)
    spins, E = ising.random_state(L, rng)
    cb = ising.make_ising_callbacks(L)
    state = (spins, E)
    for _ in range(200):
        new_state, lpr = cb["propose_move_fn"](state, rng)
        assert lpr == 0.0  # symmetric proposal
        true_E = ising.total_energy(new_state[0])
        assert abs(true_E - new_state[1]) < 1e-9, (
            f"cached E ({new_state[1]}) drifted from recomputed ({true_E})"
        )
        state = new_state


def test_energy_in_valid_range_L8():
    """For L=8, every random initial config has E in [-128, +128]."""
    L = 8
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, E = ising.random_state(L, rng)
        assert -2 * L * L <= E <= +2 * L * L


def test_single_spin_flip_changes_exactly_one_site():
    """The proposal must only ever change one spin (spec §4.2)."""
    L = 4
    rng = np.random.default_rng(0)
    spins, E = ising.random_state(L, rng)
    cb = ising.make_ising_callbacks(L)
    for _ in range(200):
        new_state, _ = cb["propose_move_fn"]((spins, E), rng)
        diff = (new_state[0] != spins).sum()
        assert diff == 1


def test_bin_scheme_centres_land_on_allowed_energies():
    """The Bin1D constructor parameters from `ising_energy_bins` should
    produce bin centres exactly on the allowed Ising spectrum."""
    from flatwalk import Bin1D

    L = 8
    low, high, n_bins = ising.ising_energy_bins(L)
    scheme = Bin1D(low, high, n_bins)
    expected = np.arange(-2 * L * L, 2 * L * L + 1, 4, dtype=float)
    np.testing.assert_allclose(scheme.centers, expected)
