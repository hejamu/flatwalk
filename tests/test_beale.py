"""Cross-validate Beale's transfer-matrix recursion against brute-force
enumeration on small lattices.

If Beale and brute force agree exactly on L=3 (512 configs) and L=4 (65536
configs), then the L=8 reference used by the Ising validation script
(`examples/ising_validation.py`) can be trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``examples/`` importable for tests.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import beale  # noqa: E402


@pytest.mark.parametrize("L", [3, 4])
def test_beale_matches_brute_force(L):
    g_beale = beale.beale_g_E(L)
    g_brute = beale.brute_force_g_E(L)
    assert g_beale == g_brute, (
        f"Beale L={L} disagrees with brute force; "
        f"symmetric diff: {set(g_beale.items()) ^ set(g_brute.items())}"
    )


def test_beale_total_count_l5():
    """L=5 is too large for brute force (2^25 configs) but Beale should still
    satisfy the easy sanity Σ n(E) = 2^(L²)."""
    n_E = beale.beale_g_E(5)
    assert sum(n_E.values()) == 1 << (5 * 5)


def test_beale_z2_symmetry_l4():
    """n(E) should be even — every config and its spin-flip image both exist
    (Z2 symmetry of the Hamiltonian with no field)."""
    n_E = beale.beale_g_E(4)
    odd_counts = [E for E, n in n_E.items() if n % 2 != 0]
    assert odd_counts == [], f"non-Z2-paired counts at E = {odd_counts}"
