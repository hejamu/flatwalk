"""2D Ising model on an L×L periodic lattice — callbacks for `WLDriver`.

State convention: ``state = (spins, energy)`` where

- ``spins`` is an ``np.ndarray`` of shape ``(L, L)``, ``dtype=int8``, values in ``{-1, +1}``.
- ``energy`` is the cached integer (in ``J`` units) energy of ``spins``.

Caching the energy is essential: a single-spin-flip Monte Carlo move
changes the energy in O(1), but recomputing it from scratch is O(L²).
The WL driver calls ``energy_fn`` and ``order_parameter_fn`` on every
trial, so we keep both as O(1) lookups on the cached value.

Single-spin-flip is symmetric, so ``propose_move_fn`` returns
``log_proposal_ratio = 0``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def total_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """``E = -J Σ_<ij> σ_i σ_j`` on a periodic L×L lattice (each bond once)."""
    right = np.roll(spins, -1, axis=1)
    down = np.roll(spins, -1, axis=0)
    return float(
        -J
        * (
            (spins.astype(np.int32) * right.astype(np.int32))
            + (spins.astype(np.int32) * down.astype(np.int32))
        ).sum()
    )


def random_state(L: int, rng: np.random.Generator, J: float = 1.0) -> tuple:
    """Random Ising configuration and its cached energy."""
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    return (spins, total_energy(spins, J=J))


def make_ising_callbacks(L: int, J: float = 1.0) -> dict[str, Callable]:
    """Return WL-compatible callbacks for the L×L Ising on torus.

    All callbacks act on a ``state = (spins, cached_E)`` tuple. The move
    proposal is single-spin-flip: pick one site uniformly, propose
    σ_i → −σ_i, with ``log_proposal_ratio = 0``.
    """

    def energy_fn(state) -> float:
        return state[1]

    def order_parameter_fn(state) -> float:
        # For "WL on E" the order parameter coincides with the energy.
        return state[1]

    def propose_move_fn(state, rng: np.random.Generator) -> tuple[tuple, float]:
        spins, E = state
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        s = int(spins[i, j])
        # Neighbour sum with PBC
        neighbour_sum = int(
            spins[(i - 1) % L, j]
            + spins[(i + 1) % L, j]
            + spins[i, (j - 1) % L]
            + spins[i, (j + 1) % L]
        )
        # Single-flip ΔE = 2 J σ_i (sum of neighbours).
        # Each affected bond's contribution flips sign, so
        # E_new - E_old = +2 J σ_old · neighbour_sum.
        dE = 2.0 * J * s * neighbour_sum
        new_spins = spins.copy()
        new_spins[i, j] = -s
        return (new_spins, E + dE), 0.0

    return {
        "energy_fn": energy_fn,
        "order_parameter_fn": order_parameter_fn,
        "propose_move_fn": propose_move_fn,
    }


# ---------------------------------------------------------------------------
# Bin scheme convenience: bin centres on allowed Ising energies (step 4J)
# ---------------------------------------------------------------------------


def ising_energy_bins(L: int, J: float = 1.0):
    """Return (low, high, n_bins) for a Bin1D placing centres on allowed energies.

    Allowed energies are ``E ∈ {-2L², -2L²+4, ..., 2L²}`` (step 4J). With
    ``n_bins = L²+1`` and width ``4J``, bin centres land exactly on these
    integers.
    """
    E_min = -2.0 * J * L * L
    E_max = +2.0 * J * L * L
    n_bins = L * L + 1
    low = E_min - 2.0 * J
    high = E_max + 2.0 * J
    return low, high, n_bins
