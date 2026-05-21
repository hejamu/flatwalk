"""Batched 2D Ising callbacks for the ≥2-walker / REWL paths.

The scalar single-lattice callbacks live in ``examples/ising.py`` and are left
exactly as they are; this is the stacked-``W``-lattices version used by
``WLDriver.run_batched`` and ``RewlDriver``.

State convention: a single ``np.ndarray`` of shape ``(W, L, L)``, ``int8``,
values in ``{-1, +1}`` — ``W`` lattices stacked along axis 0. Keeping the state
as one array (rather than an ``(spins, energy)`` tuple as in the scalar
example) is what lets the driver apply accepted moves with boolean-mask
assignment ``state[accept] = new_state[accept]`` and reorder it on an exchange.

For "WL on E" the order parameter *is* the energy, recomputed each tick by a
single batched pass over the ``W`` lattices — exactly the one-batched-call-per
-tick pattern the batched layer exists for.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def total_energy_batched(spins: np.ndarray, J: float = 1.0) -> np.ndarray:
    """``E_w = -J Σ_<ij> σ_i σ_j`` for each of the ``W`` lattices (each bond once).

    ``spins`` has shape ``(W, L, L)``; returns ``(W,)`` float64.
    """
    s = spins.astype(np.int32)
    right = np.roll(s, -1, axis=2)
    down = np.roll(s, -1, axis=1)
    bonds = (s * right) + (s * down)
    return (-J * bonds.sum(axis=(1, 2))).astype(np.float64)


def make_batched_ising_callbacks(L: int, J: float = 1.0) -> dict[str, Callable]:
    """Batched WL callbacks for a stack of ``W`` L×L Ising lattices on the torus.

    The move is a single-spin flip per lattice (one random site each), so the
    proposal is symmetric and ``log_proposal_ratio = 0`` for every walker.
    """

    def energy_fn(spins: np.ndarray) -> np.ndarray:
        return total_energy_batched(spins, J)

    def order_parameter_fn(spins: np.ndarray) -> np.ndarray:
        return total_energy_batched(spins, J)

    def propose_move_fn(spins: np.ndarray, rng: np.random.Generator):
        W = spins.shape[0]
        w = np.arange(W)
        i = rng.integers(0, L, size=W)
        j = rng.integers(0, L, size=W)
        new_spins = spins.copy()
        new_spins[w, i, j] = -new_spins[w, i, j]
        return new_spins, np.zeros(W)

    return {
        "energy_fn": energy_fn,
        "order_parameter_fn": order_parameter_fn,
        "propose_move_fn": propose_move_fn,
    }


def config_in_window(
    L: int,
    low: float,
    high: float,
    rng: np.random.Generator,
    J: float = 1.0,
    max_iter: int = 200_000,
) -> np.ndarray:
    """Build one L×L config whose energy lands in ``[low, high]``.

    Random Ising configs cluster near ``E ≈ 0``, so a window near the energy
    extremes needs a targeted start. We walk greedily toward the window centre
    (accepting any flip that does not increase ``|E - target|``) with an
    occasional random kick to avoid stalls, and stop once inside the window.
    """
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    E = float(total_energy_batched(spins[None, :, :], J)[0])
    target = 0.5 * (low + high)
    for _ in range(max_iter):
        if low <= E <= high:
            return spins
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        s = int(spins[i, j])
        nb = int(
            spins[(i - 1) % L, j]
            + spins[(i + 1) % L, j]
            + spins[i, (j - 1) % L]
            + spins[i, (j + 1) % L]
        )
        dE = 2.0 * J * s * nb
        if abs((E + dE) - target) <= abs(E - target) or rng.random() < 0.05:
            spins[i, j] = -s
            E += dE
    raise RuntimeError(f"could not construct a config with energy in [{low}, {high}]")


def initial_states_for_windows(
    L: int,
    windows: list[tuple[float, float]],
    rng: np.random.Generator,
    J: float = 1.0,
) -> np.ndarray:
    """Stack one in-window config per window into a ``(W, L, L)`` int8 array."""
    return np.stack([config_in_window(L, lo, hi, rng, J) for lo, hi in windows]).astype(
        np.int8
    )
