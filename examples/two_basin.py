"""A minimal model that *needs* replica exchange.

Most simple systems (the Ising model included) do not require configuration
exchange between windows — windowing plus gluing already recovers ``g``. This
toy is built to be the exception, so the replica-exchange tutorial has something
to sink its teeth into.

The picture: two "conformations" (basins) of a molecule share a single
**gateway** state at order parameter ``q = 0`` and have *different* densities of
states. A move flips one bit, changing ``q`` by ``±1`` within a basin; the basin
can change *only* at the gateway. So a walker confined to a window that excludes
``q = 0`` is trapped in whichever basin it started in. Windowing alone therefore
gets the wrong ``g`` — each window sees only one basin. Only replica exchange,
shuttling configurations through the gateway window and back, lets every window
sample *both* basins.

State convention: a single ``(W, 1 + M1)`` int8 array — column 0 is the basin
label, the remaining ``M1`` columns are bits. Basin 0 uses the first ``M0`` bits
(the rest stay 0); basin 1 uses all ``M1``. The order parameter is the popcount,
so ``q = bits.sum()`` works for both. The exact density of states is closed
form: ``g(0) = 1`` and ``g(q) = C(M0, q) + C(M1, q)`` for ``q ≥ 1``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np


def exact_log_g(M0: int, M1: int) -> np.ndarray:
    """Exact ``log g(q)`` over the sampled range ``q ∈ [0, M0]``."""
    g = np.array([1.0 if q == 0 else comb(M0, q) + comb(M1, q) for q in range(M0 + 1)])
    return np.log(g)


def make_two_basin_callbacks(M0: int, M1: int) -> dict[str, Callable]:
    """Batched WL callbacks for the two-basin gateway model (WL on ``q``).

    The proposal picks a uniform valid bit-flip; because the number of valid
    moves differs between the gateway (``M0 + M1``) and a basin interior
    (``M0`` or ``M1``), the log proposal ratio ``log(deg_old / deg_new)`` is
    non-zero around the gateway (an exercise in the ``log_proposal_ratio`` term).
    """

    def order_parameter_fn(state: np.ndarray) -> np.ndarray:
        return state[:, 1 : 1 + M1].sum(1).astype(float)

    energy_fn = order_parameter_fn  # WL on q with beta = 0; energy is unused

    def propose_move_fn(state: np.ndarray, rng: np.random.Generator):
        W = state.shape[0]
        basin = state[:, 0].copy()
        bits = state[:, 1 : 1 + M1].copy()
        q = bits.sum(1)
        at_gate = q == 0
        deg_old = np.where(at_gate, M0 + M1, np.where(basin == 0, M0, M1)).astype(float)
        k = (rng.random(W) * deg_old).astype(int)
        rows = np.arange(W)

        nbits = bits.copy()
        nbasin = basin.copy()
        g0 = at_gate & (k < M0)  # leave gateway into basin 0
        g1 = at_gate & (k >= M0)  # leave gateway into basin 1
        inter = ~at_gate  # ordinary bit-flip inside the current basin
        nbits[rows[g0], k[g0]] = 1
        nbasin[g0] = 0
        nbits[rows[g1], k[g1] - M0] = 1
        nbasin[g1] = 1
        nbits[rows[inter], k[inter]] ^= 1

        nq = nbits.sum(1)
        nbasin[nq == 0] = 0  # canonicalise the single gateway state
        deg_new = np.where(nq == 0, M0 + M1, np.where(nbasin == 0, M0, M1)).astype(float)
        lpr = np.log(deg_old / deg_new)

        new = state.copy()
        new[:, 0] = nbasin
        new[:, 1 : 1 + M1] = nbits
        return new, lpr

    return {
        "energy_fn": energy_fn,
        "order_parameter_fn": order_parameter_fn,
        "propose_move_fn": propose_move_fn,
    }


def initial_states_for_windows(
    M0: int, M1: int, windows, rng: np.random.Generator
) -> np.ndarray:
    """One in-window config per window, alternating the starting basin.

    Alternating basins ensures both are present in the population, so the
    no-exchange run is visibly trapped (each window keeps its basin) while
    exchange can shuffle them.
    """
    states = []
    for w, (lo, hi) in enumerate(windows):
        basin = w % 2
        n = M0 if basin == 0 else M1
        qlo = max(int(np.ceil(lo)), 0)
        qhi = min(int(np.floor(hi)), n)
        q = int(rng.integers(qlo, qhi + 1))
        bits = np.zeros(M1, dtype=np.int8)
        if q > 0:
            bits[rng.choice(n, size=q, replace=False)] = 1
        st = np.zeros(1 + M1, dtype=np.int8)
        st[0] = basin if q > 0 else 0
        st[1:] = bits
        states.append(st)
    return np.stack(states)
