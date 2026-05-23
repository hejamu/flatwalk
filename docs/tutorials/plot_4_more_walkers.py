"""
4. More walkers, fewer backend calls
====================================

Going deeper costs iterations. To halve the error you run far longer, and every
one of those steps calls back into your energy function. When that backend is
expensive — a GPU model, an MPI kernel — the number of *calls* is what sets the
wall-clock.

This is what multiple walkers are for. flatwalk advances ``N`` walkers as a
single batched call per tick (:meth:`~flatwalk.WLDriver.run_batched` — no Python
loop over walkers), all feeding *one shared* ``g``. So each tick gathers ``N``
times the samples for the price of one backend call. Reaching a given quality
then takes about ``N`` times fewer ticks — an ``N``× throughput win on a
vectorised backend, for the same-quality ``g``. The maths is in
:doc:`/theory/07-multiple-walkers`.
"""

# %%
# Setup
# -----
#
# We reach the *same* target ``ln_f_final`` with ``N = 1, 2, 4, 8, 16`` walkers
# and count two things: the final error against Beale, and the number of **ticks**
# — one tick is one batched backend call, regardless of ``N``.

import sys

try:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
except NameError:
    pass  # sphinx-gallery exec context: __file__ undefined, sys.path is already set

import beale  # noqa: E402
import ising  # noqa: E402
import ising_batched  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flatwalk import Bin1D, WLConfig, WLDriver  # noqa: E402

L = 4
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)
log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
finite = np.isfinite(log_g_exact)
batched_cb = ising_batched.make_batched_ising_callbacks(L)


def max_central_eps(result):
    valid = result.visited & finite
    n_exact = np.exp(log_g_exact[valid])
    n_wl = np.exp(result.g[valid] - result.g[valid].max())
    n_wl *= n_exact.sum() / n_wl.sum()
    eps = np.abs(n_wl - n_exact) / n_exact
    eps[0] = eps[-1] = 0.0  # drop the g = 2 corners
    return eps.max()


# %%
# Run each walker count to the same target
# ----------------------------------------
#
# ``t_total`` counts individual moves (``N`` per tick), so ``ticks = t_total / N``
# is the number of batched backend calls.

N_LIST = [1, 2, 4, 8, 16]
ticks = []
errors = []
for n in N_LIST:
    rng = np.random.default_rng(0)
    init = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, L, L))
    res = WLDriver(
        WLConfig(bin_scheme=scheme, beta=0.0, n_check=1_000, ln_f_final=1e-4)
    ).run_batched(
        initial_state=init,
        energy_fn=batched_cb["energy_fn"],
        order_parameter_fn=batched_cb["order_parameter_fn"],
        propose_move_fn=batched_cb["propose_move_fn"],
        n_walkers=n,
        rng=rng,
    )
    ticks.append(res.t_total // n)
    errors.append(max_central_eps(res))
    print(f"N={n:2d}: {res.t_total:>8,} moves  ->  {ticks[-1]:>8,} backend calls  "
          f"max ε={errors[-1]:.3f}")

# Same target reached in ~N× fewer backend calls, at the same quality.
assert ticks[-1] < ticks[0] / 4
assert max(errors) < 0.3

# %%
# The win
# -------
#
# Left: ticks (backend calls) to reach the target fall almost exactly as ``1/N``
# (dashed guide) — 16 walkers need ~16× fewer calls than one. Right: the final
# error is essentially flat across ``N``: the speedup is *not* paid for in
# quality. More walkers buy the same ``g`` in fewer passes over your backend.

N = np.array(N_LIST)
fig, (axT, axE) = plt.subplots(1, 2, figsize=(10, 4))

axT.loglog(N, ticks, "o-", color="C0", label="measured")
axT.loglog(N, ticks[0] / N, "k--", lw=1.0, label="ideal 1/N")
axT.set_xlabel("walkers N")
axT.set_ylabel("ticks to reach ln_f_final = 1e-4")
axT.set_title("Fewer backend calls")
axT.legend()
axT.grid(alpha=0.3, which="both")

axE.semilogx(N, errors, "o-", color="C1")
axE.set_xlabel("walkers N")
axE.set_ylabel("final max ε vs Beale")
axE.set_ylim(0, max(errors) * 1.5)
axE.set_title("Same quality")
axE.grid(alpha=0.3, which="both")

fig.suptitle(f"L={L} Ising: many walkers, one shared g")
fig.tight_layout()
plt.show()

# %%
# Equivalently — read at *fixed* ticks rather than a fixed target — ``N`` walkers
# fold ``N``× the samples into ``g`` per call, so the estimate is both more
# converged and steadier from seed to seed. Either way the lever is the same:
# more configurations per backend call.
#
# But all the walkers still roam the *entire* energy range, and the steep tails
# of ``g`` stay the slowest part for every one of them. The last tutorial gives
# each walker an easier, *local* job and lets neighbours trade — replica-exchange
# Wang-Landau.
