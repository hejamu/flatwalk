"""
4. More walkers, less variance
==============================

A single Wang-Landau walker has to traverse the whole energy axis by itself. It
reaches one tail before the other and piles up large-``ln f`` updates there, so
even with the 1/t schedule a lone run leaves a few-percent error that wobbles
from seed to seed — and a faint ``E ↔ −E`` asymmetry, even though the Ising
``g(E)`` is exactly symmetric.

Running several walkers through *one shared* ``g`` averages that away: each bin
collects contributions from many walkers per stage. With flatwalk this is
:meth:`~flatwalk.WLDriver.run_batched` — one stacked call per tick, no Python
loop over walkers. The maths is in :doc:`/theory/07-multiple-walkers`.
"""

# %%
# Setup
# -----

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

scalar_cb = ising.make_ising_callbacks(L)
batched_cb = ising_batched.make_batched_ising_callbacks(L)

LN_F_FINAL = 1e-4
N_SEEDS = 4
N_WALKERS = 8


def signed_dev(result):
    """log g_WL − log g_exact over valid bins, both shifted to min 0."""
    valid = result.visited & finite
    gw = result.g[valid] - result.g[valid].min()
    ge = log_g_exact[valid] - log_g_exact[valid].min()
    return scheme.centers[valid], gw - ge


def max_central_eps(result):
    valid = result.visited & finite
    n_exact = np.exp(log_g_exact[valid])
    n_wl = np.exp(result.g[valid] - result.g[valid].max())
    n_wl *= n_exact.sum() / n_wl.sum()
    eps = np.abs(n_wl - n_exact) / n_exact
    eps[0] = eps[-1] = 0.0
    return eps.max()


# %%
# Single walker, several seeds
# ----------------------------

single_eps = []
single_dev = None
for seed in range(N_SEEDS):
    res = WLDriver(
        WLConfig(bin_scheme=scheme, beta=0.0, n_check=1_000, ln_f_final=LN_F_FINAL)
    ).run(
        initial_state=ising.random_state(L, np.random.default_rng(seed)),
        energy_fn=scalar_cb["energy_fn"],
        order_parameter_fn=scalar_cb["order_parameter_fn"],
        propose_move_fn=scalar_cb["propose_move_fn"],
        rng=np.random.default_rng(seed),
    )
    single_eps.append(max_central_eps(res))
    if seed == 0:
        single_dev = signed_dev(res)

# %%
# Eight walkers sharing one g, same seeds
# ---------------------------------------

batched_eps = []
batched_dev = None
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    init = rng.choice(np.array([-1, 1], dtype=np.int8), size=(N_WALKERS, L, L))
    res = WLDriver(
        WLConfig(bin_scheme=scheme, beta=0.0, n_check=1_000, ln_f_final=LN_F_FINAL)
    ).run_batched(
        initial_state=init,
        energy_fn=batched_cb["energy_fn"],
        order_parameter_fn=batched_cb["order_parameter_fn"],
        propose_move_fn=batched_cb["propose_move_fn"],
        n_walkers=N_WALKERS,
        rng=rng,
    )
    batched_eps.append(max_central_eps(res))
    if seed == 0:
        batched_dev = signed_dev(res)

print(
    f"single-walker  max ε: mean {np.mean(single_eps):.3f}, spread {np.ptp(single_eps):.3f}"
)
print(
    f"{N_WALKERS}-walker      max ε: mean {np.mean(batched_eps):.3f}, "
    f"spread {np.ptp(batched_eps):.3f}"
)
assert np.mean(batched_eps) <= np.mean(single_eps) + 0.05

# %%
# The win
# -------
#
# Left: the per-seed error — many walkers give a lower, tighter result. Right:
# the signed deviation from the exact ``g(E)`` for one run of each; the single
# walker's error grows and tilts in the tails (the ``E ↔ −E`` asymmetry), while
# the shared-``g`` run stays flat and centred.

fig, (axB, axD) = plt.subplots(1, 2, figsize=(10, 4))

axB.plot(np.zeros(N_SEEDS), single_eps, "o", color="C0", label="1 walker")
axB.plot(np.ones(N_SEEDS), batched_eps, "o", color="C1", label=f"{N_WALKERS} walkers")
axB.set_xticks([0, 1])
axB.set_xticklabels(["1 walker", f"{N_WALKERS} walkers"])
axB.set_ylabel("max ε vs Beale (per seed)")
axB.set_title("Lower and tighter error")
axB.grid(alpha=0.3)

axD.axhline(0.0, color="k", lw=0.8)
axD.plot(single_dev[0], single_dev[1], "o-", color="C0", ms=3, label="1 walker")
axD.plot(
    batched_dev[0], batched_dev[1], "o-", color="C1", ms=3, label=f"{N_WALKERS} walkers"
)
axD.set_xlabel("E")
axD.set_ylabel("log g_WL − log g_exact")
axD.set_title("Deviation across the spectrum")
axD.legend()
axD.grid(alpha=0.3)

fig.suptitle(f"L={L} Ising: one walker vs {N_WALKERS} sharing g")
fig.tight_layout()
plt.show()

# %%
# Sharing one ``g`` cuts the variance, and because the walkers advance as a
# single batched call, an expensive vectorised energy backend is paid once per
# tick rather than once per walker. But all the walkers still roam the *entire*
# energy range. The last tutorial confines them to overlapping **windows** and
# lets them exchange — replica-exchange Wang-Landau.
