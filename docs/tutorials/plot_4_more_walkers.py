"""
4. More walkers: symmetry and efficiency
========================================

A single walker on a hard system has two distinct weaknesses, and several
walkers sharing one ``g`` (:meth:`~flatwalk.WLDriver.run_batched` — one batched
call per tick, no Python loop) cure them for two distinct reasons. This tutorial
takes them one at a time, each with its own controlled experiment:

1. **Symmetry.** A lone walker has to diffuse across the whole energy axis and
   reaches one tail long before the other, piling up large-``ln f`` updates on
   the side it found first. Its ``g`` comes out *tilted* — a spurious
   ``E ↔ −E`` asymmetry, even though the Ising ``g`` is exactly symmetric. Many
   walkers sit at different energies at once and cover both ends together, so the
   tilt averages out. We show this at **equal total compute**.

2. **Efficiency.** Each tick is one call into your (possibly expensive) energy
   backend. ``N`` walkers ride one tick together, so the same number of calls
   buys ``N`` times the samples. We show this at a **fixed number of ticks**.

We use ``L=8`` — large enough that a lone walker genuinely struggles, where the
toy ``L=4`` of the earlier tutorials hides both effects. The maths is in
:doc:`/theory/07-multiple-walkers`.
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

L = 8
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)
log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
finite = np.isfinite(log_g_exact)
batched_cb = ising_batched.make_batched_ising_callbacks(L)

N_LIST = [1, 4, 16]


def run(n_walkers, max_trials):
    """One shared-g run of ``n_walkers`` to a fixed move budget (no early stop)."""
    rng = np.random.default_rng(0)
    init = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_walkers, L, L))
    return WLDriver(
        WLConfig(bin_scheme=scheme, beta=0.0, n_check=2_000, ln_f_final=1e-30)
    ).run_batched(
        initial_state=init,
        energy_fn=batched_cb["energy_fn"],
        order_parameter_fn=batched_cb["order_parameter_fn"],
        propose_move_fn=batched_cb["propose_move_fn"],
        n_walkers=n_walkers,
        rng=rng,
        max_trials=max_trials,
    )


def max_central_eps(result):
    valid = result.visited & finite
    n_exact = np.exp(log_g_exact[valid])
    n_wl = np.exp(result.g[valid] - result.g[valid].max())
    n_wl *= n_exact.sum() / n_wl.sum()
    eps = np.abs(n_wl - n_exact) / n_exact
    eps[0] = eps[-1] = 0.0  # drop the g = 2 corners
    return eps.max()


def signed_deviation(result):
    """log g_WL − log g_exact over valid bins, each shifted to mean 0."""
    valid = result.visited & finite
    dev = (result.g[valid] - result.g[valid].mean()) - (
        log_g_exact[valid] - log_g_exact[valid].mean()
    )
    return scheme.centers[valid], dev


# %%
# Experiment 1 — symmetry, at equal compute
# -----------------------------------------
#
# Every run gets the **same total number of moves** (2,000,000), just split among
# more walkers. So the energy backend does the same total work; only the number
# of walkers spreading over the spectrum changes.

MOVES = 2_000_000
sym = {}
for n in N_LIST:
    res = run(n, max_trials=MOVES)
    E, dev = signed_deviation(res)
    rms = float(np.sqrt((dev**2).mean()))
    sym[n] = (E, dev, rms)
    print(f"N={n:2d}: {res.t_total:>9,} moves   rms deviation = {rms:.3f}")

# The shared g gets flatter and more symmetric as walkers are added.
assert sym[N_LIST[-1]][2] < sym[N_LIST[0]][2] / 2

# %%
# The single walker's curve is tilted and ragged — it reached one tail first.
# Adding walkers (same total compute) flattens it and restores the ``E ↔ −E``
# symmetry the exact ``g`` has.

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.axhline(0.0, color="k", lw=0.8)
for n in N_LIST:
    E, dev, _ = sym[n]
    lw = 2.0 if n == N_LIST[-1] else 1.0
    ax.plot(E, dev, "-", lw=lw, label=f"{n} walker" + ("s" if n > 1 else ""))
ax.set_xlabel("E")
ax.set_ylabel("log g_WL − log g_exact")
ax.set_title(f"L={L}: same compute, more walkers → symmetric g")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# Experiment 2 — efficiency, at a fixed number of ticks
# -----------------------------------------------------
#
# Now every run makes the **same number of backend calls** (400,000 ticks) — the
# same wall-clock on a vectorised backend — but ``N`` walkers ride each call, so
# they accumulate ``N`` times the moves. ``max_trials = N × ticks`` holds the tick
# count fixed.

TICKS = 400_000
eff_eps = []
for n in N_LIST:
    res = run(n, max_trials=n * TICKS)
    eff_eps.append(max_central_eps(res))
    print(f"N={n:2d}: {TICKS:,} ticks = {res.t_total:>9,} moves   max ε = {eff_eps[-1]:.3f}")

# Same number of (expensive) calls, far lower error with more walkers.
assert eff_eps[-1] < eff_eps[0] / 2

# %%
# For the same number of backend calls, the error falls steeply with ``N``: each
# call simply does more useful work. This is why batching is the point — an
# expensive GPU or MPI energy evaluation is paid once per tick whether it scores
# one configuration or many.

N = np.array(N_LIST)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(N, eff_eps, "o-", color="C3")
ax.set_xlabel("walkers N (at fixed 400,000 ticks)")
ax.set_ylabel("max ε vs Beale")
ax.set_title(f"L={L}: same backend calls, more walkers → lower error")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
plt.show()

# %%
# Two levers, then: spend a fixed budget across more walkers and the ``g`` comes
# out symmetric; make a fixed number of calls with more walkers and it comes out
# accurate. Both rest on the same primitive — many configurations through one
# shared ``g`` per backend call.
#
# But every walker still roams the *entire* energy range, and the steep tails of
# ``g`` stay the slowest part for each of them. The last tutorial gives each
# walker an easier, *local* job — its own slice of the spectrum — and lets
# neighbours trade: replica-exchange Wang-Landau.
