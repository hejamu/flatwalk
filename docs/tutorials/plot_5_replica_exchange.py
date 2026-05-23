"""
5. Replica exchange: windows that talk to each other
====================================================

More walkers buy throughput, but in tutorial 4 every walker still had to roam
the *entire* energy axis — and the steep tails of ``g`` stayed the slowest part
for each of them. Replica-exchange Wang-Landau (REWL) takes the opposite tack: split
the energy range into overlapping **windows**, confine one walker to each, and
let neighbouring windows **exchange** configurations now and then. Each walker
now has an easy local job; the exchanges keep the whole thing globally
consistent; and afterwards the per-window curves are **joined** over their
overlaps into one ``g(E)``.

This is the most robust flat-histogram method flatwalk ships — no ``E ↔ −E``
asymmetry, fast bulk convergence, and naturally parallel. The maths is in
:doc:`/theory/08-replica-exchange`; a compact recipe is the
:doc:`REWL example </auto_examples/plot_5_replica_exchange_ising>`.
"""

# %%
# Build the windows
# -----------------
#
# :func:`~flatwalk.make_windows` tiles the energy grid into equal-width,
# overlapping windows. REWL uses the batched callbacks (one stacked call per
# tick over all windows); the single-spin-flip physics is unchanged.

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

from flatwalk import Bin1D, RewlDriver, WLConfig, join_g, make_windows  # noqa: E402

L = 4
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)
windows = make_windows(scheme, n_windows=3, overlap=4)
for w, (lo, hi) in enumerate(windows):
    print(f"window {w}: E ∈ [{lo:.0f}, {hi:.0f}]")

cb = ising_batched.make_batched_ising_callbacks(L)
rng = np.random.default_rng(0)
initial_state = ising_batched.initial_states_for_windows(L, windows, rng)

# %%
# Run REWL
# --------
#
# Each window builds its own ``g`` confined to its energy range; every 100 ticks
# the driver attempts swaps between adjacent windows.

cfg = WLConfig(bin_scheme=scheme, beta=0.0, n_check=1_000, ln_f_final=1e-4)
result = RewlDriver(cfg, windows).run(
    initial_state=initial_state,
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    n_exchange=100,
    rng=rng,
)
acc = int(result.exchange_accepts.sum())
att = max(int(result.exchange_attempts.sum()), 1)
print(
    f"{result.t_total:,} ticks, converged={result.converged}, "
    f"exchange accept = {acc / att:.2f}"
)

# %%
# Join the windows
# ----------------
#
# :func:`~flatwalk.join_g` aligns each window to its neighbour over the shared
# bins (a constant log-shift) and averages the overlaps into one curve.

joined, visited_joined = join_g(result.g_windows, result.visited_windows)

log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
valid = visited_joined & np.isfinite(log_g_exact)
n_exact = np.exp(log_g_exact[valid])
n_wl = np.exp(joined[valid] - joined[valid].max())
n_wl *= n_exact.sum() / n_wl.sum()
eps = np.abs(n_wl - n_exact) / n_exact
eps[0] = eps[-1] = 0.0
print(f"joined: max ε = {eps.max():.3f}, mean ε = {eps.mean():.3f}")
assert eps.mean() < 0.6, "smoke REWL mean ε too large"

# %%
# The joined curve and the window spans
# -------------------------------------
#
# Shaded bands mark each window's energy range (overlaps appear darker). The
# joined curve (dots) tracks Beale (dashed) across all of them — assembled from
# three independent local walks.

E = scheme.centers[valid]
fig, ax = plt.subplots(figsize=(6, 4))
for lo, hi in windows:
    ax.axvspan(lo, hi, color="C1", alpha=0.12)
ax.plot(E, joined[valid] - joined[valid].min(), "o-", color="C0", label="REWL (joined)")
ax.plot(
    E, log_g_exact[valid] - log_g_exact[valid].min(), "k--", lw=1.0, label="Beale exact"
)
ax.set_xlabel("E")
ax.set_ylabel("log g(E)   (shifted to min = 0)")
ax.set_title(f"L={L} Ising: replica-exchange WL vs Beale")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# That completes the journey from a single fixed-temperature Monte Carlo run to
# a robust, parallel estimate of the whole density of states. The
# :doc:`final tutorial <plot_6_thermodynamics>` turns a converged ``g(E)`` into
# the full thermodynamics — free energy, entropy, energy, and heat capacity
# across temperature.
