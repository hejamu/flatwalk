"""
5. Windows, gluing, and replica exchange
========================================

The last idea splits the energy axis: cut it into overlapping **windows**,
confine one walker to each, and afterwards **glue** the per-window ``g`` together
over the overlaps. Replica exchange adds one more move on top — neighbouring
windows swap configurations. These are *two separate ingredients*, and conflating
them is a common confusion, so this tutorial pulls them apart:

* **Gluing** reassembles ``g`` from the windows, using their overlaps alone — no
  communication between windows needed. We show it recovers ``g`` for the Ising
  model with exchange switched *off*.
* **Exchange** does something gluing cannot: it mixes the degrees of freedom
  *orthogonal* to the binned energy. Ising never needs that (shown below). So we
  then build a small model that *does*, where windowing alone gets the wrong
  answer and only exchange fixes it.

The maths is in :doc:`/theory/08-replica-exchange`; a compact recipe is the
:doc:`REWL example </auto_examples/plot_5_replica_exchange_ising>`.
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
import two_basin  # noqa: E402

from flatwalk import Bin1D, RewlDriver, WLConfig, join_g, make_windows  # noqa: E402

# %%
# Part A — gluing recovers g (Ising, no exchange)
# -----------------------------------------------
#
# :func:`~flatwalk.make_windows` tiles the energy grid into overlapping windows.
# We run one *independent* confined walker per window (``n_exchange=None``) and
# :func:`~flatwalk.join_g` aligns neighbours over their shared bins (a constant
# log-shift) — the overlaps alone carry the alignment.

L = 8
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)
windows = make_windows(scheme, n_windows=4, overlap=8)
cb = ising_batched.make_batched_ising_callbacks(L)
log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
finite = np.isfinite(log_g_exact)

rng = np.random.default_rng(0)
init = ising_batched.initial_states_for_windows(L, windows, rng)
res = RewlDriver(
    WLConfig(bin_scheme=scheme, beta=0.0, n_check=2_000, ln_f_final=1e-4), windows
).run(
    initial_state=init,
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    n_exchange=None,  # exchange OFF — pure windowing + gluing
    rng=rng,
)
joined, vis = join_g(res.g_windows, res.visited_windows)
val = vis & finite
n_exact = np.exp(log_g_exact[val])
n_wl = np.exp(joined[val] - joined[val].max())
n_wl *= n_exact.sum() / n_wl.sum()
eps_ising = np.abs(n_wl - n_exact) / n_exact
eps_ising[0] = eps_ising[-1] = 0.0
print(f"Ising, gluing only (no exchange): joined max ε = {eps_ising.max():.3f}")
assert eps_ising.max() < 0.5

E = scheme.centers[val]
fig, ax = plt.subplots(figsize=(6, 4))
for lo, hi in windows:
    ax.axvspan(lo, hi, color="C1", alpha=0.12)
ax.plot(E, joined[val] - joined[val].min(), "o-", color="C0", label="windowed WL (joined)")
ax.plot(E, log_g_exact[val] - log_g_exact[val].min(), "k--", lw=1.0, label="Beale exact")
ax.set_xlabel("E")
ax.set_ylabel("log g(E)   (shifted to min = 0)")
ax.set_title(f"L={L} Ising: windows + gluing, no exchange")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# Gluing alone tracks Beale across all four windows. **Ising never needs
# exchange:** WL flattens the *energy*, so a confined walker random-walks freely
# across its window; and the one barrier it has — the two ferromagnetic basins —
# is *symmetric*, so a basin-trapped window has the right shape up to a constant,
# which ``join_g`` absorbs. The mixing exchange provides is simply not required.

# %%
# Part B — a system that needs exchange
# -------------------------------------
#
# Now a model built to break windowing. Two "conformations" (basins) share a
# single **gateway** at order parameter ``q = 0`` and have *different* densities
# of states. A move flips one bit (``q → q ± 1``) within a basin; the basin can
# change *only* at the gateway. So a walker confined to a window that excludes
# ``q = 0`` is **trapped** in the basin it started in — and the two basins have
# different shapes, so a trapped window reports the wrong ``g`` (not just a
# constant offset that gluing could absorb). The exact ``g`` is known:
# ``g(q) = C(M0, q) + C(M1, q)``.

M0, M1 = 8, 12
scheme2 = Bin1D(-0.5, M0 + 0.5, M0 + 1)
windows2 = make_windows(scheme2, n_windows=4, overlap=3)
log_g_exact2 = two_basin.exact_log_g(M0, M1)
cb2 = two_basin.make_two_basin_callbacks(M0, M1)


def run_two_basin(n_exchange, seed):
    rng = np.random.default_rng(seed)
    init = two_basin.initial_states_for_windows(M0, M1, windows2, rng)
    r = RewlDriver(
        WLConfig(bin_scheme=scheme2, beta=0.0, n_check=1_000, ln_f_final=1e-5), windows2
    ).run(
        initial_state=init,
        energy_fn=cb2["energy_fn"],
        order_parameter_fn=cb2["order_parameter_fn"],
        propose_move_fn=cb2["propose_move_fn"],
        n_exchange=n_exchange,
        rng=rng,
        max_trials=5_000_000,
    )
    joined, vis = join_g(r.g_windows, r.visited_windows)
    g_exact = np.exp(log_g_exact2[vis])
    g_wl = np.exp(joined[vis] - joined[vis].max())
    g_wl *= g_exact.sum() / g_wl.sum()
    eps = float((np.abs(g_wl - g_exact) / g_exact).max())
    return scheme2.centers[vis], joined[vis] - joined[vis].min(), eps


q_off, g_off, eps_off = run_two_basin(n_exchange=None, seed=0)
q_on, g_on, eps_on = run_two_basin(n_exchange=5, seed=0)
print(f"two-basin, gluing only (no exchange): max ε = {eps_off:.2f}")
print(f"two-basin, REWL (exchange on):        max ε = {eps_on:.2f}")
assert eps_off > 0.5, "expected windowing-without-exchange to fail here"
assert eps_on < 0.35, "expected exchange to recover g"

# %%
# Without exchange the joined curve is badly wrong — each window is stuck in one
# basin and reports only that basin's density. With exchange it lands on the exact
# curve: configurations migrate through the gateway window, switch basins there,
# and carry the mix back, so every window samples both.

qg = scheme2.centers
g_ex = log_g_exact2 - log_g_exact2.min()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(qg, g_ex, "k--", lw=1.2, label="exact")
ax.plot(q_off, g_off, "o-", color="C3", label=f"windowing only (max ε={eps_off:.2f})")
ax.plot(q_on, g_on, "o-", color="C0", label=f"REWL (max ε={eps_on:.2f})")
ax.set_xlabel("q")
ax.set_ylabel("log g(q)   (shifted to min = 0)")
ax.set_title("Two-basin model: only exchange recovers g")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# This is the mechanism REWL exists for, and it is exactly parallel tempering's:
# a barrier the walker cannot cross *inside* a window can be crossed in the
# gateway window, and exchange ferries configurations there and back. Real rugged
# landscapes — first-order transitions, biomolecules — are full of such barriers;
# Ising simply is not, which is why its windows glued cleanly without any of this.
#
# That completes the journey from a single fixed-temperature Monte Carlo run to a
# robust, parallel estimate of the whole density of states. The
# :doc:`final tutorial <plot_6_thermodynamics>` turns a converged ``g(E)`` into
# the full thermodynamics — free energy, entropy, energy, and heat capacity
# across temperature.
