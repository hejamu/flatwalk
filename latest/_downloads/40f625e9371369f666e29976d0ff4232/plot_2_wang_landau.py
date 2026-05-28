"""
2. Wang-Landau: one run, every temperature
==========================================

The previous tutorial left us stuck: plain Monte Carlo only ever sees the
energies that matter at its own temperature. Wang-Landau breaks that tie. By
estimating the density of states ``g(E)`` over the *whole* energy axis, a single
run gives us the thermodynamics at **every** temperature at once.

Here we run Wang-Landau on the same ``L=6`` Ising model, then reconstruct the
mean energy and heat capacity across a range of ``T`` from the one ``g(E)`` —
the curve the plain-MC points from tutorial 1 only sampled three dots of.

The method is derived in :doc:`/theory/05-wang-landau`; the thermodynamics from
``g`` in :doc:`/theory/04-density-of-states`.
"""

# %%
# Run Wang-Landau on Ising
# ------------------------
#
# We hand flatwalk the same single-spin-flip physics, now via
# :meth:`~flatwalk.WLDriver.run`. With ``beta = 0`` and the order parameter set
# to the energy, this is a canonical "WL on E" run.

import sys

try:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
except NameError:
    pass  # sphinx-gallery exec context: __file__ undefined, sys.path is already set

import beale  # noqa: E402
import ising  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flatwalk import Bin1D, WLConfig, WLDriver  # noqa: E402

L = 6
cb = ising.make_ising_callbacks(L)
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)

cfg = WLConfig(bin_scheme=scheme, beta=0.0, n_check=2_000, ln_f_final=1e-5)
result = WLDriver(cfg).run(
    initial_state=ising.random_state(L, np.random.default_rng(0)),
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    rng=np.random.default_rng(0),
)
print(
    f"{result.t_total:,} trials, {result.n_f_stages} f-stages, converged={result.converged}"
)

# %%
# Thermodynamics from g(E)
# ------------------------
#
# Given ``log g(E)``, every canonical average follows by reweighting:
#
# .. math::
#
#    \langle A \rangle_\beta =
#    \frac{\sum_E A(E)\, g(E)\, e^{-\beta E}}{\sum_E g(E)\, e^{-\beta E}} .
#
# We evaluate it stably in log space (subtract the max before exponentiating)
# and read off the mean energy and the heat capacity
# ``C_V = β² (⟨E²⟩ − ⟨E⟩²)``.


def thermo(log_g, E, temperatures):
    mean_E = np.empty(len(temperatures))
    cv = np.empty(len(temperatures))
    for k, T in enumerate(temperatures):
        beta = 1.0 / T
        w = log_g - beta * E
        w -= w.max()
        p = np.exp(w)
        p /= p.sum()
        eav = float((p * E).sum())
        e2 = float((p * E * E).sum())
        mean_E[k] = eav
        cv[k] = beta * beta * (e2 - eav * eav)
    return mean_E, cv


# Compare on bins visited by WL and known to Beale.
log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
valid = result.visited & np.isfinite(log_g_exact)
E = scheme.centers[valid]

T_grid = np.linspace(1.2, 4.0, 60)
meanE_wl, cv_wl = thermo(result.g[valid], E, T_grid)
meanE_ex, cv_ex = thermo(log_g_exact[valid], E, T_grid)

# %%
# A few plain-MC points, to close tutorial 1's loop
# -------------------------------------------------
#
# These are short fixed-temperature Metropolis runs — exactly the kind of
# per-temperature experiment we were stuck with before. Each yields *one* point;
# the WL curve gives all of them from a single run.


def metropolis_mean_E(T, n_steps, burn_in, rng):
    spins = ising.random_state(L, rng)[0]
    En = ising.total_energy(spins)
    acc = []
    for t in range(n_steps):
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        s = int(spins[i, j])
        nb = int(
            spins[(i - 1) % L, j]
            + spins[(i + 1) % L, j]
            + spins[i, (j - 1) % L]
            + spins[i, (j + 1) % L]
        )
        dE = 2.0 * s * nb
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            spins[i, j] = -s
            En += dE
        if t >= burn_in:
            acc.append(En)
    return np.mean(acc)


rng = np.random.default_rng(1)
T_mc = [1.5, 2.27, 3.5]
meanE_mc = [metropolis_mean_E(T, 30_000, 5_000, rng) for T in T_mc]

# %%
# The payoff
# ----------
#
# Left: mean energy per spin. The single WL run (line) reproduces both the exact
# curve (dashed) and the individual plain-MC points (dots). Right: the heat
# capacity, whose peak near ``T_c`` we now get for free — drawing it the
# plain-MC way would have meant a separate run at every temperature.

fig, (axE, axC) = plt.subplots(1, 2, figsize=(10, 4))
axE.plot(T_grid, meanE_wl / (L * L), "-", color="C0", label="Wang-Landau (one run)")
axE.plot(T_grid, meanE_ex / (L * L), "k--", lw=1.0, label="exact (Beale)")
axE.plot(T_mc, np.array(meanE_mc) / (L * L), "o", color="C3", label="plain MC (per T)")
axE.set_xlabel("T")
axE.set_ylabel("⟨E⟩ / N")
axE.set_title("Mean energy")
axE.legend()
axE.grid(alpha=0.3)

axC.plot(T_grid, cv_wl / (L * L), "-", color="C0", label="Wang-Landau")
axC.plot(T_grid, cv_ex / (L * L), "k--", lw=1.0, label="exact (Beale)")
axC.set_xlabel("T")
axC.set_ylabel("C_V / N")
axC.set_title("Heat capacity")
axC.legend()
axC.grid(alpha=0.3)
fig.suptitle(f"L={L} Ising: thermodynamics from a single g(E)")
fig.tight_layout()
plt.show()

# %%
# That is the whole promise made good: **one** Wang-Landau run, **every**
# temperature. The next tutorial looks closer at the convergence — the curve
# above is right in the bulk, but the steep tails of ``g`` hide an error floor
# that standard Wang-Landau alone cannot push past.
