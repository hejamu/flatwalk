"""
6. Thermodynamics from a converged g(E)
=======================================

The journey ends where flat-histogram sampling was always headed: a single
density of states ``g(E)`` carries the *entire* thermodynamics of the system.
Here we take a converged ``g(E)`` and read off the free energy, internal energy,
entropy, and heat capacity across temperature — comparing each against the exact
answer from Beale's ``n(E)``.

One subtlety made concrete: Wang-Landau returns ``log g`` only up to an additive
constant. We pin it with a fact we know — the total number of states is
``2^N`` — which turns the free energy and entropy into *absolute* quantities.
The reweighting formulae are derived in :doc:`/theory/04-density-of-states`.
"""

# %%
# Get a good g(E)
# ---------------
#
# We use a batch of walkers sharing one ``g`` (tutorial 4) for a clean curve.

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

L = 6
N = L * L  # number of spins
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)

cb = ising_batched.make_batched_ising_callbacks(L)
rng = np.random.default_rng(0)
initial_state = rng.choice(np.array([-1, 1], dtype=np.int8), size=(8, L, L))
result = WLDriver(
    WLConfig(bin_scheme=scheme, beta=0.0, n_check=2_000, ln_f_final=1e-5)
).run_batched(
    initial_state=initial_state,
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    n_walkers=8,
    rng=rng,
)
print(f"{result.t_total:,} moves, converged={result.converged}")

# %%
# Pin the additive constant, then reweight
# ----------------------------------------
#
# Normalising ``log g`` so that ``Σ_E g(E) = 2^N`` fixes the constant. Then for
# each temperature ``Z = Σ_E g(E) e^{-βE}`` gives everything:
# ``F = −T ln Z``, ``⟨E⟩`` and ``C_V = β²(⟨E²⟩−⟨E⟩²)`` by reweighting, and
# ``S = ln Z + β⟨E⟩`` (with ``k_B = 1``).


def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def normalize(log_g):
    """Shift log g so the total number of states is 2^N."""
    return log_g + (N * np.log(2.0) - logsumexp(log_g))


def thermo(log_g, E, temperatures):
    F = np.empty(len(temperatures))
    U = np.empty(len(temperatures))
    Cv = np.empty(len(temperatures))
    S = np.empty(len(temperatures))
    for k, T in enumerate(temperatures):
        beta = 1.0 / T
        lw = log_g - beta * E
        lnZ = logsumexp(lw)
        p = np.exp(lw - lnZ)
        u = float((p * E).sum())
        u2 = float((p * E * E).sum())
        F[k] = -T * lnZ
        U[k] = u
        Cv[k] = beta * beta * (u2 - u * u)
        S[k] = lnZ + beta * u
    return F, U, Cv, S


log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
valid = result.visited & np.isfinite(log_g_exact)
E = scheme.centers[valid]

T_grid = np.linspace(1.0, 4.0, 80)
F_wl, U_wl, Cv_wl, S_wl = thermo(normalize(result.g[valid]), E, T_grid)
F_ex, U_ex, Cv_ex, S_ex = thermo(normalize(log_g_exact[valid]), E, T_grid)

# %%
# All four, from one curve
# ------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
panels = [
    (axes[0, 0], "free energy  F / N", F_wl, F_ex),
    (axes[0, 1], "internal energy  ⟨E⟩ / N", U_wl, U_ex),
    (axes[1, 0], "heat capacity  C_V / N", Cv_wl, Cv_ex),
    (axes[1, 1], "entropy  S / N", S_wl, S_ex),
]
for ax, label, wl, ex in panels:
    ax.plot(T_grid, wl / N, "-", color="C0", label="from WL g(E)")
    ax.plot(T_grid, ex / N, "k--", lw=1.0, label="exact (Beale)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[1, 0].set_xlabel("T")
axes[1, 1].set_xlabel("T")
axes[0, 0].legend()
fig.suptitle(f"L={L} Ising: full thermodynamics from a single g(E)")
fig.tight_layout()
plt.show()

# %%
# At ``T → ∞`` the entropy approaches ``N ln 2`` (all ``2^N`` states equally
# likely) — the check that our additive-constant fix was right:

print(
    f"S/N at T={T_grid[-1]:.1f}:  WL = {S_wl[-1] / N:.4f}, "
    f"exact = {S_ex[-1] / N:.4f},  ln 2 = {np.log(2):.4f}"
)

# %%
# From plain Monte Carlo stuck at one temperature, to one ``g(E)`` that yields
# the free energy, energy, entropy, and heat capacity at *every* temperature —
# that is what flat-histogram sampling buys, and what flatwalk exists to make
# routine. Where to go next: the :doc:`examples </auto_examples/index>` for
# recipes to adapt to your own system, and the :doc:`theory </theory/index>` for
# the full derivations.
