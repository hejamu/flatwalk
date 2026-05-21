"""
2D Ising validation against Beale's exact n(E) (L=4 smoke version)
==================================================================

Runs Wang-Landau sampling on a small 2D Ising lattice, compares the
recovered density of states against Beale's exact ``n(E)``, and plots
both curves. This is the smoke version of the project's
spec §4.4 validation — same pipeline as the full L=8 runner at
``examples/ising_validation.py``, just with a small lattice and a loose
``ln_f_final`` so it finishes in a few seconds. The numerical pass
criteria below are deliberately loose to keep the docs build fast;
they're tightened in the full CI runner.
"""

# %%
# Setup
# -----
#
# ``conf.py`` adds the repo's ``examples/`` directory to ``sys.path``
# for the docs build; the ``try`` block here is only for standalone
# execution of the script.

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

# %%
# Compute the exact reference
# ---------------------------

L = 4
n_E_exact = beale.beale_g_E(L)
print(f"L={L}: {len(n_E_exact)} distinct energies, total = 2^{L * L}")

# %%
# Run Wang-Landau
# ---------------
#
# Single seed, ``ln_f_final = 1e-5`` — well above the spec target of
# ``1e-8``, but enough for the curve to take its right shape on L=4.

cb = ising.make_ising_callbacks(L)
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)

rng = np.random.default_rng(0)
initial_state = ising.random_state(L, rng)
cfg = WLConfig(
    bin_scheme=scheme,
    beta=0.0,
    flatness_threshold=0.8,
    n_check=1_000,
    ln_f_initial=1.0,
    ln_f_final=1e-5,
)
result = WLDriver(cfg).run(
    initial_state=initial_state,
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    rng=rng,
)
print(
    f"  {result.t_total:,} trials, {result.n_f_stages} f-stages, "
    f"converged={result.converged}"
)

# %%
# Compare against Beale
# ---------------------
#
# Normalise ``g_WL`` to match the exact total over valid bins, then
# compute per-bin relative error and exclude the two extreme bins
# (g = 2) where relative noise dominates.

log_g_exact = beale.log_g_E_array(L, n_E_exact, scheme.centers)
n_E_exact_arr = np.exp(log_g_exact)

valid = result.visited & np.isfinite(log_g_exact) & (n_E_exact_arr > 0)
shifted = result.g[valid] - result.g[valid].max()
n_E_WL = np.exp(shifted)
n_E_WL *= n_E_exact_arr[valid].sum() / n_E_WL.sum()
eps = np.abs(n_E_WL - n_E_exact_arr[valid]) / n_E_exact_arr[valid]

central = np.ones_like(eps, dtype=bool)
central[0] = central[-1] = False
eps_c = eps[central]

print(f"  Compared on {int(central.sum())} central bins:")
print(f"    max  ε = {eps_c.max():.4f}")
print(f"    mean ε = {eps_c.mean():.4f}")

# Loose, gallery-friendly bounds (the full L=8 CI runner uses 0.05 / 0.01).
assert eps_c.max() < 0.5, f"smoke validation max ε too large: {eps_c.max():.3f}"
assert eps_c.mean() < 0.2, f"smoke validation mean ε too large: {eps_c.mean():.3f}"

# %%
# Plot log g(E)
# -------------
#
# Both curves shifted so their minimum over visited bins is 0. The WL
# line (dots) should track the Beale reference (dashed) within the
# per-bin error reported above.

E_axis = scheme.centers[valid]
g_WL_shift = result.g[valid] - result.g[valid].min()
g_ref_shift = log_g_exact[valid] - log_g_exact[valid].min()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(E_axis, g_WL_shift, "o-", color="C0", label="WL (single seed)")
ax.plot(E_axis, g_ref_shift, "k--", lw=1.0, label="Beale exact")
ax.set_xlabel("E")
ax.set_ylabel("log g(E)   (shifted to min = 0)")
ax.set_title(f"L={L} Ising: Wang-Landau vs Beale")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
