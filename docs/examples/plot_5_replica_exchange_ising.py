"""
Replica-exchange Wang-Landau on the 2D Ising model
==================================================

Replica-exchange Wang-Landau (REWL) splits the energy axis into overlapping
**windows**, runs one walker per window (each confined to its sub-range),
periodically **exchanges** configurations between adjacent windows, and
**joins** the per-window ``g`` into one curve over the overlaps.

We run it on ``L=4`` against Beale's exact ``n(E)`` — the same target as the
single-walker recipe, now recovered window by window. The method is derived in
:doc:`/theory/08-replica-exchange`.
"""

# %%
# Setup
# -----
#
# REWL uses the batched callbacks (one stacked call per tick over all
# windows) from ``examples/ising_batched.py``; the scalar single-spin-flip
# physics is identical to the single-walker example.

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

# %%
# Build the windows
# -----------------
#
# ``make_windows`` tiles the global bin grid into equal-width, overlapping
# windows. Each is given a walker started inside its energy range.

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
# ``ln_f_final = 1e-4`` keeps the smoke run to a few seconds. Exchanges are
# attempted every 100 ticks (one tick = one move per window).

cfg = WLConfig(
    bin_scheme=scheme,
    beta=0.0,
    flatness_threshold=0.8,
    n_check=1_000,
    ln_f_initial=1.0,
    ln_f_final=1e-4,
)
result = RewlDriver(cfg, windows).run(
    initial_state=initial_state,
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    n_exchange=100,
    rng=rng,
)
acc = result.exchange_accepts.sum()
att = max(int(result.exchange_attempts.sum()), 1)
print(
    f"{result.t_total:,} ticks, {result.n_f_stages} f-stages, "
    f"converged={result.converged}, exchange accept = {acc / att:.2f}"
)

# %%
# Join the windows and compare
# ----------------------------
#
# ``join_g`` aligns each window to its neighbour over the shared bins (a
# constant log-shift) and averages the overlaps into one curve.

joined, visited_joined = join_g(result.g_windows, result.visited_windows)

log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
n_E_exact = np.exp(log_g_exact)
valid = visited_joined & np.isfinite(log_g_exact) & (n_E_exact > 0)
shifted = joined[valid] - joined[valid].max()
n_E_WL = np.exp(shifted)
n_E_WL *= n_E_exact[valid].sum() / n_E_WL.sum()
eps = np.abs(n_E_WL - n_E_exact[valid]) / n_E_exact[valid]

central = np.ones_like(eps, dtype=bool)
central[0] = central[-1] = False
print(f"  joined: max ε = {eps[central].max():.3f}, mean ε = {eps[central].mean():.3f}")
assert eps[central].mean() < 0.6, "smoke REWL mean ε too large"

# %%
# Plot the joined g(E) with the window spans
# ------------------------------------------
#
# Shaded bands mark each window's energy range (overlaps appear darker). The
# joined curve (dots) should track Beale (dashed) across all of them.

E_axis = scheme.centers[valid]
g_joined_shift = joined[valid] - joined[valid].min()
g_ref_shift = log_g_exact[valid] - log_g_exact[valid].min()

fig, ax = plt.subplots(figsize=(6, 4))
for lo, hi in windows:
    ax.axvspan(lo, hi, color="C1", alpha=0.12)
ax.plot(E_axis, g_joined_shift, "o-", color="C0", label="REWL (joined)")
ax.plot(E_axis, g_ref_shift, "k--", lw=1.0, label="Beale exact")
ax.set_xlabel("E")
ax.set_ylabel("log g(E)   (shifted to min = 0)")
ax.set_title(f"L={L} Ising: replica-exchange WL vs Beale")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# A note on where the error lives
# -------------------------------
#
# REWL nails the bulk quickly — thermodynamic averages converge fast — but
# the exponentially steep tails (where ``n(E)`` spans many orders of
# magnitude) are the last bins to converge in *any* flat-histogram method
# and need a deep ``ln_f``. The win over a single walker is robustness (no
# ``E ↔ −E`` asymmetry, no multi-seed averaging) and parallelism: the windows
# advance as one batched call, so an expensive batched energy backend is
# evaluated once per tick rather than once per walker.
