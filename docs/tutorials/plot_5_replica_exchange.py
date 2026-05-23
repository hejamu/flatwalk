"""
5. Windows, gluing, and replica exchange
========================================

More walkers (tutorial 4) buy throughput and symmetry, but every walker still
roamed the *entire* energy axis. The last idea splits the axis instead: cut it
into overlapping **windows**, confine one walker to each, and afterwards **glue**
the per-window ``g`` together over the overlaps into one curve. That alone — no
communication between windows — is already a complete method, and we show it
first.

**Replica exchange** then adds one more move: neighbouring windows swap
configurations now and then. A common misconception is that the swapping is what
stitches the windows together — it is not; the gluing does that, from the
overlaps alone. What exchange adds is *mixing*. For a rugged system that buys
accuracy; for the symmetric Ising ``g(E)`` it buys something subtler but real,
which the last section measures. The maths is in
:doc:`/theory/08-replica-exchange`; a compact recipe is the
:doc:`REWL example </auto_examples/plot_5_replica_exchange_ising>`.

We use ``L=8`` (as in tutorial 4), where windowing actually earns its keep.
"""

# %%
# Build the windows
# -----------------
#
# :func:`~flatwalk.make_windows` tiles the energy grid into equal-width,
# overlapping windows. REWL uses the batched callbacks (one stacked call per tick
# over all windows); the single-spin-flip physics is unchanged.

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

L = 8
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)
windows = make_windows(scheme, n_windows=4, overlap=8)
for w, (lo, hi) in enumerate(windows):
    print(f"window {w}: E ∈ [{lo:.0f}, {hi:.0f}]")

cb = ising_batched.make_batched_ising_callbacks(L)
log_g_exact = beale.log_g_E_array(L, beale.beale_g_E(L), scheme.centers)
finite = np.isfinite(log_g_exact)


def run(n_exchange, seed):
    """One windowed run; ``n_exchange=None`` disables exchange entirely."""
    rng = np.random.default_rng(seed)
    init = ising_batched.initial_states_for_windows(L, windows, rng)
    cfg = WLConfig(bin_scheme=scheme, beta=0.0, n_check=2_000, ln_f_final=1e-4)
    return RewlDriver(cfg, windows).run(
        initial_state=init,
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        n_exchange=n_exchange,
        rng=rng,
    )


def joined_eval(result):
    """Join, normalise to the exact total, return (E, log-deviation, max ε)."""
    joined, vis = join_g(result.g_windows, result.visited_windows)
    val = vis & finite & (np.exp(log_g_exact) > 0)
    n_exact = np.exp(log_g_exact[val])
    n_wl = np.exp(joined[val] - joined[val].max())
    n_wl *= n_exact.sum() / n_wl.sum()
    eps = np.abs(n_wl - n_exact) / n_exact
    eps[0] = eps[-1] = 0.0
    dev = np.log(n_wl) - log_g_exact[val]
    return scheme.centers[val], dev, eps.max()


# %%
# Windows + gluing — no exchange yet
# ----------------------------------
#
# Each window runs an *independent* confined walker (``n_exchange=None``); then
# :func:`~flatwalk.join_g` aligns neighbours over their shared bins (a constant
# log-shift) and averages the overlaps into one curve. The overlaps alone carry
# the alignment — the windows never talk to each other during the run.

noex = {0: run(n_exchange=None, seed=0)}
joined0, vis0 = join_g(noex[0].g_windows, noex[0].visited_windows)
val0 = vis0 & finite
_, _, eps0 = joined_eval(noex[0])
print(f"windowed WL (no exchange): joined max ε = {eps0:.3f}")
assert eps0 < 0.5, "gluing without exchange failed to recover g"

E0 = scheme.centers[val0]
fig, ax = plt.subplots(figsize=(6, 4))
for lo, hi in windows:
    ax.axvspan(lo, hi, color="C1", alpha=0.12)
ax.plot(
    E0, joined0[val0] - joined0[val0].min(), "o-", color="C0", label="windowed WL (joined)"
)
ax.plot(
    E0,
    log_g_exact[val0] - log_g_exact[val0].min(),
    "k--",
    lw=1.0,
    label="Beale exact",
)
ax.set_xlabel("E")
ax.set_ylabel("log g(E)   (shifted to min = 0)")
ax.set_title(f"L={L}: windows + gluing, no exchange")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# That already tracks Beale across all four windows — gluing, not exchange, is
# what reassembles ``g``.

# %%
# Adding exchange
# ---------------
#
# Now every 100 ticks the driver proposes swaps between adjacent windows. It
# recovers ``g`` just as well — exchange does not change the gluing.

rex = {0: run(n_exchange=100, seed=0)}
acc = int(rex[0].exchange_accepts.sum())
att = max(int(rex[0].exchange_attempts.sum()), 1)
_, _, eps_rex0 = joined_eval(rex[0])
print(
    f"REWL (n_exchange=100): joined max ε = {eps_rex0:.3f}, "
    f"exchange accept = {acc / att:.2f}"
)

# %%
# What exchange buys here: reproducibility
# ----------------------------------------
#
# So if both recover ``g``, why exchange? For this symmetric problem it does not
# sharpen the *typical* result — windowing already nails the shape. What it
# removes is the **run-to-run luck**. Without exchange, each window's lone walker
# explores on its own, and one unlucky window skews the joined curve; *which*
# seed goes wrong is a coin toss. Exchange keeps the windows mutually consistent,
# so every seed lands in the same place.
#
# We run three seeds each way and plot the joined deviation ``log g − log g_exact``
# for all of them.

for seed in (1, 2):
    noex[seed] = run(n_exchange=None, seed=seed)
    rex[seed] = run(n_exchange=100, seed=seed)

seeds = (0, 1, 2)
noex_eps = [joined_eval(noex[s])[2] for s in seeds]
rex_eps = [joined_eval(rex[s])[2] for s in seeds]
print(
    f"no exchange max ε per seed: {[round(e, 3) for e in noex_eps]}  "
    f"(spread {np.ptp(noex_eps):.3f})"
)
print(
    f"REWL        max ε per seed: {[round(e, 3) for e in rex_eps]}  "
    f"(spread {np.ptp(rex_eps):.3f})"
)
assert np.ptp(rex_eps) < np.ptp(noex_eps), "exchange did not reduce run-to-run spread"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for s in seeds:
    E, dev, _ = joined_eval(noex[s])
    axA.plot(E, dev, lw=1.0, label=f"seed {s}")
    E, dev, _ = joined_eval(rex[s])
    axB.plot(E, dev, lw=1.0, label=f"seed {s}")
for ax, title in (
    (axA, "Without exchange: run-to-run scatter"),
    (axB, "With exchange: reproducible"),
):
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("E")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
axA.set_ylabel("log g_joined − log g_exact")
fig.suptitle(f"L={L} Ising: exchange buys reproducibility")
fig.tight_layout()
plt.show()

# %%
# The left curves fan apart — independent windows are a lottery; the right curves
# lie on top of each other. On a *rugged*, non-symmetric system the same mixing
# also buys accuracy (a walker can otherwise sit trapped behind a barrier
# orthogonal to the energy for an entire run); here, where ``g(E)`` is symmetric,
# it shows up as reproducibility instead.
#
# That completes the journey from a single fixed-temperature Monte Carlo run to a
# robust, parallel estimate of the whole density of states. The
# :doc:`final tutorial <plot_6_thermodynamics>` turns a converged ``g(E)`` into
# the full thermodynamics — free energy, entropy, energy, and heat capacity
# across temperature.
