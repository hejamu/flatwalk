"""
The minimal contract: a 1D random walk
=======================================

The smallest complete flatwalk program — a template to copy. It fills the
four-callback contract for a toy whose answer we know, so the wiring stands out
with no physics in the way.

A walker hops ``±1`` on the integers ``0..n-1``. Every position is one
configuration, so the true density of states is **flat**: ``g(Q)`` is identical
in every bin, and a correct sampler must recover a constant.

For a narrative walk-through of *why* this works, start the
:doc:`tutorials </auto_tutorials/index>`; for the acceptance rule it relies on,
see :doc:`/theory/05-wang-landau`.
"""

# %%
# Block 1 — your physics
# ----------------------
#
# Replace this block to sample a different system; ``flatwalk`` never inspects
# ``state``. Here ``state`` is just the integer position.

import matplotlib.pyplot as plt
import numpy as np

from flatwalk import Bin1D, WLConfig, WLDriver

n_states = 20
scheme = Bin1D(-0.5, n_states - 0.5, n_states)  # one bin per integer position


def energy_fn(state):
    return 0.0  # beta = 0 below, so this never enters acceptance


def order_parameter_fn(state):
    return float(state)  # Q = the walker position


def propose_move_fn(state, rng):
    step = 1 if rng.random() < 0.5 else -1
    return state + step, 0.0  # symmetric move → log_proposal_ratio = 0


initial_state = n_states // 2

# %%
# Block 2 — generic flatwalk wiring (unchanged across systems)
# ------------------------------------------------------------
#
# Out-of-range proposals (off either end) are rejected, so the walk reflects at
# the boundaries and every position stays reachable.

cfg = WLConfig(bin_scheme=scheme, beta=0.0, n_check=2_000, ln_f_final=1e-6)
result = WLDriver(cfg).run(
    initial_state=initial_state,
    energy_fn=energy_fn,
    order_parameter_fn=order_parameter_fn,
    propose_move_fn=propose_move_fn,
    rng=np.random.default_rng(0),
)
print(
    f"{result.t_total:,} trials, {result.n_f_stages} f-stages, converged={result.converged}"
)

# %%
# Read off the result
# -------------------
#
# ``result.g`` is the **log** density of states, known up to an additive
# constant; we shift it to mean zero. The exact answer is a flat line, so the
# spread of the recovered ``g`` is the whole story.

g = result.g - result.g.mean()
print(f"g spread (max - min) = {g.max() - g.min():.3f}  (exact: 0)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.axhline(0.0, color="k", ls="--", lw=1.0, label="exact (flat)")
ax.plot(scheme.centers, g, "o-", color="C0", label="Wang-Landau")
ax.set_xlabel("position Q")
ax.set_ylabel("log g(Q)   (shifted to mean 0)")
ax.set_title("1D random walk: a flat density of states")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# To use this on your own system, change **block 1** only — the bin scheme, the
# three callbacks, and ``initial_state``. The
# :doc:`single-walker Ising recipe <plot_3_single_walker_ising>` does exactly
# that for a real Hamiltonian.
