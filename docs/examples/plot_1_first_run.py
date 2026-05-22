"""
A first flat-histogram run: a 1D random walk
============================================

Before any physics, this is the smallest possible Wang-Landau run — enough
to meet the four-callback contract and see what ``g`` means.

We sample a walker that hops ``±1`` on the integers ``0..n-1``. Every
position is a single "configuration", so the true density of states is
**flat**: ``g(Q)`` is the same for every bin. A correct flat-histogram
sampler must recover that constant — a result we can eyeball.
"""

# %%
# The four-callback contract
# --------------------------
#
# ``flatwalk`` never inspects your state; it only calls back into your code.
# You supply four things:
#
# * a ``bin_scheme`` mapping the order parameter ``Q`` to a bin index,
# * ``energy_fn(state)`` — here identically zero (``beta = 0``, so the
#   energy term drops out of acceptance),
# * ``order_parameter_fn(state)`` — the quantity ``g`` is estimated over;
#   here the walker position itself,
# * ``propose_move_fn(state, rng)`` returning ``(new_state, log_proposal_ratio)``.

import matplotlib.pyplot as plt
import numpy as np

from flatwalk import Bin1D, WLConfig, WLDriver

n_states = 20
scheme = Bin1D(-0.5, n_states - 0.5, n_states)  # one bin per integer position


def energy_fn(state):
    return 0.0


def order_parameter_fn(state):
    return float(state)


def propose_move_fn(state, rng):
    step = 1 if rng.random() < 0.5 else -1
    return state + step, 0.0  # symmetric move → log_proposal_ratio = 0


# %%
# Run Wang-Landau
# ---------------
#
# Out-of-range proposals (off either end) are simply rejected, so the walk
# reflects at the boundaries and every position stays reachable.

cfg = WLConfig(
    bin_scheme=scheme,
    beta=0.0,
    flatness_threshold=0.8,
    n_check=2_000,
    ln_f_initial=1.0,
    ln_f_final=1e-6,
)
result = WLDriver(cfg).run(
    initial_state=n_states // 2,
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
# ``result.g`` is the log density of states, known up to an additive
# constant; we shift it to mean zero. The true answer is a flat line, so the
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
# That flat line is the entire point of flat-histogram sampling: the bias
# ``g`` builds up exactly enough to cancel the system's own entropy, so the
# walker spends equal time in every bin. The Ising tutorials that follow
# swap the toy walk for a real Hamiltonian — the contract is identical.
