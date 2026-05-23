"""
Checkpoint and bit-identical resume
===================================

Long flat-histogram runs need to survive interruption. flatwalk checkpoints
the *entire* driver state — ``g``, ``H``, the f-stage schedule, the walker, and
the full RNG state — so a run that is stopped and resumed from disk reproduces
an uninterrupted run **bit for bit**. This recipe demonstrates that guarantee
on the Ising model.

Set ``checkpoint_path`` (and optionally ``checkpoint_every_t``) to write
periodic checkpoints; pass ``resume_from`` to continue from one. On resume the
RNG is restored from the checkpoint, so you do *not* pass ``rng``.
"""

# %%
# Setup
# -----

import sys
import tempfile

try:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
except NameError:
    pass  # sphinx-gallery exec context: __file__ undefined, sys.path is already set

import ising  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flatwalk import Bin1D, WLConfig, WLDriver  # noqa: E402

L = 4
cb = ising.make_ising_callbacks(L)
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)

N = 8_000  # total moves
seed = 12345


def fresh_state():
    return ising.random_state(L, np.random.default_rng(seed))


# %%
# Run A — uninterrupted, N moves
# ------------------------------
#
# We cap the run with ``max_trials`` and set ``ln_f_final`` far below reach so
# the comparison is over an identical number of moves.

cfg_a = WLConfig(bin_scheme=scheme, beta=0.0, n_check=500, ln_f_final=1e-30)
result_a = WLDriver(cfg_a).run(
    initial_state=fresh_state(),
    energy_fn=cb["energy_fn"],
    order_parameter_fn=cb["order_parameter_fn"],
    propose_move_fn=cb["propose_move_fn"],
    rng=np.random.default_rng(seed),
    max_trials=N,
)

# %%
# Run B — stop at N/2, checkpoint, resume to N
# --------------------------------------------
#
# Part 1 runs the same seed for ``N/2`` moves and writes a checkpoint at exactly
# that point. Part 2 resumes from disk (no ``rng`` argument) and runs to ``N``.

with tempfile.TemporaryDirectory() as tmp:
    cp = Path(tmp) / "checkpoint.npz"

    cfg_b1 = WLConfig(
        bin_scheme=scheme,
        beta=0.0,
        n_check=500,
        ln_f_final=1e-30,
        checkpoint_path=cp,
        checkpoint_every_t=N // 2,
    )
    WLDriver(cfg_b1).run(
        initial_state=fresh_state(),
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        rng=np.random.default_rng(seed),
        max_trials=N // 2,
    )

    cfg_b2 = WLConfig(bin_scheme=scheme, beta=0.0, n_check=500, ln_f_final=1e-30)
    result_b = WLDriver(cfg_b2).run(
        initial_state=None,  # ignored on resume
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        resume_from=cp,
        max_trials=N,
    )

# %%
# The resumed run equals the uninterrupted run, exactly
# -----------------------------------------------------

max_g_diff = np.abs(result_a.g - result_b.g).max()
print(f"max |g_uninterrupted - g_resumed| = {max_g_diff:.3e}")
np.testing.assert_array_equal(result_a.g, result_b.g)
np.testing.assert_array_equal(result_a.H, result_b.H)
print("g and H are bit-identical.")

valid = result_a.visited
E = scheme.centers[valid]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    E, result_a.g[valid] - result_a.g[valid].min(), "o", color="C0", label="uninterrupted"
)
ax.plot(
    E,
    result_b.g[valid] - result_b.g[valid].min(),
    "x",
    color="C3",
    label="checkpoint → resume",
)
ax.set_xlabel("E")
ax.set_ylabel("log g(E)   (shifted to min = 0)")
ax.set_title(f"L={L} Ising: resume reproduces the run bit-for-bit")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
