"""
Diagnosing a run from its trace
===============================

Set ``trace_path`` and flatwalk writes one TSV row per flatness check: the
current ``ln_f``, the histogram flatness, the acceptance rate, the spread of
``H`` over visited bins, and whether the 1/t regime has kicked in. Reading it
back with :func:`~flatwalk.read_trace` lets you see *how* a run converged —
when each f-stage ended and when standard halving handed off to the 1/t
schedule (the subject of :doc:`/theory/06-one-over-t`).
"""

# %%
# Run with a trace
# ----------------

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

from flatwalk import Bin1D, WLConfig, WLDriver, read_trace  # noqa: E402

L = 4
cb = ising.make_ising_callbacks(L)
low, high, n_bins = ising.ising_energy_bins(L)
scheme = Bin1D(low, high, n_bins)

with tempfile.TemporaryDirectory() as tmp:
    trace_path = Path(tmp) / "trace.tsv"
    cfg = WLConfig(
        bin_scheme=scheme,
        beta=0.0,
        n_check=1_000,
        ln_f_final=1e-5,
        trace_path=trace_path,
    )
    result = WLDriver(cfg).run(
        initial_state=ising.random_state(L, np.random.default_rng(0)),
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        rng=np.random.default_rng(0),
    )
    rows = read_trace(trace_path)

print(f"{len(rows)} trace rows over {result.t_total:,} trials")

# %%
# Plot ln_f over time, marking the 1/t handoff
# --------------------------------------------
#
# ``ln_f`` falls by halving in the standard regime, then follows ``1/t`` once
# the schedule switches — visible as the change of slope on the log axis. The
# shaded span marks the 1/t regime.

t = np.array([r.t for r in rows])
ln_f = np.array([r.ln_f for r in rows])
in_1overt = np.array([r.in_1overt for r in rows])

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(t, ln_f, "o-", color="C0", ms=3, label="ln f")
if in_1overt.any():
    ax.axvspan(t[in_1overt].min(), t.max(), color="C1", alpha=0.12, label="1/t regime")
ax.set_xlabel("trials t")
ax.set_ylabel("ln f")
ax.set_title(f"L={L} Ising: the modification factor over a run")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
plt.show()

# %%
# Every column of :class:`~flatwalk.TraceRow` (flatness, acceptance rate,
# ``min``/``mean``/``max`` of ``H`` over visited bins, ``n_visited``,
# ``stage_index``) is available the same way — enough to diagnose a stalled run
# offline without re-running it.
