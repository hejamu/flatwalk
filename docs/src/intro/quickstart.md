# Quickstart

Below, **block 1** fills the {doc}`four-callback contract <the-contract>` for
the 2D Ising model; **block 2** is the flatwalk setup and run — verbatim across
systems. To run a different model you replace block 1 only (your callbacks,
`initial_state`, and the {class}`~flatwalk.Bin1D` range for your `Q`); block 2
stays as it is.

```python
import numpy as np
from flatwalk import Bin1D, WLConfig, WLDriver

# ──────────────────────────────────────────────────────────────────
# 1. Your physics — replace this block to use a different system.
#    flatwalk doesn't know or care what `state` is.
# ──────────────────────────────────────────────────────────────────
L = 8

def energy_fn(state):
    return state[1]                                # cached E, O(1)

def order_parameter_fn(state):
    return state[1]                                # WL on E: Q = E

def propose_move_fn(state, rng):                   # single-spin flip
    spins, E = state
    i, j = int(rng.integers(0, L)), int(rng.integers(0, L))
    s = int(spins[i, j])
    nb_sum = int(spins[(i-1)%L, j] + spins[(i+1)%L, j] +
                 spins[i, (j-1)%L] + spins[i, (j+1)%L])
    dE = 2.0 * s * nb_sum                          # ΔE in O(1)
    new_spins = spins.copy(); new_spins[i, j] = -s
    return (new_spins, E + dE), 0.0                # symmetric → lpr = 0

initial_state = (np.ones((L, L), dtype=np.int8), -2.0 * L * L)
bin_scheme = Bin1D(low=-2*L*L - 2, high=2*L*L + 2, n_bins=L*L + 1)

# ──────────────────────────────────────────────────────────────────
# 2. Generic flatwalk wiring — unchanged across systems.
# ──────────────────────────────────────────────────────────────────
cfg = WLConfig(bin_scheme=bin_scheme, beta=0.0, ln_f_final=1e-8,
               trace_path="trace.tsv")
result = WLDriver(cfg).run(
    initial_state, energy_fn, order_parameter_fn, propose_move_fn,
    rng=np.random.default_rng(0),
)
print(result.g)                                    # log density of states
```

Two choices in block 1 are worth calling out, because they recur in every
flatwalk system:

- **The state carries the cached energy** as `(spins, E)`. A single-spin flip
  changes the energy by a local `ΔE`, so keeping `E` in the state makes
  `energy_fn` and `order_parameter_fn` O(1) instead of O(L²). The driver treats
  `state` as opaque and never looks inside.
- **The order parameter *is* the energy** ("WL on E"), and `beta = 0`. The
  acceptance `Δ = −β·(E_new − E_old) + g[bin_old] − g[bin_new]` then collapses
  to `g[bin_old] − g[bin_new]`.

See [`examples/ising.py`](../../../examples/ising.py) for the production Ising
implementation used by the validation, and
[`examples/ising_validation.py`](../../../examples/ising_validation.py) for the
full pass/fail run.

```{seealso}
The {doc}`first tutorial <../auto_tutorials/index>` runs this very system
step by step, starting from plain Monte Carlo to motivate why Wang-Landau is
needed at all.
```
