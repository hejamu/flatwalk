# flatwalk

A production-quality Wang-Landau (flat-histogram random walk) sampling driver
in Python, designed so the same core loop carries forward to ≥2D order
parameters and replica-exchange WL without a rewrite.

## Why flatwalk

Wang-Landau sampling estimates the density of states `g(Q)` of an arbitrary
order parameter `Q` by performing a random walk in `Q`-space biased toward
the running histogram, refining the bias factor on a schedule that
converges to the true density.

The driver here is **order-parameter agnostic** and **energy-backend
agnostic** — the user supplies four callbacks (energy, order parameter,
move proposal, initial state) and the driver handles the WL bookkeeping.

The current scope is the 1D order-parameter case validated against the 2D
Ising model. The architecture is set up so 2D `g(Q1, Q2)` sampling and
replica-exchange WL drop in additively — see the architectural notes below.

## Install

Editable install via [uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

Plain pip works too (`pip install -e ".[test]"`) but Homebrew Python may
require `--break-system-packages` or a venv.

## Quick start

```python
import numpy as np
from flatwalk import Bin1D, WLConfig, WLDriver

scheme = Bin1D(low=-128.5, high=128.5, n_bins=65)  # bins for L=8 Ising energy

def energy_fn(state):
    return state[1]                                # state = (spins, cached_E)

def order_parameter_fn(state):
    return state[1]                                # WL on E for Ising

def propose_move_fn(state, rng):
    spins, E = state
    L = spins.shape[0]
    i, j = int(rng.integers(0, L)), int(rng.integers(0, L))
    s = int(spins[i, j])
    nb_sum = int(spins[(i-1)%L, j] + spins[(i+1)%L, j] +
                 spins[i, (j-1)%L] + spins[i, (j+1)%L])
    dE = 2.0 * s * nb_sum
    new_spins = spins.copy()
    new_spins[i, j] = -s
    return (new_spins, E + dE), 0.0                # symmetric proposal

cfg = WLConfig(bin_scheme=scheme, beta=0.0, ln_f_final=1e-8,
               trace_path="trace.tsv")
driver = WLDriver(cfg)
initial_state = (np.ones((8, 8), dtype=np.int8), -128.0)
result = driver.run(initial_state, energy_fn, order_parameter_fn,
                    propose_move_fn, rng=np.random.default_rng(0))
print("log g(E):", result.g)
```

See [examples/ising_validation.py](examples/ising_validation.py) for the
full Ising L=8 validation.

## API surface

| Symbol | Purpose |
| --- | --- |
| `Bin1D`, `BinScheme` | Map order-parameter values to flat bin indices. `BinScheme` is an ABC — implement `BinND` for higher dimensions. |
| `WLConfig` | One-shot config: bin scheme, β, flatness threshold, n_check, ln_f targets, checkpoint path, trace path. |
| `WLDriver` | The sampler. `.run(...)` returns a `WLResult`. |
| `WLResult` | g, H, visited mask, bin geometry, t_total, n_f_stages, ln_f_final, converged, final state, RNG state. |
| `Walker` | Per-replica state container (state, bin_current, energy, RNG, counters). The driver loops over `Walker`s — single-walker today, multi-walker tomorrow. |
| `TraceWriter`, `TraceRow`, `read_trace` | TSV-backed per-check diagnostics (`t`, `ln_f`, flatness, acceptance rate, min/max/mean H, n_visited, 1/t-regime flag, stage index). Abstraction allows swapping to Parquet without changing callers. |
| `ExchangeHandler` | Abstract hook for replica-exchange WL. Not implemented; the driver loop already has the call site so REWL plugs in additively. |
| `save_checkpoint`, `load_checkpoint` | Atomic .npz checkpoints (`.tmp` + `os.replace`) preserving full RNG state. |

## Validation: 2D Ising

The driver is validated against the exact density of states `n(E)` for the
2D Ising model on an L×L torus, computed via a Beale-style transfer-matrix
recursion with modular CRT (see [examples/beale.py](examples/beale.py)).

[examples/ising_validation.py](examples/ising_validation.py) runs the
driver to `ln_f_final = 1e-8` on L=8 and compares against Beale:

```bash
.venv/bin/python examples/ising_validation.py --seed 0
```

Pass criteria (spec §4.4):
- `max ε(E) < 0.05` over visited central bins (excluding the two extremes).
- `mean ε(E) < 0.01`.
- `max |⟨E⟩_WL − ⟨E⟩_exact| / |⟨E⟩_exact| < 0.5%` over T ∈ [1, 4].
- C_V peak temperature within 2% of exact.

Beale's recursion is cross-validated against brute-force enumeration on
L=3 (512 configs) and L=4 (65,536 configs) in
[tests/test_beale.py](tests/test_beale.py).

A `--quick` flag runs to `ln_f_final = 1e-5` (~30 s) for smoke testing the
pipeline; the resulting `g_WL` will NOT meet the spec criteria but is
useful for development.

### Divergences from spec, and why

To meet the spec §4.4 pass criteria on L=8 (`max ε < 5%`, `mean ε < 1%`)
the validation script makes two script-level tuning choices and one
multi-run averaging choice, *none of which touch the `flatwalk` driver*:

1. **WL hyperparameters** `n_check = 1000`, `flatness_threshold = 0.95`
   (spec defaults: 10_000, 0.8). The spec marks both as "Tunable" in
   §1.5, so this is within bounds. Smaller `n_check` triggers the 1/t
   regime sooner; stricter flatness gives each f-stage more samples so
   `g[bin]` is better-equilibrated at each halving.

2. **Multi-seed averaging** (`--n-seeds 3`). A single-seed single-walker
   1/t-WL on L=8 produces a `g_WL` with ~5–10% per-bin error in the
   high-|E| tails. The asymmetry is *between* `E` and `−E` and arises
   from the trajectory: the walker reaches one tail before the other and
   accumulates more early (large-`ln_f`) updates there. Averaging the
   `log g` arrays from `K` independent seeds reduces the variance by
   `~1/K`. This is standard practice in WL literature; REWL (see
   [`flatwalk.exchange`](flatwalk/exchange.py)) is the more elegant
   solution but is out of scope for M3.

   `--n-seeds 1` recovers the pure spec interpretation ("Run the
   driver"). The driver itself is single-walker and bit-identical on a
   fixed seed.

## Architectural notes (for future extension)

These choices distinguish flatwalk from a throwaway driver. They cost ~150
extra lines of code now and save rewriting the core later.

1. **`BinScheme` abstraction.** All bin indexing goes through
   `bin_scheme.value_to_index(q)` and `bin_scheme.index_to_center(idx)`.
   Adding `Bin2D(...)` for a 2D order parameter requires no driver changes
   — just implement the abstract methods and pass the new scheme to
   `WLConfig`.

2. **Vector-typed order parameter.** `order_parameter_fn` returns
   `Union[float, np.ndarray]`. The 1D Ising case returns a float; a future
   `(E, M)` case returns a length-2 ndarray. The driver passes the value
   directly to `bin_scheme.value_to_index`.

3. **`Walker` ownership.** Per-trial state (current configuration, cached
   energy, RNG, counters) lives on a `Walker`, not on `WLDriver`. The
   per-trial logic is `WLDriver._trial_step(walker, …)`. A shared-`g`
   multi-walker variant becomes a loop over walkers rather than a rewrite.

4. **`ExchangeHandler` hook point.** The main loop already has the call
   site for REWL exchanges every `n_exchange` trials. The current build
   doesn't ship an implementation, so single-walker runs pay zero cost
   (the field is `None`).

### 2D-WL validation target

When the driver is later extended to 2D order parameters, the analogous
validation is **2D Ising in (E, M)**: Beale's recursion can be extended
to give exact `g(E, M)` for moderate L, and the same `max ε < 5%`,
`mean ε < 1%` criteria apply per bin.

### REWL validation target

When REWL is implemented, the canonical validation is to run L=8 Ising
with N_windows = 4 overlapping windows on E, exchange every N_exchange
trials, and verify the joined `g(E)` matches the single-window result
within statistical noise.

## Layout

```
flatwalk/
  binning.py        — BinScheme ABC + Bin1D
  walker.py         — Walker dataclass
  core.py           — WLConfig, WLResult, WLDriver
  exchange.py       — ExchangeHandler ABC (REWL hook)
  diagnostics.py    — TraceWriter + TraceRow + read_trace
  io.py             — save_checkpoint / load_checkpoint
tests/
  test_binning.py
  test_diagnostics.py
  test_core.py
  test_checkpoint.py
  test_beale.py
  test_ising.py
  test_imports.py
examples/
  beale.py             — Exact n(E) via transfer matrix + CRT
  ising.py             — Ising callbacks for the WL driver
  ising_validation.py  — end-to-end pass/fail run
  cache/               — Beale results cached as TSV (created on first run)
```

## License

MIT.
