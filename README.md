# flatwalk

[![CI](https://github.com/hejamu/flatwalk/actions/workflows/ci.yml/badge.svg)](https://github.com/hejamu/flatwalk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

flatwalk is an enhanced sampling library implementing flat-histogram
methods while being order-parameter and energy-backend agnostic.
flatwalk does the sampling, the user provides the system to sample.
The contract between flatwalk and the user is the following:

| You supply | Type | What flatwalk does with it |
| --- | --- | --- |
| `bin_scheme` | `BinScheme` instance | maps `Q → bin index` |
| `energy_fn(state)` | `→ float` | the `−β·ΔE` term in WL acceptance (skip when `β=0` and `Q=E`) |
| `order_parameter_fn(state)` | `→ float \| np.ndarray` | the quantity `g(Q)` is estimated over (vector for ≥2D) |
| `propose_move_fn(state, rng)` | `→ (new_state, log_proposal_ratio)` | one Markov step |

`state` is opaque to flatwalk — whatever your callbacks recognise:
tuple, dataclass, numpy array, torch tensor, anything. You hand one
initial `state` object to `driver.run(...)` to start; from there the
callbacks do all state manipulation.

## Capabilities

### Implemented

- **Single-walker Wang-Landau** on a 1D order parameter, with the
  Belardinelli-Pereyra 1/t-WL transition (`WLDriver.run`).
- **Batched walkers** — N walkers advanced through a shared `g` in one
  stacked callback call per tick, never a Python loop over walkers
  (`WLDriver.run_batched`). This is the path by which a GPU energy
  backend (PyTorch, JAX) gets its speedup: one forward pass per tick,
  not N sequential ones.
- **Replica-exchange Wang-Landau** — W overlapping windows, one walker
  each, with batched entropy-based exchange and a `join_g` step that
  stitches the per-window `g` into a single curve (`RewlDriver`,
  `make_windows`, `join_g`).
- **Atomic checkpoint and bit-identical resume** (full RNG state
  preserved) for the scalar and batched drivers.
- **TSV trace writer** for offline diagnostics.
- **Validated against Beale's exact `n(E)`** on the 2D Ising L=8 torus,
  cross-checked against brute-force enumeration on L=3 and L=4; both the
  single-walker and REWL validations run in CI.

### Planned

See [docs/src/storyline.md](docs/src/storyline.md) for design rationale.

- **≥2D order parameters.** `BinND` as a sibling of `Bin1D`; the
  driver's `g` stays a flat 1D ndarray, only the binning changes.
- **2D Ising in (E, M) Beale extension.** Exact reference for the
  multi-D order-parameter validation.

## Install

Editable install via [uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[test]"
```

Plain pip works too (`pip install -e ".[test]"`) but Homebrew Python may
require `--break-system-packages` or a venv.

## Quick start

Below, block 1 fills the contract above for the 2D Ising model;
block 2 is the `flatwalk` setup and run — verbatim across systems.

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

To run a different model you'd replace block 1 only (your callbacks,
`initial_state`, and the `Bin1D` range for your `Q`); block 2 stays
verbatim. See [examples/ising.py](examples/ising.py) for the
production Ising implementation used by the validation, and
[examples/ising_validation.py](examples/ising_validation.py) for the
full pass/fail run.

## Documentation

Full documentation lives in [docs/](docs/) as a Sphinx site: a getting-started
guide, the API reference, the design storyline, and a **runnable example
gallery** that walks the methods as tutorials — a toy first run, the exact
Ising reference, single-walker Wang-Landau, then replica exchange. It is not
yet hosted; build it locally with:

```bash
uv pip install --python .venv/bin/python -e ".[docs]"
tox -e docs        # writes docs/build/html
```

Then open `docs/build/html/index.html`.

## API surface

| Symbol | Purpose |
| --- | --- |
| `Bin1D`, `BinScheme` | Map order-parameter values to flat bin indices. `BinScheme` is an ABC — implement `BinND` for higher dimensions. |
| `WLConfig` | One-shot config: bin scheme, β, flatness threshold, n_check, ln_f targets, checkpoint path, trace path. |
| `WLDriver` | The sampler. `.run(...)` runs one walker; `.run_batched(..., n_walkers=N)` runs N walkers through a shared `g`. Both return a `WLResult`. |
| `WLResult` | g, H, visited mask, bin geometry, t_total, n_f_stages, ln_f_final, converged, final state, RNG state. |
| `Walker`, `WalkerBatch` | Per-walker state container. `Walker` holds one walker; `WalkerBatch` carries N walkers in stacked arrays for the batched and REWL paths. |
| `BatchedEnergyFn`, `BatchedOrderParamFn`, `BatchedProposeMoveFn` | Type aliases for the stacked (N-at-once) callbacks consumed by `run_batched` and `RewlDriver`. |
| `RewlDriver`, `ReplicaExchangeHandler`, `make_windows`, `join_g`, `RewlResult` | Replica-exchange WL: build overlapping windows, run one walker per window with batched exchange, then stitch the per-window `g` into one curve. |
| `TraceWriter`, `TraceRow`, `read_trace` | TSV-backed per-check diagnostics (`t`, `ln_f`, flatness, acceptance rate, min/max/mean H, n_visited, 1/t-regime flag, stage index). Abstraction allows swapping to Parquet without changing callers. |
| `ExchangeHandler`, `ExchangeResult` | Abstract hook for shared-`g` exchange inside `run_batched` (the `exchange_handler` argument); not yet wired. REWL itself ships as `RewlDriver` above. |
| `save_checkpoint`, `load_checkpoint` | Atomic .npz checkpoints (`.tmp` + `os.replace`) preserving full RNG state. |

## Validation: 2D Ising

The driver is validated end-to-end against the exact density of states
`n(E)` for the 2D Ising model on an L×L periodic lattice, computed via a
Beale-style transfer-matrix recursion. The full methodology, pass
criteria, and the script-level tuning choices used to meet them are
described in [docs/src/validation.md](docs/src/validation.md); the
short version is:

```bash
.venv/bin/python examples/ising_validation.py --seed 0
```

Runs 3 independent seeds to `ln_f_final = 1e-8`, averages the per-seed
`log g`, compares to Beale's exact `n(E)`, and exits 0 only if all four
spec §4.4 criteria pass (`max ε < 5%`, `mean ε < 1%`, `‹E›(T)` agreement
within 0.5%, `C_V` peak temperature within 2%). The slow lane of CI
runs this on every push.

The replica-exchange path has its own end-to-end check,
[examples/ising_rewl_validation.py](examples/ising_rewl_validation.py): L=8
with overlapping windows on E, exchanged periodically, with the joined `g(E)`
held to the same criteria. It runs in the same slow CI lane.

## Design and roadmap

flatwalk is deliberately built so the unbuilt pieces drop in without rewriting
the core: bin indexing is behind the `BinScheme` ABC (so a future `BinND` is
additive), the order parameter is vector-typed (so an `(E, M)` parameter needs
no driver change), and per-walker state lives on `Walker`/`WalkerBatch` rather
than on the driver. The full rationale, the batched-walker design, and the
validation targets for the planned ≥2D / `(E, M)` work are in
[docs/src/storyline.md](docs/src/storyline.md).

## Layout

```
flatwalk/             — the package
  binning.py            BinScheme ABC + Bin1D
  walker.py             Walker + WalkerBatch state containers
  core.py               WLConfig, WLResult, WLDriver (.run / .run_batched)
  exchange.py           ExchangeHandler ABC (shared-g exchange hook)
  rewl.py               RewlDriver, ReplicaExchangeHandler, make_windows, join_g
  diagnostics.py        TraceWriter + TraceRow + read_trace
  io.py                 save_checkpoint / load_checkpoint
tests/                — pytest suite (one module per package module)
examples/             — user-side code that fills the contract
  beale.py              Exact n(E) via transfer matrix + CRT
  ising.py              Ising callbacks for the WL driver
  ising_batched.py      Batched-walker Ising run
  ising_validation.py   Single-walker end-to-end pass/fail run
  ising_rewl_validation.py  Replica-exchange end-to-end pass/fail run
  cache/                Beale results cached as TSV (created on first run)
docs/src/             — Sphinx docs source (guide, gallery, API, storyline)
tox.ini               — tests / lint / format / docs / build envs
```

## License

Released under the [MIT License](LICENSE).
