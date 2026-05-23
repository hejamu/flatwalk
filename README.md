<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/_static/flatwalk-logo-dark.svg">
    <img alt="flatwalk" src="docs/src/_static/flatwalk-logo-light.svg" width="320">
  </picture>
</p>

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
- **Batched walkers** — run N walkers at once through a shared `g`, so a
  vectorised energy backend (GPU, JAX, MPI, …) evaluates them in one call per
  tick (`WLDriver.run_batched`).
- **Replica-exchange Wang-Landau** — one walker per overlapping window, each
  building its own `g`; neighbouring windows exchange configurations, and
  `join_g` stitches the per-window curves into one (`RewlDriver`,
  `make_windows`, `join_g`).
- **Checkpoint and bit-identical resume**, with the full RNG state
  preserved, for the scalar and batched drivers.
- **TSV trace writer** for offline diagnostics.
- **Validated against Beale's exact `n(E)`** on the 2D Ising L=8 torus,
  cross-checked against brute-force enumeration on L=3 and L=4; both the
  single-walker and REWL validations run in CI.

### Planned

- **Multiple walkers per window in REWL.** The shared batched trial step
  already scatters correctly into per-window `g`, so this needs only the
  walker→window map, pooled per-window flatness, and cross-window pair
  exchange in `RewlDriver`.
- **≥2D order parameters** (`BinND` alongside `Bin1D`).
- **2D Ising in (E, M)** as the exact reference for the ≥2D validation.

See [docs/src/background/storyline.md](docs/src/background/storyline.md) for the
design rationale behind these.

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
described in [docs/src/theory/10-validation.md](docs/src/theory/10-validation.md);
the short version is:

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
[docs/src/background/storyline.md](docs/src/background/storyline.md).

Both batched drivers share one trial step: `run_batched` (single shared `g`)
and `RewlDriver` (one `g` per window) are thin adapters over the same
primitive, parameterised by a walker→group map and per-walker bin bounds. That
unification — and how it makes multiple walkers per window fall out — is
written up in
[docs/src/background/design-unified-batched-step.md](docs/src/background/design-unified-batched-step.md).

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

## Related work and other Monte Carlo codes

flatwalk is a deliberately small, NumPy-only Wang-Landau *driver*: it owns the
flat-histogram bookkeeping and stays agnostic to what a configuration is and
where its energy comes from (you supply callbacks — no particle model, no
recompile). The codes below are mature and far broader; most are tied to a
specific state representation or simulation engine. They are the right tools
for production molecular and materials simulation, and useful references for
the methods flatwalk implements.

### Wang-Landau and flat-histogram

- **[OWL](https://github.com/owl-suite/OWL)** — Open-source / Oak-Ridge
  Wang-Landau. A C++ (MPI+X) suite for large-scale Wang-Landau and other
  classical/parallel MC, with first-principles energies via Quantum ESPRESSO
  or LSMS.
- **[FEASST](https://pages.nist.gov/feasst/)** — NIST's Free Energy and
  Advanced Sampling Simulation Toolkit. C++ with a Python module; Metropolis,
  Wang-Landau, and transition-matrix MC across canonical, grand-canonical, and
  Gibbs ensembles.
- **[DL_MONTE](https://gitlab.com/dl_monte)** — a general-purpose molecular MC
  code (CCP5 / Daresbury) with umbrella sampling, Wang-Landau, and
  transition-matrix free-energy methods; the companion
  [dlmontepython](https://gitlab.com/dl_monte/dlmontepython) adds automation,
  histogram reweighting, and analysis.
- **[icet / mchammer](https://icet.materialsmodeling.org)** — a Python
  cluster-expansion toolkit whose `mchammer` Monte Carlo module provides a
  `WangLandauEnsemble` alongside canonical, semi-grand-canonical, and VCSGC
  ensembles.
- **[SSAGES](https://github.com/SSAGESproject/SSAGES)** — an enhanced-sampling
  suite for LAMMPS/GROMACS/OpenMD; its Basis Function Sampling is a continuous
  Wang-Landau variant (the free energy as a projection onto orthogonal basis
  functions).

### Replica exchange / parallel tempering

- **[openmmtools](https://github.com/choderalab/openmmtools)** — a
  batteries-included toolkit on the GPU-accelerated OpenMM engine, with
  multistate samplers (`ReplicaExchangeSampler`, `ParallelTemperingSampler`)
  for temperature and Hamiltonian replica exchange.
- Replica exchange is also standard in the major MD engines —
  **[GROMACS](https://www.gromacs.org)**, **[LAMMPS](https://www.lammps.org)**,
  **[OpenMM](https://openmm.org)** — and exposed through the CV plugins below.

### Adaptive biasing on a collective variable (Wang-Landau's neighbours)

Wang-Landau is adaptive biasing on an order parameter; these bias a collective
variable instead, and are the molecular-dynamics-side analogues.

- **[PLUMED](https://www.plumed.org)** — the de facto enhanced-sampling plugin
  for MD engines: metadynamics, umbrella sampling, and many CV-based biases.
- **[Colvars](https://colvars.github.io)** — a collective-variables library
  embedded in NAMD, LAMMPS, GROMACS, VMD, and Tinker-HP; ABF, metadynamics,
  and umbrella sampling on user-defined CVs.
- **[PySAGES](https://github.com/SSAGESLabs/PySAGES)** — JAX-based, GPU/TPU
  enhanced sampling (ABF, metadynamics, forward-flux, string method) coupling
  to HOOMD-blue, LAMMPS, OpenMM, JAX-MD, and ASE.

### General-purpose molecular and materials Monte Carlo

- **[Cassandra](https://cassandra.nd.edu/)** — open-source atomistic MC
  (Maginn group, Notre Dame) for fluids and phase equilibria across
  NVT/NPT/μVT/Gibbs ensembles; a MoSDeF-Cassandra Python interface also exists.
- **[RASPA](https://github.com/iRASPA/RASPA2)** — classical MC/MD for
  adsorption and diffusion in nanoporous materials (zeolites, MOFs); GCMC and
  Gibbs-ensemble.
- **[MCCCS Towhee](https://towhee.sourceforge.net/)** — configurational-bias
  MC for fluid phase equilibria in the Gibbs ensemble, with a large built-in
  force-field library.
- **[ALPS](http://alps.comp-phys.org/)** — Algorithms and Libraries for
  Physics Simulations: classical and quantum MC for lattice models, including
  extended-ensemble methods.

### General-purpose statistical MCMC (a different problem)

Bayesian-inference samplers like **[emcee](https://emcee.readthedocs.io)** and
**[PyMC](https://www.pymc.io)** also "do Monte Carlo," but for sampling
posterior distributions rather than estimating a density of states — noted only
to head off the ambiguity.

## License

Released under the [MIT License](LICENSE).
