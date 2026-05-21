# flatwalk: storyline and architecture proposal

## 1. Premise

Wang-Landau (WL) sampling estimates the density of states `g(Q)` of an
arbitrary order parameter `Q` by random-walking in `Q`-space with a
bias refined on a flat-histogram schedule, converging to the true
density.

No existing Wang-Landau implementation is simultaneously **order-parameter
agnostic** and **energy-backend agnostic**. Mature WL tools are tightly
coupled to a particle-based state representation and a curated catalogue
of potentials; plugging in a custom Hamiltonian, a non-particle lattice
model, or a modern framework (PyTorch, JAX, …) means subclassing in the
host language and recompiling. The WL bookkeeping isn't separated from
the physics by a callable boundary — it's separated by an inheritance
hierarchy in someone else's C++.

`flatwalk` makes the cut at the right place. The bookkeeping (bins,
bias, flatness check, f-stage schedule, 1/t transition, checkpointing,
diagnostics) lives in the driver; the physics (what a configuration is,
how to flip one, what its energy and order parameter are) lives in
user-supplied callables. Five callables, no inheritance, no recompile,
no host-language constraint on `state` or the energy backend.

This document describes:

- what's in place today and why those choices were made (§3),
- the central design commitment that makes everything else possible:
  **batched walkers as the canonical path for ≥2 walkers** (§4),
- how replica-exchange WL (§5) and ≥2D order parameters (§6) drop in
  additively on top of that commitment, and
- the validation targets and effort estimates (§7).

## 2. The cut at callable boundaries

The driver's contract with user code is five things:

| You supply | Type | What the driver does with it |
| --- | --- | --- |
| `bin_scheme` | `BinScheme` instance | maps `Q → bin index` |
| `initial_state` | anything (opaque) | passed through to your callbacks |
| `energy_fn(state)` | `→ float` | the `−β·ΔE` term in WL acceptance |
| `order_parameter_fn(state)` | `→ float \| np.ndarray` | the quantity `g(Q)` is estimated over |
| `propose_move_fn(state, rng)` | `→ (new_state, log_proposal_ratio)` | one Markov step |

`state` is *never inspected* by the driver. The driver only:
1. Calls callbacks with whatever you passed in.
2. Calls `bin_scheme.value_to_index(q)` on the value returned by
   `order_parameter_fn`.
3. Maintains `g`, `H`, the f-stage schedule, the 1/t transition.

That's it. The energy backend can be NumPy, PyTorch, JAX, a Fortran
kernel via `ctypes`, a remote queue, a learned surrogate — the driver
doesn't know and doesn't need to.

## 3. What's in place today

The single-walker driver has shipped through M1–M4 with the spec
validation passing on L=8 Ising. The architectural pieces below were
chosen specifically to make the next-step extensions (batched walkers,
REWL, ≥2D OP) *additive* rather than rewrites.

| Piece | What it is | Why it matters for what's next |
| --- | --- | --- |
| `BinScheme` ABC + `Bin1D` | Abstract mapping `Q → bin index` | A vectorized `value_to_index_batched` is a one-method addition; a `BinND` for ≥2D OP slots in as a sibling. |
| `WLConfig` / `WLResult` | Plain dataclasses | Add fields without breaking call sites. |
| `WLDriver.run(…)` | The main loop with 1/t-WL transition, checkpointing, periodic diagnostics | The trial logic is factored into `_trial_step(walker, …)`. A batched variant `_trial_step_batched(walker_batch, …)` is additive; the existing scalar path stays. |
| `Walker` | Per-replica state container (state, bin_current, energy, RNG, counters) | Already separated from the driver. A `WalkerBatch` is its N-walker generalisation. |
| `ExchangeHandler` ABC + call site | Empty hook fired every `n_exchange` trials | REWL drops in as a concrete handler — no change to the loop. |
| `TraceWriter` / `TraceRow` | Per-check TSV diagnostic writer | Used by `WLConfig.trace_path`; offline analysis with grep / awk / pandas without re-running. |
| Beale exact validation | Modular-CRT transfer-matrix recursion for `n(E)` on the 2D Ising torus, cross-validated against brute-force enumeration on L≤4 | A sharp correctness test for the driver. Same machinery will validate the batched path against the scalar path bit-for-bit on small systems. |

Code is ~2k lines of pure Python, 80 tests, CI on three Python
versions plus the slow Ising L=8 validation in its own job.

## 4. The central commitment: batched walkers

> **Design rule.** For ≥2 walkers, the per-tick primitives operate on
> *N walker states at once*, not on one walker at a time. There is no
> `for w in walkers: …` in the inner loop. Anywhere.

This is the performance decision. The whole reason a user wires
`flatwalk` to PyTorch instead of an in-tree LJ pair potential is that
they want one GPU call to evaluate N stacked configurations. A loop
over walkers makes that single GPU call N times, throws away the
batching opportunity, and reduces a planned 100× speedup to 1×.

### The batched primitives

A new `BatchedCallbacks` API, parallel to the existing scalar one:

| Scalar (today) | Batched (proposed) |
| --- | --- |
| `energy_fn(state) → float` | `energy_fn(state_batch) → ndarray[N]` |
| `order_parameter_fn(state) → float \| ndarray[D]` | `order_parameter_fn(state_batch) → ndarray[N]` or `ndarray[N, D]` |
| `propose_move_fn(state, rng) → (new_state, lpr)` | `propose_move_fn(state_batch, rng) → (new_state_batch, lpr_batch[N])` |

`state_batch` is opaque exactly the same way `state` is opaque today —
whatever the user's batched callable accepts as input. For torch-CPM
it's a stacked tensor on the GPU; for a polymer chain it might be a
list of N graphs; for an MNIST-like state space it might be `ndarray[N,
28, 28]`. The driver never inspects it.

### `WalkerBatch` (the N-walker generalisation of `Walker`)

```text
WalkerBatch:
  state          # whatever the user's batched callables understand
  bin_current    # ndarray[N], int
  energy         # ndarray[N], float
  rng            # one Generator; vectorized draws shared across walkers
  n_attempted    # ndarray[N], int (per-walker counters)
  n_accepted     # ndarray[N], int
```

The `for w in walkers:` anti-pattern is structurally absent: there is
no Python-side list of `Walker` instances to iterate. There is one
`WalkerBatch` carrying N walkers' worth of state in flat arrays.

### `Bin1D.value_to_index_batched`

A one-method extension on `Bin1D` (and any future `BinND`):
`value_to_index_batched(q[N]) → idx[N]`. Vectorized in NumPy; ~10
lines. Out-of-range entries get a sentinel index (e.g. `-1`) and the
trial step's `in_range` check becomes a boolean mask.

### The batched trial step

Per tick, the inner loop is *six vectorised ops, no Python loop*:

```text
new_state_batch, lpr           = propose_move_fn(state_batch, rng)   # 1 call
q_new                          = order_parameter_fn(new_state_batch) # 1 call
in_range                       = bin_scheme.in_range_batched(q_new)
bin_new                        = bin_scheme.value_to_index_batched(q_new)  # masked
e_new                          = energy_fn(new_state_batch) if β ≠ 0 else 0  # 1 call
Δ                              = -β·(e_new − energy) + g[bin_cur] - g[bin_new] + lpr
accept                         = in_range & ((Δ ≥ 0) | (random[N] < exp(Δ)))
# Vectorized in-place update of WalkerBatch on `accept`:
state_batch[accept]            = new_state_batch[accept]    # or torch.where(...)
bin_current[accept]            = bin_new[accept]
energy[accept]                 = e_new[accept]
# Scatter add into g/H at the post-trial bin (np.add.at is the canonical
# safe scatter for repeated indices):
np.add.at(g, bin_current, ln_f)
np.add.at(H, bin_current, 1)
```

For *shared g* (single-window multi-walker WL), all N walkers
contribute to the same `g` array via the scatter add; `np.add.at`
handles repeated indices correctly. For *per-window g* (the REWL
case below), each walker has its own `g_window` and the scatter is
into the right window's array.

### `WLDriver.run` dispatch

`run(…)` looks at the callable shape (or a `batched=True` flag, or a
separate `WLDriver.run_batched(…)` method — small surface choice; the
mechanics are the same). For a single walker with scalar callbacks
the existing path runs unchanged. For ≥2 walkers, the user passes
`BatchedCallbacks` and the driver builds a `WalkerBatch` and runs the
batched loop.

The scalar path is *not* deprecated. For systems where each
configuration is heavy but the energy evaluation is cheap (e.g. a
Python-side custom Monte Carlo move with N=1), the scalar path remains
the right answer. Batching is the right answer when the energy backend
itself wants to be batched.

## 5. Replica-exchange WL on top of batched walkers

REWL solves most of the practical problems we hit in the single-walker
validation (Z₂-asymmetry from a single seed, slow convergence in the
tails, no good way to share information across the energy range).
With batched walkers it's a small addition.

### `ReplicaExchangeHandler`

The existing `ExchangeHandler` ABC and the driver's call site already
fit; we add a concrete handler:

- N windows, each with its own `Bin1D(low_w, high_w, n_bins_w)` and
  its own `g_window[w]`. The `WalkerBatch` carries one walker per
  window (or k walkers per window for further parallelism).
- Every `n_exchange` ticks, the handler proposes swaps between
  adjacent windows using the standard Metropolis criterion:
  ```text
  Δ = g_i(E_j) − g_i(E_i) + g_j(E_i) − g_j(E_j)
  accept ⇔ U < exp(min(0, Δ))
  ```
  Evaluated *batched* over all adjacent pairs (alternating even/odd
  offsets per call to satisfy detailed balance), so the handler is
  three numpy ops, not a loop.
- Accepted swaps permute the state indices in the `WalkerBatch`
  in place. Per-walker `bin_current`, `energy`, and counters are
  permuted likewise. There is *still* no Python loop over walkers.

### `join_g` post-processing

After the run, the per-window log-`g` arrays are joined in their
overlap regions via least-squares shift on the log scale, stitched
into the full range. ~50 lines, runs in milliseconds.

### Validation target

L=8 Ising with N=4 windows on E with explicit overlap. Run to
`ln_f_final = 1e-8`. Join `g(E)` across windows. Compare to Beale
exact. Pass criteria mirror §4.4: `max ε < 5%`, `mean ε < 1%`. Tests
include a bit-for-bit equivalence check with the scalar single-walker
path on a tiny synthetic system (no windowing, N=1, batched mode) so
the batched primitives are themselves verified.

## 6. ≥2D order parameters on top of batched walkers

A future Paper 2 target (joint `(Q_charge, N_adsorbed)` sampling) is
essentially free once §4 is in place:

- `order_parameter_fn(state_batch)` returns `ndarray[N, 2]` instead of
  `ndarray[N]`.
- `BinND(low[D], high[D], n_bins[D])` is a sibling of `Bin1D`. Its
  `value_to_index_batched(q[N, D])` flattens a D-D bin index into a
  single int per walker. `g` and `H` stay 1D ndarrays in the driver —
  the driver never knows the order parameter has higher dimensionality.
- Validation: 2D Ising in `(E, M)`. Beale's recursion can be extended
  to give exact `g(E, M)` for moderate L. Same pass criteria per bin.

Estimated cost: ~100 lines for `BinND` + tests, ~80 lines for the
2D-Ising Beale extension, ~50 lines for the validation script.

## 7. Roadmap and effort

Step-by-step, all additive (no rewrites):

| Step | Adds | Notes |
| --- | --- | --- |
| `Bin1D.value_to_index_batched` + `in_range_batched` | ~15 lines + tests | Trivial. |
| `BatchedCallbacks` type aliases + signature documentation | ~30 lines | Mirrors the existing `EnergyFn` / `OrderParamFn` / `ProposeMoveFn` aliases. |
| `WalkerBatch` dataclass | ~80 lines + tests | Same factoring as `Walker`. |
| `WLDriver` batched trial step + dispatch in `run` | ~150 lines + tests | The existing scalar path stays untouched. |
| `ReplicaExchangeHandler` + `join_g` helper | ~200 lines + tests | Uses the existing ABC + call site. |
| `BinND` | ~100 lines + tests | When the first ≥2D-OP project hits. |
| 2D Ising Beale extension + REWL/2D validation scripts | ~250 lines | Mirrors `examples/beale.py` and `examples/ising_validation.py`. |
| **Total** | **~825 lines of code + ~400 lines of tests** | ~1 week of focused work for the batched + REWL capability; +1–2 days for 2D OP. |

The existing ~2k lines stay as they are. There is no flag day, no
deprecation, no breaking change to the scalar single-walker API. The
batched path is the canonical answer for ≥2 walkers, and the scalar
path remains the canonical answer for one.

## 8. Why this matters

- **Modern frameworks become a one-line callback.** A PyTorch model
  evaluating energy for a batch of configurations is exactly what
  `BatchedCallbacks.energy_fn` is for. The driver doesn't need to know
  about CUDA, autograd, or device placement.
- **REWL becomes ordinary.** REWL is the practical answer to most of
  WL's known pathologies (Z₂-asymmetry, slow tail convergence,
  single-window saturation). With batched walkers it's small and
  testable.
- **≥2D order parameters become a binning change, not a driver
  change.** The same proven core loop estimates `g(Q1)`, `g(Q1, Q2)`,
  `g(Q1, …, Qk)`, with the only swap being `Bin1D` → `BinND`.
- **The validation lever stays sharp.** Beale's exact `n(E)` (and its
  extension to `n(E, M)`) gives a precise correctness test for every
  step of the roadmap, on a system where the right answer is known.

The design rule from §4 is what holds it all together: ≥2 walkers
move through the system as a batch, not as a loop. Everything else
follows from that.
