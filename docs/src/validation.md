# Validation

A flat-histogram sampler is only as trustworthy as the density of states
it returns, so `flatwalk` is validated the direct way: run it on systems
whose `g(Q)` is known exactly, and compare bin for bin.

## The exact-reference lever

The 2D Ising model on an `L×L` periodic lattice has an exactly computable
integer density of states `n(E)`. [`examples/beale.py`](../../examples/beale.py)
obtains it from a Beale-style transfer-matrix recursion (the trace of
`T^L` evaluated modulo several primes and reconstructed by the Chinese
Remainder Theorem, which keeps the `~10^19` coefficients exact where naive
integer arithmetic would not). On small lattices the recursion is itself
cross-checked against brute-force enumeration, so the reference is sound
before the driver leans on it at `L=8`.

This gives a sharp target: Wang-Landau sampling `g(E)` on the same lattice
should reproduce `n(E)` across the whole spectrum.

## What counts as a pass

The spec §4.4 criteria, evaluated over visited, non-gap central bins
(the two extreme `E = ±2L²` bins are excluded — there `n = 2` and relative
noise dominates):

| Metric | Pass |
| --- | --- |
| `max ε(E)` per bin | < 0.05 |
| `mean ε(E)` per bin | < 0.01 |
| `max |⟨E⟩_WL − ⟨E⟩_exact| / |⟨E⟩_exact|`, `T ∈ [1, 4]` | < 0.5% |
| `C_V(T)` peak-temperature error | < 2% |

The full `ln_f_final = 1e-8` `L=8` runs that meet these criteria are too
slow for a docs build (~15 min single-walker; longer for REWL). They live
at the repo root —
[`examples/ising_validation.py`](../../examples/ising_validation.py) and
[`examples/ising_rewl_validation.py`](../../examples/ising_rewl_validation.py) —
and run in CI's slow lane. The {doc}`gallery <auto_examples/index>` below
runs fast smoke versions of the same pipelines on every build.

## Reproducibility

Two driver-correctness properties are pinned in the test suite rather than
the docs:

- **Fixed-seed bit-identicality** — two runs with the same seed return
  bit-identical `g` and `H`
  ([`tests/test_core.py`](../../tests/test_core.py)).
- **Checkpoint/restart bit-identicality** — a run interrupted, resumed from
  disk, and continued reproduces the uninterrupted run exactly
  ([`tests/test_checkpoint.py`](../../tests/test_checkpoint.py); the batched
  path is covered in [`tests/test_batched.py`](../../tests/test_batched.py)).

## Worked examples

The {doc}`gallery <auto_examples/index>` walks the methods as runnable
tutorials, each executed live on every docs build:

1. **A first flat-histogram run** — recovering a known flat `g` on a 1D
   bounded random walk, introducing the four-callback contract.
2. **The exact reference** — Beale's recursion vs brute force.
3. **Single-walker Wang-Landau on the Ising model** — the canonical
   `g(E)` recovery, with the callback design and the multi-seed averaging
   used to reach the strict criteria.
4. **Replica-exchange Wang-Landau** — windows, exchange, and joining the
   per-window `g` into one curve.
