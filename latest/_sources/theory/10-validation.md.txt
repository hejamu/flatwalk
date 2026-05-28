# How we know it's right

The preceding chapters derived the methods; this one is the proof they are
implemented correctly. A flat-histogram sampler is only as trustworthy as the
density of states it returns, so flatwalk is validated the direct way: run it on
a system whose $g(E)$ is known *exactly*, and compare bin for bin.

## The exact-reference lever

The 2D Ising model on an $L\times L$ periodic lattice has an exactly computable
integer density of states $n(E)$. [`examples/beale.py`](../../../examples/beale.py)
obtains it from a Beale-style transfer-matrix recursion (the trace of $T^L$
evaluated modulo several primes and reconstructed by the Chinese Remainder
Theorem, which keeps the $\sim 10^{19}$ coefficients exact where naive integer
arithmetic would not). On small lattices the recursion is itself cross-checked
against brute-force enumeration, so the reference is sound before the driver
leans on it at $L=8$.

This gives a sharp target: Wang-Landau sampling $g(E)$ on the same lattice should
reproduce $n(E)$ across the whole spectrum — and the thermodynamics derived from
it ({doc}`04-density-of-states`) should match the exact curves.

## What counts as a pass

The criteria, evaluated over visited, non-gap central bins (the two extreme
$E = \pm 2L^2$ bins are excluded — there $n = 2$ and relative noise dominates):

| Metric | Pass |
| --- | --- |
| `max ε(E)` per bin | < 0.05 |
| `mean ε(E)` per bin | < 0.01 |
| `max |⟨E⟩_WL − ⟨E⟩_exact| / |⟨E⟩_exact|`, `T ∈ [1, 4]` | < 0.5% |
| `C_V(T)` peak-temperature error | < 2% |

The full $\ln f_{\text{final}} = 10^{-8}$, $L=8$ runs that meet these criteria
are too slow for a docs build. They live at the repo root —
[`examples/ising_validation.py`](../../../examples/ising_validation.py) and
[`examples/ising_rewl_validation.py`](../../../examples/ising_rewl_validation.py) —
and run in CI's slow lane. The galleries below run fast smoke versions of the
same pipelines on every build, with loosened bounds.

## Reproducibility

Two driver-correctness properties are pinned in the test suite rather than the
docs:

- **Fixed-seed bit-identicality** — two runs with the same seed return
  bit-identical $g$ and $H$
  ([`tests/test_core.py`](../../../tests/test_core.py)).
- **Checkpoint/restart bit-identicality** — a run interrupted, resumed from disk,
  and continued reproduces the uninterrupted run exactly
  ([`tests/test_checkpoint.py`](../../../tests/test_checkpoint.py); the batched
  path is covered in [`tests/test_batched.py`](../../../tests/test_batched.py)).
  The {doc}`checkpoint recipe </auto_examples/plot_6_checkpoint_resume>`
  demonstrates it live.

## Where to watch it run

Both galleries execute against this reference on every docs build:

- The {doc}`tutorials </auto_tutorials/index>` rebuild the validation as a story
  — recovering $g(E)$ with {doc}`Wang-Landau </auto_tutorials/plot_2_wang_landau>`,
  sharpening it with the {doc}`1/t schedule </auto_tutorials/plot_3_one_over_t>`,
  cutting variance with {doc}`more walkers </auto_tutorials/plot_4_more_walkers>`
  and {doc}`replica exchange </auto_tutorials/plot_6_replica_exchange>`, and
  checking the {doc}`thermodynamics </auto_tutorials/plot_7_thermodynamics>`
  against Beale.
- The {doc}`examples </auto_examples/index>` give the same comparisons as compact
  recipes: the {doc}`exact reference </auto_examples/plot_2_beale_reference>`
  (recursion vs brute force), {doc}`single-walker
  </auto_examples/plot_3_single_walker_ising>`, {doc}`batched
  </auto_examples/plot_4_batched_ising>`, and {doc}`replica-exchange
  </auto_examples/plot_5_replica_exchange_ising>` $g(E)$ against $n(E)$.

```{seealso}
**Previous:** {doc}`09-higher-d` extends the same lever to $g(E, M)$.
**Back to** the {doc}`theory index <index>`.
```
