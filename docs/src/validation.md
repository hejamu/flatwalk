# Validation: 2D Ising against Beale's exact n(E)

The 1D order-parameter case is validated end-to-end by Wang-Landau
sampling the 2D Ising model on an L×L periodic lattice and comparing
the recovered density of states against the exact ``n(E)`` computed
from a Beale-style transfer-matrix recursion. The pass criteria match
the project specification §4.4.

This page walks through the example end-to-end:

- §[The exact reference](#the-exact-reference-beales-recursion):
  Beale's modular-CRT recursion for ``n(E)`` and its cross-check
  against brute-force enumeration on small lattices.
- §[The user-side callbacks](#the-user-side-callbacks): how the Ising
  Hamiltonian and the single-spin-flip move fill the four-piece
  contract.
- §[The script and pass criteria](#the-script-and-pass-criteria):
  invocation, what counts as a pass, typical numbers.
- §[Divergences from the literal spec](#divergences-from-the-literal-spec):
  the script-level tuning choices and multi-seed averaging used to
  satisfy the strict pass criteria, all *outside* the `flatwalk`
  driver.

## The exact reference: Beale's recursion

The partition function on an L×L periodic Ising lattice is

```{math}
Z(x) = \sum_{\text{configs}} x^{\sum_{\langle i,j\rangle} \sigma_i \sigma_j}
     = \sum_E n(E)\, x^{-E/J}
     = \mathrm{Tr}\, T^L,
```

where `T` is the `2^L × 2^L` row transfer matrix and each `T[s, s']`
is a single monomial in `x`. `T^L` is then a matrix of integer
polynomials in `x` whose trace recovers `n(E)` as polynomial
coefficients.

Direct integer arithmetic is intractable for `L=8` (coefficients reach
~10^{19}), so [`examples/beale.py`](../../examples/beale.py) computes
the trace polynomial modulo several primes chosen so that float64
matrix-matrix multiplication stays mantissa-exact through every
accumulation, then reconstructs the integer coefficients via the
Chinese Remainder Theorem.

```python
from examples.beale import beale_g_E
n_E = beale_g_E(L=8)       # → {E: integer n(E)} on 2^64 configurations
n_E[-128]                  # 2 (the two FM ground states)
n_E[+128]                  # 2 (the two AF "checkerboard" states)
n_E[-124]                  # 0 (spectrum gap — no config has this energy)
```

**Cross-validation.** The recursion is sanity-checked against direct
enumeration on small lattices in
[`tests/test_beale.py`](../../tests/test_beale.py): brute-force ``n(E)``
on the 2^9 = 512 configurations of L=3 and the 2^16 = 65,536
configurations of L=4 must agree bin-for-bin with the recursion output.

The L=8 reference is cached on first run as a TSV file at
`examples/cache/beale_L8.tsv`; subsequent invocations load it
instantly.

## The user-side callbacks

The Ising side of the four-piece contract lives in
[`examples/ising.py`](../../examples/ising.py). Two implementation
choices matter:

1. **The state is `(spins, cached_E)`.** Each accepted move updates
   ``cached_E`` by the local ΔE so the driver's ``energy_fn`` and
   ``order_parameter_fn`` are O(1) lookups rather than O(L²)
   recomputations. flatwalk doesn't know or care about this — it
   treats `state` as opaque.

2. **The order parameter is the energy.** "WL on E" means
   ``order_parameter_fn(state) == energy_fn(state)``. The driver's
   acceptance criterion ``Δ = −β·(E_new − E_old) + g[bin_old] −
   g[bin_new] + log_proposal_ratio`` reduces to ``Δ = g[bin_old] −
   g[bin_new]`` at β = 0 (since E_new and bin_new are tied).

The single-spin-flip move is symmetric, so the callback returns
``(new_state, 0.0)``. The ΔE for flipping site (i, j) is
``2·J·σ_{i,j}·(sum of four neighbours)``.

## The script and pass criteria

[`examples/ising_validation.py`](../../examples/ising_validation.py)
runs the end-to-end validation. Default invocation:

```bash
python examples/ising_validation.py --seed 0
```

This:

1. Loads (or computes + caches) Beale's exact ``n(E)`` for L=8.
2. Runs `--n-seeds` independent WL runs (default 3) to ``ln_f_final =
   1e-8`` with tuned hyperparameters (``n_check = 1000``,
   ``flatness_threshold = 0.95``).
3. Averages the per-seed ``log g`` arrays (geometric mean of n(E)
   estimates per bin; unbiased, variance ``≈ var_single / K``).
4. Compares against Beale's exact `n(E)` and reports four metrics:

| Metric | Pass criterion (spec §4.4) |
| --- | --- |
| `max ε(E)` over central bins (visited & non-gap, excluding the two extremes) | < 0.05 |
| `mean ε(E)` over the same bins | < 0.01 |
| `max |⟨E⟩_WL − ⟨E⟩_exact| / |⟨E⟩_exact|` over T ∈ [1, 4] | < 0.5% |
| `C_V(T)` peak-temperature error vs exact | < 2% |

The script exits 0 only if all four pass.

**Smoke mode** for development: ``--quick`` runs to ``ln_f_final =
1e-5`` (~30 s) and exercises the whole pipeline without satisfying the
strict criteria.

Run time is ~15 min total at the default settings; the slow lane in CI
runs this on every push.

## Divergences from the literal spec

The strict spec §4.4 pass criteria on L=8 require two script-level
tuning choices and one multi-run averaging choice. None of them touch
the `flatwalk` driver itself.

1. **WL hyperparameters** `n_check = 1000`, `flatness_threshold = 0.95`
   (spec defaults: 10_000, 0.8). The spec marks both as "Tunable" in
   §1.5, so this is within bounds. Smaller `n_check` triggers the 1/t
   regime sooner; stricter flatness gives each f-stage more samples,
   so `g[bin]` is better-equilibrated at each halving.

2. **Multi-seed averaging** (``--n-seeds 3``). A single-seed
   single-walker 1/t-WL on L=8 produces a `g_WL` with ~5–10% per-bin
   error in the high-|E| tails. The asymmetry is *between* `E` and
   `−E` and arises from the trajectory: the walker reaches one tail
   before the other and accumulates more early (large-`ln_f`) updates
   there. Averaging the `log g` arrays from `K` independent seeds
   reduces the variance by `~1/K`. This is standard practice in the
   WL literature; REWL is the more elegant solution and is the
   roadmap path (see {doc}`storyline`).

   ``--n-seeds 1`` recovers the pure spec interpretation ("Run the
   driver"). The driver itself is single-walker and bit-identical on a
   fixed seed.

## Reproducibility

Two related correctness properties of the driver, validated in the
test suite rather than in the script:

- **Fixed-seed bit-identicality.** Two runs of `WLDriver.run` with the
  same RNG seed produce bit-identical `g` and `H`. Tested in
  [`tests/test_core.py::test_reproducibility`](../../tests/test_core.py).
- **Checkpoint/restart bit-identicality.** A run interrupted mid-way,
  resumed from disk, and continued to the same total trial count
  produces *exactly* the `g` and `H` arrays of the uninterrupted run
  on the same seed. Tested in
  [`tests/test_checkpoint.py::test_resume_is_bit_identical`](../../tests/test_checkpoint.py)
  over a 4,000-trial random-walk system with a checkpoint at
  `t = 2,000`.
