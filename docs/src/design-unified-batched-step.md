# Design: one batched trial step for both drivers

Status: accepted, in progress.

## Background

flatwalk has two batched Wang-Landau drivers, and each carries its own
copy of the per-tick acceptance logic:

| | `WLDriver._trial_step_batched` (`core.py`) | `_rewl_trial_step` (`rewl.py`) |
| --- | --- | --- |
| `g` addressing | 1D `g[bin]` — one shared group | 2D `g_windows[w, bin]` — one row per window |
| confinement | `in_range_batched(value)` (full grid) | `b_lo ≤ bin ≤ b_hi` (per window) |
| histogram update | `np.add.at(g, bin, ln_f)` (scatter) | `g_windows[w, bin] += ln_f` (buffered) |

Everything else — the proposal call, the `−β·ΔE + Δg + log_proposal_ratio`
delta, the `exp(min(0, Δ))` acceptance, the masked state/energy writes, the
counter bookkeeping — is line-for-line identical between the two. The
duplication is real: a fix or feature in one must be mirrored in the other,
and the two have already drifted in one important way (see below).

The drift that matters: the REWL copy uses a buffered `+=` because, with one
walker per window, the `(window, bin)` pairs touched in a tick are unique.
That shortcut is exactly what blocks **multiple walkers per window** — the
canonical massively-parallel REWL configuration — because two walkers in the
same window can land on the same bin in one tick, and a buffered `+=` would
drop one of the two updates. `_trial_step_batched` already does the correct
thing (`np.add.at`); REWL's copy is the one that took the unsafe path.

## Goal

A single batched trial-step primitive that both drivers call, so that:

1. there is one implementation of the acceptance rule to test and maintain;
2. the correct scatter (`np.add.at`) is used everywhere, which is also the
   prerequisite for multi-walker-per-window REWL; and
3. no public surface changes — `WLResult.g` stays 1D, the checkpoint format
   is untouched, and the existing tests that call `_trial_step_batched` and
   `_rewl_trial_step` keep their signatures.

## Options considered

- **A — keep `g` 1D, branch inside the primitive on `group is None`.**
  Rejected: it recreates the two-path duplication the refactor exists to
  remove, now inside one function and in the hot path.
- **B1 — make `run_batched`'s `g` genuinely `(1, B)` end to end.** Rejected:
  it leaks an internal refactor into the public contract — `WLResult.g`
  becomes 2D and the checkpoint format changes, breaking callers and tests.
- **B2 — primitive is always `(G, B)`; drivers translate at the call site.**
  Chosen.

## Decision: B2

One module-level primitive in `core.py`:

```python
def _grouped_trial_step(
    bin_scheme, wb, g, H, visited, group, b_lo, b_hi, ln_f,
    energy_fn, order_parameter_fn, propose_move_fn, beta,
) -> np.ndarray:
    ...
```

with this contract:

- `g`, `H`, `visited` have shape `(G, B)`.
- `group` is an `int[N]` array mapping each of the `N` walkers to its row of
  `g`/`H`/`visited`.
- `b_lo`, `b_hi` are inclusive bin bounds (scalars or `int[N]`) confining each
  walker; a proposal landing outside `[b_lo, b_hi]` is rejected.
- the post-trial update is `np.add.at(g, (group, bin), ln_f)` (likewise `H`),
  so several walkers sharing a `(group, bin)` in one tick all count.

Each driver becomes a thin adapter that keeps its own representation:

- **`WLDriver._trial_step_batched`** keeps its 1D `g`/`H`/`visited` and its
  signature, and delegates with a view bridge:

  ```python
  N = wb.n_walkers
  return _grouped_trial_step(
      self.bin_scheme, wb, g[None], H[None], visited[None],
      np.zeros(N, dtype=np.intp), 0, self.bin_scheme.n_bins - 1,
      ln_f, energy_fn, order_parameter_fn, propose_move_fn, beta,
  )
  ```

  `g[None]` is a *view* of the 1D buffer, so `np.add.at` through it writes
  straight back into `g`. The driver's result and checkpoint stay 1D; the
  whole seam is one line and lives in the caller, not the primitive.

- **`_rewl_trial_step`** keeps its signature and delegates with `group =
  np.arange(W)` and the window bounds it already holds.

Neither driver loop changes.

## Why this is bit-identical

The refactor must not perturb the fixed-seed and checkpoint/resume
reproducibility tests, so the primitive preserves the exact observable
behaviour of both copies:

- **RNG draw order** is unchanged: `propose_move_fn(...)` first, then one
  `rng.random(N)` for the acceptance draw — the same order both copies use.
- **The confinement mask is identical.** `value_to_index_batched` returns
  `-1` for any out-of-domain value, so `(bin >= 0) & (bin <= B-1)` equals
  `in_range_batched(value)` exactly. The full-grid bounds reproduce
  `run_batched`'s old `in_range` mask bit for bit; the window bounds reproduce
  REWL's old `in_win` mask.
- **The scatter is identical on the existing paths.** With one walker per
  window the `(group, bin)` pairs are unique, and `np.add.at` over unique
  indices performs the same single additions in the same order as the old
  buffered `+=` — same floats, same result. `run_batched` already used
  `np.add.at`, so it is untouched.

So all current tests pass unchanged; the only new behaviour is the one that
was previously impossible.

## What it unlocks

Multiple walkers per window becomes a `group` map with repeats. For `W`
windows and `m` walkers each:

```python
group = np.repeat(np.arange(W), m)        # int[N], N = W * m
b_lo_w, b_hi_w = window_bounds[group]      # per-walker bounds
```

`g`/`H`/`visited` are `(W, B)` as today; the primitive already scatters
correctly into shared rows. The remaining work to expose the feature lives in
`RewlDriver` (build the group map, pool per-window flatness over the `m`
walkers, generalise exchange to pick pairs across adjacent windows) — not in
the step. That is a follow-on; this design only unifies the step.

## Performance

`np.add.at` on a `(1, B)` view is marginally slower than a bare 1D
`np.add.at`, and the per-call `np.zeros(N)` group array is a small allocation.
Both are negligible next to the batched energy callback, and `run_batched`
already paid `np.add.at` rather than buffered `+=`, so this introduces no new
cost category. If profiling ever flags it, the group array can be hoisted out
of the loop; it is constant per run.

## Testing and rollout

1. Land `_grouped_trial_step` and reduce both step functions to adapters in
   one change; the existing `test_batched.py` and `test_rewl.py` suites are
   the regression guard (bit-for-bit vs the scalar step, and window
   confinement).
2. Confirm the fixed-seed and checkpoint/resume tests still pass — they are
   the bit-identicality proof.
3. Multi-walker-per-window REWL is a separate, additive change on top.
