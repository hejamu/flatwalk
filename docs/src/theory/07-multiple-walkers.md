# Multiple and batched walkers

A single Wang-Landau walker must traverse the whole order-parameter range by
itself, and one run carries the statistical luck of one trajectory. Running
several walkers fixes both — and flatwalk's batched design makes it the natural
path to a vectorised energy backend.

## Many walkers, one shared g

The simplest parallel scheme is $N$ independent walkers all updating the **same**
$\ln g$ and histogram $H$. Each walker proposes and accepts by the usual rule
(see {doc}`03-detailed-balance`), and each contributes its visit to the shared
arrays:

$$
\ln g(b_i) \mathrel{+}= \ln f, \qquad H(b_i) \mathrel{+}= 1
\qquad \text{for every walker } i = 1,\dots,N .
$$

Because all walkers see the same evolving bias, they spread out and cooperate to
flatten one histogram. A stage now accumulates $N$ times the statistics in the
same number of ticks, so $g$ converges faster and the per-bin variance falls
roughly as $1/N$. The leftover single-walker pathologies — most visibly an
$E\leftrightarrow -E$ asymmetry on a symmetric system, from whichever tail the
lone walker happened to reach first — average out across walkers.
{doc}`Tutorial 4 </auto_tutorials/plot_4_more_walkers>` measures both effects.

## The scatter, done correctly

Two walkers can land in the **same** bin on the same tick. The shared update must
then add *both* contributions, so flatwalk scatters with `np.add.at`, which
accumulates repeated indices correctly, rather than a plain buffered `+=` that
would drop one of a colliding pair. This is not a detail — it is the exact
property that later lets several walkers share one window in {doc}`replica
exchange <08-replica-exchange>`.

## Why batched, not looped

flatwalk's design rule is that for $N \ge 2$ walkers the per-tick primitives
operate on **all $N$ states at once** — there is no `for walker in walkers:` in
the inner loop. The reason is performance. The whole point of wiring flatwalk to
a vectorised backend (a GPU tensor library, a JAX kernel, an MPI energy call) is
that one stacked call evaluates $N$ configurations together. A Python loop over
walkers would make that call $N$ separate times and throw the speedup away.

So the batched callbacks take one opaque `state_batch` of $N$ walkers and return
$N$ results in a single call each:

| scalar callback | batched callback |
| --- | --- |
| `energy_fn(state) → float` | `energy_fn(state_batch) → ndarray[N]` |
| `order_parameter_fn(state) → float` | `order_parameter_fn(state_batch) → ndarray[N]` |
| `propose_move_fn(state, rng) → (new_state, lpr)` | `propose_move_fn(state_batch, rng) → (new_state_batch, lpr[N])` |

`state_batch` is opaque exactly as `state` is: the driver only ever applies
accepted moves to it by boolean-mask assignment, so a stacked `ndarray[N, …]` or
a torch tensor works without the driver knowing anything about the backend. One
tick is then a handful of vectorised array operations — propose, evaluate, mask,
accept, scatter — with no per-walker Python.

## Bookkeeping

{meth}`~flatwalk.WLDriver.run_batched` counts `t_total` in individual moves
($N$ per tick), so `n_check`, `ln_f_final`, and the {doc}`1/t schedule
<06-one-over-t>` use the same units as the scalar {meth}`~flatwalk.WLDriver.run`;
with $N = 1$ the batched path reduces to the scalar schedule exactly. The
architectural rationale is written up in the {doc}`storyline
</background/storyline>` and the {doc}`unified-step design note
</background/design-unified-batched-step>`.

```{seealso}
**See it run:** {doc}`Tutorial 4 </auto_tutorials/plot_4_more_walkers>` and the
{doc}`batched recipe </auto_examples/plot_4_batched_ising>`.
**Next:** {doc}`08-replica-exchange`.
```
