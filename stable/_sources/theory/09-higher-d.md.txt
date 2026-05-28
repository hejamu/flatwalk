# Higher-dimensional order parameters

```{admonition} Planned, not yet shipped
:class: important
The capability described here is on the roadmap, not in the current release.
flatwalk today samples a one-dimensional order parameter via
{class}`~flatwalk.Bin1D`. This chapter records *why* the extension is
straightforward.
```

## Joint densities of states

Nothing in the flat-histogram method requires the order parameter to be a single
scalar. The same machinery estimates a **joint** density of states over several
coordinates at once,

$$
g(Q_1, Q_2, \dots, Q_k),
$$

for example the 2D Ising model resolved in both energy and magnetisation,
$g(E, M)$. From a joint density you get free-energy *surfaces* and the response
along any combination of the coordinates — the natural target when two order
parameters are coupled.

## Why it is a binning change, not a driver change

The acceptance rule ({doc}`03-detailed-balance`) only ever touches $\ln g$
through a **bin index**. It never inspects how that index was computed. So
raising the order parameter's dimension is contained entirely in the bin scheme:

- `order_parameter_fn(state)` returns a length-$k$ vector instead of a scalar
  (and the batched form returns `ndarray[N, k]`).
- A `BinND(low[k], high[k], n_bins[k])` — a sibling of {class}`~flatwalk.Bin1D`
  under the same {class}`~flatwalk.BinScheme` interface — maps that vector to a
  single flattened integer bin index.
- $\ln g$ and $H$ stay one-dimensional arrays in the driver, indexed by that
  flattened bin. The driver never learns the order parameter has higher
  dimensionality; `result.g` is reshaped back to $k$ dimensions only for
  analysis.

Everything else — the f-stage schedule, the {doc}`1/t handoff <06-one-over-t>`,
{doc}`batched walkers <07-multiple-walkers>`, and {doc}`replica exchange
<08-replica-exchange>` — carries over unchanged, because none of it depends on
how a bin index is produced.

## Validation target

The correctness lever stays sharp in higher dimensions: Beale's transfer-matrix
recursion extends to give the exact $g(E, M)$ for moderate $L$, so the 2D Ising
model in $(E, M)$ would be the exact reference for a $\ge 2$-D run, mirroring the
1-D validation in the {doc}`next chapter <10-validation>`.

```{seealso}
**Next:** {doc}`10-validation`.
```
