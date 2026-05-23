# The contract

flatwalk never inspects your system. It does the flat-histogram bookkeeping and
calls back into your code for everything physical. You supply four things:

| You supply | Type | What flatwalk does with it |
| --- | --- | --- |
| `bin_scheme` | {class}`~flatwalk.BinScheme` instance | maps `Q → bin index` |
| `energy_fn(state)` | `→ float` | the `−β·ΔE` term in WL acceptance (skip when `β=0` and `Q=E`) |
| `order_parameter_fn(state)` | `→ float \| np.ndarray` | the quantity `g(Q)` is estimated over (vector for ≥2D) |
| `propose_move_fn(state, rng)` | `→ (new_state, log_proposal_ratio)` | one Markov step |

`state` is **opaque** to flatwalk — whatever your callbacks recognise: a tuple,
a dataclass, a NumPy array, a torch tensor, anything. You hand one initial
`state` object to {meth}`~flatwalk.WLDriver.run` to start; from there the
callbacks do all state manipulation. The driver only ever:

1. calls your callbacks with whatever you passed in,
2. calls `bin_scheme.value_to_index(q)` on the value `order_parameter_fn`
   returns, and
3. maintains `g`, the histogram `H`, the f-stage schedule, and the 1/t
   transition.

That is the whole boundary. The energy backend can be NumPy, PyTorch, JAX, a
Fortran kernel via `ctypes`, a remote queue, or a learned surrogate — flatwalk
neither knows nor needs to.

```{note}
`g` is the **log** density of states, known up to an additive constant. Every
page in these docs means $\log g(Q)$ when it writes `result.g`.
```

## The pieces in detail

`bin_scheme`
: A {class}`~flatwalk.BinScheme` (today {class}`~flatwalk.Bin1D`) carries the
  order-parameter domain and resolution: the `[low, high)` range and the number
  of bins. It maps each `Q` to a bin index, and reports when a value falls
  outside the range.

`energy_fn(state) → float`
: The configurational energy. It enters acceptance only through `−β·ΔE`. For a
  canonical "WL on energy" run you set `beta = 0` and this term drops out
  entirely — see {doc}`quickstart`.

`order_parameter_fn(state) → float | np.ndarray`
: The quantity $Q$ whose density of states you want. Returning the energy makes
  it a $g(E)$ run; returning some other scalar (magnetisation, a reaction
  coordinate, …) estimates $g$ over that instead. A vector return targets a
  joint, higher-dimensional order parameter.

`propose_move_fn(state, rng) → (new_state, log_proposal_ratio)`
: One trial Markov move. Return the proposed `new_state` and the log of the
  proposal ratio $\log[\pi(\text{old}\,|\,\text{new})/\pi(\text{new}\,|\,\text{old})]$.
  For a **symmetric** proposal this is `0.0`; the term exists so asymmetric
  moves still satisfy detailed balance (derived in
  {doc}`../theory/03-detailed-balance`).

For two or more walkers, each callable has a **batched** sibling that takes one
opaque `state_batch` of `N` walkers and returns `N` results in one call — so a
vectorised backend evaluates them all at once. See
{doc}`../theory/07-multiple-walkers` and the batched
{doc}`example <../auto_examples/index>`.
