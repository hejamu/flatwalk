# Replica-exchange Wang-Landau

Sharing one $g$ across walkers ({doc}`07-multiple-walkers`) cuts the variance but
leaves every walker responsible for the *entire* energy range — and the steep
tails of $g$ stay the slowest part to converge. Replica-exchange Wang-Landau
(REWL) divides that labour: confine each walker to a sub-range, and let
neighbours trade configurations to keep the whole consistent.

## Windows

Split the order-parameter range into $W$ **overlapping windows**, each a
contiguous band of bins. Window $w$ runs its own walker, confined to its band
(proposals that leave it are rejected, the reflecting-boundary rule of
{doc}`03-detailed-balance`), and builds its **own** $\ln g_w$ over just those
bins. A narrow window is an easy flat-histogram problem — short to traverse, fast
to flatten — so all $W$ converge in parallel where one global walker would
struggle. {func}`~flatwalk.make_windows` tiles the grid into equal-width windows
with a prescribed overlap.

## The exchange move

Independent windows would each recover $g$ only up to their own constant, with no
way to align them and no mixing across boundaries. The fix is a second kind of
move. Every `n_exchange` ticks, propose swapping the configurations of two
walkers in **adjacent, overlapping** windows $i$ and $j$. Swapping is itself a
Markov move and must satisfy detailed balance against the flat-in-each-window
target; the resulting acceptance exponent is

$$
\Delta = \ln g_i(E_j) - \ln g_i(E_i)
       + \ln g_j(E_i) - \ln g_j(E_j),
$$

accepted when $U < e^{\min(0,\Delta)}$. Here $E_i$ is the energy of the
configuration currently in window $i$, and $\ln g_i(\cdot)$ is window $i$'s own
bias evaluated at that energy — so a swap is favoured when each configuration is
comparatively under-represented in the *other* window. A swap is only possible
when both energies lie in the windows' shared overlap, which is why the windows
must overlap at all.

To keep the exchange itself in detailed balance, the adjacent pairs are chosen in
**alternating even/odd offsets** on successive exchange attempts, so no window
participates in two swaps at once. Accepted swaps permute the walkers'
configurations; the per-window biases $\ln g_w$ are untouched by a swap — only
the configurations move.

## Joining the windows

Each window finishes with its own $\ln g_w$, correct *within* the window but
carrying an independent additive constant. {func}`~flatwalk.join_g` stitches them
into one curve: over each adjacent pair's overlap region the two log-curves
should differ by a constant, so it finds the shift that best aligns them (a
least-squares match on the log scale) and averages the overlap. The result is a
single $\ln g(E)$ across the full range, assembled from the local pieces.

## Why REWL is the robust choice

- **No global traversal.** Each walker only crosses its own short window, so the
  hard, slow long-range diffusion of a single walker disappears.
- **Symmetry restored.** With windows tiling both tails, the
  $E\leftrightarrow -E$ asymmetry of a lone walker does not arise.
- **Naturally parallel.** The windows advance as one batched call (one walker
  per window, or several — the {doc}`shared scatter <07-multiple-walkers>` makes
  multiple-per-window correct), so a vectorised energy backend is paid once per
  tick across all windows.

{doc}`Tutorial 5 </auto_tutorials/plot_5_replica_exchange>` runs the whole
pipeline — windows, exchange, join — against the exact reference.

```{seealso}
**See it run:** {doc}`Tutorial 5 </auto_tutorials/plot_5_replica_exchange>` and
the {doc}`REWL recipe </auto_examples/plot_5_replica_exchange_ising>`.
**Next:** {doc}`09-higher-d`.
```
