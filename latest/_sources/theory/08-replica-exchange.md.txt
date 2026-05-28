# Replica-exchange Wang-Landau

Sharing one $g$ across walkers ({doc}`07-multiple-walkers`) cuts the variance but
leaves every walker responsible for the *entire* energy range — and the steep
tails of $g$ stay the slowest part to converge. Replica-exchange Wang-Landau
(REWL) divides that labour into overlapping **windows**, each an easy local
problem; reassembles the pieces by **gluing** their overlaps; and — the step that
gives the method its name — **exchanges** configurations between neighbouring
windows to keep each one well mixed.

The first two steps already form a complete, usable sampler on their own
("windowed" or "multiple-range" WL). Exchange is an *enhancement* layered on top,
not a requirement for assembling the windows — a distinction worth keeping clear,
since the two solve different problems.

## Windows

Split the order-parameter range into $W$ **overlapping windows**, each a
contiguous band of bins. Window $w$ runs its own walker, confined to its band
(proposals that leave it are rejected, the reflecting-boundary rule of
{doc}`03-detailed-balance`), and builds its **own** $\ln g_w$ over just those
bins. A narrow window is an easy flat-histogram problem — short to traverse, fast
to flatten — so all $W$ converge in parallel where one global walker would
struggle. {func}`~flatwalk.make_windows` tiles the grid into equal-width windows
with a prescribed overlap.

## Gluing the windows

Each window finishes with its own $\ln g_w$, correct in *shape* within the window
but carrying an independent additive constant. {func}`~flatwalk.join_g` stitches
them into one curve: over each adjacent pair's overlap region the two log-curves
should differ only by a constant, so it finds the shift that best aligns them (a
least-squares match on the log scale) and averages the overlap. The result is a
single $\ln g(E)$ across the full range, assembled from the local pieces.

The alignment uses **only the overlaps** — it needs no communication between
windows while they run. So windows + gluing is *already* a working method:
independent confined walkers, stitched at the end, recover $g$ with no exchange
at all. The overlaps are what make this possible, which is why the windows must
overlap regardless of whether exchange is used.

## The exchange move

What windowed WL alone lacks is **mixing**. A confined walker is ergodic *in
energy* within its band — the flat-histogram bias makes it random-walk across the
window's energies — but nothing refreshes the degrees of freedom *orthogonal* to
the energy. Many configurations share an energy, and a barrier in those
directions can trap a walker for an entire run. For the Ising model the two
ferromagnetic basins (mostly-up and mostly-down) sit at the same low energies but
are separated by a barrier the walker cannot cross without leaving its
low-energy window — so a lone confined walker samples only the basin it started
in.

Replica exchange supplies the missing mixing with a second kind of move. Every
`n_exchange` ticks, propose swapping the configurations of two walkers in
**adjacent, overlapping** windows $i$ and $j$. Swapping is itself a Markov move
and must satisfy detailed balance against the flat-in-each-window target; the
acceptance exponent is

$$
\Delta = \ln g_i(E_j) - \ln g_i(E_i)
       + \ln g_j(E_i) - \ln g_j(E_j),
$$

accepted when $U < e^{\min(0,\Delta)}$. Here $E_i$ is the energy of the
configuration currently in window $i$, and $\ln g_i(\cdot)$ is window $i$'s own
bias evaluated at that energy — so a swap is favoured when each configuration is
comparatively under-represented in the *other* window. A swap is only possible
when both energies lie in the windows' shared overlap (a second reason the
windows overlap).

To keep the exchange itself in detailed balance, the adjacent pairs are chosen in
**alternating even/odd offsets** on successive exchange attempts, so no window
participates in two swaps at once. Accepted swaps permute the walkers'
configurations; the per-window biases $\ln g_w$ are untouched by a swap — only
the configurations move. The effect is that each window keeps receiving fresh,
decorrelated configurations from its neighbours — exactly the mixing that
confinement removed.

## Why REWL beats plain windowed WL

- **No global traversal.** Each walker only crosses its own short window, so the
  hard, slow long-range diffusion of a single walker disappears (windowing).
- **Mixing restored.** Exchange refreshes the modes orthogonal to the energy, so
  a walker can no longer sit trapped in one basin or behind a barrier for the
  whole run — the benefit that windowing alone does *not* provide.
- **Symmetry restored.** With both the windowing and the exchange, the
  $E\leftrightarrow -E$ asymmetry of a lone walker does not arise.
- **Naturally parallel.** The windows advance as one batched call (one walker
  per window, or several — the {doc}`shared scatter <07-multiple-walkers>` makes
  multiple-per-window correct), so a vectorised energy backend is paid once per
  tick across all windows.

This is built up over two tutorials. {doc}`Tutorial 5
</auto_tutorials/plot_5_windows_gluing>` shows windows + gluing recovering $g$ on
the Ising model with no exchange (the symmetric basins make it benign);
{doc}`Tutorial 6 </auto_tutorials/plot_6_replica_exchange>` then builds a
deliberately asymmetric two-basin model where a single full-range walker is fine,
windowing alone gets the wrong $g$, and only exchange recovers it.

```{seealso}
**See it run:** {doc}`Tutorial 5 (windows + gluing)
</auto_tutorials/plot_5_windows_gluing>`, {doc}`Tutorial 6 (replica exchange)
</auto_tutorials/plot_6_replica_exchange>`, and the {doc}`REWL recipe
</auto_examples/plot_5_replica_exchange_ising>`.
**Next:** {doc}`09-higher-d`.
```
