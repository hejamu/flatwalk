# Detailed balance and ergodicity

This chapter derives the acceptance rule flatwalk actually uses — including the
`log_proposal_ratio` term you return from `propose_move_fn` — and shows where
the flat-histogram target enters. It is the formal backing for the
{doc}`contract </intro/the-contract>`.

## Detailed balance

A Markov chain has stationary distribution $\pi$ if it satisfies the **global
balance** equation $\pi(s') = \sum_s \pi(s) P(s\to s')$. A simpler *sufficient*
condition, easy to enforce move by move, is **detailed balance**:

$$
\pi(s)\, P(s \to s') = \pi(s')\, P(s' \to s)
\qquad \text{for all } s, s'.
$$

Each pairwise probability flow is balanced, so no net probability circulates and
$\pi$ is stationary. Together with **ergodicity** — the proposals must connect
every state to every other, with no part of configuration space left
unreachable — detailed balance guarantees the chain converges to $\pi$ from any
start.

## Metropolis-Hastings: splitting proposal and acceptance

Write each transition as a proposal times an acceptance,
$P(s\to s') = \pi_{\text{prop}}(s'\mid s)\, A(s\to s')$. Substituting into
detailed balance and solving for the acceptance ratio gives the
**Metropolis-Hastings** rule

$$
A(s \to s') = \min\!\left(1,\;
\frac{\pi(s')\,\pi_{\text{prop}}(s\mid s')}
     {\pi(s)\,\pi_{\text{prop}}(s'\mid s)}\right).
$$

The second factor — the ratio of forward and reverse proposal probabilities —
corrects for any **asymmetry** in how moves are generated. For a symmetric
proposal ($\pi_{\text{prop}}(s'\mid s) = \pi_{\text{prop}}(s\mid s')$, like a
single-spin flip) it is $1$ and drops out. flatwalk asks `propose_move_fn` to
return its logarithm,

$$
\texttt{log\_proposal\_ratio} =
\ln\frac{\pi_{\text{prop}}(s\mid s')}{\pi_{\text{prop}}(s'\mid s)},
$$

which is `0.0` for symmetric moves and non-zero only when you bias the proposal.

## The flat-histogram target

So far $\pi$ has been the Boltzmann distribution. Flat-histogram sampling instead
targets a distribution that is **uniform in the order parameter**. We want
$\pi(s) \propto 1/g(Q(s))$, so that the induced distribution over $Q$ is flat
(see {doc}`01-sampling-problem`). Substituting this $\pi$ into the
Metropolis-Hastings ratio, with $g$ stored on the log scale, the acceptance of a
move from bin $b = Q(s)$ to bin $b' = Q(s')$ becomes

$$
A = \min\!\left(1,\; e^{\Delta}\right),
\qquad
\Delta = \ln g(b) - \ln g(b') + \texttt{log\_proposal\_ratio}.
$$

The walker is *pushed away from well-populated bins* (large $\ln g$) and *toward
empty ones*, which is exactly a flat random walk in $Q$.

## The general acceptance flatwalk evaluates

flatwalk supports a mixed target: flat in the order parameter $Q$ while still
carrying a Boltzmann factor at inverse temperature $\beta$ — useful when $Q$ is
*not* the energy (sampling $g$ over magnetisation at temperature $T$, say). The
full per-trial exponent is

$$
\boxed{\;
\Delta = -\beta\,(E_{s'} - E_s)
       + \ln g(b) - \ln g(b')
       + \texttt{log\_proposal\_ratio}
\;}
$$

with the move accepted when $U < e^{\min(0,\Delta)}$ for a uniform draw
$U\in[0,1)$. This is precisely what {meth}`~flatwalk.WLDriver.run` computes each
step. The two common cases:

- **WL on energy** ($Q = E$, $\beta = 0$): the energy term vanishes and
  $\Delta = \ln g(b) - \ln g(b')$ — the canonical density-of-states run used in
  every Ising example here.
- **WL on another $Q$ at finite $\beta$:** both terms are live, and
  `energy_fn` must return the configurational energy.

## Boundaries

A proposal whose $Q$ falls outside the bin scheme's range is **rejected**; the
walker stays put, and $g$ and $H$ are updated at the *current* bin. This
reflecting-boundary convention keeps detailed balance intact at the edges of the
sampled range while confining the walk to the bins you asked for — the same
mechanism that confines each window in {doc}`replica exchange <08-replica-exchange>`.

```{seealso}
**See it applied:** the {doc}`contract </intro/the-contract>` and the
{doc}`minimal recipe </auto_examples/plot_1_minimal_contract>`.
**Next:** {doc}`04-density-of-states`.
```
