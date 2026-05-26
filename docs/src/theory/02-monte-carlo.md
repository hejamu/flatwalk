# Monte Carlo and Markov chains

The previous chapter set the goal: sample configurations from a target
distribution. This chapter is the standard machinery for doing that — Markov
chain Monte Carlo — and why, in its plain canonical form, it is tied to one
temperature. Wang-Landau ({doc}`05-wang-landau`) reuses every piece of it,
changing only the target distribution.

## Importance sampling

We cannot enumerate the $2^N$ configurations, and drawing them uniformly is
useless: the states that dominate an average $\langle A\rangle$ are
exponentially rare under the uniform measure. **Importance sampling** instead
draws configurations *with* the target probability $p(s)$ and estimates the
average as a plain mean over the sample,

$$
\langle A\rangle \approx \frac{1}{M}\sum_{m=1}^{M} A(s_m),
\qquad s_m \sim p(s).
$$

The remaining problem is generating samples from $p(s)$ when we can only compute
$p(s)$ up to its normalisation $Z$ — which we never know.

## Markov chains

A **Markov chain** sidesteps the normalisation. We construct a stochastic
process that hops from configuration to configuration, $s_0 \to s_1 \to \dots$,
with transition probabilities $P(s \to s')$ that depend only on the current
state. If the chain is **ergodic** (every state reachable from every other in
finite time, no trapping cycles) it has a unique stationary distribution
$\pi(s)$ satisfying

$$
\pi(s') = \sum_s \pi(s)\, P(s \to s').
$$

The art is to *design* $P$ so that its stationary distribution is the target we
want. Run the chain long enough and its visited states are distributed as
$\pi$ — exactly the samples importance sampling needs. The sufficient condition
that makes this easy to arrange is **detailed balance**, the subject of the
{doc}`next chapter <03-detailed-balance>`.

## The Metropolis algorithm

The classic recipe splits each step into a **proposal** and an **acceptance**.
From the current state $s$:

1. propose a candidate $s'$ from some proposal distribution $\pi_{\text{prop}}(s'\mid s)$;
2. accept it with probability

$$
A(s \to s') = \min\!\left(1,\; \frac{p(s')}{p(s)}\right)
= \min\!\left(1,\; e^{-\beta (E_{s'} - E_s)}\right)
$$

for a symmetric proposal at the canonical target; otherwise stay at $s$.

Because only the *ratio* $p(s')/p(s)$ appears, the unknown $Z$ cancels — the
whole reason Markov chain Monte Carlo is practical. For the Ising model the
proposal is a single-spin flip and $\Delta E$ is a cheap local computation; this
is the move flatwalk's examples reuse throughout.

## Why canonical Monte Carlo is temperature-bound

Run the Metropolis chain with the canonical target $p(s)\propto e^{-\beta E_s}$
and it equilibrates to that one $\beta$. The energies it visits cluster around
$\langle E\rangle_\beta$ with a width set by the heat capacity; the rare states
in the tails of $g(E)$ are visited with vanishing probability, and a *different*
temperature would concentrate the walk somewhere else entirely. To map a whole
curve $C_V(T)$ you would re-run at each temperature, and even then never resolve
the tails. {doc}`Tutorial 1 </auto_tutorials/plot_1_plain_mc>` shows the visited
energies pinned to a sliver of the axis.

The fix is not a better Markov chain — Metropolis is fine — but a better
*target*. If we aim the same machinery at a distribution that is flat in energy
rather than Boltzmann-weighted, the walk covers the entire spectrum. Choosing
and reaching that target is what the rest of this section is about.

```{seealso}
**See it run:** {doc}`Tutorial 1 </auto_tutorials/plot_1_plain_mc>`.
**Next:** {doc}`03-detailed-balance` derives the acceptance rule and the flat
target precisely.
```
