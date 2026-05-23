# The density of states and thermodynamics

A converged $g(E)$ is not the end — it is the input to everything thermodynamic.
This chapter is the reweighting that turns the array `result.g` into the free
energy, internal energy, entropy, and heat capacity at any temperature, and the
numerical care it takes to do so.

## Everything from one curve

With $g(E)$ in hand the partition function at any inverse temperature is a single
sum,

$$
Z(\beta) = \sum_E g(E)\, e^{-\beta E},
$$

and the standard thermodynamic relations follow:

$$
F(\beta) = -\frac{1}{\beta}\ln Z(\beta), \qquad
\langle E\rangle_\beta = \frac{\sum_E E\, g(E)\, e^{-\beta E}}{Z(\beta)},
$$

$$
C_V(\beta) = \frac{1}{k_B T^2}\Big(\langle E^2\rangle_\beta - \langle E\rangle_\beta^2\Big)
           = \beta^2 \operatorname{Var}_\beta(E), \qquad
\frac{S}{k_B} = \ln Z(\beta) + \beta \langle E\rangle_\beta .
$$

No new sampling is needed for a new temperature — only re-evaluating these sums.
A single Wang-Landau run therefore replaces an entire temperature scan of
canonical simulations. {doc}`Tutorial 2 </auto_tutorials/plot_2_wang_landau>`
draws $\langle E\rangle(T)$ and $C_V(T)$ this way; {doc}`Tutorial 6
</auto_tutorials/plot_6_thermodynamics>` adds $F$ and $S$.

## The additive constant

Wang-Landau determines $\ln g$ only **up to an additive constant**: the
acceptance rule (see {doc}`03-detailed-balance`) depends only on *differences*
$\ln g(b) - \ln g(b')$, so a uniform shift of `result.g` is invisible to the
sampler. For ratios like $\langle E\rangle$ and $C_V$ the constant cancels and
you can ignore it. For *absolute* quantities — $F$ and $S$ — it must be pinned.

The clean way to fix it is a sum rule you know independently. The total number of
configurations is fixed, so

$$
\sum_E g(E) = \Omega_{\text{tot}}
\quad(\text{for the Ising model, } \Omega_{\text{tot}} = 2^{N}),
$$

which determines the shift exactly. After applying it, $S/k_B \to \ln
\Omega_{\text{tot}} = N\ln 2$ as $T\to\infty$ — every state equally likely — a
useful check that the normalisation is right. {doc}`Tutorial 6
</auto_tutorials/plot_6_thermodynamics>` does exactly this and verifies the
high-temperature entropy.

## Doing the sums without overflow

$\ln g$ runs to hundreds, so $g = e^{\ln g}$ and the Boltzmann factor both
overflow `float64` immediately. Evaluate the sums in **log space** with the
log-sum-exp identity: for the partition function,

$$
\ln Z(\beta) = \ln\!\sum_E e^{\,\ln g(E) - \beta E}
= M + \ln\!\sum_E e^{\,\ln g(E) - \beta E - M},
\qquad M = \max_E\big(\ln g(E) - \beta E\big).
$$

Subtracting the maximum $M$ before exponentiating keeps every term in $[0,1]$ and
the largest at exactly $1$. Averages use the same shifted weights, normalised to
sum to one:

$$
p_\beta(E) = \frac{e^{\,\ln g(E) - \beta E - M}}{\sum_{E'} e^{\,\ln g(E') - \beta E' - M}},
\qquad
\langle A\rangle_\beta = \sum_E A(E)\, p_\beta(E).
$$

This is the few-line `thermo` helper that appears in the tutorials. Restrict the
sums to **visited, finite** bins; unvisited bins carry no information and
$\ln g = -\infty$ contributes zero weight automatically.

```{seealso}
**See it run:** {doc}`Tutorial 2 </auto_tutorials/plot_2_wang_landau>` and the
capstone {doc}`Tutorial 6 </auto_tutorials/plot_6_thermodynamics>`.
**Next:** {doc}`05-wang-landau` — how `result.g` is produced in the first place.
```
