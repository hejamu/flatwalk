# The sampling problem

## Canonical averages

A classical system in equilibrium at temperature $T$ visits each configuration
$s$ with the Boltzmann probability

$$
p(s) = \frac{e^{-\beta E_s}}{Z}, \qquad
Z(\beta) = \sum_s e^{-\beta E_s}, \qquad
\beta = \frac{1}{k_B T},
$$

where $Z$ is the partition function. Everything thermodynamic is an average over
this distribution — the internal energy $\langle E\rangle$, the heat capacity,
the magnetisation, and so on. The partition function itself gives the free
energy $F = -k_B T \ln Z$, from which the rest follow by differentiation.

## The density of states

Most observables we care about depend on the configuration only through some
**order parameter** $Q(s)$ — often the energy itself, but it could be the
magnetisation, a reaction coordinate, or a particle count. Group the
configurations by their value of $Q$ and the sum over $2^{N}$-or-more states
collapses to a sum over the (few) distinct values of $Q$:

$$
Z(\beta) = \sum_E g(E)\, e^{-\beta E},
$$

where the **density of states**

$$
g(E) = \sum_s \delta\!\left(E - E_s\right)
$$

counts how many configurations have energy $E$. This is the central object. Once
you know $g(E)$ you know $Z$ at *every* temperature, and with it every canonical
average — from a single, temperature-independent quantity:

$$
\langle A\rangle_\beta =
\frac{\sum_E A(E)\, g(E)\, e^{-\beta E}}{\sum_E g(E)\, e^{-\beta E}} .
$$

The reweighting that turns $g(E)$ into thermodynamics is
{doc}`its own chapter <04-density-of-states>`.

## Why straightforward sampling fails

The trouble is scale. For a system of $N$ degrees of freedom $g$ ranges over
*tens of orders of magnitude* between its peak and its tails. A canonical Monte
Carlo run at temperature $T$ draws configurations with weight
$g(E)\,e^{-\beta E}$, which is sharply peaked: the walk spends essentially all
its time in a narrow energy window around the equilibrium energy for that $T$
and never visits the rare states. You can recover $g$ near that window, but not
the tails — and not the other temperatures. {doc}`Tutorial 1
</auto_tutorials/plot_1_plain_mc>` runs straight into this wall.

## The flat-histogram idea

Suppose instead we sample from a distribution that is **flat in $Q$** — equally
likely to be at any value of the order parameter. To do that we would weight each
configuration by $1/g(Q)$, exactly cancelling the system's own degeneracy:

$$
p(s) \propto \frac{1}{g\!\left(Q(s)\right)}
\quad\Longrightarrow\quad
p(Q) \propto g(Q)\cdot\frac{1}{g(Q)} = \text{const}.
$$

A walker under this weighting performs a random walk across the whole range of
$Q$, tails included. The catch is circular — we would need $g$ to build the
weighting that measures $g$. **Wang-Landau** sampling ({doc}`05-wang-landau`)
breaks the circle by learning $g$ on the fly: it starts from a flat guess and
refines the bias until the histogram of visits *is* flat, at which point the
accumulated bias is $\log g$ itself.

This is what flatwalk does. The order parameter $Q$ need not be the energy — any
$Q(s)$ your `order_parameter_fn` returns defines a $g(Q)$ the driver will
estimate. Using $Q = E$ gives the canonical density of states above; using a
different $Q$ gives a free-energy profile along that coordinate.

```{seealso}
**See it run:** {doc}`Tutorial 1 </auto_tutorials/plot_1_plain_mc>` (the wall)
and {doc}`Tutorial 2 </auto_tutorials/plot_2_wang_landau>` (one run, every
temperature). **Next:** {doc}`02-monte-carlo`.
```
