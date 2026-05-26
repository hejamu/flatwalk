# Wang-Landau sampling

The {doc}`first chapter <01-sampling-problem>` left a circular problem: to sample
flatly in energy we need a bias $1/g(E)$, but $g$ is what we are trying to
measure. Wang-Landau (Wang & Landau, *Phys. Rev. Lett.* **86**, 2050, 2001)
breaks the circle by learning $g$ while it walks — refining the bias until the
histogram of visits is flat, at which point the bias *is* $\ln g$.

## The algorithm

Keep two arrays over the order-parameter bins: the running estimate $\ln g(b)$
(the **bias**, initialised to $0$) and a visit histogram $H(b)$ (initialised to
$0$). Hold a **modification factor** $f > 1$, stored as $\ln f$ (initialised to
$\ln f = 1$). Then repeat:

1. **Propose** a move and accept it with the flat-histogram rule from
   {doc}`03-detailed-balance`, $\Delta = \ln g(b) - \ln g(b') +
   \texttt{lpr}$ (plus the $-\beta\,\Delta E$ term if $Q\neq E$).
2. **Update**, at the bin $b$ the walker now occupies — accepted or not:

$$
\ln g(b) \mathrel{+}= \ln f, \qquad H(b) \mathrel{+}= 1.
$$

The first update is the trick. Every visit *raises* the bias at the current bin,
so the walker is increasingly repelled from where it has already been and pushed
toward under-visited bins. Left alone with fixed $f$, this drives the visit
histogram toward flatness — and the accumulated $\ln g$ toward the true log
density of states, up to a constant.

## Flatness and the f-stage schedule

A fixed $f$ can only get $H$ *approximately* flat; the residual ripple is of
order $\ln f$. So Wang-Landau runs in **stages**. Every `n_check` trials the
driver tests how flat $H$ is over the visited bins,

$$
\text{flatness} = \frac{\min_b H(b)}{\operatorname{mean}_b H(b)}
\quad (\text{over visited } b),
$$

and once this exceeds `flatness_threshold` (e.g. $0.8$) the stage ends:

$$
\ln f \to \tfrac{1}{2}\ln f, \qquad H \to 0,
$$

while $\ln g$ is **kept**. The next stage refines the same $\ln g$ with a smaller
increment, so each round adds finer detail on top of the last. As $f \to 1$
($\ln f \to 0$) the updates become infinitesimal and $\ln g$ stops changing: it
has converged. The run stops when $\ln f$ falls below `ln_f_final`.

## What the knobs do

| `WLConfig` field | Role |
| --- | --- |
| `ln_f_initial` | starting increment (default $1$); how aggressively early exploration fills $g$ |
| `flatness_threshold` | how flat $H$ must be to end a stage; stricter → more samples per stage, smoother $g$ |
| `n_check` | trials between flatness checks; also sets when the {doc}`1/t regime <06-one-over-t>` can engage |
| `ln_f_final` | convergence target; how deep the refinement goes |

## Convergence and its limit

In the limit of infinitely many trials per stage and $f\to 1$, Wang-Landau
returns the exact $\ln g$ up to a constant. In practice each stage is finite, and
the simple halving schedule has a subtle failure: past a point, shrinking $\ln f$
no longer reduces the error — it settles onto a **floor** set by the
flatness-driven stage lengths rather than by how long you run. That floor, and
the modification that removes it, are the {doc}`next chapter <06-one-over-t>`.

```{seealso}
**See it run:** {doc}`Tutorial 2 </auto_tutorials/plot_2_wang_landau>` and the
{doc}`single-walker recipe </auto_examples/plot_3_single_walker_ising>`.
**Next:** {doc}`06-one-over-t`.
```
