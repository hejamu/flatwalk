# The 1/t refinement

Standard Wang-Landau halves $\ln f$ at every flat-histogram stage. That schedule
has a well-known flaw: the error stops improving past a point, no matter how long
you run. The **1/t** modification (Belardinelli & Pereyra, *Phys. Rev. E*
**75**, 046701, 2007) fixes it. flatwalk performs the switch automatically; this
chapter explains what it is switching to and why.

## The saturation of halving

After the $k$-th stage the modification factor is $\ln f_k = 2^{-k}$. The
statistical error in the recovered $\ln g$ accumulated over a stage scales like
$\sqrt{\ln f}$, so each halving should cut the error by $\sqrt{2}$. The problem
is the stages get *exponentially longer*: reaching the flatness threshold with a
tiny $\ln f$ takes ever more trials, because each visit nudges $\ln g$ by less.
Past some stage the run spends essentially all its time refining a factor too
small to matter, and the **total** error saturates at a floor:

$$
\varepsilon_{\text{sat}} \sim \sqrt{\ln f_{\text{stop}}}
\;\;\not\to\;\; 0 .
$$

Running deeper buys diminishing returns; the floor is set by *when* you stopped
halving, not by total effort. {doc}`Tutorial 3
</auto_tutorials/plot_3_one_over_t>` shows the error against run depth.

## The 1/t schedule

Belardinelli and Pereyra observed that the *right* asymptotic decay of the
modification factor is not geometric but

$$
\ln f(t) = \frac{1}{t},
$$

where $t$ is the number of trials. Tie $\ln f$ to $1/t$ and the error decays as
$t^{-1/2}$ indefinitely — no floor. Intuitively, $1/t$ is exactly the rate at
which fresh statistics arrive, so the increment shrinks no faster (which would
freeze $g$ prematurely) and no slower (which would keep injecting noise) than the
data can support.

## flatwalk's automatic handoff

The two regimes meet naturally. flatwalk runs standard halving while $\ln f$ is
large, and switches to the $1/t$ rule once halving would drop $\ln f$ *below*
$1/t$:

$$
\ln f \;\longleftarrow\;
\begin{cases}
\tfrac12\,\ln f & \text{if } \tfrac12\,\ln f \ge \dfrac{1}{t}
  \quad(\text{standard regime, reset } H),\\[1.2ex]
\dfrac{1}{t} & \text{otherwise}
  \quad(\text{1/t regime, updated every trial}).
\end{cases}
$$

Once in the $1/t$ regime there are no more discrete stages or histogram resets:
$\ln f = 1/t$ is recomputed continuously as $t$ grows. The crossover happens by
itself at the right moment, late enough that the cheap halving has done the
coarse work and early enough to avoid the saturation floor. You control only how
deep to go, through `ln_f_final`; the handoff needs no configuration.

The transition shows up plainly in a {doc}`trace
</auto_examples/plot_7_trace_diagnostics>`: $\ln f$ steps down by halves, then
bends into the smooth $1/t$ tail, with `in_1overt` flipping to true at the
crossover.

```{seealso}
**See it run:** {doc}`Tutorial 3 </auto_tutorials/plot_3_one_over_t>` (the error
floor and the handoff) and the {doc}`trace recipe
</auto_examples/plot_7_trace_diagnostics>`.
**Next:** {doc}`07-multiple-walkers`.
```
