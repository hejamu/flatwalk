# Theory

These chapters explain the methods flatwalk implements — *why* they work and
*what*, exactly, the driver computes. They build in order: the sampling
problem, ordinary Monte Carlo, detailed balance, the density of states,
Wang-Landau, the 1/t refinement, multiple walkers, and replica exchange. The
final chapter is the validation argument — how we know the implementation
reproduces a known-exact answer.

For the same methods *experienced* on a running system, see the
{doc}`tutorials <../auto_tutorials/index>`; for short scripts to adapt, the
{doc}`examples <../auto_examples/index>`.

## Notation

| Symbol | Meaning |
| --- | --- |
| $Q$ | the order parameter being sampled (often the energy $E$) |
| $g(Q)$ | the density of states; in code, `result.g` is its **logarithm** |
| $\beta$ | inverse temperature $1/(k_B T)$ |
| $H(Q)$ | the visit histogram within an f-stage |
| $f$, $\ln f$ | the modification factor and its log (the bias increment) |
| $\varepsilon$ | per-bin relative error against an exact reference |

```{toctree}
:maxdepth: 1

01-sampling-problem
02-monte-carlo
03-detailed-balance
04-density-of-states
05-wang-landau
06-one-over-t
07-multiple-walkers
08-replica-exchange
09-higher-d
10-validation
```
