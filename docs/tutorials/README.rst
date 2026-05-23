Tutorials
=========

A guided journey through flat-histogram sampling on **one system carried
throughout** — the 2D Ising model. We start with plain Metropolis Monte Carlo,
run into its limits, and fix them one method at a time: Wang-Landau to get the
whole density of states, the 1/t schedule to sharpen convergence, several
walkers to cut the variance, and replica exchange to make it robust and
parallel. A final step turns the converged ``g(E)`` into thermodynamics.

Read these in order — each picks up where the last left off. They are runnable
scripts (executed live on every documentation build) at smoke size; the
strict, slow ``L=8`` runs that meet the validation criteria live at the repo
root under ``examples/``. For short, copy-paste recipes per method rather than a
narrative, see the :doc:`examples </auto_examples/index>`; for the maths behind
each step, the :doc:`theory </theory/index>`.
