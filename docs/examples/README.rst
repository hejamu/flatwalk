Worked examples
===============

Runnable tutorials for the flat-histogram methods in ``flatwalk``, executed
live on every documentation build via ``sphinx-gallery``. They build up in
order — a toy first run, the exact Ising reference, single-walker
Wang-Landau, then replica exchange — and each is a self-contained script you
can also run from the command line.

These are fast smoke versions; the full ``ln_f_final = 1e-8`` ``L=8`` runs
that meet the spec pass criteria live at the repo root
(``examples/ising_validation.py``, ``examples/ising_rewl_validation.py``) and
run in CI's slow lane.
