Examples
========

Short, adaptable **recipes** — one per flat-histogram method — for a user
writing their own contract. Each is a self-contained script: keep the generic
flatwalk wiring, swap in your own physics. They open with a line pointing at the
tutorial or theory that explains the method, then stay terse.

These are fast smoke versions (small ``L``, loose ``ln_f_final``) executed live
on every documentation build. The full ``ln_f_final = 1e-8`` ``L=8`` runs that
meet the strict pass criteria live at the repo root
(``examples/ising_validation.py``, ``examples/ising_rewl_validation.py``) and
run in CI's slow lane.

For a guided, build-it-up narrative on a single system instead of standalone
recipes, follow the :doc:`tutorials </auto_tutorials/index>`.
