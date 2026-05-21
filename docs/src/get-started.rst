Getting started
===============

The contract
------------

The user-supplied side of the contract:

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - You supply
     - Type
     - What flatwalk does with it
   * - ``bin_scheme``
     - :class:`~flatwalk.BinScheme` instance
     - maps ``Q → bin index``
   * - ``energy_fn(state)``
     - ``→ float``
     - the ``−β·ΔE`` term in WL acceptance (skip when ``β=0`` and ``Q=E``)
   * - ``order_parameter_fn(state)``
     - ``→ float | np.ndarray``
     - the quantity ``g(Q)`` is estimated over (vector for ≥2D)
   * - ``propose_move_fn(state, rng)``
     - ``→ (new_state, log_proposal_ratio)``
     - one Markov step

``state`` is opaque to flatwalk — whatever your callbacks recognise:
tuple, dataclass, numpy array, torch tensor, anything. You hand one
initial ``state`` object to :meth:`~flatwalk.WLDriver.run` to start;
from there the callbacks do all state manipulation.


Install
-------

Editable install via `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   uv venv .venv
   uv pip install --python .venv/bin/python -e ".[test]"

Plain pip works too (``pip install -e ".[test]"``) but Homebrew Python
may require ``--break-system-packages`` or a venv.


Quick start
-----------

Below, block 1 fills the four-piece contract for the 2D Ising model;
block 2 is the flatwalk setup and run — verbatim across systems.

.. code-block:: python

   import numpy as np
   from flatwalk import Bin1D, WLConfig, WLDriver

   # ──────────────────────────────────────────────────────────────────
   # 1. Your physics — replace this block to use a different system.
   #    flatwalk doesn't know or care what `state` is.
   # ──────────────────────────────────────────────────────────────────
   L = 8

   def energy_fn(state):
       return state[1]                                # cached E, O(1)

   def order_parameter_fn(state):
       return state[1]                                # WL on E: Q = E

   def propose_move_fn(state, rng):                   # single-spin flip
       spins, E = state
       i, j = int(rng.integers(0, L)), int(rng.integers(0, L))
       s = int(spins[i, j])
       nb_sum = int(spins[(i-1)%L, j] + spins[(i+1)%L, j] +
                    spins[i, (j-1)%L] + spins[i, (j+1)%L])
       dE = 2.0 * s * nb_sum                          # ΔE in O(1)
       new_spins = spins.copy(); new_spins[i, j] = -s
       return (new_spins, E + dE), 0.0                # symmetric → lpr = 0

   initial_state = (np.ones((L, L), dtype=np.int8), -2.0 * L * L)
   bin_scheme = Bin1D(low=-2*L*L - 2, high=2*L*L + 2, n_bins=L*L + 1)

   # ──────────────────────────────────────────────────────────────────
   # 2. Generic flatwalk wiring — unchanged across systems.
   # ──────────────────────────────────────────────────────────────────
   cfg = WLConfig(bin_scheme=bin_scheme, beta=0.0, ln_f_final=1e-8,
                  trace_path="trace.tsv")
   result = WLDriver(cfg).run(
       initial_state, energy_fn, order_parameter_fn, propose_move_fn,
       rng=np.random.default_rng(0),
   )
   print(result.g)                                    # log density of states

To run a different model you replace block 1 only (your callbacks,
``initial_state``, and the :class:`~flatwalk.Bin1D` range for your
``Q``); block 2 stays verbatim. See ``examples/ising.py`` for the
production Ising implementation used by the validation, and
``examples/ising_validation.py`` for the full pass/fail run.
