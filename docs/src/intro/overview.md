# Overview

## What flatwalk does

Most Monte Carlo sampling answers a question *at one temperature*: draw
configurations with probability $\propto e^{-\beta E}$, average what you care
about. Flat-histogram methods answer a different, more powerful question. They
estimate the **density of states** $g(Q)$ — how many configurations the system
has at each value of an order parameter $Q$ — directly. From a single $g(E)$
you can reconstruct the partition function, and with it the free energy,
internal energy, entropy, and heat capacity at *every* temperature, all from
one run.

The catch is that $g$ spans many orders of magnitude, so an unbiased walker
never visits the rare states. **Wang-Landau** sampling fixes this by building
up a bias as it goes, exactly cancelling the system's own entropy, so the
walker spends equal time in every bin and the accumulated bias *is* $\log g$.
flatwalk implements Wang-Landau and its modern refinements (the 1/t schedule,
multiple walkers, replica exchange).

```{seealso}
The {doc}`theory section <../theory/index>` derives all of this from scratch;
the {doc}`tutorials <../auto_tutorials/index>` build it up on one running
example.
```

## What makes flatwalk different

No widely used Wang-Landau implementation is simultaneously
**order-parameter agnostic** and **energy-backend agnostic**. Mature tools are
coupled to a particle-based state representation and a fixed catalogue of
potentials; a custom Hamiltonian, a non-particle lattice model, or a modern
framework (PyTorch, JAX, …) means subclassing in someone else's C++.

flatwalk makes the cut at a callable boundary instead. The bookkeeping (bins,
bias, flatness check, the f-stage schedule, the 1/t transition, checkpointing,
diagnostics) lives in the driver; the physics (what a configuration is, how to
change one, what its energy and order parameter are) lives in user-supplied
functions. You hand flatwalk {doc}`four callables <the-contract>` — no
inheritance, no recompile, no constraint on what your `state` is or what
evaluates its energy.

The design rationale is written up in the {doc}`storyline
<../background/storyline>`.

## When to use it

Reach for flatwalk when you want a temperature-independent picture of a
system — a full $g(Q)$, free-energy profiles, or thermodynamics across a range
of $T$ from one run — and your system is awkward to express in an existing
sampler but easy to express as a few Python callbacks. It shines when the order
parameter has a rugged or barrier-crossing landscape that defeats plain
Metropolis sampling.

It is *not* the tool when a single canonical average at one temperature is all
you need and ordinary Metropolis already mixes well — there the extra
machinery buys you nothing.

## Where to go next

- {doc}`install` — get it running.
- {doc}`the-contract` — the four callables you supply.
- {doc}`quickstart` — a complete Ising run in two blocks.
- {doc}`Tutorials <../auto_tutorials/index>` — a guided journey from plain
  Monte Carlo to replica exchange on one system.
- {doc}`Examples <../auto_examples/index>` — short recipes to adapt for your
  own system.
- {doc}`Theory <../theory/index>` — the methods, derived.
