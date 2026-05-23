# Overview

Monte Carlo is a standard method to compute equilibrium properties. You run a
Markov chain — a **walker** that hops from one configuration to the next —
drawing them with the Boltzmann probability $\propto e^{-\beta E}$ at a fixed
temperature $T$, then average whatever you care about over the samples. It works
well — but each run is tied to the one temperature it was run at. The walk
concentrates on the configurations that matter at that $T$, so a different
temperature means another run, and the rare configurations that dominate
elsewhere are barely sampled.

What you would often rather have is the **density of states** $g(E)$ — how many
configurations the system has at each energy. It is temperature-independent, and
from a single $g(E)$ you can reconstruct the partition function, and with it the
free energy, internal energy, entropy, and heat capacity at *every* temperature
at once. The trouble is you cannot get it by sampling directly: $g$ spans tens of
orders of magnitude between its peak and its tails, so sampling that visits
states in proportion to how common they are essentially never reaches the rare
ones, and the tails of $g$ go unmeasured.

**Wang-Landau** sampling breaks the impasse by learning $g$ as it goes. It builds
up a bias that pushes the walker away from energies it has already visited; once
that bias cancels the system's own degeneracy the walker spends equal time at
every energy — a flat histogram — and the accumulated bias *is* $\log g$.

## What flatwalk does

flatwalk implements Wang-Landau and its modern refinements: the 1/t schedule,
multiple walkers, and replica exchange. The order parameter need not be the
energy — any quantity $Q$ you choose gives a $g(Q)$ the same way. Multiple order
parameters are also possible.

## What makes flatwalk different

No widely used Wang-Landau implementation is simultaneously
**order-parameter agnostic** and **energy-backend agnostic**. Most mature tools
are coupled to a given state representation and a fixed catalogue of potentials,
making them hard to adapt to new systems.

flatwalk makes a deliberate cut at a callable boundary instead. The bookkeeping
lives in the driver; the physics lives in user-supplied functions. You hand
flatwalk {doc}`four callables <the-contract>` (the "contract") — your `state` can
be anything they understand, and flatwalk never inspects it: no inheritance, no
fixed catalogue, no recompile.

## When to use it

Reach for flatwalk when you want flat-histogram sampling and your system is
awkward to express in an existing tool but easy to express as a few Python
callbacks — and you'd rather not write the Wang-Landau machinery yourself.

If you have a system that fits into an existing sampler, that sampler is likely
to be more efficient than flatwalk.

## Where to go next

- {doc}`install` — get it running.
- {doc}`the-contract` — the four callables you supply.
- {doc}`quickstart` — a complete Ising run in two blocks.
- {doc}`Tutorials <../auto_tutorials/index>` — a guided journey from plain
  Monte Carlo to replica exchange on one system.
- {doc}`Examples <../auto_examples/index>` — short recipes to adapt for your
  own system.
- {doc}`Theory <../theory/index>` — the methods, derived.
