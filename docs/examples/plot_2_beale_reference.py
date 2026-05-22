"""
Beale's recursion vs brute-force enumeration on small lattices
==============================================================

flatwalk's reference for the 2D Ising validation is the integer
density of states ``n(E)``, computed by a Beale-style transfer-matrix
recursion (``examples/beale.py``). For small enough lattices we can
also enumerate every spin configuration directly and bin by energy.
This example runs both on L=3 (512 configurations) and L=4 (65,536
configurations) and checks bin-for-bin agreement, sanity-checking the
recursion before we lean on it to validate the WL driver on L=8.
"""

# %%
# Setup
# -----
#
# ``examples/beale.py`` lives next to the package. ``conf.py`` adds the
# repo's ``examples/`` directory to ``sys.path`` for the docs build; the
# ``try`` block here is only for standalone execution of the script.

import sys

try:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
except NameError:
    pass  # sphinx-gallery exec context: __file__ undefined, sys.path is already set

import beale  # noqa: E402

# %%
# L=3: 2^9 = 512 configurations
# -----------------------------
#
# At this scale we can print every bin side-by-side.

L = 3
n_brute = beale.brute_force_g_E(L)
n_beale = beale.beale_g_E(L)

print(
    f"L={L}: {len(n_beale)} distinct energies, total = {sum(n_beale.values())} = 2^{L * L}"
)
print(f"{'E':>6} {'Beale n(E)':>14} {'brute n(E)':>14}")
for E in sorted(n_beale):
    print(f"{E:>6} {n_beale[E]:>14d} {n_brute[E]:>14d}")

assert n_brute == n_beale, "L=3 Beale and brute force disagree!"

# %%
# L=4: 2^16 = 65,536 configurations
# ---------------------------------
#
# Brute force still finishes in ~1 s here; this is the largest lattice
# where direct enumeration is comfortable. Beale's recursion scales
# polynomially in ``L`` (the transfer matrix is ``2^L × 2^L``) and is
# what allows the L=8 validation later.

L = 4
n_brute = beale.brute_force_g_E(L)
n_beale = beale.beale_g_E(L)
assert n_brute == n_beale, "L=4 Beale and brute force disagree!"

print(
    f"L={L}: {len(n_beale)} distinct energies, total = {sum(n_beale.values())} = 2^{L * L}"
)
print(f"  All {len(n_beale)} bins match brute force.")
