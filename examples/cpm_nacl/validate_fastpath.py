"""Accuracy + speed check of the incremental fast paths for a SINGLE-atom move.

Builds a small Au||Au capacitor with Na+/Cl- point ions and, for one ion
translate, compares the incremental ``look_move`` (dE and induced electrode
charges) of three calculator configurations against an exact full re-solve
(``set_state`` of the moved configuration):

  * ewald, full SR      (incremental_sr=False)            -- exact baseline
  * incremental_sr=True, ewald  k-space
  * incremental_sr=True, green  k-space

Reports |dE error|, max charge error, and ms/look for each.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from torchcpm import IncrementalCalculator

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)


def build(n_pairs=40, nps=4, a=4.08, gap=20.0, seed=0):
    bot = [[(i + 0.5) * a, (j + 0.5) * a, 0.0] for i in range(nps) for j in range(nps)]
    top = [[(i + 0.5) * a, (j + 0.5) * a, gap] for i in range(nps) for j in range(nps)]
    el = np.array(bot + top)
    n_el = len(el)
    rng = np.random.default_rng(seed)
    ions, q = [], []
    for _ in range(n_pairs):
        ions.append([rng.uniform(0, nps * a), rng.uniform(0, nps * a), rng.uniform(3, gap - 3)])
        q.append(+1.0)
        ions.append([rng.uniform(0, nps * a), rng.uniform(0, nps * a), rng.uniform(3, gap - 3)])
        q.append(-1.0)
    pos = torch.tensor(np.concatenate([el, np.array(ions)]))
    chg = torch.tensor(np.concatenate([np.zeros(n_el), q]))
    cell = torch.diag(torch.tensor([nps * a, nps * a, gap + 30.0]))
    phi = [-0.5] * (n_el // 2) + [0.5] * (n_el // 2)
    return pos, chg, cell, phi, n_el


def make(pos, chg, cell, phi, n_el, **kw):
    c = IncrementalCalculator(
        electrode_indices=list(range(n_el)),
        electrode_eta=1.0 / math.sqrt(2.0),
        applied_potential=phi,
        cell=cell,
        pbc=(True, True, False),
        sr_cutoff=12.0,
        lr_wavelength=0.5,
        **kw,
    )
    c.set_state(pos, chg)
    return c


def main():
    pos, chg, cell, phi, n_el = build()
    print(f"system: {n_el} electrode + {pos.shape[0] - n_el} ions = {pos.shape[0]} atoms")
    ion0 = n_el
    gi = torch.tensor([ion0])
    newp = pos[ion0 : ion0 + 1].clone() + torch.tensor([[0.3, -0.2, 0.1]])

    # Exact reference: full re-solve of the moved configuration.
    cref = make(pos, chg, cell, phi, n_el)
    E0 = float(cref.energy)
    mp = pos.clone()
    mp[ion0] = newp[0]
    cref.set_state(mp, chg)
    dE_ref = float(cref.energy) - E0
    q_ref = cref.electrode_charges.clone()

    configs = [
        ("ewald, full SR", dict()),
        ("incremental_sr + ewald", dict(incremental_sr=True)),
        ("incremental_sr + green", dict(incremental_sr=True, incremental_kspace="green")),
    ]
    print(f"{'config':<26} {'ms/look':>8} {'|dE err| eV':>13} {'max|dq| e':>12}")
    for label, kw in configs:
        try:
            c = make(pos, chg, cell, phi, n_el, **kw)
            dE, qmv = c.look_move(gi, newp)
            e_err = abs(float(dE) - dE_ref)
            q_err = float((qmv - q_ref).abs().max())
            for _ in range(10):
                c.look_move(gi, newp)
            t0 = time.perf_counter()
            K = 800
            for _ in range(K):
                c.look_move(gi, newp)
            dt = (time.perf_counter() - t0) / K * 1000
            print(f"{label:<26} {dt:>8.3f} {e_err:>13.2e} {q_err:>12.2e}")
        except Exception as ex:
            print(f"{label:<26}   ERROR: {ex}")


if __name__ == "__main__":
    main()
