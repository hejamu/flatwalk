"""Dump a viewable extxyz trajectory of the CPM water capacitor MC.

Writes electrode (Au) + SPC/E water sites per frame, with the CPM-induced
per-atom charge as an extxyz field (``initial_charges``) so you can colour by
it in OVITO/VMD/ASE. Frame info carries top/bottom-plate charge, energy, step.

Uses the fast *incremental* contract (~4x the snapshot path), so it can run long
enough to actually melt the seeded grid and show the charge developing. Saves
the trajectory incrementally every ``--flush-every`` frames so you can open it
while it grows.

    python make_trajectory.py --n-steps 300000 --dump-every 300 --out water_capacitor.xyz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # examples/ for gcmc
from flatwalk_cpm.wl_electrode_charge import (  # noqa: E402
    KB_EV,
    IncrementalContract,
    build_capacitor,
    seed_waters,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=300000)
    ap.add_argument("--dump-every", type=int, default=300, help="record a frame every k steps")
    ap.add_argument("--flush-every", type=int, default=50, help="write file every k frames")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--n-water", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="water_capacitor.xyz")
    args = ap.parse_args()

    from ase import Atoms
    from ase.io import write

    beta = 1.0 / (KB_EV * args.temperature)
    geom = build_capacitor()
    n = seed_waters(geom, n_target=args.n_water, seed=args.seed)
    backend = geom.backend
    n_per_plate = geom.n_per_plate
    contract = IncrementalContract(geom)
    n_frames_planned = args.n_steps // args.dump_every
    print(
        f"Seeded {n} waters; {args.n_steps:,} MC steps (incremental), frame every "
        f"{args.dump_every} -> ~{n_frames_planned} frames to {args.out}"
    )

    el_pos = backend._electrode_pos.detach().cpu().numpy()
    el_sym = ["Au"] * el_pos.shape[0]
    cell = backend.cell().detach().cpu().numpy()
    water_sym: list[str] = []
    for mi in range(backend.n_molecules):
        water_sym += list(backend.molecule_templates()[backend.molecule_kind(mi)].site_types)

    rng = np.random.default_rng(args.seed)
    cur = contract.initial_state
    e_cur = contract.energy_fn(cur)
    frames: list = []
    out = Path(args.out)

    for i in range(args.n_steps):
        new, lpr = contract.propose_move_fn(cur, rng)
        e_new = contract.energy_fn(new)
        if e_new - e_cur < 0 or rng.random() < math.exp(min(-beta * (e_new - e_cur) + lpr, 0.0)):
            cur, e_cur = new, e_new
        if i % args.dump_every == 0:
            contract._resolve(cur)  # sync backend to the kept state
            wpos = backend.ion_positions().detach().cpu().numpy()
            qel = backend.electrode_charges().detach().cpu().numpy()
            wq = backend.ion_charges().detach().cpu().numpy()
            atoms = Atoms(
                symbols=el_sym + water_sym,
                positions=np.vstack([el_pos, wpos]),
                cell=cell,
                pbc=(True, True, False),
            )
            atoms.set_initial_charges(np.concatenate([qel, wq]))
            atoms.info["step"] = i
            atoms.info["Q_top"] = float(qel[n_per_plate:].sum())
            atoms.info["Q_bot"] = float(qel[:n_per_plate].sum())
            atoms.info["energy_eV"] = float(e_cur)
            frames.append(atoms)
            if len(frames) % args.flush_every == 0:
                write(str(out), frames, format="extxyz")

    write(str(out), frames, format="extxyz")
    qts = [f.info["Q_top"] for f in frames]
    print(f"Wrote {len(frames)} frames -> {out}")
    print(f"Q_top over trajectory: {min(qts):+.3f} .. {max(qts):+.3f} e")


if __name__ == "__main__":
    main()
