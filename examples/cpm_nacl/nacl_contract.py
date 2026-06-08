"""flatwalk contract for a NaCl-in-capacitor cell on the fast exact CPM path.

Single-atom Na+/Cl- ions hit torch-CPM's incremental_sr fast path (rigid water
does not), so this builds the MC contract directly on
:class:`torchcpm.IncrementalCalculator` with ``incremental_sr=True`` -- *exact*
as long as ``sr_cutoff <= L/2`` (the calculator now guards this). A small
incremental Lennard-Jones core (recomputed only for the moved ion) keeps the
point ions from collapsing; LJ uses the same minimum-image regime.

Order parameter Q = induced charge on the top (positive) plate.

Modes:
    --validate            check incremental (dE, Q) vs a full re-solve
    --benchmark           ms/step
    --plain-mc N          canonical MC, measure Var(Q) -> differential capacitance
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from torchcpm import IncrementalCalculator

KB_EV = 8.617333262e-5
KE = 14.399645478425668  # e^2 / (4 pi eps0) in eV*Angstrom


# ======================================================================
# Geometry
# ======================================================================


@dataclass
class NaClCell:
    calc: IncrementalCalculator
    cell: torch.Tensor
    inv_cell: torch.Tensor
    n_el: int
    top_idx: torch.Tensor          # electrode indices of the +V plate
    ion_charges: torch.Tensor      # (n_ion,)
    lj_sigma: float
    lj_eps: float
    lj_rc: float
    z_lo: float
    z_hi: float
    dtype: torch.dtype = torch.float64
    device: torch.device = torch.device("cpu")

    @property
    def n_ion(self) -> int:
        return int(self.ion_charges.shape[0])


def build_nacl_capacitor(
    *,
    nps: int = 8,
    a: float = 4.08,
    gap: float = 20.0,
    vacuum: float = 30.0,
    n_pairs: int = 64,
    phi_volt: float = 1.0,
    sr_cutoff: float = 12.0,
    lr_wavelength: float = 0.5,
    eta: float = 1.0 / math.sqrt(2.0),
    lj_sigma: float = 2.5,
    lj_eps: float = 0.01,
    lj_rc: float = 8.0,
    wall_clearance: float = 2.5,
    init: str = "random",
    layer: float = 4.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> NaClCell:
    dev = torch.device(device)
    L = nps * a
    if max(sr_cutoff, lj_rc) > 0.5 * L + 1e-9:
        raise ValueError(
            f"need L/2 ({0.5 * L:.2f}) >= max(sr_cutoff,lj_rc) ({max(sr_cutoff, lj_rc):.2f}); "
            f"increase nps."
        )
    bot = [[(i + 0.5) * a, (j + 0.5) * a, 0.0] for i in range(nps) for j in range(nps)]
    top = [[(i + 0.5) * a, (j + 0.5) * a, gap] for i in range(nps) for j in range(nps)]
    el = np.array(bot + top)
    n_el = len(el)
    n_per_plate = nps * nps

    rng = np.random.default_rng(seed)
    z_lo, z_hi = wall_clearance, gap - wall_clearance
    ions, q = [], []
    for _ in range(n_pairs):
        for charge in (+1.0, -1.0):
            if init == "layered":
                # Pre-form the equilibrium double layer: cations (Na+) at the
                # negative (bottom) plate, anions (Cl-) at the positive (top)
                # plate -> start near equilibrium, skip the slow ion migration.
                z = rng.uniform(z_lo, z_lo + layer) if charge > 0 else rng.uniform(z_hi - layer, z_hi)
            elif init == "random":
                z = rng.uniform(z_lo, z_hi)
            else:
                raise ValueError(f"init must be 'random' or 'layered'; got {init!r}")
            ions.append([rng.uniform(0, L), rng.uniform(0, L), z])
            q.append(charge)
    ions = np.array(ions)
    ion_q = np.array(q)

    pos = torch.tensor(np.concatenate([el, ions]), dtype=dtype, device=dev)
    chg = torch.tensor(np.concatenate([np.zeros(n_el), ion_q]), dtype=dtype, device=dev)
    cell = torch.diag(torch.tensor([L, L, gap + vacuum], dtype=dtype, device=dev))
    phi = [-phi_volt / 2.0] * n_per_plate + [+phi_volt / 2.0] * n_per_plate

    calc = IncrementalCalculator(
        electrode_indices=list(range(n_el)),
        electrode_eta=eta,
        applied_potential=phi,
        cell=cell,
        pbc=(True, True, False),
        sr_cutoff=sr_cutoff,
        lr_wavelength=lr_wavelength,
        incremental_sr=True,  # exact here because sr_cutoff <= L/2
        device=device,
        dtype=dtype,
    )
    calc.set_state(pos, chg)
    top_idx = torch.arange(n_per_plate, 2 * n_per_plate, device=dev)
    return NaClCell(
        calc=calc,
        cell=cell,
        inv_cell=torch.linalg.inv(cell),
        n_el=n_el,
        top_idx=top_idx,
        ion_charges=chg[n_el:].clone(),
        lj_sigma=lj_sigma,
        lj_eps=lj_eps,
        lj_rc=lj_rc,
        z_lo=z_lo,
        z_hi=z_hi,
        dtype=dtype,
        device=dev,
    )


# ======================================================================
# Lennard-Jones (generic repulsive core on all atoms; excl. electrode-electrode)
# ======================================================================


def _mic_dists(point: torch.Tensor, others: torch.Tensor, cell: torch.Tensor, inv_cell: torch.Tensor):
    """Minimum-image distances from one ``point`` (3,) to each row of ``others``.
    Exact in the minimum-image regime (cutoff <= L/2)."""
    frac = (others - point) @ inv_cell
    frac = frac - torch.round(frac)
    return (frac @ cell).norm(dim=1)


def _lj(d: torch.Tensor, sigma: float, eps: float, rc: float) -> torch.Tensor:
    inside = (d < rc) & (d > 0)
    dc = d.clamp_min(1e-9)
    s6 = (sigma / dc) ** 6
    return torch.where(inside, 4.0 * eps * (s6 * s6 - s6), torch.zeros_like(d))


def _lj_total_clean(cell: NaClCell, positions: torch.Tensor) -> float:
    """Full LJ over all pairs except electrode-electrode (computed once at init).
    Vectorised over the ion rows: for each ion, sum LJ to every later atom so
    each pair is counted once; ion-electrode pairs are all captured because
    every such pair has its ion endpoint iterated."""
    n = positions.shape[0]
    n_el = cell.n_el
    total = 0.0
    for a in range(n_el, n):  # iterate ions only
        others = positions[a + 1 :]  # later atoms -> each pair counted once
        if others.shape[0] == 0:
            continue
        d = _mic_dists(positions[a], others, cell.cell, cell.inv_cell)
        total += float(_lj(d, cell.lj_sigma, cell.lj_eps, cell.lj_rc).sum())
    # ion-electrode pairs with the electrode BEFORE the ion in the array are
    # missed by the "later atoms" trick (electrodes occupy indices < n_el),
    # so add ion->electrode explicitly.
    el = positions[:n_el]
    for a in range(n_el, n):
        d = _mic_dists(positions[a], el, cell.cell, cell.inv_cell)
        total += float(_lj(d, cell.lj_sigma, cell.lj_eps, cell.lj_rc).sum())
    return total


def lj_delta_move(cell: NaClCell, positions: torch.Tensor, a: int, r_new: torch.Tensor) -> float:
    """LJ energy change if ion ``a`` (global index) moves to ``r_new``.
    Only pairs involving ``a`` change; minimum-image, O(N)."""
    mask = torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)
    mask[a] = False
    others = positions[mask]
    d_old = _mic_dists(positions[a], others, cell.cell, cell.inv_cell)
    d_new = _mic_dists(r_new, others, cell.cell, cell.inv_cell)
    e_old = _lj(d_old, cell.lj_sigma, cell.lj_eps, cell.lj_rc).sum()
    e_new = _lj(d_new, cell.lj_sigma, cell.lj_eps, cell.lj_rc).sum()
    return float(e_new - e_old)


# ======================================================================
# flatwalk contract (single-ion translate; lazy commit/reject)
# ======================================================================


@dataclass
class IonToken:
    Q: float
    E: float


class NaClContract:
    def __init__(self, cell: NaClCell, *, max_disp: float = 0.5):
        self.c = cell
        self.calc = cell.calc
        self.max_disp = max_disp
        self.n_el = cell.n_el
        self.n_ion = cell.ion_charges.shape[0]
        self.top = cell.top_idx
        # absolute committed energy = CPM (Omega) + LJ, tracked via the token chain
        e_cpm = float(self.calc.energy)
        e_lj = _lj_total_clean(cell, self.calc.positions)
        q0 = float(self.calc.electrode_charges[self.top].sum())
        self.initial_state = IonToken(Q=q0, E=e_cpm + e_lj)
        self._committed_E = self.initial_state.E
        self._committed_token = self.initial_state
        self._pending: dict | None = None  # {token, a, r_new}

    def _resolve(self, incoming: IonToken) -> None:
        if self._pending is not None:
            if incoming is self._pending["token"]:
                a = self._pending["a"]
                self.calc.move_atoms(torch.tensor([a]), self._pending["r_new"].unsqueeze(0))
                self._committed_E = self._pending["token"].E
                self._committed_token = self._pending["token"]
            # reject: look_move was non-mutating -> nothing to roll back
            self._pending = None

    def energy_fn(self, s: IonToken) -> float:
        return s.E

    def order_parameter_fn(self, s: IonToken) -> float:
        return s.Q

    def propose_move_fn(self, s: IonToken, rng: np.random.Generator):
        self._resolve(s)
        a_ion = int(rng.integers(self.n_ion))
        a = self.n_el + a_ion
        r_old = self.calc.positions[a]
        disp = torch.tensor(
            (rng.random(3) * 2.0 - 1.0) * self.max_disp, dtype=self.c.dtype, device=self.c.device
        )
        r_new = r_old + disp
        # hard wall: keep ion COM between the plates (z non-periodic)
        z_new = float(r_new[2])
        if z_new < self.c.z_lo or z_new > self.c.z_hi:
            return IonToken(Q=s.Q, E=math.inf), 0.0
        gi = torch.tensor([a], device=self.c.device)
        dE_cpm, q_pending = self.calc.look_move(gi, r_new.unsqueeze(0))
        dE_lj = lj_delta_move(self.c, self.calc.positions, a, r_new)
        tok = IonToken(
            Q=float(q_pending[self.top].sum()),
            E=self._committed_E + float(dE_cpm) + dE_lj,
        )
        self._pending = {"token": tok, "a": a, "r_new": r_new}
        return tok, 0.0


# ======================================================================
# validate / benchmark / run
# ======================================================================


def _block_analysis(q: np.ndarray, beta: float, n_blocks: int = 20) -> dict:
    n = len(q)
    bs = n // n_blocks
    if bs < 1:
        return {"n_samples": n}
    blk = q[: bs * n_blocks].reshape(n_blocks, bs)
    var = float(q.var())
    var_err = float(blk.var(axis=1).std(ddof=1) / math.sqrt(n_blocks))
    mean = float(q.mean())
    mean_err = float(blk.mean(axis=1).std(ddof=1) / math.sqrt(n_blocks))
    tau = float(bs * blk.mean(axis=1).var(ddof=1) / var) if var > 0 else float("nan")
    return {
        "q_mean": mean, "q_mean_err": mean_err, "q_var": var, "q_var_err": var_err,
        "C_eV": var * beta, "C_err_eV": var_err * beta,
        "tau": tau, "n_eff": n / max(tau, 1.0),
    }


def cmd_validate(args):
    cell = build_nacl_capacitor(nps=args.nps, n_pairs=args.n_pairs, seed=args.seed, init=args.init, device=args.device, dtype=args.torch_dtype)
    print(f"NaCl: {cell.n_el} electrode + {cell.n_ion} ions, L={float(cell.cell[0,0]):.1f} "
          f"(L/2={float(cell.cell[0,0])/2:.1f}), sr_cutoff/L2 ok")
    calc = cell.calc
    pos0 = calc.positions.clone()
    chg = calc.charges.clone()
    e_cpm0 = float(calc.energy)
    e_lj0 = _lj_total_clean(cell, pos0)
    rng = np.random.default_rng(1)
    max_err_cpm = max_err_lj = max_err_q = 0.0
    for _ in range(8):
        a = cell.n_el + int(rng.integers(cell.n_ion))
        r_new = pos0[a] + torch.tensor(
            (rng.random(3) * 2 - 1) * 0.4, dtype=cell.dtype, device=cell.device
        )
        dE_cpm, q_inc = calc.look_move(torch.tensor([a], device=cell.device), r_new.unsqueeze(0))
        dE_lj = lj_delta_move(cell, pos0, a, r_new)
        # reference: full re-solve + full LJ of the moved config
        mp = pos0.clone(); mp[a] = r_new
        calc.set_state(mp, chg)
        dE_cpm_ref = float(calc.energy) - e_cpm0
        q_ref = calc.electrode_charges[cell.top_idx].sum()
        e_lj_ref = _lj_total_clean(cell, mp) - e_lj0
        calc.set_state(pos0, chg)  # restore
        max_err_cpm = max(max_err_cpm, abs(float(dE_cpm) - dE_cpm_ref))
        max_err_lj = max(max_err_lj, abs(dE_lj - e_lj_ref))
        max_err_q = max(max_err_q, abs(float(q_inc[cell.top_idx].sum()) - float(q_ref)))
    print(f"max |dE_cpm err| = {max_err_cpm:.2e} eV")
    print(f"max |dE_lj  err| = {max_err_lj:.2e} eV")
    print(f"max |Q err|      = {max_err_q:.2e} e")
    ok = max(max_err_cpm, max_err_lj, max_err_q) < 1e-10
    print("VALIDATION:", "PASS" if ok else "FAIL")


def cmd_benchmark(args):
    cell = build_nacl_capacitor(nps=args.nps, n_pairs=args.n_pairs, seed=args.seed, init=args.init, device=args.device, dtype=args.torch_dtype)
    con = NaClContract(cell)
    rng = np.random.default_rng(0)
    s = con.initial_state
    for _ in range(20):
        s2, _ = con.propose_move_fn(s, rng); con.energy_fn(s2); con.order_parameter_fn(s2)
        s = s2 if (s2.E != math.inf and rng.random() < 0.5) else s
    t0 = time.perf_counter(); K = 2000
    for _ in range(K):
        s2, _ = con.propose_move_fn(s, rng); con.energy_fn(s2); con.order_parameter_fn(s2)
        s = s2 if (s2.E != math.inf and rng.random() < 0.5) else s
    print(f"{(time.perf_counter()-t0)/K*1000:.3f} ms/step  ({cell.n_el+cell.n_ion} atoms)")


def cmd_plain_mc(args):
    cell = build_nacl_capacitor(nps=args.nps, n_pairs=args.n_pairs, seed=args.seed, init=args.init, device=args.device, dtype=args.torch_dtype)
    con = NaClContract(cell, max_disp=args.max_disp)
    beta = 1.0 / (KB_EV * args.temperature)
    print(f"NaCl {cell.n_el} el + {cell.n_ion} ions; beta={beta:.2f}; plain MC {args.plain_mc} steps")
    rng = np.random.default_rng(args.seed + 1)
    cur = con.initial_state
    e_cur = con.energy_fn(cur)
    qs = np.empty(args.plain_mc // args.trace_every + 1)
    j = 0; nacc = 0
    for i in range(args.plain_mc):
        new, lpr = con.propose_move_fn(cur, rng)
        e_new = con.energy_fn(new)
        if e_new - e_cur < 0 or rng.random() < math.exp(min(-beta * (e_new - e_cur) + lpr, 0.0)):
            cur, e_cur = new, e_new; nacc += 1
        if i % args.trace_every == 0:
            qs[j] = con.order_parameter_fn(cur); j += 1
        if i and i % args.save_every == 0:
            np.savez(args.out, q_trace=qs[:j], trace_every=args.trace_every, beta=beta, step=i)
            a = _block_analysis(qs[int(0.3 * j):j], beta)
            print(f"  step {i} acc={nacc/(i+1):.2f} <Q>={a.get('q_mean',float('nan')):+.4f} "
                  f"C={a.get('C_eV',float('nan')):.4f}+-{a.get('C_err_eV',float('nan')):.4f} e/V "
                  f"n_eff~{a.get('n_eff',float('nan')):.0f}", flush=True)
    np.savez(args.out, q_trace=qs[:j], trace_every=args.trace_every, beta=beta, step=args.plain_mc)
    a = _block_analysis(qs[int(0.3 * j):j], beta)
    e = 1.602176634e-19; A = float(cell.cell[0, 0]) * float(cell.cell[1, 1]) * 1e-16
    print(f"FINAL <Q>={a['q_mean']:+.4f}+-{a['q_mean_err']:.4f} e | "
          f"C={a['C_eV']:.4f}+-{a['C_err_eV']:.4f} e/V = {a['C_eV']*e/A*1e6:.2f} uF/cm^2 | "
          f"acc={nacc/args.plain_mc:.2f} n_eff~{a['n_eff']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nps", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--init", choices=["random", "layered"], default="random",
                    help="ion seeding: 'layered' pre-forms the double layer (faster equilibration)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (single-walker; no batching)")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--max-disp", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100000)
    ap.add_argument("--out", type=str, default="nacl_qtrace.npz")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--plain-mc", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="single precision (float32) instead of float64")
    args = ap.parse_args()
    args.torch_dtype = torch.float32 if args.fp32 else torch.float64
    torch.set_default_dtype(args.torch_dtype)
    print(f"precision: {'float32' if args.fp32 else 'float64'} | device: {args.device}")
    if args.validate:
        cmd_validate(args)
    elif args.benchmark:
        cmd_benchmark(args)
    elif args.plain_mc > 0:
        cmd_plain_mc(args)
    else:
        print("nothing to do; pass --validate / --benchmark / --plain-mc N")


if __name__ == "__main__":
    main()
