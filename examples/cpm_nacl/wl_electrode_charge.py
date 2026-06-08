"""Wang-Landau free-energy profile of the electrode charge (flatwalk x torch-CPM).

This is the *torch-CPM contract* for flatwalk: it fills flatwalk's four-callback
interface so Wang-Landau flat-histogram sampling runs on a constant-potential
electrode/water cell, recovering

    g(Q) = -ln P_T(Q) + const = beta * F(Q) + const,

the free-energy profile of the **electrode charge Q** (charge on one plate) at
a fixed applied potential and temperature.

Design (the "snapshot / functional" adapter)
--------------------------------------------
flatwalk's contract is stateless -- ``propose_move_fn(state) -> new_state`` and a
pure ``energy_fn(state)`` / ``order_parameter_fn(state)`` -- whereas the CPM
backend is a stateful look/leap engine. We bridge them by making ``state`` a
*snapshot* of the mobile water configuration and using
``CPMWaterBackend.evaluate(...)`` as a pure evaluator (one full CPM solve per
proposed state). No incremental commit/reject bookkeeping leaks into flatwalk.

Physics choices (see the conversation that produced this file):
* ``Q`` = induced charge on the **top (positive) plate**.  In a two-plate
  ``constant_V`` cell the *total* electrode charge is ~0 by neutrality, so Q must
  be per-plate.
* ``beta = 1 / (k_B T)`` with T = 300 K (NOT beta = 0; that would be the
  microcanonical g(E) case, which only applies when Q is the energy itself).
* moves = rigid-water translate + rotate, fixed step sizes (WL needs a
  stationary proposal), symmetric so ``log_proposal_ratio = 0``.
* a hard wall keeps water COMs between the plates: dilute water would otherwise
  drift through the porous top plate into the slab's vacuum buffer and be lost.

Run ``python wl_electrode_charge.py --help`` for knobs; ``--smoke`` runs a tiny
end-to-end wiring check.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --- make the sibling ``gcmc`` package and the flatwalk repo importable -------
_THIS = Path(__file__).resolve()
_EXAMPLES_DIR = _THIS.parent.parent  # torch_CPM/examples
sys.path.insert(0, str(_EXAMPLES_DIR))
_FLATWALK_REPO = Path("/Users/hjaeger/repos/flatwalk")
if _FLATWALK_REPO.exists():
    sys.path.insert(0, str(_FLATWALK_REPO))

from gcmc import SPC_E, CPMWaterBackend  # noqa: E402
from gcmc.rigid_body import random_rotation_matrix  # noqa: E402

from flatwalk import Bin1D, WLConfig, WLDriver  # noqa: E402

# Boltzmann constant in eV/K (ASE convention) -- matches the GCMC driver.
KB_EV = 8.617333262e-5
# Bulk water number density at ambient conditions, molecules / A^3.
RHO_BULK = 0.03342


# ======================================================================
# 1. Build the capacitor + seed water at ~bulk density
# ======================================================================


@dataclass
class CellGeometry:
    """Static geometry the contract closures need to know about."""

    backend: CPMWaterBackend
    n_per_plate: int
    top_indices: torch.Tensor  # electrode indices whose charge sum is Q
    z_lo: float  # water-COM confinement (inclusive) between the plates
    z_hi: float
    lx: float
    ly: float


def build_capacitor(
    *,
    n_plate: int = 4,
    lateral_a: float = 4.08,
    plate_gap: float = 22.0,
    vacuum_top: float = 30.0,
    eta_au: float = 1.0 / math.sqrt(2.0),
    phi_volt: float = 1.0,
    sr_cutoff: float = 9.0,
    wall_clearance: float = 2.5,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> CellGeometry:
    """Two Au(100)-like plates, bottom at ``-phi/2`` V and top at ``+phi/2`` V,
    slab geometry (periodic xy, vacuum above the top plate)."""
    lx = ly = n_plate * lateral_a
    bot, top = [], []
    for i in range(n_plate):
        for j in range(n_plate):
            x = (i + 0.5) * lateral_a
            y = (j + 0.5) * lateral_a
            bot.append([x, y, 0.0])
            top.append([x, y, plate_gap])
    electrode_positions = torch.tensor(bot + top, dtype=dtype, device=device)
    n_per_plate = n_plate * n_plate
    electrode_types = ["Au"] * (2 * n_per_plate)
    phi = [-phi_volt / 2.0] * n_per_plate + [+phi_volt / 2.0] * n_per_plate
    cell = [lx, ly, plate_gap + vacuum_top]

    lj_params = {
        "O": (3.16555789, 0.0067392),  # SPC/E oxygen
        "Au": (2.951, 0.2266),
    }

    backend = CPMWaterBackend(
        electrode_positions=electrode_positions,
        electrode_types=electrode_types,
        electrode_eta=eta_au,
        cell=cell,
        templates=[SPC_E],
        applied_potential=phi,
        boundary_condition="constant_V",
        sr_cutoff=sr_cutoff,
        lj_params=lj_params,
        pbc=(True, True, False),
        dtype=dtype,
        device=device,
    )
    # Top plate is the second half of the electrode index range.
    top_indices = torch.arange(n_per_plate, 2 * n_per_plate, device=device)
    return CellGeometry(
        backend=backend,
        n_per_plate=n_per_plate,
        top_indices=top_indices,
        z_lo=wall_clearance,
        z_hi=plate_gap - wall_clearance,
        lx=lx,
        ly=ly,
    )


def seed_waters(
    geom: CellGeometry,
    *,
    n_target: int | None = None,
    min_spacing: float = 2.9,
    seed: int = 0,
) -> int:
    """Seed rigid waters on a clearance grid between the plates at ~bulk
    density, with random orientations. Returns the number actually placed
    (capped by what the grid holds at ``min_spacing``)."""
    backend = geom.backend
    dtype = backend.dtype
    device = backend.device
    lx, ly = geom.lx, geom.ly
    z_lo, z_hi = geom.z_lo, geom.z_hi
    depth = z_hi - z_lo

    if n_target is None:
        n_target = int(round(RHO_BULK * lx * ly * depth))

    nx = max(1, int(lx // min_spacing))
    ny = max(1, int(ly // min_spacing))
    nz = max(1, int(depth // min_spacing))
    capacity = nx * ny * nz
    n = min(n_target, capacity)

    rng = torch.Generator(device=device).manual_seed(seed)
    placed = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if placed >= n:
                    break
                com = torch.tensor(
                    [
                        (i + 0.5) * lx / nx,
                        (j + 0.5) * ly / ny,
                        z_lo + (k + 0.5) * depth / nz,
                    ],
                    dtype=dtype,
                    device=device,
                )
                R = random_rotation_matrix(rng, device=device, dtype=dtype)
                backend.add_molecule(0, com, R)
                placed += 1
    return placed


# ======================================================================
# 2. The flatwalk contract (snapshot / functional adapter)
# ======================================================================


@dataclass
class WaterState:
    """flatwalk's opaque ``state``: just the mobile water-site positions.

    Topology (charges, types, molecule slices) is fixed across translate /
    rotate moves, so it lives in the closure, not here. ``energy`` / ``Q`` are
    a lazy one-solve cache filled on first access; a hard-wall-rejected
    proposal is pre-stamped with ``energy = +inf`` so it never triggers a solve
    and is always rejected by the Metropolis / WL test.
    """

    positions: torch.Tensor
    energy: float | None = None
    Q: float | None = None


def _small_rotation(rng: np.random.Generator, max_angle: float, dtype, device) -> torch.Tensor:
    """Symmetric small-step rotation matrix (Rodrigues), randomness from the
    numpy generator flatwalk hands us so runs stay reproducible."""
    v = rng.standard_normal(3)
    v = v / np.linalg.norm(v)
    angle = (2.0 * rng.random() - 1.0) * max_angle
    c, s = math.cos(angle), math.sin(angle)
    K = torch.tensor(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=dtype,
        device=device,
    )
    return torch.eye(3, dtype=dtype, device=device) + s * K + (1.0 - c) * (K @ K)


def make_contract(
    geom: CellGeometry,
    *,
    translate_weight: float = 0.6,
    rotate_weight: float = 0.4,
    max_displacement: float = 0.30,
    max_angle: float = 0.30,
):
    """Return ``(initial_state, energy_fn, order_parameter_fn, propose_move_fn)``
    filling the flatwalk contract over the seeded ``geom.backend``."""
    backend = geom.backend
    dtype = backend.dtype
    device = backend.device
    top_indices = geom.top_indices
    z_lo, z_hi = geom.z_lo, geom.z_hi

    # ---- fixed topology, read once from the seeded backend (public API) ----
    charges = backend.ion_charges().clone()
    templates = backend.molecule_templates()
    site_types: list[str] = []
    mol_slices: list[tuple[int, int]] = []
    mol_kinds: list[int] = []
    mol_masses: list[torch.Tensor] = []
    cursor = 0
    for mi in range(backend.n_molecules):
        kind = backend.molecule_kind(mi)
        tmpl = templates[kind]
        n = tmpl.n_sites
        site_types.extend(tmpl.site_types)
        mol_slices.append((cursor, n))
        mol_kinds.append(kind)
        mol_masses.append(tmpl.site_mass)
        cursor += n
    n_mol = len(mol_slices)
    if n_mol == 0:
        raise ValueError("seed waters before building the contract")
    p_translate = translate_weight / (translate_weight + rotate_weight)

    def _ensure_evaluated(state: WaterState) -> None:
        if state.energy is None:
            e, q = backend.evaluate(state.positions, charges, site_types, mol_kinds, mol_slices)
            state.energy = float(e.item())
            state.Q = float(q[top_indices].sum().item())

    def energy_fn(state: WaterState) -> float:
        _ensure_evaluated(state)
        return state.energy

    def order_parameter_fn(state: WaterState) -> float:
        _ensure_evaluated(state)
        return state.Q

    def propose_move_fn(state: WaterState, rng: np.random.Generator):
        mol = int(rng.integers(n_mol))
        s, n = mol_slices[mol]
        m = mol_masses[mol]
        new_pos = state.positions.clone()
        sites = new_pos[s : s + n]
        com = (m.unsqueeze(-1) * sites).sum(dim=0) / m.sum()

        if rng.random() < p_translate:
            delta = torch.tensor(
                (rng.random(3) * 2.0 - 1.0) * max_displacement, dtype=dtype, device=device
            )
            new_sites = sites + delta
            new_com = com + delta
        else:
            R = _small_rotation(rng, max_angle, dtype, device)
            new_sites = (sites - com) @ R.T + com
            new_com = com  # rotation about COM leaves it fixed

        new_com_z = float(new_com[2].item())
        if new_com_z < z_lo or new_com_z > z_hi:
            # Hard wall: water would leave the inter-plate slab. Reject via
            # infinite energy at the *current* bin (detailed-balance safe).
            _ensure_evaluated(state)
            return WaterState(positions=state.positions, energy=math.inf, Q=state.Q), 0.0

        new_pos[s : s + n] = new_sites
        return WaterState(positions=new_pos), 0.0  # symmetric -> log ratio 0

    initial_state = WaterState(positions=backend.ion_positions().clone())
    return initial_state, energy_fn, order_parameter_fn, propose_move_fn


# ======================================================================
# 2b. Incremental contract adapter (O(N_el) per step -- the fast path)
# ======================================================================


@dataclass
class MoveToken:
    """flatwalk's opaque ``state`` for the incremental adapter. Carries only
    the proposal's order parameter and energy; the actual configuration lives
    in the (stateful) backend. ``E`` is built from the incremental look-ΔE
    chain so it matches what flatwalk caches as ``walker.energy``."""

    Q: float
    E: float


class IncrementalContract:
    """Bridge flatwalk's stateless contract to the stateful look/leap backend
    at O(N_el) per step.

    flatwalk only ever proposes from the current ``walker.state`` and keeps
    *that exact object* (accept) or the previous one (reject). So we reconcile
    lazily: at the top of each ``propose_move_fn`` we look at which token
    flatwalk handed back and ``commit()`` (it kept our last proposal) or
    ``reject()`` (it didn't) the outstanding backend look. The order parameter
    of a *pending* move comes from ``backend.pending_electrode_charges()``;
    its energy from ``committed_E + delta_E`` (no extra full solve).
    """

    def __init__(
        self,
        geom: CellGeometry,
        *,
        translate_weight: float = 0.6,
        rotate_weight: float = 0.4,
        max_displacement: float = 0.30,
        max_angle: float = 0.30,
    ) -> None:
        self.backend = geom.backend
        self.dtype = self.backend.dtype
        self.device = self.backend.device
        self.top = geom.top_indices
        self.z_lo, self.z_hi = geom.z_lo, geom.z_hi
        self.max_disp = max_displacement
        self.max_angle = max_angle
        self.p_translate = translate_weight / (translate_weight + rotate_weight)

        templates = self.backend.molecule_templates()
        self.slices: list[tuple[int, int]] = []
        self.masses: list[torch.Tensor] = []
        cursor = 0
        for mi in range(self.backend.n_molecules):
            tmpl = templates[self.backend.molecule_kind(mi)]
            self.slices.append((cursor, tmpl.n_sites))
            self.masses.append(tmpl.site_mass)
            cursor += tmpl.n_sites
        self.n_mol = len(self.slices)
        if self.n_mol == 0:
            raise ValueError("seed waters before building the contract")

        q0 = float(self.backend.electrode_charges()[self.top].sum().item())
        e0 = float(self.backend.current_energy().item())
        self.initial_state = MoveToken(Q=q0, E=e0)
        self._committed_token: MoveToken = self.initial_state
        self._committed_E = e0
        self._pending_token: MoveToken | None = None

    def _resolve(self, incoming: MoveToken) -> None:
        """Reconcile the backend's committed state with whichever token
        flatwalk kept, then leave no move pending."""
        if self._pending_token is not None:
            if incoming is self._pending_token:  # flatwalk accepted our proposal
                self.backend.commit()
                self._committed_token = self._pending_token
                self._committed_E = self._pending_token.E
            else:  # rejected (incl. out-of-range / wall): roll the look back
                self.backend.reject()
            self._pending_token = None

    def energy_fn(self, state: MoveToken) -> float:
        return state.E

    def order_parameter_fn(self, state: MoveToken) -> float:
        return state.Q

    def propose_move_fn(self, state: MoveToken, rng: np.random.Generator):
        self._resolve(state)  # backend now committed to `state`
        mol = int(rng.integers(self.n_mol))
        s, n = self.slices[mol]
        m = self.masses[mol]
        sites = self.backend.ion_positions()[s : s + n]
        com = (m.unsqueeze(-1) * sites).sum(dim=0) / m.sum()

        if rng.random() < self.p_translate:
            disp_np = (rng.random(3) * 2.0 - 1.0) * self.max_disp
            new_com_z = float(com[2].item()) + float(disp_np[2])
            if new_com_z < self.z_lo or new_com_z > self.z_hi:
                # hard wall: reject without touching the backend.
                return MoveToken(Q=state.Q, E=math.inf), 0.0
            disp = torch.tensor(disp_np, dtype=self.dtype, device=self.device)
            dE = self.backend.propose_molecule_translate(mol, disp)
        else:
            R = _small_rotation(rng, self.max_angle, self.dtype, self.device)
            dE = self.backend.propose_molecule_rotate(mol, R)

        q_pending = self.backend.pending_electrode_charges()
        token = MoveToken(
            Q=float(q_pending[self.top].sum().item()),
            E=self._committed_E + float(dE.item()),
        )
        self._pending_token = token
        return token, 0.0  # symmetric proposal -> log ratio 0


# ======================================================================
# 3. Pilot (plain Metropolis) to locate <Q> and its spread
# ======================================================================


def pilot_metropolis(
    initial_state, energy_fn, order_parameter_fn, propose_move_fn, *, beta, n_steps, seed, burn=0.5
):
    """Plain canonical Metropolis using the same callbacks, to read the natural
    Q distribution before choosing the WL window. Returns (final_state, stats)."""
    rng = np.random.default_rng(seed)
    cur = initial_state
    e_cur = energy_fn(cur)
    qs = np.empty(n_steps, dtype=np.float64)
    n_acc = 0
    for i in range(n_steps):
        new, lpr = propose_move_fn(cur, rng)
        e_new = energy_fn(new)
        delta = -beta * (e_new - e_cur) + lpr
        if delta >= 0.0 or rng.random() < math.exp(min(delta, 0.0)):
            cur, e_cur = new, e_new
            n_acc += 1
        qs[i] = order_parameter_fn(cur)
    tail = qs[int(burn * n_steps) :]
    stats = {
        "q_mean": float(tail.mean()),
        "q_std": float(tail.std()),
        "q_min": float(tail.min()),
        "q_max": float(tail.max()),
        "acceptance": n_acc / n_steps,
    }
    return cur, stats


def _block_analysis(q: np.ndarray, beta: float, n_blocks: int = 20) -> dict:
    """Block-averaged <Q>, Var(Q) and their errors, plus a capacitance estimate.

    Var(Q)/kT is the differential capacitance; the block scatter gives an honest
    error bar that accounts for the (long) autocorrelation time -- the only way
    to know whether a slowly-mixing chain has actually converged."""
    n = len(q)
    bs = n // n_blocks
    if bs < 1:
        return {"n_samples": n, "insufficient": True}
    blocks = q[: bs * n_blocks].reshape(n_blocks, bs)
    block_means = blocks.mean(axis=1)
    block_vars = blocks.var(axis=1)
    mean = float(q.mean())
    mean_err = float(block_means.std(ddof=1) / math.sqrt(n_blocks))
    var = float(q.var())
    var_err = float(block_vars.std(ddof=1) / math.sqrt(n_blocks))
    # Crude integrated autocorrelation time from blocking: tau ~ bs * Var(block_means)/Var.
    tau = float(bs * block_means.var(ddof=1) / var) if var > 0 else float("nan")
    return {
        "n_samples": n,
        "q_mean": mean,
        "q_mean_err": mean_err,
        "q_var": var,
        "q_var_err": var_err,
        "capacitance_e_per_V": var * beta,
        "capacitance_err_e_per_V": var_err * beta,
        "autocorr_steps_per_sample": tau,
        "n_eff": float(n / max(tau, 1.0)),
    }


def run_plain_mc(contract, *, beta, n_steps, seed, trace_every, out, save_every):
    """Long canonical Metropolis chain recording the full Q(t) trace, so we can
    SEE whether the (slow) electrode charge equilibrates and measure Var(Q)
    directly. Saves the trace periodically for live / partial analysis."""
    e_fn = contract.energy_fn
    q_fn = contract.order_parameter_fn
    prop = contract.propose_move_fn
    rng = np.random.default_rng(seed)
    cur = contract.initial_state
    e_cur = e_fn(cur)
    n_rec = n_steps // trace_every + 1
    qtr = np.empty(n_rec, dtype=np.float64)
    etr = np.empty(n_rec, dtype=np.float64)
    j = 0
    n_acc = 0

    def _save(upto_step):
        np.savez(
            out,
            q_trace=qtr[:j],
            e_trace=etr[:j],
            trace_every=trace_every,
            beta=beta,
            step=upto_step,
            acceptance=n_acc / max(upto_step, 1),
        )

    for i in range(n_steps):
        new, lpr = prop(cur, rng)
        e_new = e_fn(new)
        delta = -beta * (e_new - e_cur) + lpr
        if delta >= 0.0 or rng.random() < math.exp(min(delta, 0.0)):
            cur, e_cur = new, e_new
            n_acc += 1
        if i % trace_every == 0:
            qtr[j] = q_fn(cur)
            etr[j] = e_cur
            j += 1
        if i > 0 and i % save_every == 0:
            _save(i)
            a = _block_analysis(qtr[int(0.3 * j) : j], beta)
            logging.info(
                "plain MC %d/%d acc=%.3f <Q>=%.4f Var=%.3e C=%.4f+-%.4f e/V tau~%.0f n_eff~%.0f",
                i, n_steps, n_acc / (i + 1), a.get("q_mean", float("nan")),
                a.get("q_var", float("nan")), a.get("capacitance_e_per_V", float("nan")),
                a.get("capacitance_err_e_per_V", float("nan")),
                a.get("autocorr_steps_per_sample", float("nan")) * trace_every,
                a.get("n_eff", float("nan")),
            )
    _save(n_steps)
    return qtr[:j]


# ======================================================================
# 4. Drive Wang-Landau
# ======================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("incremental", "snapshot"),
        default="incremental",
        help="incremental: O(N_el)/step look-leap (fast). snapshot: full solve/step.",
    )
    ap.add_argument("--temperature", type=float, default=300.0, help="K")
    ap.add_argument("--n-water", type=int, default=None, help="default: ~bulk density")
    ap.add_argument("--n-bins", type=int, default=40)
    ap.add_argument("--window", type=float, default=1.0, help="max half-window for Q, in e (cap)")
    ap.add_argument("--n-sigma", type=float, default=4.0, help="window = <Q> +- n_sigma * std")
    ap.add_argument(
        "--q-half", type=float, default=None,
        help="explicit window half-width in e (overrides pilot/n_sigma sizing; "
        "use when the true std is already known to avoid pilot under/over-estimation)",
    )
    ap.add_argument(
        "--q-center", type=float, default=None,
        help="explicit window center in e (default: pilot <Q>)",
    )
    ap.add_argument("--pilot-steps", type=int, default=4000)
    ap.add_argument("--ln-f-final", type=float, default=1e-4)
    ap.add_argument("--n-check", type=int, default=2000)
    ap.add_argument("--max-trials", type=int, default=None, help="hard cap on WL trials")
    ap.add_argument("--checkpoint", type=str, default=None, help="WL checkpoint path (.npz)")
    ap.add_argument("--checkpoint-every", type=int, default=200_000)
    ap.add_argument("--resume-from", type=str, default=None, help="resume WL from a checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="wl_electrode_charge_result.npz")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end wiring check")
    ap.add_argument(
        "--plain-mc", type=int, default=0,
        help="if >0, run plain canonical MC for this many steps (measure Var(Q) "
        "directly) instead of Wang-Landau",
    )
    ap.add_argument("--trace-every", type=int, default=10, help="record Q every k steps (plain MC)")
    ap.add_argument("--save-every", type=int, default=200_000, help="save trace cadence (plain MC)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.smoke:
        args.n_water = 12
        args.pilot_steps = 150
        args.ln_f_final = 0.2
        args.n_check = 100

    beta = 1.0 / (KB_EV * args.temperature)
    print(f"T = {args.temperature} K  ->  beta = {beta:.3f} eV^-1")

    geom = build_capacitor()
    n = seed_waters(geom, n_target=args.n_water, seed=args.seed)
    gap_vol = geom.lx * geom.ly * (geom.z_hi - geom.z_lo)
    print(
        f"Seeded {n} waters  ({n / gap_vol:.4f} /A^3 vs bulk {RHO_BULK:.4f}; "
        f"{100 * n / gap_vol / RHO_BULK:.0f}% of bulk) between plates."
    )

    # --- plain canonical MC (direct Var(Q) measurement, no Wang-Landau) -------
    if args.plain_mc > 0:
        contract = IncrementalContract(geom)
        print(
            f"Plain canonical MC: {args.plain_mc:,} steps, recording Q every "
            f"{args.trace_every} -> {args.out}"
        )
        q = run_plain_mc(
            contract,
            beta=beta,
            n_steps=args.plain_mc,
            seed=args.seed,
            trace_every=args.trace_every,
            out=args.out,
            save_every=args.save_every,
        )
        a = _block_analysis(q[int(0.3 * len(q)) :], beta)
        e = 1.602176634e-19
        a_cm2 = (geom.lx * geom.ly) * 1e-16
        print("Plain-MC result (post-30% burn-in, block-averaged):")
        print(f"  <Q>   = {a['q_mean']:+.4f} +- {a['q_mean_err']:.4f} e")
        print(f"  Var(Q)= {a['q_var']:.3e} +- {a['q_var_err']:.3e} e^2")
        print(
            f"  C = Var/kT = {a['capacitance_e_per_V']:.4f} +- "
            f"{a['capacitance_err_e_per_V']:.4f} e/V  = "
            f"{a['capacitance_e_per_V'] * e / a_cm2 * 1e6:.2f} uF/cm^2"
        )
        print(
            f"  autocorr ~ {a['autocorr_steps_per_sample'] * args.trace_every:.0f} steps, "
            f"n_eff ~ {a['n_eff']:.0f}  (need n_eff >> 1 to trust the error bar)"
        )
        return

    if args.mode == "incremental":
        contract = IncrementalContract(geom)
        initial_state = contract.initial_state
        energy_fn = contract.energy_fn
        order_parameter_fn = contract.order_parameter_fn
        propose_move_fn = contract.propose_move_fn
    else:
        initial_state, energy_fn, order_parameter_fn, propose_move_fn = make_contract(geom)
    print(f"Contract mode: {args.mode}")

    print(f"Pilot: {args.pilot_steps} Metropolis steps to equilibrate and locate <Q> ...")
    equilibrated_state, stats = pilot_metropolis(
        initial_state,
        energy_fn,
        order_parameter_fn,
        propose_move_fn,
        beta=beta,
        n_steps=args.pilot_steps,
        seed=args.seed,
    )
    print(
        "  <Q> = {q_mean:+.4f} e, std = {q_std:.4f}, "
        "range [{q_min:+.4f}, {q_max:+.4f}], acc = {acceptance:.2f}".format(**stats)
    )

    # Window: explicit override if given, else data-driven (<Q> +- n_sigma * std)
    # capped at +-args.window e.
    center = args.q_center if args.q_center is not None else stats["q_mean"]
    if args.q_half is not None:
        half = args.q_half
        how = f"explicit half={half:.4f} (~{half / max(stats['q_std'], 1e-9):.1f} pilot-sigma)"
    else:
        half = min(args.window, max(args.n_sigma * stats["q_std"], 1e-3))
        how = f"half={half:.4f} = {args.n_sigma:g} pilot-sigma"
    lo, hi = center - half, center + half
    # flatwalk requires the initial state's Q to be inside the bin domain. The
    # electrode charge mixes slowly, so the equilibrated state can sit outside a
    # pilot-mean-centred window -- expand to include it rather than crash.
    q_init = float(order_parameter_fn(equilibrated_state))
    if not (lo < q_init < hi):
        pad = 0.05 * (hi - lo)
        lo, hi = min(lo, q_init - pad), max(hi, q_init + pad)
        how += " [expanded to include initial Q]"
    scheme = Bin1D(lo, hi, args.n_bins)
    print(
        f"WL window: [{lo:+.4f}, {hi:+.4f}] e over {args.n_bins} bins ({how}); "
        f"initial Q={q_init:+.4f}."
    )

    cfg = WLConfig(
        bin_scheme=scheme,
        beta=beta,
        n_check=args.n_check,
        ln_f_final=args.ln_f_final,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        checkpoint_every_t=args.checkpoint_every,
    )
    print("Running Wang-Landau ...")
    result = WLDriver(cfg).run(
        initial_state=equilibrated_state,
        energy_fn=energy_fn,
        order_parameter_fn=order_parameter_fn,
        propose_move_fn=propose_move_fn,
        rng=np.random.default_rng(args.seed + 1),
        max_trials=args.max_trials,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )
    print(
        f"  {result.t_total:,} trials, {result.n_f_stages} f-stages, "
        f"converged = {result.converged}"
    )

    # For a finite-beta WL run over an order parameter, g(Q) = ln P_T(Q) + const
    # (the canonical distribution of Q), so the free-energy profile is the WELL
    #   F(Q) = -g(Q)/beta + const,
    # shifted so its minimum (the most probable Q) sits at 0.
    g = np.asarray(result.g)
    log_g = g - float(g.min())  # >= 0, peak = most probable Q
    F = (float(g.max()) - g) / beta  # eV, well with min 0 at the most probable Q
    centers = np.asarray(scheme.centers)

    # Distribution / capacitance from the WL g, and a window-adequacy check.
    P = np.exp(g - g.max())
    P /= P.sum()
    q_mean = float((centers * P).sum())
    q_var = float((P * (centers - q_mean) ** 2).sum())
    cap_eV = q_var * beta  # Var(Q)/kT, units e/V
    print(f"WL distribution: <Q>={q_mean:+.4f} e, std={q_var**0.5:.4f} e")
    print(f"Differential capacitance Var(Q)/kT = {cap_eV:.4f} e/V")
    edge_kt = min(F[0], F[-1]) * beta
    if edge_kt < 4.0:
        print(
            f"  WARNING: nearest window edge is only {edge_kt:.1f} kT above the "
            f"minimum -- the charge distribution is CLIPPED; widen --n-sigma / --window "
            f"(true std {q_var**0.5:.4f} e vs pilot {stats['q_std']:.4f} e) and rerun."
        )

    np.savez(
        args.out,
        Q=centers,
        log_g=log_g,
        F_eV=F,
        beta=beta,
        temperature=args.temperature,
        n_water=n,
        converged=bool(result.converged),
        q_mean=q_mean,
        q_var=q_var,
        capacitance_e_per_V=cap_eV,
    )
    print(f"Saved -> {args.out}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(centers, F, "o-", color="C0")
        ax.set_xlabel("electrode charge  Q (top plate) [e]")
        ax.set_ylabel("free energy  F(Q) [eV]  (shifted to min 0)")
        ax.set_title(f"Electrode-charge PMF, T = {args.temperature:.0f} K, N = {n} waters")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png = Path(args.out).with_suffix(".png")
        fig.savefig(png, dpi=130)
        print(f"Saved -> {png}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
