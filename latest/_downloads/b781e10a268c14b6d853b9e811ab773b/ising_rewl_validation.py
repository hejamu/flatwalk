"""Replica-exchange Wang-Landau validation on the 2D Ising model (L=8).

This is the REWL counterpart of ``examples/ising_validation.py`` (single
walker). Instead of averaging independent runs, it splits the energy axis into
``--n-windows`` overlapping windows, runs one walker per window with periodic
adjacent-window exchanges, joins the per-window log-``g`` over their overlaps,
and compares the result to Beale's exact ``n(E)``.

Pass criteria (mirroring the single-walker validation)
------------------------------------------------------
- ``max ε(E) < 0.05`` over visited central bins (excluding the two extremes).
- ``mean ε(E) < 0.01`` over the same bins.
- ``⟨E⟩(T)`` within 0.5% of exact across T ∈ [1.0, 4.0].
- ``C_V(T)`` peak temperature within 2% of exact.

The exact reference and the comparison / thermodynamics helpers are reused
from ``ising_validation`` so the two scripts apply identical scoring.

Usage
-----
    python examples/ising_rewl_validation.py [-L 8] [--n-windows 4]
        [--overlap 8] [--n-exchange 10] [--ln-f-final 1e-8] [--seed 0]
        [--cache-dir examples/cache] [--quick]

``--quick`` runs to ln_f_final=1e-5 to smoke-test the pipeline; it will NOT
meet the strict criteria (the script exits 0 only when they are satisfied).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import ising  # noqa: E402
import ising_batched  # noqa: E402
import ising_validation as iv  # noqa: E402

from flatwalk import Bin1D, RewlDriver, WLConfig, join_g, make_windows  # noqa: E402


def run_rewl(
    L: int,
    n_windows: int,
    overlap: int,
    n_exchange: int,
    ln_f_final: float,
    seed: int,
    trace_path: Path | None,
    n_check: int,
    flatness_threshold: float,
):
    low, high, n_bins = ising.ising_energy_bins(L)
    scheme = Bin1D(low, high, n_bins)
    windows = make_windows(scheme, n_windows, overlap)

    cb = ising_batched.make_batched_ising_callbacks(L)
    rng = np.random.default_rng(seed)
    initial_state = ising_batched.initial_states_for_windows(L, windows, rng)

    cfg = WLConfig(
        bin_scheme=scheme,
        beta=0.0,
        flatness_threshold=flatness_threshold,
        n_check=n_check,
        ln_f_initial=1.0,
        ln_f_final=ln_f_final,
        trace_path=trace_path,
    )
    driver = RewlDriver(cfg, windows)

    t0 = time.perf_counter()
    result = driver.run(
        initial_state=initial_state,
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        n_exchange=n_exchange,
        rng=rng,
    )
    dt = time.perf_counter() - t0
    moves = result.t_total * n_windows
    acc = result.exchange_accepts.sum()
    att = max(int(result.exchange_attempts.sum()), 1)
    print(
        f"  {result.t_total:,} ticks ({moves:,} moves) in {dt:.1f}s "
        f"({moves / dt / 1e3:.0f} kmoves/s), {result.n_f_stages} f-stages, "
        f"converged={result.converged}, exchange accept={acc / att:.2f}"
    )
    for w, (lo, hi) in enumerate(windows):
        cov = int(result.visited_windows[w].sum())
        print(f"    window {w}: E∈[{lo:.0f}, {hi:.0f}], {cov} bins visited")

    joined, visited_joined = join_g(result.g_windows, result.visited_windows)
    return joined, visited_joined, scheme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", type=int, default=8, help="lattice size (default 8)")
    parser.add_argument("--n-windows", type=int, default=4, help="number of windows")
    parser.add_argument("--overlap", type=int, default=8, help="overlap in bins")
    parser.add_argument(
        "--n-exchange", type=int, default=10, help="ticks between exchange sweeps"
    )
    parser.add_argument("--ln-f-final", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=EXAMPLES / "cache")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="ln-f-final=1e-5 smoke run")
    parser.add_argument("--n-check", type=int, default=1_000)
    parser.add_argument("--flatness", type=float, default=0.9)
    args = parser.parse_args(argv)

    if args.quick:
        args.ln_f_final = max(args.ln_f_final, 1e-5)

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )

    print(f"=== flatwalk Ising L={args.L} REWL validation ===")
    print(
        f"  ln_f_final = {args.ln_f_final:.0e}, n_windows = {args.n_windows}, "
        f"overlap = {args.overlap}, n_exchange = {args.n_exchange}, seed = {args.seed}"
    )

    print(f"  Loading exact n(E) (Beale recursion) for L={args.L}...")
    t0 = time.perf_counter()
    g_exact = iv.load_or_compute_beale(args.L, args.cache_dir)
    print(
        f"  → {len(g_exact)} distinct energies (loaded in {time.perf_counter() - t0:.1f}s)"
    )

    print("  Running replica-exchange Wang-Landau...")
    joined, visited_joined, scheme = run_rewl(
        args.L,
        args.n_windows,
        args.overlap,
        args.n_exchange,
        args.ln_f_final,
        args.seed,
        args.trace,
        args.n_check,
        args.flatness,
    )

    cmp = iv.compare_g(joined, visited_joined, scheme, g_exact)
    print(
        f"  Compared on {cmp['n_compared']} central bins "
        f"(visited & non-gap, excluding the two extremes):"
    )
    print(f"    max  ε = {cmp['max_eps']:.4f}  (pass < 0.05)")
    print(f"    mean ε = {cmp['mean_eps']:.4f}  (pass < 0.01)")

    T_grid = np.linspace(1.0, 4.0, 31)
    E_exact, CV_exact = iv.thermodynamics(scheme.centers, g_exact, T_grid)
    n_E_WL = iv.wl_to_n_E_dict(
        scheme.centers, joined, exact_total_for_normalization=sum(g_exact.values())
    )
    E_WL, CV_WL = iv.thermodynamics(scheme.centers, n_E_WL, T_grid)
    E_rel = np.abs(E_WL - E_exact) / np.abs(E_exact + 1e-12)
    max_E_rel = float(E_rel.max())
    print(f"    max |Δ⟨E⟩/⟨E⟩| = {max_E_rel * 100:.3f}%  (pass < 0.5%)")

    T_peak_exact = float(T_grid[CV_exact.argmax()])
    T_peak_WL = float(T_grid[CV_WL.argmax()])
    T_peak_err = abs(T_peak_WL - T_peak_exact) / T_peak_exact
    print(
        f"    C_V peak T: exact = {T_peak_exact:.3f}, WL = {T_peak_WL:.3f}, "
        f"rel err = {T_peak_err * 100:.3f}%  (pass < 2%)"
    )

    criteria = {
        "max ε < 0.05": cmp["max_eps"] < 0.05,
        "mean ε < 0.01": cmp["mean_eps"] < 0.01,
        "max |ΔE/E| < 0.5%": max_E_rel < 0.005,
        "C_V peak T err < 2%": T_peak_err < 0.02,
    }
    print("\n  Pass criteria:")
    for name, ok in criteria.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(criteria.values())
    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
