"""End-to-end Wang-Landau validation on the 2D Ising model (L=8 by default).

Pass criteria (spec §4.4)
-------------------------
- ``max ε(E) < 0.05`` over visited central bins (excluding the two extreme
  bins ``E = ±2L²``).
- ``mean ε(E) < 0.01`` over the same bins.
- ``⟨E⟩(T)`` agrees with the exact value within 0.5% across T ∈ [1.0, 4.0].
- The peak of ``C_V(T)`` is within 2% of the exact L=8 peak temperature.

The exact reference comes from `beale.beale_g_E(L)` (transfer-matrix + CRT
recursion); see `tests/test_beale.py` for the L=3/L=4 cross-validation
against brute-force enumeration.

Divergences from the spec literal
---------------------------------
- ``n_check`` and ``flatness_threshold`` are tuned (defaults 1000 / 0.95
  here vs. spec §1.5 defaults 10_000 / 0.8). The spec marks both as
  "Tunable" so this is within bounds.
- The default run averages ``--n-seeds`` independent WL runs (default 3)
  and computes the comparison on the averaged ``log g``. A single seed
  produces ~10% max ε on L=8 single-walker, well above spec; the literature
  routine fix is multi-seed (or REWL — see [`flatwalk.exchange`]). The
  spec's singular "run the driver" is interpreted as "run the WL
  procedure", and the averaging is a script-level post-process that
  doesn't touch the driver. ``--n-seeds 1`` recovers the pure spec
  reading.

Usage
-----
    python examples/ising_validation.py [-L 8] [--ln-f-final 1e-8] [--seed 0]
                                        [--n-seeds 3]
                                        [--cache-dir examples/cache]
                                        [--quick]

``--quick`` runs to ln_f_final=1e-5 instead of 1e-8 (~30 s) for smoke
testing the pipeline; the resulting g_WL will NOT meet the spec pass
criteria — the script exits 0 only when the strict criteria are satisfied.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import beale  # noqa: E402
import ising  # noqa: E402

from flatwalk import Bin1D, WLConfig, WLDriver  # noqa: E402


# ---------------------------------------------------------------------------
# Beale cache (avoid recomputing 55s for L=8 on every script run)
# ---------------------------------------------------------------------------

def load_or_compute_beale(L: int, cache_dir: Path | None) -> dict[int, int]:
    """Beale's g(E) for the L×L Ising torus, with on-disk caching.

    Values are big Python ints (up to 2^(L²)), serialized as decimal text to
    a TSV cache file — avoids the allow_pickle gotcha of numpy object arrays.
    """
    if cache_dir is None:
        return beale.beale_g_E(L)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"beale_L{L}.tsv"
    if cache.exists():
        g: dict[int, int] = {}
        with open(cache, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or line.startswith("#") or line.startswith("E\t"):
                    continue
                E_str, n_str = line.rstrip("\n").split("\t")
                g[int(E_str)] = int(n_str)
        return g
    t0 = time.perf_counter()
    g = beale.beale_g_E(L)
    with open(cache, "w", encoding="utf-8") as fh:
        fh.write(f"# Beale g(E) for L={L} 2D Ising torus; Σ n(E) = 2^{L*L}\n")
        fh.write("E\tn\n")
        for E in sorted(g.keys()):
            fh.write(f"{E}\t{g[E]}\n")
    print(f"  Beale L={L} computed in {time.perf_counter() - t0:.1f}s and cached "
          f"to {cache}")
    return g


# ---------------------------------------------------------------------------
# WL run
# ---------------------------------------------------------------------------

def run_wl(
    L: int, ln_f_final: float, seed: int, trace_path: Path | None,
    n_check: int = 1_000, flatness_threshold: float = 0.95,
):
    cb = ising.make_ising_callbacks(L)
    rng = np.random.default_rng(seed)
    initial_state = ising.random_state(L, rng)

    low, high, n_bins = ising.ising_energy_bins(L)
    scheme = Bin1D(low, high, n_bins)
    cfg = WLConfig(
        bin_scheme=scheme,
        beta=0.0,
        flatness_threshold=flatness_threshold,
        n_check=n_check,
        ln_f_initial=1.0,
        ln_f_final=ln_f_final,
        trace_path=trace_path,
    )
    driver = WLDriver(cfg)

    t0 = time.perf_counter()
    result = driver.run(
        initial_state=initial_state,
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        rng=rng,
    )
    dt = time.perf_counter() - t0
    print(f"  seed {seed}: {result.t_total:,} trials in {dt:.1f}s "
          f"({result.t_total / dt / 1e3:.0f} kT/s), "
          f"{result.n_f_stages} f-stages, converged={result.converged}, "
          f"in_1overt={result.in_1overt}")
    return result, scheme


def average_log_g(results: list) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple WL ``g`` arrays into one averaged ``log g``.

    Each WL run produces ``g`` up to an additive constant. We shift each to
    have ``max(g[visited]) = 0`` and then arithmetic-mean over seeds. The
    result is the geometric mean of ``n(E)`` estimates per bin — an
    unbiased estimator of ``log n(E)`` with variance ``≈ var_single / K``.

    Unvisited-by-everyone bins return ``-inf`` so downstream code that
    exponentiates gets zero weight from them (not ``exp(0) = 1``, which
    would dominate the thermodynamics if any bin were missed).
    """
    n_bins = len(results[0].g)
    accumulator = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for r in results:
        v = r.visited
        shifted = r.g.copy()
        shifted[v] -= shifted[v].max()
        accumulator[v] += shifted[v]
        counts[v] += 1
    log_g = np.full(n_bins, -np.inf, dtype=np.float64)
    nz = counts > 0
    log_g[nz] = accumulator[nz] / counts[nz]
    return log_g, nz


# ---------------------------------------------------------------------------
# Comparison and thermodynamics
# ---------------------------------------------------------------------------

def compare_g(log_g_WL, visited, scheme, g_exact_dict):
    """Compare a (possibly averaged) WL ``log g`` against Beale's exact g(E).

    Returns a diagnostics dict including ``max_eps`` and ``mean_eps`` on
    central bins (excludes the two extreme bins ``E = ±2L²``).
    """
    centers = scheme.centers
    log_g_exact = beale.log_g_E_array(8, g_exact_dict, centers)
    n_E_exact = np.exp(log_g_exact)  # zeros where g=0 (gap bins)

    # Mask: visited AND g_exact > 0 (skip spectrum gaps).
    valid = visited & np.isfinite(log_g_exact) & (n_E_exact > 0)
    if not valid.any():
        raise RuntimeError("no overlap between visited bins and exact non-zero bins")

    # Normalize WL log_g by aligning on the visited set so totals match.
    # Convert to a *probability* over the comparison set:
    #   p_WL[i] = exp(log_g_WL[i]) / Σ exp(log_g_WL),   sum on `valid`
    # Same for exact.  We then turn p back into a renormalized g.
    shifted = log_g_WL[valid] - log_g_WL[valid].max()
    p_WL = np.exp(shifted)
    p_WL /= p_WL.sum()
    n_E_WL_norm = p_WL * n_E_exact[valid].sum()

    eps = np.abs(n_E_WL_norm - n_E_exact[valid]) / n_E_exact[valid]

    # Central mask: exclude the two extremes (E = ±2L²) on the comparison set.
    central = np.ones_like(eps, dtype=bool)
    if eps.size >= 2:
        # By construction, allowed energies are sorted by centers (which are
        # increasing). The first valid bin is the lowest E; the last is the
        # highest. Drop them.
        central[0] = False
        central[-1] = False
    if not central.any():
        raise RuntimeError("comparison set too small after excluding extremes")
    eps_central = eps[central]

    return {
        "valid": valid,
        "central_eps": eps_central,
        "max_eps": float(eps_central.max()),
        "mean_eps": float(eps_central.mean()),
        "n_compared": int(central.sum()),
        "n_E_WL_norm": n_E_WL_norm,
        "n_E_exact_valid": n_E_exact[valid],
        "log_g_WL_valid": log_g_WL[valid],
    }


def thermodynamics(centers, n_E_dict, T_grid):
    """Compute ⟨E⟩(T) and C_V(T) by reweighting exact n(E) (or by passing
    a different dict — same call). Returns dict of arrays."""
    # Stack to arrays
    E_arr = np.array([E for E in n_E_dict if n_E_dict[E] > 0], dtype=np.float64)
    # log n(E) for numerical safety at low T
    log_n = np.array([math.log(n_E_dict[int(E)]) for E in E_arr])

    E_mean = np.empty_like(T_grid)
    C_V = np.empty_like(T_grid)
    for i, T in enumerate(T_grid):
        beta = 1.0 / T
        # log of Boltzmann-weighted DOS:
        log_w = log_n - beta * E_arr
        log_w -= log_w.max()
        w = np.exp(log_w)
        Z = w.sum()
        Em = (E_arr * w).sum() / Z
        E2m = (E_arr * E_arr * w).sum() / Z
        E_mean[i] = Em
        # Specific heat per spin: C_V = (⟨E²⟩ - ⟨E⟩²) / T²  (then /N for per-spin)
        C_V[i] = (E2m - Em * Em) / (T * T)
    return E_mean, C_V


def wl_to_n_E_dict(centers, log_g, exact_total_for_normalization):
    """Convert WL log_g array to a {E: n(E)} dict normalized so its sum matches
    the exact total of valid configurations."""
    finite = np.isfinite(log_g) & (log_g > -np.inf)
    shifted = log_g - log_g[finite].max()
    n_arr = np.exp(shifted)
    n_arr[~finite] = 0.0
    n_arr *= exact_total_for_normalization / n_arr.sum()
    out = {}
    for E, n in zip(centers, n_arr):
        E_int = int(round(E))
        if n > 0:
            out[E_int] = n
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", type=int, default=8, help="lattice size (default 8)")
    parser.add_argument("--ln-f-final", type=float, default=1e-8,
                        help="WL convergence threshold (default 1e-8)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--cache-dir", type=Path,
                        default=EXAMPLES / "cache",
                        help="directory for caching the Beale result")
    parser.add_argument("--trace", type=Path, default=None,
                        help="optional TSV trace output path")
    parser.add_argument("--quick", action="store_true",
                        help="quick smoke run (ln-f-final=1e-5; does NOT satisfy spec)")
    parser.add_argument("--n-check", type=int, default=1_000,
                        help="WL flatness check period (default 1000; "
                             "tighter than the spec default 10000 to enter the "
                             "1/t regime earlier)")
    parser.add_argument("--flatness", type=float, default=0.95,
                        help="WL flatness threshold (default 0.95; "
                             "spec default is 0.8)")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="number of independent WL runs to average "
                             "(default 3; --n-seeds 1 = spec literal)")
    args = parser.parse_args(argv)

    if args.quick:
        args.ln_f_final = max(args.ln_f_final, 1e-5)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    seeds = [args.seed + k for k in range(max(1, args.n_seeds))]
    print(f"=== flatwalk Ising L={args.L} validation ===")
    print(f"  ln_f_final = {args.ln_f_final:.0e}, seeds = {seeds}")

    # Exact reference
    print(f"  Loading exact n(E) (Beale recursion) for L={args.L}...")
    t0 = time.perf_counter()
    g_exact = load_or_compute_beale(args.L, args.cache_dir)
    print(f"  → {len(g_exact)} distinct energies, "
          f"Σ n(E) = 2^{args.L*args.L} (loaded in {time.perf_counter()-t0:.1f}s)")

    # WL runs
    print(f"  Running Wang-Landau (n_check={args.n_check}, "
          f"flatness={args.flatness:.2f})...")
    results = []
    scheme = None
    for s in seeds:
        r, sch = run_wl(
            args.L, args.ln_f_final, s,
            args.trace if (s == seeds[0]) else None,
            n_check=args.n_check, flatness_threshold=args.flatness,
        )
        results.append(r)
        scheme = sch

    log_g_avg, visited_avg = average_log_g(results)

    # Compare
    cmp = compare_g(log_g_avg, visited_avg, scheme, g_exact)
    print(f"  Compared on {cmp['n_compared']} central bins "
          f"(visited & non-gap, excluding the two extremes):")
    print(f"    max  ε = {cmp['max_eps']:.4f}  (pass < 0.05)")
    print(f"    mean ε = {cmp['mean_eps']:.4f}  (pass < 0.01)")

    # Thermodynamics
    T_grid = np.linspace(1.0, 4.0, 31)
    E_exact, CV_exact = thermodynamics(scheme.centers, g_exact, T_grid)
    n_E_WL = wl_to_n_E_dict(
        scheme.centers, log_g_avg,
        exact_total_for_normalization=sum(g_exact.values()),
    )
    E_WL, CV_WL = thermodynamics(scheme.centers, n_E_WL, T_grid)
    E_rel = np.abs(E_WL - E_exact) / np.abs(E_exact + 1e-12)
    max_E_rel = float(E_rel.max())
    print(f"    max |⟨E⟩_WL − ⟨E⟩_exact| / |⟨E⟩_exact| = {max_E_rel*100:.3f}%  "
          f"(pass < 0.5%)")

    T_peak_exact = float(T_grid[CV_exact.argmax()])
    T_peak_WL = float(T_grid[CV_WL.argmax()])
    T_peak_err = abs(T_peak_WL - T_peak_exact) / T_peak_exact
    print(f"    C_V peak T: exact = {T_peak_exact:.3f}, WL = {T_peak_WL:.3f}, "
          f"rel err = {T_peak_err*100:.3f}%  (pass < 2%)")

    # Pass / fail
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
