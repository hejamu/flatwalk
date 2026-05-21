"""Render a per-trial video of a short Wang-Landau run.

This is the "every step" companion to `wl_demo.mp4` (which samples
log-spaced check intervals). Here every single trial gets a frame, so
the walker is visible hopping bin-by-bin while the histogram and ``log
g(E)`` build up from zero.

Use a short run (default 1500 trials). Recording every trial of a
production run (10⁸ trials on L=8 to ``ln_f = 1e-8``) would be both
useless visually and impossible in memory.

Usage
-----
    python examples/wl_trajectory_demo.py [-L 8] [-n 1500] [-o out.mp4]
                                          [--seed 0] [--fps 30]

The script forces ``MPLBACKEND=Agg`` if the env var isn't set, so it
runs headlessly. Use ``.mp4`` / ``.mov`` / ``.webm`` for ffmpeg-rendered
video or ``.gif`` for Pillow.
"""

from __future__ import annotations

import argparse
import logging
import os
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", type=int, default=8)
    parser.add_argument("-n", "--n-trials", type=int, default=1500,
                        help="trials to record (= frames in the movie)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("examples/wl_trajectory.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cache-dir", type=Path,
                        default=EXAMPLES / "cache",
                        help="Beale cache directory (read-only here)")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the Beale exact-reference overlay")
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLBACKEND", "Agg")

    # Late imports so MPLBACKEND env takes effect.
    from wl_viewer import TrialRecorder, make_trajectory_movie

    logging.basicConfig(level=logging.WARNING)

    cb = ising.make_ising_callbacks(args.L)
    rng = np.random.default_rng(args.seed)
    initial_state = ising.random_state(args.L, rng)
    low, high, n_bins = ising.ising_energy_bins(args.L)
    scheme = Bin1D(low, high, n_bins)

    log_g_exact = None
    if not args.no_reference:
        cache = args.cache_dir / f"beale_L{args.L}.tsv"
        if cache.exists():
            print(f"Loading Beale reference from {cache}...")
            g_exact: dict[int, int] = {}
            for line in cache.read_text().splitlines():
                if not line.strip() or line.startswith("#") or line.startswith("E\t"):
                    continue
                E_str, n_str = line.split("\t")
                g_exact[int(E_str)] = int(n_str)
            log_g_exact = beale.log_g_E_array(args.L, g_exact, scheme.centers)
        else:
            print(f"No Beale cache for L={args.L}; rendering without reference.")

    # WL run with trial recorder. n_check=2*n_trials so no halve fires
    # within the recorded window — the walker stays at ln_f=1 throughout
    # so each visit contributes a large +1 to g, making the build-up
    # dramatic.
    cfg = WLConfig(
        bin_scheme=scheme, beta=0.0,
        flatness_threshold=0.95,
        n_check=2 * args.n_trials,
        ln_f_initial=1.0, ln_f_final=1e-30,
    )
    driver = WLDriver(cfg)
    recorder = TrialRecorder(max_records=args.n_trials)

    print(f"Running WL for {args.n_trials:,} trials (seed={args.seed})...")
    t0 = time.perf_counter()
    driver.run(
        initial_state=initial_state,
        energy_fn=cb["energy_fn"],
        order_parameter_fn=cb["order_parameter_fn"],
        propose_move_fn=cb["propose_move_fn"],
        rng=rng,
        max_trials=args.n_trials,
        trial_callback=recorder,
    )
    print(f"  recorded {len(recorder.t):,} trials in {time.perf_counter() - t0:.2f}s")

    print(f"Rendering {len(recorder.t)} frames → {args.output} ...")
    t0 = time.perf_counter()
    make_trajectory_movie(
        recorder.as_arrays(),
        args.output,
        bin_centers=scheme.centers,
        log_g_exact=log_g_exact,
        title=f"Wang-Landau trajectory   |   L={args.L} Ising",
        fps=args.fps,
    )
    print(f"  rendered in {time.perf_counter() - t0:.1f}s → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
