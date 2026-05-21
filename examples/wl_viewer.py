"""Live matplotlib viewer for a Wang-Landau run.

Usage
-----
Pass `LiveViewer(...).callback` as the ``progress_callback`` argument of
`WLDriver.run`::

    from wl_viewer import LiveViewer
    viewer = LiveViewer(scheme.centers, flatness_threshold=0.95)
    result = driver.run(..., progress_callback=viewer.callback)
    viewer.keep_open()  # block until the user closes the figure

The viewer shows three live panels stacked vertically:

1. **log g(E)** — the WL density of states (shifted so its minimum over
   visited bins is 0). Optionally overlaid with a reference curve
   (e.g. Beale's exact ``log n(E)`` for Ising).
2. **H(E)** — current-stage histogram. Bars drop visibly each f-stage
   transition (because ``H`` resets in the standard regime); a dashed
   line marks ``flatness_threshold × mean(H)``.
3. **ln_f and flatness vs t** — log-log ``ln_f`` time series with the
   ``1/t`` reference line dashed (the 1/t-WL asymptote), plus the
   flatness on a right-hand axis. Vertical lines mark f-stage transitions
   and the 1/t regime entry.

The viewer rate-limits drawing to ``update_every_s`` (default 0.1 s) so
the WL run doesn't pay matplotlib overhead on every check.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

# Defer matplotlib import to the constructor so this module can be imported
# in a headless context for testing without a display backend.


class LiveViewer:
    """Three-panel matplotlib viewer that updates as a WL run progresses."""

    def __init__(
        self,
        bin_centers: np.ndarray,
        *,
        flatness_threshold: float = 0.95,
        log_g_exact: Optional[np.ndarray] = None,
        update_every_s: float = 0.1,
        title: str = "Wang-Landau live",
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()

        self._bin_centers = np.asarray(bin_centers, dtype=np.float64)
        # Bar width: use the median bin spacing.
        bin_width = float(np.median(np.diff(self._bin_centers))) if len(self._bin_centers) > 1 else 1.0
        self._bin_width = 0.9 * bin_width
        self._flatness_threshold = flatness_threshold
        self._update_every_s = update_every_s
        self._title = title

        # Per-snapshot time series (always recorded; drawing rate-limited).
        self._t_hist: list[int] = []
        self._lnf_hist: list[float] = []
        self._flat_hist: list[float] = []
        self._stage_transitions: list[tuple[int, int]] = []  # (t, new_stage)
        self._one_over_t_t: Optional[int] = None
        self._last_n_f_stages = 0
        self._last_draw_time = 0.0

        # Reference curve (optional).
        self._log_g_exact = (
            np.asarray(log_g_exact, dtype=np.float64) if log_g_exact is not None else None
        )

        # Figure + axes
        self._fig, axes = plt.subplots(
            3, 1, figsize=(11, 9),
            gridspec_kw={"height_ratios": [3, 2, 2]},
        )
        self._ax_g, self._ax_H, self._ax_t = axes

        # Panel 1: log g(E)
        (self._line_g,) = self._ax_g.plot(
            [], [], color="C0", lw=2.0, label="WL log g (shifted)",
        )
        if self._log_g_exact is not None:
            finite = np.isfinite(self._log_g_exact)
            ref = self._log_g_exact.copy()
            if finite.any():
                ref[finite] -= ref[finite].min()
            (self._line_g_ref,) = self._ax_g.plot(
                self._bin_centers[finite], ref[finite],
                color="k", lw=1.0, ls="--", alpha=0.6, label="reference (exact)",
            )
        else:
            self._line_g_ref = None
        self._ax_g.set_ylabel("log g(E)  (shifted)")
        self._ax_g.legend(loc="upper right")
        self._ax_g.grid(alpha=0.3)

        # Panel 2: H(E)
        self._bars_H = self._ax_H.bar(
            self._bin_centers, np.zeros_like(self._bin_centers),
            width=self._bin_width, color="C2", alpha=0.85,
        )
        self._line_H_threshold = self._ax_H.axhline(
            0.0, color="C3", ls="--", alpha=0.7,
            label=f"{flatness_threshold:.2f}·mean(H)",
        )
        self._ax_H.set_ylabel("H (current stage)")
        self._ax_H.legend(loc="upper right")
        self._ax_H.grid(alpha=0.3)

        # Panel 3: ln_f + flatness over time
        (self._line_lnf,) = self._ax_t.plot(
            [], [], color="C0", lw=1.5, label="ln_f",
        )
        # 1/t reference (dashed): drawn once we have data
        (self._line_oneovert,) = self._ax_t.plot(
            [], [], color="k", lw=1.0, ls="--", alpha=0.5, label="1/t",
        )
        self._ax_t.set_yscale("log")
        self._ax_t.set_xscale("log")
        # Placeholder limits so the log scale has positive data even before
        # the first snapshot arrives (matplotlib's tight_layout refuses to
        # render otherwise).
        self._ax_t.set_xlim(1, 10)
        self._ax_t.set_ylim(1e-10, 2.0)
        self._ax_t.set_xlabel("t (trials)")
        self._ax_t.set_ylabel("ln_f")
        self._ax_t.grid(alpha=0.3, which="both")
        self._ax_t.legend(loc="upper right")

        self._ax_flat = self._ax_t.twinx()
        (self._line_flat,) = self._ax_flat.plot(
            [], [], color="C1", lw=1.2, alpha=0.8, label="flatness",
        )
        self._ax_flat.axhline(
            flatness_threshold, color="C3", ls=":", alpha=0.5,
        )
        self._ax_flat.set_ylim(0.0, 1.05)
        self._ax_flat.set_ylabel("flatness", color="C1")
        self._ax_flat.tick_params(axis="y", labelcolor="C1")

        self._fig.suptitle(title)
        self._fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        self._fig.canvas.draw_idle()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass

    # ------------------------------------------------------------------ API

    @property
    def callback(self):
        """Pass this as ``progress_callback=...`` to `WLDriver.run`."""
        return self._on_snapshot

    def save(self, path) -> None:
        """Save the current figure to ``path`` (any format matplotlib supports)."""
        self._draw_final()
        self._fig.savefig(str(path), dpi=110, bbox_inches="tight")

    def keep_open(self) -> None:
        """Block until the user closes the figure window.

        Returns immediately on a non-interactive (e.g. ``Agg``) backend, so
        the same script can drive a live window in an interactive session
        or a save-only run in CI/headless contexts.
        """
        import matplotlib

        self._plt.ioff()
        self._draw_final()
        if matplotlib.get_backend().lower() == "agg":
            return
        self._plt.show()

    # ---------------------------------------------------------------- impl

    def _on_snapshot(self, snap) -> None:
        # Always record time-series points (cheap).
        self._t_hist.append(snap.t)
        self._lnf_hist.append(snap.ln_f)
        self._flat_hist.append(snap.flatness)
        if snap.n_f_stages > self._last_n_f_stages:
            self._stage_transitions.append((snap.t, snap.n_f_stages))
            self._last_n_f_stages = snap.n_f_stages
        if snap.in_1overt and self._one_over_t_t is None:
            self._one_over_t_t = snap.t

        # Rate-limit drawing.
        now = time.perf_counter()
        if now - self._last_draw_time < self._update_every_s:
            return
        self._last_draw_time = now
        self.draw(snap)

    def draw(self, snap) -> None:
        """Force a draw with this snapshot (no rate-limit).

        Movie renderers call this directly; the live callback path goes
        through `_on_snapshot` which rate-limits.
        """
        self._draw(snap)

    def _draw(self, snap) -> None:
        # ---- Panel 1: log g(E) ----
        visited = snap.visited
        if visited.any():
            shifted = snap.g.copy()
            shifted[visited] -= shifted[visited].min()
            xs = self._bin_centers[visited]
            ys = shifted[visited]
            self._line_g.set_data(xs, ys)
            ymin, ymax = float(ys.min()), float(ys.max())
            if self._log_g_exact is not None:
                ref_y = self._log_g_exact[visited] - self._log_g_exact[visited].min()
                ymax = max(ymax, float(ref_y.max()))
            self._ax_g.set_xlim(self._bin_centers.min(), self._bin_centers.max())
            self._ax_g.set_ylim(ymin - 0.5, ymax * 1.05 + 0.5)

        # ---- Panel 2: H(E) ----
        for bar, h in zip(self._bars_H, snap.H):
            bar.set_height(float(h))
        if visited.any() and snap.H[visited].mean() > 0:
            mean_H = float(snap.H[visited].mean())
            self._line_H_threshold.set_ydata(
                [self._flatness_threshold * mean_H, self._flatness_threshold * mean_H]
            )
        max_h = float(snap.H.max()) if snap.H.size > 0 else 1.0
        self._ax_H.set_ylim(0, max(1.0, max_h * 1.1))
        self._ax_H.set_xlim(self._bin_centers.min(), self._bin_centers.max())
        regime = "1/t-WL" if snap.in_1overt else f"standard (stage {snap.n_f_stages})"
        self._ax_H.set_xlabel(f"E   —   {regime},  accept = {snap.acceptance_rate:.2f}")

        # ---- Panel 3: ln_f and flatness vs t ----
        t_arr = np.asarray(self._t_hist)
        lnf_arr = np.asarray(self._lnf_hist)
        flat_arr = np.asarray(self._flat_hist)
        self._line_lnf.set_data(t_arr, lnf_arr)
        self._line_flat.set_data(t_arr, flat_arr)
        if t_arr.size > 1:
            # 1/t reference matches the asymptote of the 1/t-WL regime.
            one_over_t = 1.0 / t_arr.astype(np.float64)
            self._line_oneovert.set_data(t_arr, one_over_t)
            self._ax_t.set_xlim(max(1, int(t_arr.min())), int(t_arr.max() * 1.1) + 1)
            self._ax_t.set_ylim(min(lnf_arr.min(), 1e-9) * 0.5, max(lnf_arr.max(), 1.0) * 2)

        # Vertical markers for f-stage transitions (drawn fresh each time).
        # Cheap because there are at most ~30.
        for collection in list(self._ax_t.collections):
            collection.remove()
        if self._stage_transitions:
            xs = [t for t, _ in self._stage_transitions]
            self._ax_t.vlines(
                xs, 1e-30, 1e30, color="gray", lw=0.5, alpha=0.4, zorder=-1,
            )
        if self._one_over_t_t is not None:
            self._ax_t.vlines(
                [self._one_over_t_t], 1e-30, 1e30,
                color="C3", lw=1.0, alpha=0.7, zorder=-1,
            )

        # ---- title ----
        self._fig.suptitle(
            f"{self._title}    t = {snap.t:,}    "
            f"ln_f = {snap.ln_f:.3e}    "
            f"flatness = {snap.flatness:.3f}    "
            f"n_visited = {int(snap.visited.sum())}/{snap.visited.size}"
        )

        self._fig.canvas.draw_idle()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass

    def _draw_final(self) -> None:
        """One last full draw before blocking. Uses the most recent snapshot."""
        if not self._t_hist:
            return
        # Re-issue draw on stored arrays with no snapshot-specific bits
        # (the bars/g lines were last updated in _draw, which is fine).
        self._fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Recording and movie rendering
# ---------------------------------------------------------------------------

class SnapshotRecorder:
    """`progress_callback` that buffers a sub-sampled history of snapshots.

    Sampling is log-spaced in ``t`` so the early stages (where g and H
    change visibly between checks) get many frames and the late 1/t
    regime (where things change slowly) gets few. Aim for ~``n_frames``
    total over a typical run.

    Usage::

        recorder = SnapshotRecorder(n_frames=600)
        driver.run(..., progress_callback=recorder)
        make_movie(recorder.snapshots, ...)
    """

    def __init__(self, n_frames: int = 600, t_min: int = 1) -> None:
        self.n_frames = int(n_frames)
        self.snapshots: list = []
        # Geometric ratio so ~n_frames samples span t ∈ [1, ~10^8]:
        # we recompute the ratio adaptively from the run's actual t range.
        self._next_t = max(1, int(t_min))
        self._last_t = 0

    def __call__(self, snap) -> None:
        if snap.t >= self._next_t:
            self.snapshots.append(snap)
            # Aim for ~n_frames over a run that may reach 10^8 trials.
            # ratio ≈ 10^(8 / n_frames) gives even log-spacing.
            ratio = 10.0 ** (8.0 / self.n_frames)
            self._next_t = max(int(snap.t * ratio), snap.t + 1)
            self._last_t = snap.t


class TrialRecorder:
    """`trial_callback` that buffers per-trial walker state.

    Records ``(t, bin_current, energy, ln_f, accepted)`` for every trial
    up to ``max_records`` (or unlimited). Storage is ~32 bytes/trial;
    10⁵ trials → ~3 MB. After the run, replay via
    :func:`make_trajectory_movie` to see the walker stepping bin by bin
    while the histogram and ``g`` build up.

    Pair this with ``max_trials=N`` on ``WLDriver.run`` for a short
    self-contained demo of the per-trial dynamics. Trying to record an
    entire production run (10⁸ trials) is impractical — the recorder
    would consume gigabytes.
    """

    def __init__(self, max_records: Optional[int] = None) -> None:
        self.max_records = max_records
        self.t: list[int] = []
        self.bin: list[int] = []
        self.energy: list[float] = []
        self.ln_f: list[float] = []
        self.accepted: list[bool] = []
        self._stop = False

    def __call__(self, t, bin_current, energy, ln_f, accepted) -> None:
        if self._stop:
            return
        self.t.append(t)
        self.bin.append(bin_current)
        self.energy.append(energy)
        self.ln_f.append(ln_f)
        self.accepted.append(accepted)
        if self.max_records is not None and len(self.t) >= self.max_records:
            self._stop = True

    def as_arrays(self) -> dict:
        return {
            "t": np.asarray(self.t, dtype=np.int64),
            "bin": np.asarray(self.bin, dtype=np.int64),
            "energy": np.asarray(self.energy, dtype=np.float64),
            "ln_f": np.asarray(self.ln_f, dtype=np.float64),
            "accepted": np.asarray(self.accepted, dtype=bool),
        }


def make_trajectory_movie(
    history,
    output_path,
    *,
    bin_centers,
    log_g_exact=None,
    title: str = "Wang-Landau trajectory",
    fps: int = 30,
    n_frames: Optional[int] = None,
    dpi: int = 110,
) -> None:
    """Per-trial animation: walker hopping between bins, with H and ``g`` building up.

    Three panels:

    1. **log g(E)** building up across the visited bins (one bin gets
       ``+ln_f`` per trial). Optional dashed reference overlay.
    2. **H(E)** bars rising as visits accumulate, with the **current bin
       highlighted** in red. ``H`` is reset whenever the recorded ``ln_f``
       drops (f-stage halve), exactly mirroring the WL driver.
    3. **E vs t** time series — the walker's energy trajectory, with the
       current step marked as a red dot.

    ``history`` is the dict returned by ``TrialRecorder.as_arrays()``.

    If ``n_frames`` is given and < ``len(history['t'])``, frames are
    chosen on a log-spaced schedule of trial index — so the visible
    walker advances one-step-at-a-time early on and skips through more
    trials per frame later. Combined with constant playback FPS, this
    makes the video "speed up" as the histogram approaches flatness.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    n_bins = len(bin_centers)
    bin_width = float(np.median(np.diff(bin_centers))) if n_bins > 1 else 1.0
    bar_w = 0.9 * bin_width

    t_arr = history["t"]
    bin_arr = history["bin"]
    energy_arr = history["energy"]
    ln_f_arr = history["ln_f"]
    accepted_arr = history["accepted"]
    n_recorded = len(t_arr)
    if n_recorded == 0:
        raise ValueError("history is empty")

    # ---- choose frame indices ----
    if n_frames is None or n_frames >= n_recorded:
        frame_indices = np.arange(n_recorded, dtype=np.int64)
    else:
        # Geometric spacing of trial indices: early frames hit consecutive
        # trials (because t_i+1 = ceil(t_i * r) ≈ t_i + 1 while r-1 < 1/t_i),
        # then progressively skip more.
        log_idx = np.linspace(0.0, np.log(n_recorded), n_frames)
        raw = np.exp(log_idx).astype(np.int64) - 1
        frame_indices = np.unique(np.clip(raw, 0, n_recorded - 1))

    # ---- precompute cumulative g, H per displayed frame ----
    # Replay every trial up to each frame index; reset H at any ln_f drop.
    cum_g = np.zeros((len(frame_indices), n_bins), dtype=np.float64)
    cum_H = np.zeros((len(frame_indices), n_bins), dtype=np.int64)
    visited_at_frame = np.zeros((len(frame_indices), n_bins), dtype=bool)
    halve_at_frame = np.zeros(len(frame_indices), dtype=bool)

    g_run = np.zeros(n_bins, dtype=np.float64)
    H_run = np.zeros(n_bins, dtype=np.int64)
    v_run = np.zeros(n_bins, dtype=bool)
    prev_ln_f = float(ln_f_arr[0])
    saw_halve = False

    next_frame = 0
    for j in range(n_recorded):
        cur_ln_f = float(ln_f_arr[j])
        if cur_ln_f < prev_ln_f - 1e-30:
            # f-stage halve: WL resets H. g and visited persist.
            H_run[:] = 0
            saw_halve = True
        prev_ln_f = cur_ln_f
        b = int(bin_arr[j])
        g_run[b] += cur_ln_f
        H_run[b] += 1
        v_run[b] = True
        if next_frame < len(frame_indices) and j == int(frame_indices[next_frame]):
            cum_g[next_frame] = g_run
            cum_H[next_frame] = H_run
            visited_at_frame[next_frame] = v_run
            halve_at_frame[next_frame] = saw_halve
            saw_halve = False
            next_frame += 1

    # ---- sub-sample dense trajectory for plotting (≤ ~3000 points) ----
    traj_stride = max(1, n_recorded // 3000)
    traj_t = t_arr[::traj_stride]
    traj_E = energy_arr[::traj_stride]
    # index in the sub-sampled arrays corresponding to each frame
    traj_idx_for_frame = np.searchsorted(traj_t, t_arr[frame_indices], side="right") - 1
    traj_idx_for_frame = np.clip(traj_idx_for_frame, 0, len(traj_t) - 1)

    # ---- figure + axes ----
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [3, 3, 2]},
    )
    ax_g, ax_H, ax_traj = axes

    (line_g,) = ax_g.plot([], [], color="C0", lw=2.0, label="WL log g (shifted)")
    if log_g_exact is not None:
        log_g_exact = np.asarray(log_g_exact)
        finite = np.isfinite(log_g_exact)
        ref = log_g_exact.copy()
        if finite.any():
            ref[finite] -= ref[finite].min()
        ax_g.plot(
            bin_centers[finite], ref[finite],
            color="k", ls="--", lw=1.0, alpha=0.6, label="reference (exact)",
        )
    ax_g.set_ylabel("log g(E)  (shifted)")
    ax_g.grid(alpha=0.3)
    ax_g.legend(loc="upper right")
    ax_g.set_xlim(bin_centers.min(), bin_centers.max())

    bars_H = ax_H.bar(bin_centers, np.zeros(n_bins), width=bar_w, color="C2", alpha=0.85)
    current_bar = ax_H.bar(
        [bin_centers[bin_arr[int(frame_indices[0])]]], [0.0],
        width=bar_w, color="red", alpha=0.95, zorder=5,
    )[0]
    ax_H.set_ylabel("H(E)")
    ax_H.set_xlim(bin_centers.min(), bin_centers.max())
    ax_H.grid(alpha=0.3)

    (line_traj,) = ax_traj.plot([], [], color="C1", lw=0.7, alpha=0.7)
    (walker_dot,) = ax_traj.plot([], [], "ro", markersize=8, zorder=5)
    ax_traj.set_xlabel("t (trials)")
    ax_traj.set_ylabel("E(t)")
    ax_traj.set_xlim(0, int(t_arr[-1]))
    ax_traj.set_ylim(float(energy_arr.min()) - 4, float(energy_arr.max()) + 4)
    ax_traj.grid(alpha=0.3)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def animate(i):
        idx = int(frame_indices[i])
        b = int(bin_arr[idx])
        g_frame = cum_g[i]
        H_frame = cum_H[i]
        v_frame = visited_at_frame[i]

        if v_frame.any():
            shifted = g_frame.copy()
            shifted[v_frame] -= shifted[v_frame].min()
            line_g.set_data(bin_centers[v_frame], shifted[v_frame])
            ymax = float(shifted[v_frame].max())
            if log_g_exact is not None:
                fin = np.isfinite(log_g_exact)
                if fin.any():
                    ref = log_g_exact[fin] - log_g_exact[fin].min()
                    ymax = max(ymax, float(ref.max()))
            ax_g.set_ylim(-0.5, ymax * 1.05 + 0.5)

        for bar, h in zip(bars_H, H_frame):
            bar.set_height(int(h))
        current_bar.set_x(bin_centers[b] - bar_w / 2)
        current_bar.set_height(int(H_frame[b]))
        ax_H.set_ylim(0, max(1, int(H_frame.max()) + 1))

        traj_end = int(traj_idx_for_frame[i]) + 1
        line_traj.set_data(traj_t[:traj_end], traj_E[:traj_end])
        walker_dot.set_data([t_arr[idx]], [energy_arr[idx]])

        halve_tag = "  ← halve" if halve_at_frame[i] else ""
        fig.suptitle(
            f"{title}    trial {int(t_arr[idx]):,}    "
            f"bin E = {bin_centers[b]:+.0f}    "
            f"ln_f = {float(ln_f_arr[idx]):.3g}    "
            f"n_visited = {int(v_frame.sum())}/{n_bins}    "
            f"{'accepted' if bool(accepted_arr[idx]) else 'rejected'}"
            f"{halve_tag}"
        )
        return ()

    ani = animation.FuncAnimation(
        fig, animate, frames=len(frame_indices),
        interval=int(1000 / fps), blit=False, repeat=False,
    )

    ext = output_path.suffix.lower()
    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif ext in (".mp4", ".mov", ".m4v", ".webm"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    else:
        raise ValueError(
            f"unsupported video extension {ext!r}; use .mp4, .mov, .webm, or .gif"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)


def make_movie(
    snapshots,
    output_path,
    *,
    bin_centers,
    flatness_threshold: float = 0.95,
    log_g_exact=None,
    title: str = "Wang-Landau",
    fps: int = 20,
    dpi: int = 110,
) -> None:
    """Render a recorded list of `ProgressSnapshot` as an mp4 (or gif).

    Output format is chosen from ``output_path`` extension. ``.mp4``,
    ``.mov``, ``.webm`` use ffmpeg (must be on PATH); ``.gif`` uses
    Pillow (always available).

    Each frame draws the full viewer state at that snapshot — log g(E),
    H(E), and the running ln_f / flatness time series. Time-series
    progress builds up naturally as frames advance.
    """
    import os
    from pathlib import Path

    import matplotlib

    # Force a non-interactive backend; rendering is offline.
    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation  # noqa: E402

    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    if not snapshots:
        raise ValueError("snapshots list is empty")

    viewer = LiveViewer(
        bin_centers,
        flatness_threshold=flatness_threshold,
        log_g_exact=log_g_exact,
        update_every_s=0.0,
        title=title,
    )

    def animate(idx):
        viewer._on_snapshot(snapshots[idx])
        return ()

    ani = animation.FuncAnimation(
        viewer._fig,
        animate,
        frames=len(snapshots),
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    if ext == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif ext in (".mp4", ".mov", ".m4v", ".webm"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    else:
        raise ValueError(
            f"unsupported video extension {ext!r}; use .mp4, .mov, .webm, or .gif"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer, dpi=dpi)
    # Close the figure to free resources.
    import matplotlib.pyplot as plt
    plt.close(viewer._fig)
