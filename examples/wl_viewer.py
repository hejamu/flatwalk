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

    def __init__(
        self,
        max_records: Optional[int] = None,
        state_capturer=None,
        capture_at: Optional[set] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_records:
            Stop recording after this many trials (None = unlimited).
        state_capturer:
            Optional ``callable(walker) -> Any``. When supplied, the
            captured value is appended to ``self.states`` on each trial
            (or only at trials whose 1-based ``t`` is in ``capture_at``).
            Useful for snapshotting e.g. an Ising spin grid for a
            companion visualization.
        capture_at:
            Optional set of 1-based ``t`` values to selectively capture
            state. Without it (and with a ``state_capturer``), every
            trial captures state.
        """
        self.max_records = max_records
        self.state_capturer = state_capturer
        self.capture_at = capture_at
        self.t: list[int] = []
        self.bin: list[int] = []
        self.energy: list[float] = []
        self.ln_f: list[float] = []
        self.accepted: list[bool] = []
        self.states: dict = {}  # 1-based t → captured state
        self._stop = False

    def __call__(self, t, walker, ln_f, accepted) -> None:
        if self._stop:
            return
        self.t.append(t)
        self.bin.append(walker.bin_current)
        self.energy.append(walker.energy)
        self.ln_f.append(ln_f)
        self.accepted.append(accepted)
        if self.state_capturer is not None and (
            self.capture_at is None or t in self.capture_at
        ):
            self.states[t] = self.state_capturer(walker)
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


def build_frame_indices_from_schedule(schedule, n_recorded: int) -> np.ndarray:
    """Build a list of trial indices to render from a piecewise stride schedule.

    ``schedule`` is a sequence of ``(t_end, stride)`` tuples. Frames are
    emitted for ``range(prev_end, t_end, stride)`` per segment. The last
    valid trial index is always included to anchor the animation.

    Example::

        schedule = [(1500, 1), (30_000, 20), (1_000_000, 300)]
        # → trials 0,1,2,...,1499, then 1500,1520,...,29980,
        #   then 30000,30300,...,999900, then 999999.
    """
    indices: list[int] = []
    start = 0
    for end, stride in schedule:
        end = min(int(end), n_recorded)
        stride = max(1, int(stride))
        if end > start:
            indices.extend(range(start, end, stride))
        start = end
        if start >= n_recorded:
            break
    if start < n_recorded:
        # remaining tail at the last stride
        last_stride = max(1, int(schedule[-1][1]))
        indices.extend(range(start, n_recorded, last_stride))
    if not indices or indices[-1] != n_recorded - 1:
        indices.append(n_recorded - 1)
    return np.array(sorted(set(indices)), dtype=np.int64)


def make_trajectory_movie(
    history,
    output_path,
    *,
    bin_centers,
    log_g_exact=None,
    title: str = "Wang-Landau trajectory",
    fps: int = 30,
    n_frames: Optional[int] = None,
    frame_schedule: Optional[list] = None,
    flatness_threshold: float = 0.8,
    spin_grids: Optional[dict] = None,
    dpi: int = 110,
) -> None:
    """Per-trial animation: walker hopping between bins, with H and ``g`` building up.

    Three panels:

    1. **log g(E)** building up across the visited bins (one bin gets
       ``+ln_f`` per trial). Optional dashed reference overlay.
    2. **H(E)** is *cumulative* across the whole run — bars only ever
       grow, segments are stacked and **colored by f-stage** so the
       breakdown into early (large-``ln_f``) and late (small-``ln_f``)
       contributions is visible. A red vertical line marks the current
       bin.
    3. **E vs t** time series — the walker's energy trajectory, with the
       current step marked as a red dot.

    ``history`` is the dict returned by ``TrialRecorder.as_arrays()``.

    If ``n_frames`` is given and < ``len(history['t'])``, frames are
    chosen on a log-spaced schedule of trial index — so the visible
    walker advances one-step-at-a-time early on and skips through more
    trials per frame later. Combined with constant playback FPS, this
    makes the video "speed up" as the histogram approaches flatness.

    ``frame_schedule`` (preferred over ``n_frames`` when given) is a list
    of ``(t_end, stride)`` segments giving piecewise-constant playback
    speed. See :func:`build_frame_indices_from_schedule`.

    The middle panel also shows the **current-stage** ``H(E)`` as an
    orange line on a secondary right-hand axis — this is what the
    algorithm actually evaluates against ``flatness_threshold``. The
    line resets to zero whenever a halve fires; its ``min/mean`` ratio
    is the flatness number printed in the title.

    If ``spin_grids`` is given (mapping 1-based trial number → 2-D
    spin array), a fourth panel on the right shows the current spin
    configuration. Useful for visualizing the Ising lattice alongside
    the WL bookkeeping.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib import cm

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

    # ---- stage label per trial: every ln_f drop = new f-stage ----
    if n_recorded > 1:
        halve_mask = np.diff(ln_f_arr) < -1e-30
        stage_at_trial = np.concatenate(
            ([0], np.cumsum(halve_mask).astype(np.int64))
        )
    else:
        stage_at_trial = np.zeros(1, dtype=np.int64)
    n_stages = int(stage_at_trial[-1]) + 1

    # ---- choose frame indices ----
    if frame_schedule is not None:
        frame_indices = build_frame_indices_from_schedule(frame_schedule, n_recorded)
    elif n_frames is None or n_frames >= n_recorded:
        frame_indices = np.arange(n_recorded, dtype=np.int64)
    else:
        log_idx = np.linspace(0.0, np.log(n_recorded), n_frames)
        raw = np.exp(log_idx).astype(np.int64) - 1
        frame_indices = np.unique(np.clip(raw, 0, n_recorded - 1))

    # ---- precompute per-frame state ----
    # H is *cumulative* (no halve reset); broken down per stage so the
    # renderer can stack the bars by stage colour. We also keep the
    # *current-stage* H separately so the algorithm's flatness check is
    # plottable.
    n_frames_actual = len(frame_indices)
    cum_g = np.zeros((n_frames_actual, n_bins), dtype=np.float64)
    cum_H_stage = np.zeros((n_frames_actual, n_stages, n_bins), dtype=np.int64)
    cur_stage_H_at_frame = np.zeros((n_frames_actual, n_bins), dtype=np.int64)
    visited_at_frame = np.zeros((n_frames_actual, n_bins), dtype=bool)
    stage_at_frame = np.zeros(n_frames_actual, dtype=np.int64)
    halve_at_frame = np.zeros(n_frames_actual, dtype=bool)
    flatness_at_frame = np.zeros(n_frames_actual, dtype=np.float64)

    g_run = np.zeros(n_bins, dtype=np.float64)
    H_stage_run = np.zeros((n_stages, n_bins), dtype=np.int64)
    cur_stage_H = np.zeros(n_bins, dtype=np.int64)
    v_run = np.zeros(n_bins, dtype=bool)
    current_stage_seen = 0

    next_frame = 0
    prev_stage_for_frame = 0
    for j in range(n_recorded):
        b = int(bin_arr[j])
        s = int(stage_at_trial[j])
        if s != current_stage_seen:
            # New stage starts — reset the per-stage H tracker.
            cur_stage_H[:] = 0
            current_stage_seen = s
        g_run[b] += float(ln_f_arr[j])
        H_stage_run[s, b] += 1
        cur_stage_H[b] += 1
        v_run[b] = True
        if next_frame < n_frames_actual and j == int(frame_indices[next_frame]):
            cum_g[next_frame] = g_run
            cum_H_stage[next_frame] = H_stage_run
            cur_stage_H_at_frame[next_frame] = cur_stage_H
            visited_at_frame[next_frame] = v_run
            stage_at_frame[next_frame] = s
            halve_at_frame[next_frame] = (s > prev_stage_for_frame)
            prev_stage_for_frame = s
            # Flatness of current per-stage H over visited bins.
            mask = v_run
            hv = cur_stage_H[mask]
            if hv.size > 0 and hv.mean() > 0:
                flatness_at_frame[next_frame] = float(hv.min()) / float(hv.mean())
            next_frame += 1

    # ---- sub-sample dense trajectory for plotting (≤ ~3000 points) ----
    traj_stride = max(1, n_recorded // 3000)
    traj_t = t_arr[::traj_stride]
    traj_E = energy_arr[::traj_stride]
    traj_idx_for_frame = np.searchsorted(traj_t, t_arr[frame_indices], side="right") - 1
    traj_idx_for_frame = np.clip(traj_idx_for_frame, 0, len(traj_t) - 1)

    # ---- stage colors (viridis; stage k -> colors[k]) ----
    if n_stages == 1:
        stage_colors = np.array([cm.viridis(0.55)])
    else:
        stage_colors = cm.viridis(np.linspace(0.05, 0.92, n_stages))

    # ---- figure + axes ----
    from matplotlib.colors import ListedColormap

    has_spins = spin_grids is not None and len(spin_grids) > 0

    if has_spins:
        # Two columns: stacked plots on the left, spin grid + horizontal
        # stage colorbar in the right column.
        fig = plt.figure(figsize=(13, 8.5))
        gs = fig.add_gridspec(
            3, 2,
            width_ratios=[3, 1],
            height_ratios=[3, 3, 2],
            hspace=0.32, wspace=0.08,
        )
        ax_g = fig.add_subplot(gs[0, 0])
        ax_H = fig.add_subplot(gs[1, 0])
        ax_traj = fig.add_subplot(gs[2, 0])
        right_gs = gs[:, 1].subgridspec(2, 1, height_ratios=[14, 1], hspace=0.12)
        ax_spin = fig.add_subplot(right_gs[0])
        cax_stage = fig.add_subplot(right_gs[1])
        _colorbar_orientation = "horizontal"
    else:
        # No spin grid: stacked layout with the colorbar as a thin strip
        # below the histogram panel.
        fig = plt.figure(figsize=(11, 9))
        gs = fig.add_gridspec(
            4, 1, height_ratios=[3, 3, 0.18, 2], hspace=0.3,
        )
        ax_g = fig.add_subplot(gs[0])
        ax_H = fig.add_subplot(gs[1])
        cax_stage = fig.add_subplot(gs[2])
        ax_traj = fig.add_subplot(gs[3])
        ax_spin = None
        _colorbar_orientation = "horizontal"

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

    # One bar collection per stage; stack via per-bin ``bottom``.
    bars_per_stage = []
    for k in range(n_stages):
        bars = ax_H.bar(
            bin_centers, np.zeros(n_bins),
            width=bar_w, color=stage_colors[k], alpha=0.92,
            edgecolor="white", linewidth=0.0, label=f"stage {k}",
        )
        bars_per_stage.append(bars)
    current_line = ax_H.axvline(
        bin_centers[bin_arr[int(frame_indices[0])]],
        color="red", lw=2.5, alpha=0.85, zorder=10,
    )
    ax_H.set_ylabel("H(E)  (cumulative)")
    ax_H.set_xlim(bin_centers.min(), bin_centers.max())
    ax_H.grid(alpha=0.3, axis="y")

    # Stage → colour mapping shown as a discrete colorbar (replaces the
    # previous in-axes legend, which competed for space with the bars).
    stage_cmap = ListedColormap(list(stage_colors))
    norm = plt.Normalize(vmin=0, vmax=max(n_stages - 1, 1))
    sm = plt.cm.ScalarMappable(cmap=stage_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax_stage, orientation=_colorbar_orientation)
    if n_stages <= 15:
        cbar.set_ticks(range(n_stages))
    cbar.set_label("f-stage")

    # Per-stage H overlay on the right-hand axis (what the algorithm
    # checks against ``flatness_threshold``).
    ax_H_right = ax_H.twinx()
    (line_cur_stage,) = ax_H_right.plot(
        [], [], color="darkorange", lw=2.4, marker="o", markersize=3.5,
        alpha=0.95, label="current-stage H",
    )
    max_cur_H = int(cur_stage_H_at_frame.max()) if cur_stage_H_at_frame.size > 0 else 1
    ax_H_right.set_ylim(0, max(1, int(max_cur_H * 1.15)))
    ax_H_right.set_ylabel("per-stage H", color="darkorange")
    ax_H_right.tick_params(axis="y", labelcolor="darkorange")

    (line_traj,) = ax_traj.plot([], [], color="C1", lw=0.7, alpha=0.7)
    (walker_dot,) = ax_traj.plot([], [], "ro", markersize=8, zorder=5)
    ax_traj.set_xlabel("t (trials)")
    ax_traj.set_ylabel("E(t)")
    ax_traj.set_xlim(0, int(t_arr[-1]))
    ax_traj.set_ylim(float(energy_arr.min()) - 4, float(energy_arr.max()) + 4)
    ax_traj.grid(alpha=0.3)

    # ---- right column: spin configuration ----
    spin_img = None
    spin_keys_sorted: Optional[np.ndarray] = None
    if has_spins:
        # Build a sorted array of trial-number keys for fast searchsorted lookup.
        spin_keys_sorted = np.array(sorted(spin_grids.keys()), dtype=np.int64)
        first_spins = np.asarray(spin_grids[int(spin_keys_sorted[0])])
        spin_img = ax_spin.imshow(
            first_spins, cmap="RdBu_r", vmin=-1.4, vmax=1.4,
            interpolation="nearest", aspect="equal",
        )
        ax_spin.set_xticks([])
        ax_spin.set_yticks([])
        ax_spin.set_title(f"spin configuration   ({first_spins.shape[0]}×{first_spins.shape[1]})")

    # Don't call tight_layout: it tends to drift the colorbar off-axis under
    # FuncAnimation. We've already laid out via gridspec.

    def animate(i):
        idx = int(frame_indices[i])
        b = int(bin_arr[idx])
        g_frame = cum_g[i]
        H_stage_frame = cum_H_stage[i]  # shape (n_stages, n_bins)
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

        # Stacked H bars: per-stage height with cumulative bottom.
        bottoms = np.zeros(n_bins, dtype=np.int64)
        for k in range(n_stages):
            h_k = H_stage_frame[k]
            for bin_idx, bar in enumerate(bars_per_stage[k]):
                bar.set_y(int(bottoms[bin_idx]))
                bar.set_height(int(h_k[bin_idx]))
            bottoms += h_k
        total_H = int(bottoms.max())
        ax_H.set_ylim(0, max(1, total_H + 1))
        current_line.set_xdata([bin_centers[b], bin_centers[b]])

        # Current-stage H overlay (right axis).
        cur_H_f = cur_stage_H_at_frame[i]
        line_cur_stage.set_data(bin_centers[v_frame], cur_H_f[v_frame])

        traj_end = int(traj_idx_for_frame[i]) + 1
        line_traj.set_data(traj_t[:traj_end], traj_E[:traj_end])
        walker_dot.set_data([t_arr[idx]], [energy_arr[idx]])

        # Spin grid (most-recent captured snapshot at or before this trial).
        if spin_img is not None and spin_keys_sorted is not None:
            cur_t = int(t_arr[idx])
            pos = int(np.searchsorted(spin_keys_sorted, cur_t, side="right")) - 1
            if pos >= 0:
                key = int(spin_keys_sorted[pos])
                spin_img.set_data(np.asarray(spin_grids[key]))

        halve_tag = f"  ← halve to stage {int(stage_at_frame[i])}" if halve_at_frame[i] else ""
        flat = float(flatness_at_frame[i])
        flat_tag = (
            f"flatness {flat:.3f} / {flatness_threshold:.2f}"
            if flat > 0 else "flatness —"
        )
        fig.suptitle(
            f"{title}    trial {int(t_arr[idx]):,}    "
            f"bin E = {bin_centers[b]:+.0f}    "
            f"ln_f = {float(ln_f_arr[idx]):.3g}    "
            f"stage {int(stage_at_frame[i])}/{n_stages - 1}    "
            f"{flat_tag}    "
            f"n_visited = {int(v_frame.sum())}/{n_bins}"
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
