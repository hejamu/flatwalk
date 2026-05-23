#!/usr/bin/env python3
"""Generate the flatwalk logo assets (light/dark lockups + favicon).

The mark is the physics: an asymmetric double-well ``-log g(E)`` potential
(deep left well, shallow right well, central barrier) filled with histogram
bars up to a flat waterline -- the Wang-Landau picture of biasing the sampled
histogram flat. The wordmark rides the waterline inside the wells.

The whole outline is the real potential ``V(x)`` (a tilted quartic), so the
walls steepen toward the ends like x^4; the bars are clipped to the well region
so their bottoms follow the floor exactly instead of poking through it.

Writes three SVGs into ``docs/src/_static/``:

    flatwalk-logo-light.svg   light-background lockup (mark + wordmark)
    flatwalk-logo-dark.svg    dark-background lockup
    flatwalk-favicon.svg      mark only, for the browser tab

Referenced by ``docs/src/conf.py`` (furo ``light_logo``/``dark_logo`` and
``html_favicon``) and the README header. Re-run after tweaking a parameter:

    python docs/tools/genlogo.py
"""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "docs" / "src" / "_static"
FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"


def double_well(x0: float, scale: float, depth: float, base: float):
    """Return ``V(x)``: an asymmetric double well in screen y (larger = lower).

    A quartic double well ``t**4 - t**2`` with a linear tilt for asymmetry, so
    the left well is deeper than the right with a barrier between them. Being a
    quartic, it steepens toward the ends -- the walls are near-vertical at top.
    """

    def V(x: float) -> float:
        t = (x - x0) / scale
        f = t**4 - t**2 + 0.18 * t
        return base - depth * f

    return V


def geometry(V, fill_level: float, top: float, x0: float, pitch: float, bw: float):
    """Build the outline, the well clip region, and bar positions from ``V``.

    Returns a dict with the wall-top x-range (``xL``/``xR``), the waterline
    x-range (``fx0``/``fx1``), the deepest floor y (``maxV``), the ``outline``
    and ``clip`` path strings, and the list of bar left-edges.
    """
    step = 0.4
    xs = [x0 - 260 + step * k for k in range(int(520 / step) + 1)]
    on = [x for x in xs if V(x) >= top]  # on-canvas span (V below the top edge)
    xL, xR = on[0], on[-1]
    fill = [x for x in xs if V(x) >= fill_level]  # underwater span
    fx0, fx1 = fill[0], fill[-1]
    maxV = max(V(x) for x in fill)

    out_xs = [xL + 1.3 * k for k in range(int((xR - xL) / 1.3))] + [xR]
    outline = "M" + " L".join(f"{x:.1f} {V(x):.2f}" for x in out_xs)

    fl_xs = [fx0 + 1.3 * k for k in range(int((fx1 - fx0) / 1.3))] + [fx1]
    floor_rev = " ".join(f"L{x:.1f} {V(x):.2f}" for x in reversed(fl_xs))
    clip = f"M{fx0:.1f} {fill_level:.0f} L{fx1:.1f} {fill_level:.0f} {floor_rev} Z"

    bars, x = [], fx0
    while x <= fx1:
        if V(x) > fill_level + 0.8:
            bars.append(x - bw / 2)
        x += pitch

    return dict(xL=xL, xR=xR, fx0=fx0, fx1=fx1, maxV=maxV,
                outline=outline, clip=clip, bars=bars)


def lockup(dark: bool) -> None:
    FL, top, pad = 30.0, 8.0, 7.0
    neutral = "#CBD5E1" if dark else "#334155"
    accent = "#2DD4BF" if dark else "#14B8A6"
    V = double_well(x0=100.0, scale=48.0, depth=62.0, base=38.0)
    g = geometry(V, FL, top, x0=100.0, pitch=5.5, bw=4.6)

    H = round(g["maxV"] + 9)
    xmin = g["xL"] - pad
    width = (g["xR"] + pad) - xmin
    cx = (g["fx0"] + g["fx1"]) / 2
    rects = "\n      ".join(
        f'<rect x="{bx:.1f}" y="{FL:.0f}" width="4.6" height="{H - FL:.0f}" rx="1"/>'
        for bx in g["bars"]
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin:.1f} 0 {width:.1f} {H}" role="img" aria-label="flatwalk">
  <title>flatwalk</title>
  <!-- asymmetric double-well -log g(E), filled with histogram bars up to a flat
       waterline; the wordmark rides the waterline inside the wells -->
  <defs><clipPath id="well"><path d="{g['clip']}"/></clipPath></defs>
  <g fill="{accent}" clip-path="url(#well)">
      {rects}
  </g>
  <path d="{g['outline']}" fill="none" stroke="{neutral}" stroke-width="3"
        stroke-linecap="round" stroke-linejoin="round"/>
  <text x="{cx:.1f}" y="{FL - 4:.0f}" text-anchor="middle" font-family="{FONT}"
        font-size="25" font-weight="600" letter-spacing="-0.5" fill="{neutral}">flatwalk</text>
</svg>
"""
    (STATIC / f"flatwalk-logo-{'dark' if dark else 'light'}.svg").write_text(svg)


def favicon() -> None:
    W = H = 48
    FL, top = 19.0, 6.0
    neutral, accent = "#475569", "#14B8A6"
    V = double_well(x0=24.0, scale=15.5, depth=40.0, base=23.0)
    g = geometry(V, FL, top, x0=24.0, pitch=3.4, bw=2.6)

    rects = "\n      ".join(
        f'<rect x="{bx:.1f}" y="{FL:.0f}" width="2.6" height="{H - FL:.0f}" rx="0.8"/>'
        for bx in g["bars"]
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="flatwalk">
  <title>flatwalk</title>
  <!-- asymmetric double well filled with histogram bars to a flat waterline -->
  <defs><clipPath id="well"><path d="{g['clip']}"/></clipPath></defs>
  <g fill="{accent}" clip-path="url(#well)">
      {rects}
  </g>
  <path d="{g['outline']}" fill="none" stroke="{neutral}" stroke-width="2.4"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
    (STATIC / "flatwalk-favicon.svg").write_text(svg)


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    lockup(dark=False)
    lockup(dark=True)
    favicon()
    print(f"wrote 3 SVGs into {STATIC}")


if __name__ == "__main__":
    main()
