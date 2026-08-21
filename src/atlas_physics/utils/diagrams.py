"""Tiny matplotlib diagram primitives shared by the docs (``docs/source/*.md``,
via the Sphinx ``.. plot::`` directive). Not used by the physics models
themselves -- this exists purely so every diagram in the docs shares one
visual language instead of several ad-hoc drawings.

Color convention, used consistently across the docs:

- ``INPUT``   -- raw inputs/outputs (currents, fields, forces)
- ``ENCODED`` -- a physics-encoded element: fixed topology or an exact
  physical relation (e.g. the Cauer ladder's incidence matrix, a co-energy
  derivative). Training cannot violate it.
- ``LEARNED`` -- a learned, positivity-constrained physical parameter
  (e.g. a layer capacitance/conductance). Training is allowed to change it.
- ``STATE``   -- a state variable carried across time steps.
- ``OTHER``   -- everything else (ports, bias terms, bookkeeping).
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

INPUT = "#0f172a"
ENCODED = "#0891b2"
LEARNED = "#f59e0b"
STATE = "#be123c"
OTHER = "#64748b"


def new_ax(figsize=(7.5, 4.2), xlim=(0, 10), ylim=(0, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, color=ENCODED, fontsize=9.5, textcolor="white"):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", color=textcolor, fontsize=fontsize, wrap=True)
    return (cx, cy, w, h)


def node(ax, cx, cy, r, text="", color=STATE, fontsize=8.5, textcolor="#334155"):
    """Circle with its label placed *below* it (unlike `box`, whose label sits
    inside the shape) -- so the default `textcolor` is a dark slate readable
    against the page background, not white.
    """
    patch = mpatches.Circle((cx, cy), r, linewidth=0, facecolor=color)
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy - r - 0.35, text, ha="center", va="top", fontsize=fontsize, color=textcolor)
    return (cx, cy, r)


def arrow(ax, start, end, color="#334155", dashed=False, curve=0.0, label=None, lw=1.6):
    connectionstyle = f"arc3,rad={curve}" if curve else None
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            linestyle="dashed" if dashed else "solid",
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        ),
    )
    if label:
        mx = (start[0] + end[0]) / 2 + (0.6 if curve else 0)
        my = (start[1] + end[1]) / 2 + (0.5 if curve else 0)
        ax.text(mx, my, label, ha="center", va="center", fontsize=8, color=color)


def edge(ax, start, end, color="#94a3b8", lw=1.4, label=None):
    """Undirected connection (no arrowhead) -- used for circuit/graph edges
    (e.g. Cauer-ladder conductances), where nothing actually "flows one way".
    """
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=1)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.3, label, ha="center", va="bottom", fontsize=8, color=color)
