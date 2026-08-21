import matplotlib.patches as mpatches
from atlas_physics.utils.diagrams import ENCODED, STATE, arrow, new_ax

fig, ax = new_ax(figsize=(8, 4), xlim=(0, 12), ylim=(0, 8))

# rail
ax.add_patch(mpatches.Rectangle((0.5, 0.8), 11, 1.0, facecolor="#334155"))
ax.text(6, 1.3, "rail head", color="white", ha="center", va="center", fontsize=9)

# pole piece with a small air gap above the rail
ax.add_patch(mpatches.Rectangle((3.5, 2.3), 5, 2.2, facecolor=ENCODED))
ax.text(6, 3.4, "pole / coil (N, I)", color="white", ha="center", va="center", fontsize=9)
ax.annotate("", xy=(6, 1.85), xytext=(6, 2.25), arrowprops=dict(arrowstyle="-", color="#334155", lw=1.2))
ax.text(6.6, 2.05, "gap g", fontsize=8, color="#334155")

# flux loop
arrow(ax, (4.3, 2.3), (4.3, 1.8), color=ENCODED, curve=0.0)
arrow(ax, (7.7, 1.8), (7.7, 2.3), color=ENCODED, curve=0.0)
ax.annotate(
    "",
    xy=(7.5, 4.6),
    xytext=(4.5, 4.6),
    arrowprops=dict(arrowstyle="-", color=ENCODED, lw=1.6, connectionstyle="arc3,rad=-0.6"),
)
ax.text(6, 5.6, "flux Phi", color=ENCODED, ha="center", fontsize=9)

# velocity arrow
arrow(ax, (9.5, 1.3), (11.3, 1.3), color=STATE, lw=2.0)
ax.text(10.4, 1.6, "v", color=STATE, fontsize=10)

ax.set_title("Magnetic track brake: pole, gap, flux, and rail motion", fontsize=11)
