from atlas_physics.utils.diagrams import ENCODED, INPUT, OTHER, arrow, box, new_ax

fig, ax = new_ax(figsize=(10, 4.6), xlim=(0, 15), ylim=(0, 8))

# -- guided --
box(ax, 1.7, 6.2, 2.2, 1.0, "physical\nfeatures", INPUT)
box(ax, 1.7, 4.2, 2.2, 1.0, "black-box\nNN", OTHER)
box(ax, 1.7, 2.2, 2.2, 1.0, "output", INPUT)
arrow(ax, (1.7, 5.7), (1.7, 4.7))
arrow(ax, (1.7, 3.7), (1.7, 2.7))
ax.text(1.7, 0.9, "GUIDED:\nphysics shapes the input", ha="center", fontsize=8.5)

# -- informed --
box(ax, 6.5, 6.2, 2.2, 1.0, "input", INPUT)
box(ax, 6.5, 4.2, 2.2, 1.0, "black-box\nNN", OTHER)
box(ax, 6.5, 2.2, 2.2, 1.0, "output", INPUT)
box(ax, 9.4, 3.2, 2.2, 1.0, "physics\nresidual loss", ENCODED)
arrow(ax, (6.5, 5.7), (6.5, 4.7))
arrow(ax, (6.5, 3.7), (6.5, 2.7))
arrow(ax, (7.6, 2.2), (9.4, 2.75), dashed=True, color=ENCODED)
ax.text(6.5, 0.9, "INFORMED:\nphysics penalizes bad outputs\n(but can't prevent them)", ha="center", fontsize=8.5)

# -- encoded --
box(ax, 12.5, 6.2, 2.2, 1.0, "input", INPUT)
box(ax, 12.5, 4.2, 2.6, 1.2, "physics equation\n(learned params only)", ENCODED)
box(ax, 12.5, 2.2, 2.2, 1.0, "output", INPUT)
arrow(ax, (12.5, 5.7), (12.5, 4.85))
arrow(ax, (12.5, 3.55), (12.5, 2.7))
ax.text(12.5, 0.9, "ENCODED:\nphysics IS the computation\n-- can't be violated", ha="center", fontsize=8.5)

ax.set_title("Where the physics lives", fontsize=11)
