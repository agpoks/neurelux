from atlas_physics.utils.diagrams import INPUT, LEARNED, STATE, box, edge, new_ax, node

fig, ax = new_ax(figsize=(9, 3.4), xlim=(0, 14), ylim=(0, 5))

box(ax, 0.9, 2.5, 1.3, 1.0, "u(t)", INPUT)
xs = [3.0, 5.2, 7.4, 9.6, 11.8]
labels = ["x_0\n(surface)", "x_1", "x_2", "x_3", "x_4\n(deep)"]
edge(ax, (1.55, 2.5), (xs[0] - 0.4, 2.5), label="G_0")
for i, (x, lab) in enumerate(zip(xs, labels)):
    node(ax, x, 2.5, 0.42, "", STATE)
    ax.text(x, 3.35, lab, ha="center", va="bottom", fontsize=8.5, color="#334155")
    # capacitance to ground, drawn as a short stub below each node
    ax.plot([x, x], [2.5 - 0.42, 1.4], color=LEARNED, lw=1.4)
    ax.plot([x - 0.25, x + 0.25], [1.4, 1.4], color=LEARNED, lw=1.4)
    ax.text(x, 1.05, "C_%d" % i, ha="center", fontsize=8, color=LEARNED)
    if i > 0:
        edge(ax, (xs[i - 1] + 0.42, 2.5), (x - 0.42, 2.5), label="G_%d" % i)

ax.set_title("1D Cauer ladder: series G, shunt C, driven at the surface", fontsize=11)
