from atlas_physics.utils.diagrams import ENCODED, INPUT, LEARNED, STATE, arrow, box, new_ax

fig, ax = new_ax(figsize=(6.5, 9.5), xlim=(0, 8), ylim=(0, 24))

steps = [
    ("I, U, g, v, T", INPUT),
    ("nonlinear magnetic circuit  (D/E)", ENCODED),
    ("reluctance graph  (E)", ENCODED),
    ("surface x depth Graph-Cauer  (G)", ENCODED),
    ("magnetic flux  Phi_i", STATE),
    ("co-energy / attraction force  (K)", ENCODED),
    ("friction residual  (guided) + thermal RC  (encoded)", LEARNED),
    ("braking force + temperature state", INPUT),
]
y = 22.5
for i, (text, color) in enumerate(steps):
    box(ax, 4, y, 7.2, 1.6, text, color, fontsize=8.5)
    if i > 0:
        arrow(ax, (4, y + 1.9), (4, y + 0.85))
    y -= 3.0

ax.set_title("Recommended architecture (provisional)", fontsize=11)
