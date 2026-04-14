import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# -----------------------------
# Style
# -----------------------------
VERTEX_COLOR = '#ff7f0e'
EDGE_COLOR = '#1f77b4'
FACE_COLOR = '#1f77b4'
FACE_ALPHA = 0.28
VERTEX_SIZE = 100
LINE_WIDTH = 2.2

def clean_axis(ax):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# -----------------------------
# Common vertices
# -----------------------------
v0 = (0.0, 0.0)
v1 = (1.6, 0.0)
v2 = (0.8, 1.35)

vertices = [v0, v1, v2]

label_offsets = [
    (-0.18, -0.18),  # v0
    (0.08, -0.18),   # v1
    (-0.02, 0.16),   # v2
]

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

# =========================================================
# Left panel: boundary of a triangle -> nontrivial H1 class
# =========================================================
ax = axes[0]

# Edges only
edges = [(v0, v1), (v1, v2), (v2, v0)]
for a, b in edges:
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color=EDGE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=2
    )

# Vertices
ax.scatter(
    [p[0] for p in vertices],
    [p[1] for p in vertices],
    s=VERTEX_SIZE,
    color=VERTEX_COLOR,
    zorder=3
)

# Labels
for i, (x, y) in enumerate(vertices):
    dx, dy = label_offsets[i]
    ax.text(x + dx, y + dy, rf'$v_{i}$', fontsize=11)

# Hole marker
hole = Circle((0.8, 0.45), 0.18, fill=False, linestyle='--', linewidth=1.5, edgecolor='black')
ax.add_patch(hole)
ax.text(0.8, 0.45, r'$H_1$', ha='center', va='center', fontsize=11)

clean_axis(ax)
ax.set_xlim(-0.5, 2.1)
ax.set_ylim(-0.45, 1.75)

# =========================================================
# Right panel: filled triangle -> H1 vanishes
# =========================================================
ax = axes[1]

# Filled 2-simplex
triangle = Polygon(
    [v0, v1, v2],
    closed=True,
    facecolor=FACE_COLOR,
    alpha=FACE_ALPHA,
    edgecolor='none',
    zorder=1
)
ax.add_patch(triangle)

# Boundary edges
for a, b in edges:
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color=EDGE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=2
    )

# Vertices
ax.scatter(
    [p[0] for p in vertices],
    [p[1] for p in vertices],
    s=VERTEX_SIZE,
    color=VERTEX_COLOR,
    zorder=3
)

# Labels
for i, (x, y) in enumerate(vertices):
    dx, dy = label_offsets[i]
    ax.text(x + dx, y + dy, rf'$v_{i}$', fontsize=11)

# Interior note
ax.text(0.8, 0.45, r'filled $2$-simplex', ha='center', va='center', fontsize=11)

clean_axis(ax)
ax.set_xlim(-0.5, 2.1)
ax.set_ylim(-0.45, 1.75)

# -----------------------------
# Bottom labels aligned
# -----------------------------
panel_labels = [
    r'cycle present: nontrivial $H_1$',
    r'cycle filled: trivial $H_1$'
]

for ax, label in zip(axes, panel_labels):
    bbox = ax.get_position()
    x_center = (bbox.x0 + bbox.x1) / 2
    fig.text(x_center, 0.08, label, ha='center', va='top', fontsize=12)

plt.subplots_adjust(wspace=0.35, bottom=0.20)

plt.savefig('homology_cycle_vs_filled.png', dpi=300, bbox_inches='tight')
plt.savefig('homology_cycle_vs_filled.pdf', bbox_inches='tight')
plt.show()