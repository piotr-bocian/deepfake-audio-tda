import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from itertools import combinations

# -----------------------------
# Style
# -----------------------------
VERTEX_COLOR = '#ff7f0e'
EDGE_COLOR = '#1f77b4'
FACE_COLOR = '#1f77b4'
FACE_ALPHA = 0.25
VERTEX_SIZE = 90
LINE_WIDTH = 2.0

def clean_axis(ax):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def rho(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def build_vr(ax, points, epsilon):
    edges = []
    for i, j in combinations(range(len(points)), 2):
        if rho(points[i], points[j]) <= epsilon:
            edges.append((i, j))

    triangles = []
    for i, j, k in combinations(range(len(points)), 3):
        if (
            rho(points[i], points[j]) <= epsilon and
            rho(points[i], points[k]) <= epsilon and
            rho(points[j], points[k]) <= epsilon
        ):
            triangles.append((i, j, k))

    # Filled triangles first
    for i, j, k in triangles:
        tri = [points[i], points[j], points[k]]
        ax.add_patch(
            Polygon(
                tri,
                closed=True,
                facecolor=FACE_COLOR,
                alpha=FACE_ALPHA,
                edgecolor='none',
                zorder=1
            )
        )

    # Edges
    for i, j in edges:
        ax.plot(
            [points[i][0], points[j][0]],
            [points[i][1], points[j][1]],
            color=EDGE_COLOR,
            linewidth=LINE_WIDTH,
            zorder=2
        )

    # Vertices
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=VERTEX_SIZE, color=VERTEX_COLOR, zorder=3)

    # Labels
    offsets = [
        (-0.18, -0.18),  # v0
        (0.08, -0.18),   # v1
        (-0.16, 0.12),   # v2
        (0.08, 0.12),    # v3
        (0.10, -0.18),   # v4
    ]
    for idx, (x, y) in enumerate(points):
        dx, dy = offsets[idx]
        ax.text(x + dx, y + dy, rf'$v_{idx}$', fontsize=11)

# -----------------------------
# More spread-out points
# -----------------------------
points = [
    (0.0, 0.0),    # v0
    (1.45, 0.10),  # v1
    (0.70, 1.25),  # v2
    (2.65, 1.25),  # v3
    (3.65, 0.10),  # v4
]

# Chosen so the growth is gradual
eps_values = [0.90, 1.40, 1.60, 2.30]

# -----------------------------
# Figure: 2x2 layout
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, eps in zip(axes, eps_values):
    build_vr(ax, points, eps)
    clean_axis(ax)
    ax.set_xlim(-0.5, 4.2)
    ax.set_ylim(-0.5, 1.8)

# Labels under each subplot
for ax, eps in zip(axes, eps_values):
    bbox = ax.get_position()
    x_center = (bbox.x0 + bbox.x1) / 2
    y_bottom = bbox.y0
    plt.figtext(
        x_center,
        y_bottom - 0.035,
        rf'$\epsilon = {eps}$',
        ha='center',
        va='top',
        fontsize=12
    )

plt.subplots_adjust(wspace=0.22, hspace=0.35, bottom=0.08)

plt.savefig('vr_filtration_2x2.png', dpi=300, bbox_inches='tight')
plt.savefig('vr_filtration_2x2.pdf', bbox_inches='tight')
plt.show()