import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# -----------------------------
# Style
# -----------------------------
VERTEX_COLOR = '#ff7f0e'
EDGE_COLOR = '#1f77b4'
FACE_COLOR = '#1f77b4'
FACE_ALPHA = 0.30
VERTEX_SIZE = 90
LINE_WIDTH = 2.0

def clean_axis(ax):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# -----------------------------
# Vertices
# -----------------------------
v0 = (0.0, 0.0)
v1 = (1.6, 0.0)
v2 = (0.8, 1.3)
v3 = (2.4, 1.3)
v4 = (3.5, 0.0)
v5 = (5.0, 0.0)
v6 = (6.0, 0.8)

vertices = {
    r'$v_0$': v0,
    r'$v_1$': v1,
    r'$v_2$': v2,
    r'$v_3$': v3,
    r'$v_4$': v4,
    r'$v_5$': v5,
    r'$v_6$': v6,
}

# -----------------------------
# Simplices
# -----------------------------
# 2-simplices
triangles = [
    [v0, v1, v2],
    [v1, v3, v2],
]

# 1-simplices not already visually obvious from triangle boundaries
extra_edges = [
    (v3, v4),
    (v5, v6),
]

# Optional: also show all triangle edges explicitly for clarity
triangle_edges = [
    (v0, v1), (v1, v2), (v2, v0),
    (v1, v3), (v3, v2), (v2, v1),
]

# Remove duplicate edge visually
unique_triangle_edges = []
seen = set()
for a, b in triangle_edges:
    key = tuple(sorted((a, b)))
    if key not in seen:
        seen.add(key)
        unique_triangle_edges.append((a, b))

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))

# Filled 2-simplices
for tri in triangles:
    patch = Polygon(tri, closed=True, facecolor=FACE_COLOR, edgecolor='none', alpha=FACE_ALPHA)
    ax.add_patch(patch)

# Edges
for a, b in unique_triangle_edges + extra_edges:
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color=EDGE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=2
    )

# Vertices
xs = [p[0] for p in vertices.values()]
ys = [p[1] for p in vertices.values()]
ax.scatter(xs, ys, s=VERTEX_SIZE, color=VERTEX_COLOR, zorder=3)

# Labels
offsets = {
    r'$v_0$': (-0.18, -0.16),
    r'$v_1$': (-0.05, -0.18),
    r'$v_2$': (-0.05, 0.16),
    r'$v_3$': (0.05, 0.16),
    r'$v_4$': (0.05, -0.18),
    r'$v_5$': (-0.12, -0.18),
    r'$v_6$': (0.05, 0.12),
}

for label, (x, y) in vertices.items():
    dx, dy = offsets[label]
    ax.text(x + dx, y + dy, label, fontsize=11)

clean_axis(ax)
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.6, 1.9)

plt.subplots_adjust(bottom=0.20)
plt.savefig('simplicial_complex.png', dpi=300, bbox_inches='tight')
plt.savefig('simplicial_complex.pdf', bbox_inches='tight')
plt.show()