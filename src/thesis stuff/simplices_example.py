import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------
# Style
# -----------------------------
VERTEX_COLOR = '#ff7f0e'  # pomarańczowy (matplotlib default orange)
EDGE_COLOR = '#1f77b4'
FACE_ALPHA = 0.30
VERTEX_SIZE_2D = 80
VERTEX_SIZE_3D = 45
LINE_WIDTH = 2.0

def clean_2d_axis(ax):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def clean_3d_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()

# -----------------------------
# Figure
# -----------------------------
fig = plt.figure(figsize=(14, 4.6))

# -----------------------------
# 0-simplex
# -----------------------------
ax1 = fig.add_subplot(1, 4, 1)
ax1.scatter([0], [0], s=120, color=VERTEX_COLOR, zorder=3)
ax1.text(0.15, 0.15, r'$v_0$', fontsize=11)
clean_2d_axis(ax1)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)

# -----------------------------
# 1-simplex
# -----------------------------
ax2 = fig.add_subplot(1, 4, 2)
p0 = (0, 0)
p1 = (1.2, 0)

ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], color=EDGE_COLOR, linewidth=LINE_WIDTH)
ax2.scatter([p0[0], p1[0]], [p0[1], p1[1]], s=VERTEX_SIZE_2D, color=VERTEX_COLOR)

ax2.text(p0[0] - 0.15, p0[1] + 0.15, r'$v_0$', fontsize=11)
ax2.text(p1[0] + 0.08, p1[1] + 0.15, r'$v_1$', fontsize=11)

clean_2d_axis(ax2)
ax2.set_xlim(-0.4, 1.6)
ax2.set_ylim(-0.6, 0.6)

# -----------------------------
# 2-simplex
# -----------------------------
ax3 = fig.add_subplot(1, 4, 3)
A = (0, 0)
B = (1.4, 0)
C = (0.7, 1.2)

ax3.fill([A[0], B[0], C[0]], [A[1], B[1], C[1]], color=EDGE_COLOR, alpha=FACE_ALPHA)
ax3.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=EDGE_COLOR, linewidth=LINE_WIDTH)
ax3.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], s=VERTEX_SIZE_2D, color=VERTEX_COLOR)

ax3.text(A[0] - 0.18, A[1] - 0.18, r'$v_0$', fontsize=11)
ax3.text(B[0] + 0.08, B[1] - 0.18, r'$v_1$', fontsize=11)
ax3.text(C[0], C[1] + 0.18, r'$v_2$', fontsize=11)

clean_2d_axis(ax3)
ax3.set_xlim(-0.3, 1.7)
ax3.set_ylim(-0.3, 1.5)

# -----------------------------
# 3-simplex
# -----------------------------
ax4 = fig.add_subplot(1, 4, 4, projection='3d')

V0 = (0.0, 0.0, 0.0)
V1 = (1.4, 0.0, 0.0)
V2 = (0.7, 1.2, 0.0)
V3 = (0.7, 0.45, 1.35)

verts = [V0, V1, V2, V3]
faces = [
    [V0, V1, V2],
    [V0, V1, V3],
    [V0, V2, V3],
    [V1, V2, V3]
]

poly = Poly3DCollection(
    faces,
    facecolors=EDGE_COLOR,
    alpha=FACE_ALPHA,
    edgecolor='black',
    linewidths=1.2
)
ax4.add_collection3d(poly)

xs = [v[0] for v in verts]
ys = [v[1] for v in verts]
zs = [v[2] for v in verts]
ax4.scatter(xs, ys, zs, s=VERTEX_SIZE_3D, color=VERTEX_COLOR, depthshade=False)

ax4.text(V0[0] - 0.25, V0[1], V0[2] - 0.05, r'$v_0$', fontsize=10)
ax4.text(V1[0] + 0.25, V1[1] - 0.12, V1[2] - 0.12, r'$v_1$', fontsize=10)
ax4.text(V2[0], V2[1] + 0.12, V2[2] - 0.12, r'$v_2$', fontsize=10)
ax4.text(V3[0], V3[1], V3[2] + 0.15, r'$v_3$', fontsize=10)

ax4.set_xlim(-0.2, 1.6)
ax4.set_ylim(-0.2, 1.4)
ax4.set_zlim(-0.2, 1.5)
ax4.view_init(elev=22, azim=35)
clean_3d_axis(ax4)

# -----------------------------
# Layout
# -----------------------------
plt.subplots_adjust(wspace=0.35, bottom=0.22)

# -----------------------------
# Bottom labels aligned
# -----------------------------
labels = [r'$0$-simplex', r'$1$-simplex', r'$2$-simplex', r'$3$-simplex']
axes = [ax1, ax2, ax3, ax4]

for ax, label in zip(axes, labels):
    bbox = ax.get_position()
    x_center = (bbox.x0 + bbox.x1) / 2
    plt.figtext(x_center, 0.3, label, ha='center', va='top', fontsize=12)

plt.savefig('simplices.png', dpi=300, bbox_inches='tight')
plt.savefig('simplices.pdf', bbox_inches='tight')
plt.show()