import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib.pyplot as plt
import pitchscapes.reader as rd
import pitchscapes.plotting as pt
from greedy_clustering import load_pc_bins, greedy_cluster, _norm

MIDI = r"n11op95_01.mid"
TSV  = r"external\ABC\notes\n11op95_01.notes.tsv"

# ---------- Tonnetz 距离（五度圈几何嵌入） ----------
_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
_pc_to_fifth = {pc: i for i, pc in enumerate(_FIFTHS_ORDER)}
_ANGLES = np.array([2 * np.pi * _pc_to_fifth[pc] / 12 for pc in range(12)])

def _tonnetz_embed(v):
    v = _norm(v)
    return np.array([np.sum(v * np.cos(_ANGLES)),
                     np.sum(v * np.sin(_ANGLES))])

def d_tonnetz(a, b):
    return np.linalg.norm(_tonnetz_embed(a) - _tonnetz_embed(b))

# ---------- 载入数据 + 聚类 ----------
pc_mat, bounds = load_pc_bins(TSV, bin_size_qb=8.0)
print(f"叶子数: {len(pc_mat)}")

scape = rd.get_pitch_scape(MIDI)
fig, ax = plt.subplots(figsize=(14, 8))
pt.key_scape_plot(scape=scape, n_samples=200, ax=ax)

root = greedy_cluster(pc_mat, bounds, d_tonnetz)
total = root.end

# ---------- 收集顶部节点 ----------
def collect(n, pts, edges, parent=None, depth=0, max_depth=8):
    if depth > max_depth:
        return
    center = (n.start + n.end) / 2
    width  = n.end - n.start
    me = (center, width, depth)
    pts.append(me)
    if parent is not None:
        edges.append((parent, me))
    for c in n.children:
        collect(c, pts, edges, me, depth+1, max_depth)

pts, edges = [], []
collect(root, pts, edges)
print(f"画 {len(pts)} 节点, {len(edges)} 边")

# ---------- 换算到 pitch scape 坐标 (0,1) x (0,1) ----------
def to_ax(center, width, depth):
    return center / total, width / total

# ---------- 画边 + 画点 ----------
for p, c in edges:
    x1, y1 = to_ax(*p)
    x2, y2 = to_ax(*c)
    ax.plot([x1, x2], [y1, y2], 'w-', linewidth=1.2, alpha=0.8, zorder=5)

for center, width, depth in pts:
    x, y = to_ax(center, width, depth)
    ax.plot(x, y, 'o', color='black', markersize=5,
            markeredgecolor='white', markeredgewidth=0.8, zorder=10)

plt.title("Op. 95 mvt.1 — greedy tree top 8 levels (tonnetz distance)")
plt.savefig("op95_greedytree_tonnetz.png", dpi=140, bbox_inches='tight')
plt.show()