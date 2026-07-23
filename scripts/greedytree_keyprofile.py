import sys
sys.path.insert(0, 'src')

import matplotlib.pyplot as plt
import pitchscapes.reader as rd
import pitchscapes.plotting as pt
from greedy_clustering import load_pc_bins, greedy_cluster, d_key_profile

MIDI = r"n11op95_01.mid"
TSV = r"external\ABC\notes\n11op95_01.notes.tsv"

# 用较大的 bin，减少叶子数
pc_mat, bounds = load_pc_bins(TSV, bin_size_qb=8.0)
print(f"叶子数: {len(pc_mat)}")

# 画 pitch scape 底图
scape = rd.get_pitch_scape(MIDI)
fig, ax = plt.subplots(figsize=(14, 8))
pt.key_scape_plot(scape=scape, n_samples=200, ax=ax)

root = greedy_cluster(pc_mat, bounds, d_key_profile)
total = root.end

# 只收集顶部 6 层节点
def collect(n, pts, edges, parent=None, depth=0, max_depth=8):
    if depth > max_depth:
        return
    center = (n.start + n.end) / 2
    width = n.end - n.start
    me = (center, width, depth)
    pts.append(me)
    if parent is not None:
        edges.append((parent, me))
    for c in n.children:
        collect(c, pts, edges, me, depth+1, max_depth)

pts, edges = [], []
collect(root, pts, edges)
print(f"画 {len(pts)} 节点, {len(edges)} 边")

# 换算：pitch scape 是 (0,1) x (0,1)
def to_ax(center, width, depth):
    return center / total, width / total

# 画边
for p, c in edges:
    x1, y1 = to_ax(*p)
    x2, y2 = to_ax(*c)
    ax.plot([x1, x2], [y1, y2], 'w-', linewidth=1.2, alpha=0.8, zorder=5)

# 画点：统一大小的黑点
for center, width, depth in pts:
    x, y = to_ax(center, width, depth)
    ax.plot(x, y, 'o', color='black', markersize=5,
            markeredgecolor='white', markeredgewidth=0.8, zorder=10)


plt.title("Op. 95 mvt.1 — greedy tree top 8 levels (keyprofile)")
plt.savefig("op95_greedytree_v2.png", dpi=140, bbox_inches='tight')
plt.show()
