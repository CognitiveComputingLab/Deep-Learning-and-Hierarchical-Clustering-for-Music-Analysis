import sys
sys.path.insert(0, 'src')
import numpy as np
from greedy_clustering import load_pc_bins, greedy_cluster, DISTANCES
d_tonnetz = DISTANCES['circle_of_fifths']
d_weighted = DISTANCES['fixed_c_major_weighted_ablation']

TSV = r"external\ABC\notes\n11op95_01.notes.tsv"

# ---------- Tonnetz 距离（五度圈几何嵌入） ----------
# ---------- Weighted chromagram distance（音级稳定性加权欧氏） ----------
# ---------- 距离函数字典（本地覆盖，含 4 个先验梯度） ----------
DISTS = {
    'euclidean':  DISTANCES['euclidean'],   # 无先验
    'weighted':   d_weighted,               # 弱先验
    'tonnetz':    d_tonnetz,                # 中先验
    'keyprofile': DISTANCES['keyprofile'],  # 强先验
}

pc_mat, bounds = load_pc_bins(TSV, bin_size_qb=8.0)
print(f"叶子数: {len(pc_mat)}, 总时长: {bounds[-1][1]} qb")
print()

def walk(n, depth=0, max_d=3):
    if depth > max_d:
        return
    indent = "  " * depth
    print(f"{indent}[{n.start:6.1f}, {n.end:6.1f}] "
          f"width={n.end-n.start:5.1f} qb")
    for c in n.children:
        walk(c, depth+1, max_d)

for name, fn in DISTS.items():
    root = greedy_cluster(pc_mat, bounds, fn)
    print(f"=== {name} 的顶层分段 ===")
    walk(root)
    print()
