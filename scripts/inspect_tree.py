import sys
sys.path.insert(0, 'src')
import numpy as np
from greedy_clustering import load_pc_bins, greedy_cluster, DISTANCES, _norm

TSV = r"external\ABC\notes\n11op95_01.notes.tsv"

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

# ---------- Weighted chromagram distance（音级稳定性加权欧氏） ----------
_STAB_WEIGHTS = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_STAB_WEIGHTS = _STAB_WEIGHTS / _STAB_WEIGHTS.sum()

def d_weighted(a, b):
    a, b = _norm(a), _norm(b)
    return np.sqrt(np.sum(_STAB_WEIGHTS * (a - b) ** 2))

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