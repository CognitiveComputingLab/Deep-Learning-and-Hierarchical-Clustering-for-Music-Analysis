"""Evaluate greedy trees against DCML localkey ground truth on Op. 95 mvt.1."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from fractions import Fraction
import zss

from greedy_clustering import (
    load_pc_bins, greedy_cluster, DISTANCES, ClusterNode, _norm
)

TSV_NOTES = r"external\ABC\notes\n11op95_01.notes.tsv"
TSV_HARM  = r"external\ABC\harmonies\n11op95_01.harmonies.tsv"
TOL_QB = 24.0
MAX_DEPTH = 4

# ---------- 补充两个距离函数 ----------
_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
_pc_to_fifth = {pc: i for i, pc in enumerate(_FIFTHS_ORDER)}
_ANGLES = np.array([2 * np.pi * _pc_to_fifth[pc] / 12 for pc in range(12)])

def _tonnetz_embed(v):
    v = _norm(v)
    return np.array([np.sum(v * np.cos(_ANGLES)),
                     np.sum(v * np.sin(_ANGLES))])

def d_tonnetz(a, b):
    return np.linalg.norm(_tonnetz_embed(a) - _tonnetz_embed(b))

_STAB_WEIGHTS = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_STAB_WEIGHTS = _STAB_WEIGHTS / _STAB_WEIGHTS.sum()

def d_weighted(a, b):
    a, b = _norm(a), _norm(b)
    return np.sqrt(np.sum(_STAB_WEIGHTS * (a - b) ** 2))

DISTS = {
    'euclidean':  DISTANCES['euclidean'],
    'weighted':   d_weighted,
    'tonnetz':    d_tonnetz,
    'keyprofile': DISTANCES['keyprofile'],
}

# ---------- Ground truth from DCML ----------

def to_float(x):
    if isinstance(x, str) and '/' in x:
        return float(Fraction(x))
    return float(x)

def dcml_localkey_segments(harm_tsv):
    h = pd.read_csv(harm_tsv, sep='\t').dropna(subset=['quarterbeats'])
    h['qb']  = h['quarterbeats'].apply(to_float)
    h['dur'] = h['duration_qb'].astype(float)
    total = (h['qb'] + h['dur']).max()

    segs = []
    cur_key = h.iloc[0]['localkey']
    cur_start = 0.0
    for _, row in h.iterrows():
        if row['localkey'] != cur_key:
            segs.append((cur_start, row['qb'], cur_key))
            cur_key = row['localkey']
            cur_start = row['qb']
    segs.append((cur_start, total, cur_key))
    return segs, total

def flat_tree_from_segments(segs):
    children = [ClusterNode(s, e, None) for s, e, _ in segs]
    root = ClusterNode(0.0, segs[-1][1], None, children=children)
    return root

# ---------- 时间桶标签 ----------

def label_tree_by_time(node, total, n_center_bins=10, n_width_bins=5):
    center = (node.start + node.end) / 2.0
    width  = node.end - node.start
    c_bin = min(n_center_bins - 1, int(n_center_bins * center / total))
    w_bin = min(n_width_bins  - 1, int(n_width_bins  * width  / total))
    node.label = f"c{c_bin}w{w_bin}"
    for c in node.children:
        label_tree_by_time(c, total, n_center_bins, n_width_bins)
    return node

# ---------- zss 适配 ----------

def _label(n): return getattr(n, 'label', 'X')
def _children(n): return n.children
def _label_dist(a, b): return 0 if a == b else 1

def ted(t1, t2):
    return zss.simple_distance(t1, t2, _children, _label, _label_dist)

# ---------- 剪枝 ----------

def prune(node, max_depth, depth=0):
    if depth >= max_depth or not node.children:
        return ClusterNode(node.start, node.end, None)
    new_children = [prune(c, max_depth, depth+1) for c in node.children]
    return ClusterNode(node.start, node.end, None, children=new_children)

def count_leaves(n):
    if not n.children:
        return 1
    return sum(count_leaves(c) for c in n.children)

def find_prune_depth_matching(root, target_leaves):
    best_depth, best_diff = 1, 1e9
    for depth in range(1, 20):
        p = prune(root, depth)
        n = count_leaves(p)
        diff = abs(n - target_leaves)
        if diff < best_diff:
            best_diff = diff
            best_depth = depth
        if n >= target_leaves:
            break
    return best_depth

# ---------- Boundary F1 ----------

def collect_top_splits(node, max_depth):
    """Return all internal-node split points within max_depth (as sorted list)."""
    splits = set()
    def walk(n, d):
        if d >= max_depth or not n.children:
            return
        for c in n.children[:-1]:
            splits.add(c.end)
        for c in n.children:
            walk(c, d+1)
    walk(node, 0)
    return sorted(splits)

def boundary_f1(pred, gt, tol_qb=24.0):
    matched_gt = set()
    tp = 0
    for p in pred:
        best_i, best_d = None, tol_qb
        for i, g in enumerate(gt):
            if i in matched_gt:
                continue
            d = abs(p - g)
            if d <= best_d:
                best_i, best_d = i, d
        if best_i is not None:
            matched_gt.add(best_i)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec  = tp / len(gt) if gt else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1

# ---------- Main ----------

if __name__ == '__main__':
    segs, total = dcml_localkey_segments(TSV_HARM)
    print(f"Total: {total} qb")
    print(f"DCML localkey segments: {len(segs)}")
    for s, e, k in segs:
        print(f"  [{s:6.1f}, {e:6.1f}]  localkey={k}")
    print()

    gt_tree = label_tree_by_time(flat_tree_from_segments(segs), total)
    gt_boundaries = [e for s, e, _ in segs[:-1]]
    K = len(segs)
    print(f"Target leaves (K): {K}")
    print(f"GT boundaries: {gt_boundaries}")
    print(f"Boundary tolerance: {TOL_QB} qb ≈ {TOL_QB/4:.1f} measures")
    print(f"Splits collected up to depth: {MAX_DEPTH}")
    print()

    pc_mat, bounds = load_pc_bins(TSV_NOTES, bin_size_qb=8.0)
    print(f"Greedy leaves (bin_size=8 qb): {len(pc_mat)}")
    print()

    header = f"{'distance':12s} {'raw TED':>10s} {'pruned TED':>12s} {'prec':>6s} {'rec':>6s} {'F1':>6s}   splits"
    print(header)
    print("-" * len(header))

    for name, fn in DISTS.items():
        root = greedy_cluster(pc_mat, bounds, fn)

        raw_lab = label_tree_by_time(root, total)
        raw_ted = ted(raw_lab, gt_tree)

        depth = find_prune_depth_matching(root, K)
        pruned_root = prune(root, depth)
        pruned_lab = label_tree_by_time(pruned_root, total)
        pruned_ted = ted(pruned_lab, gt_tree)

        # 取深度 MAX_DEPTH 以内所有内部切分点，不截断
        pred_bounds = collect_top_splits(root, max_depth=MAX_DEPTH)
        prec, rec, f1 = boundary_f1(pred_bounds, gt_boundaries, tol_qb=TOL_QB)

        # splits 太多就只显示接近 GT 的
        splits_str = str(pred_bounds)

        print(f"{name:12s} {raw_ted:>10.1f} {pruned_ted:>12.1f} "
              f"{prec:>6.2f} {rec:>6.2f} {f1:>6.2f}   {splits_str}")