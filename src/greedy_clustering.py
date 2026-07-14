"""Greedy bottom-up hierarchical clustering of music segments.

Leaves = fixed contiguous windows (in quarterbeats).
Only adjacent clusters may merge (temporal contiguity).
Cluster feature = pitch-class duration vector (additive on merge).
Pluggable distance functions.
"""
import numpy as np
import pandas as pd
from fractions import Fraction

# feature extractions

def to_float_qb(x):
    """DCML quarterbeats 可能是 '5/2' 这种分数字符串"""
    if isinstance(x, str) and '/' in x:
        return float(Fraction(x))
    return float(x)

def load_pc_bins(notes_tsv, bin_size_qb=4.0):
    """把一首曲子切成等宽 bin，返回 (n_bins, 12) 的 PC 时值矩阵和 bin 边界"""
    notes = pd.read_csv(notes_tsv, sep='\t')
    notes = notes.dropna(subset=['quarterbeats'])
    notes['qb'] = notes['quarterbeats'].apply(to_float_qb)
    notes['dur'] = notes['duration_qb'].astype(float)
    notes['pc'] = notes['midi'].astype(int) % 12

    total = (notes['qb'] + notes['dur']).max()
    n_bins = int(np.ceil(total / bin_size_qb))
    pc_mat = np.zeros((n_bins, 12))

    for _, nt in notes.iterrows():
        start, end, pc = nt['qb'], nt['qb'] + nt['dur'], int(nt['pc'])
        # 音符可能跨 bin，按重叠时长分配
        b0, b1 = int(start // bin_size_qb), int(min(end, total - 1e-9) // bin_size_qb)
        for b in range(b0, b1 + 1):
            lo, hi = b * bin_size_qb, (b + 1) * bin_size_qb
            overlap = max(0.0, min(end, hi) - max(start, lo))
            pc_mat[b, pc] += overlap

    boundaries = [(i * bin_size_qb, (i + 1) * bin_size_qb) for i in range(n_bins)]
    return pc_mat, boundaries

# distance functions

def _norm(v):
    s = v.sum()
    return v / s if s > 0 else np.full(12, 1/12)

def d_euclidean(a, b):
    return np.linalg.norm(_norm(a) - _norm(b))

def d_cosine(a, b):
    a, b = _norm(a), _norm(b)
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def d_js(a, b):
    a, b = _norm(a) + 1e-12, _norm(b) + 1e-12
    m = (a + b) / 2
    kl = lambda p, q: (p * np.log2(p / q)).sum()
    return np.sqrt((kl(a, m) + kl(b, m)) / 2)

# tonnetz
_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
_pc_to_fifth = {pc: i for i, pc in enumerate(_FIFTHS_ORDER)}
_ANGLES = np.array([2 * np.pi * _pc_to_fifth[pc] / 12 for pc in range(12)])

def _tonnetz_embed(v):
    v = _norm(v)
    return np.array([np.sum(v * np.cos(_ANGLES)), np.sum(v * np.sin(_ANGLES))])

def d_tonnetz(a, b):
    return np.linalg.norm(_tonnetz_embed(a) - _tonnetz_embed(b))

# weighted
_STAB_WEIGHTS = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_STAB_WEIGHTS = _STAB_WEIGHTS / _STAB_WEIGHTS.sum()

def d_weighted(a, b):
    a, b = _norm(a), _norm(b)
    return np.sqrt(np.sum(_STAB_WEIGHTS * (a - b) ** 2))

# Krumhansl-Schmuckler key profiles
_KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def _key_activation(v):
    """和 24 个调性模板的相关系数 → 24 维向量"""
    v = _norm(v)
    acts = []
    for shift in range(12):
        for prof in (_KS_MAJ, _KS_MIN):
            p = np.roll(prof, shift)
            acts.append(np.corrcoef(v, p)[0, 1])
    return np.array(acts)

def d_keyprofile(a, b):
    return np.linalg.norm(_key_activation(a) - _key_activation(b))

DISTANCES = {
    'euclidean':  d_euclidean,   # 无先验
    'weighted':   d_weighted,    # 弱先验
    'tonnetz':    d_tonnetz,     # 中先验
    'keyprofile': d_keyprofile,  # 强先验
    'cosine':     d_cosine,
    'js':         d_js,
}

# ---------- 贪心聚类 ----------

class ClusterNode:
    def __init__(self, start, end, feature, children=None):
        self.start, self.end = start, end     # quarterbeats
        self.feature = feature                # 12 维 PC 向量
        self.children = children or []

def greedy_cluster(pc_mat, boundaries, dist_fn):
    """相邻合并的贪心聚类，返回根节点"""
    clusters = [ClusterNode(s, e, pc_mat[i].copy())
                for i, (s, e) in enumerate(boundaries)]
    while len(clusters) > 1:
        dists = [dist_fn(clusters[i].feature, clusters[i+1].feature)
                 for i in range(len(clusters) - 1)]
        i = int(np.argmin(dists))
        a, b = clusters[i], clusters[i+1]
        merged = ClusterNode(a.start, b.end, a.feature + b.feature,
                             children=[a, b])
        clusters[i:i+2] = [merged]
    return clusters[0]