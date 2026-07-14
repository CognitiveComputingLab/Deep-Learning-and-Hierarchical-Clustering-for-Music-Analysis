import sys
sys.path.insert(0, 'src')

from greedy_clustering import (
    load_pc_bins, greedy_cluster, DISTANCES
)

TSV = r"external\ABC\notes\n11op95_01.notes.tsv"

# 4 拍一个 bin，Op. 95 mvt.1 是 4/4，一个 bin ≈ 一小节
pc_mat, bounds = load_pc_bins(TSV, bin_size_qb=4.0)
print(f"共 {len(pc_mat)} 个 bin，总时长 {bounds[-1][1]} qb")
print(f"第一个 bin PC 分布: {pc_mat[0]}")
print(f"最强的 pitch class: {pc_mat[0].argmax()}（0=C, 1=C#, ...）")
print()

for name, fn in DISTANCES.items():
    root = greedy_cluster(pc_mat, bounds, fn)
    # 数树的规模
    def count(n):
        return 1 + sum(count(c) for c in n.children)
    def depth(n):
        return 1 + max((depth(c) for c in n.children), default=0)
    print(f"{name:12s} → 节点数 {count(root):4d}, 深度 {depth(root):3d}, "
          f"根覆盖 [{root.start}, {root.end}]")