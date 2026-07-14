import pandas as pd
from fractions import Fraction

def to_float(x):
    if isinstance(x, str) and '/' in x:
        return float(Fraction(x))
    return float(x)

h = pd.read_csv(r"external\ABC\harmonies\n11op95_01.harmonies.tsv", sep='\t').dropna(subset=['quarterbeats'])
h['qb'] = h['quarterbeats'].apply(to_float)

# 打印每个 localkey 段落的开始位置对应的和声
print("localkey 变化点附近的和声上下文:")
prev_key = None
for _, row in h.iterrows():
    if row['localkey'] != prev_key:
        print(f"  qb={row['qb']:6.1f} mn={row['mn']:3d}  localkey={row['localkey']:5s} chord={row['chord']}")
        prev_key = row['localkey']