import sys, pandas as pd
import pitchscapes.reader as rd
import pitchscapes.plotting as pt
import matplotlib.pyplot as plt

MIDI = r"n11op95_01.mid"  # 转出来的位置
HARM = r"external\ABC\harmonies\n11op95_01.harmonies.tsv"

# 1. Pitch scape
scape = rd.get_pitch_scape(MIDI)
fig, ax = plt.subplots(figsize=(14, 7))
pt.key_scape_plot(scape=scape, n_samples=100, ax=ax)

# 2. 读 harmony 标注，提取 localkey 变化和 phraseend
harm = pd.read_csv(HARM, sep='\t')

# quarterbeats 有分数字符串（比如 "5/2"），转 float
def to_float(x):
    if isinstance(x, str) and '/' in x:
        a, b = x.split('/')
        return float(a) / float(b)
    return float(x)

harm['qb'] = harm['quarterbeats'].apply(to_float)
total_qb = harm['qb'].max() + harm['duration_qb'].iloc[-1]

# localkey 变化
harm['localkey_prev'] = harm['localkey'].shift(1)
key_changes = harm[harm['localkey'] != harm['localkey_prev']]

# phraseend
phrase_ends = harm[harm['phraseend'].notna()]

print(f"总时长 (quarterbeats): {total_qb}")
print(f"localkey 变化位置: {list(key_changes['qb'])}")
print(f"phraseend 位置: {list(phrase_ends['qb'])}")

# 3. 叠加：先看 pitch scape 的 x 轴范围
xlim = ax.get_xlim()
print(f"pitch scape x 轴范围: {xlim}")

# localkey 变化画红色虚线，phraseend 画蓝色点线
for _, row in key_changes.iterrows():
    x_norm = row['qb'] / total_qb
    x = xlim[0] + x_norm * (xlim[1] - xlim[0])
    ax.axvline(x=x, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax.text(x, ax.get_ylim()[1] * 0.95, f"→{row['localkey']}", 
            fontsize=9, color='red', ha='left')

for _, row in phrase_ends.iterrows():
    x_norm = row['qb'] / total_qb
    x = xlim[0] + x_norm * (xlim[1] - xlim[0])
    ax.axvline(x=x, color='blue', linewidth=1, linestyle=':', alpha=0.5)

plt.title("Op. 95 mvt.1 — pitch scape + ABC annotations")
plt.savefig("op95_scape_annotated.png", dpi=120, bbox_inches='tight')
plt.show()