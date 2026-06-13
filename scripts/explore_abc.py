"""
Quick exploration of ABC dataset.
Goal: understand localkey distribution in one file.
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
ABC_HARMONIES = os.path.join(REPO_ROOT, "external/ABC/harmonies")

# 1. List harmonies files
files = sorted(os.listdir(ABC_HARMONIES))
tsv_files = [f for f in files if f.endswith('.harmonies.tsv')]
print(f"Total TSV files: {len(tsv_files)}")
print(f"First 5: {tsv_files[:5]}\n")

# 2. Load one file and look at localkey changes
target = tsv_files[0]
print(f"Loading: {target}")
df = pd.read_csv(os.path.join(ABC_HARMONIES, target), sep='\t')

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# 3. Localkey distribution
print("Unique localkey values in this file:")
print(df['localkey'].value_counts())

# 4. When does localkey change? (find segment boundaries)
print("\nLocalkey segments (when localkey changes):")
prev_lk = None
for idx, row in df.iterrows():
    lk = row['localkey']
    mc = row['mc']
    if lk != prev_lk:
        print(f"  Measure {mc}: localkey = {lk}")
        prev_lk = lk

# 5. Pedal occurrences
print(f"\nPedal column non-empty count: {df['pedal'].notna().sum()}")
if df['pedal'].notna().sum() > 0:
    print("First 5 pedal occurrences:")
    print(df[df['pedal'].notna()][['mc', 'mc_onset', 'pedal']].head())