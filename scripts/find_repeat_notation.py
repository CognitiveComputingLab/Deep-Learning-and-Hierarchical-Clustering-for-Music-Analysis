"""
Find all rows across all Taking Form CSVs that use 'M-N=A-B' or similar
repeat-notation in the measure column. This tells us how widespread the
pattern is and what variants exist.
"""
import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
BEETHOVEN_DIR = os.path.join(REPO_ROOT, "external/Taking-Form/corpus/Beethoven_Sonatas")

csv_files = sorted(f for f in os.listdir(BEETHOVEN_DIR) if f.endswith('.csv'))
print(f"Scanning {len(csv_files)} files...\n")

# Pattern: anything that has '=' in the measure column
total_repeat_rows = 0
files_with_repeats = 0

for csv_name in csv_files:
    csv_path = os.path.join(BEETHOVEN_DIR, csv_name)
    df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    
    repeat_rows = []
    for idx, row in df.iterrows():
        m_str = str(row[0]).strip()
        if '=' in m_str:
            # Find what label this row has
            labels = []
            for col_idx in range(2, df.shape[1]):
                cell = str(row[col_idx]).strip()
                if cell and cell.lower() != 'nan':
                    labels.append(f"L{col_idx-1}:'{cell}'")
            repeat_rows.append((idx, m_str, ", ".join(labels) if labels else "(no labels)"))
    
    if repeat_rows:
        files_with_repeats += 1
        total_repeat_rows += len(repeat_rows)
        print(f"=== {csv_name} ({len(repeat_rows)} repeat rows) ===")
        for idx, m_str, labels in repeat_rows:
            print(f"  Row {idx}: measure='{m_str}' | {labels}")
        print()

print(f"\nSummary: {total_repeat_rows} repeat-notation rows across {files_with_repeats}/{len(csv_files)} files")