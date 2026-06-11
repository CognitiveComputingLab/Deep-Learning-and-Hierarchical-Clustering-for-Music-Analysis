"""
First exploration of Taking Form dataset.
Goal: confirm the submodule works and we can read the data.
"""
import os
import pandas as pd

TAKING_FORM = "external/Taking-Form"

# 1. List all files in Beethoven corpus
beethoven_dir = os.path.join(TAKING_FORM, "corpus/Beethoven_Sonatas")
files = sorted(os.listdir(beethoven_dir))
print(f"Total files in Beethoven_Sonatas: {len(files)}")
print(f"First 5: {files[:5]}\n")

# 2. Read one CSV file
csv_files = [f for f in files if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files\n")

if csv_files:
    first_csv = os.path.join(beethoven_dir, csv_files[0])
    print(f"Reading: {first_csv}\n")
    df = pd.read_csv(first_csv, header=None)
    print("First 20 rows:")
    print(df.head(20))
    print(f"\nShape: {df.shape}")

print("\n=== Exploration complete ===")