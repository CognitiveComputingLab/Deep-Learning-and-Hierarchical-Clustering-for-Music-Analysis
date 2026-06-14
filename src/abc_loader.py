"""
Loader for the Annotated Beethoven Corpus (ABC).

ABC provides harmony annotations as TSV files, where each row is a chord event.
The dataset does NOT provide an explicit hierarchical analysis. Instead, we
derive hierarchy from two columns:

  - localkey: changes mark Level-1 segment boundaries
  - pedal: non-empty values mark Level-2 (nested) segments

This loader converts a single TSV into a FormNode tree compatible with
taking_form_loader.py.

Hierarchy schema:
  ROOT  (whole piece)
  └─ Localkey Section [I]  m.1-39      ← Level 1
       └─ Pedal Section [V]  m.30-33   ← Level 2 (optional)
"""
import os
import sys
import pandas as pd

# Allow importing FormNode from taking_form_loader
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from taking_form_loader import FormNode


def load_abc_tsv(tsv_path: str) -> FormNode:
    """
    Parse one ABC harmonies TSV file into a FormNode tree.
    
    Returns the root node (whole piece). The tree structure will be:
      ROOT
      ├── Localkey [I]  (m.a-b)
      │   ├── Pedal [V]  (m.x-y)  [optional]
      │   └── Pedal [...]         [optional]
      ├── Localkey [V]  (m.c-d)
      ...
    """
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Sort by mc (just to be safe; should already be sorted)
    df = df.sort_values(by=['mc', 'mc_onset']).reset_index(drop=True)
    
    # Get the global metadata for the root label
    globalkey = df['globalkey'].iloc[0] if 'globalkey' in df.columns else "?"
    first_measure = int(df['mc'].iloc[0])
    last_measure = int(df['mc'].iloc[-1])
    
    root = FormNode(
        label=f"Piece [{globalkey}]",
        level=0,
        start_measure=first_measure,
        end_measure=last_measure,
    )
    
    # --- Pass 1: extract Level-1 segments based on localkey changes ---
    localkey_segments = []  # list of (start_mc, end_mc, localkey_label)
    
    prev_lk = None
    segment_start = None
    for _, row in df.iterrows():
        lk = row['localkey']
        mc = int(row['mc'])
        
        if pd.isna(lk):
            continue  # skip rows with no localkey
        
        if lk != prev_lk:
            # Close previous segment
            if prev_lk is not None and segment_start is not None:
                localkey_segments.append((segment_start, mc - 1, prev_lk))
            # Start new segment
            segment_start = mc
            prev_lk = lk
    
    # Close the last segment
    if prev_lk is not None and segment_start is not None:
        localkey_segments.append((segment_start, last_measure, prev_lk))
    
    # --- Pass 2: extract pedal segments ---
    # A pedal segment is a contiguous range of rows where pedal is non-empty
    # and has the same value.
    pedal_segments = []  # list of (start_mc, end_mc, pedal_label)
    
    prev_ped = None
    ped_start = None
    last_mc_with_ped = None
    
    for _, row in df.iterrows():
        ped = row['pedal']
        mc = int(row['mc'])
        
        if pd.isna(ped):
            # End of any current pedal
            if prev_ped is not None:
                pedal_segments.append((ped_start, last_mc_with_ped, prev_ped))
                prev_ped = None
                ped_start = None
        else:
            if ped != prev_ped:
                # New pedal starts
                if prev_ped is not None:
                    pedal_segments.append((ped_start, last_mc_with_ped, prev_ped))
                prev_ped = ped
                ped_start = mc
            last_mc_with_ped = mc
    
    # Close any pending pedal
    if prev_ped is not None:
        pedal_segments.append((ped_start, last_mc_with_ped, prev_ped))
    
    # --- Pass 3: build the tree ---
    # For each Level-1 segment, create a node and attach pedal segments that
    # fall within its range.
    for (lk_start, lk_end, lk_label) in localkey_segments:
        lk_node = FormNode(
            label=f"LK[{lk_label}]",
            level=1,
            start_measure=lk_start,
            end_measure=lk_end,
        )
        
        # Attach pedal segments that fall within this localkey range
        for (ped_start, ped_end, ped_label) in pedal_segments:
            # Pedal is inside this LK section if it overlaps the range
            if ped_start >= lk_start and ped_end <= lk_end:
                ped_node = FormNode(
                    label=f"Ped[{ped_label}]",
                    level=2,
                    start_measure=ped_start,
                    end_measure=ped_end,
                )
                lk_node.add_child(ped_node)
        
        root.add_child(lk_node)
    
    return root


if __name__ == "__main__":
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    ABC_HARMONIES = os.path.join(REPO_ROOT, "external/ABC/harmonies")
    
    tsv_files = sorted(f for f in os.listdir(ABC_HARMONIES) if f.endswith('.harmonies.tsv'))
    print(f"Total ABC harmonies files: {len(tsv_files)}\n")
    
    # Load the first one
    target = tsv_files[0]
    print(f"Loading: {target}")
    tree = load_abc_tsv(os.path.join(ABC_HARMONIES, target))
    
    print(f"\nTotal nodes: {tree.size()}")
    print(f"\nTree structure:")
    print(tree.pretty_print())