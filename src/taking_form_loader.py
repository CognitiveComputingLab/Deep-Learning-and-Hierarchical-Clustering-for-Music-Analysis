"""
Loader for Taking Form CSV files.

Taking Form's tabular format:
- Column 0: measure number (or 'Repeat:' / range like '9-16=1-8')
- Column 1: beat number
- Columns 2+: formal labels from coarsest (left) to finest (right)

We convert this into a labelled tree where each node has:
- label (str): e.g., 'Exposition', 'Theme a'
- level (int): depth from root (0 = root = full piece)
- start_measure (int)
- end_measure (int)
- children (list)
"""
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FormNode:
    """A node in a hierarchical form analysis tree."""
    label: str
    level: int
    start_measure: int
    end_measure: Optional[int] = None  # filled in after we know the next sibling/parent's start
    children: List['FormNode'] = field(default_factory=list)
    
    def add_child(self, child: 'FormNode'):
        self.children.append(child)
    
    def pretty_print(self, indent: int = 0) -> str:
        span = f"m.{self.start_measure}"
        if self.end_measure is not None:
            span += f"-{self.end_measure}"
        s = "  " * indent + f"[{self.label}] ({span})\n"
        for child in self.children:
            s += child.pretty_print(indent + 1)
        return s
    
    def size(self) -> int:
        """Count total nodes in subtree."""
        return 1 + sum(c.size() for c in self.children)


def load_taking_form_csv(csv_path: str) -> FormNode:
    """
    Parse a Taking Form CSV file into a FormNode tree.
    
    Returns the root node (representing the whole piece).
    """
    # Read without header; some files have headers, but Taking Form's convention
    # is that string-valued first row is treated as a header.
    df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    
    # Detect if first row is a header (non-numeric measure column)
    try:
        int(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
    
    # Build tree
    # We process row by row. Each non-empty cell at column i corresponds to
    # a new node at depth (i - 2) starting at that measure.
    
    # Skip the columns 0 (measure) and 1 (beat); the rest are form columns.
    n_levels = df.shape[1] - 2
    
    # Root represents the whole piece.
    root = FormNode(label="ROOT", level=0, start_measure=1)
    
    # Track the currently-open node at each depth.
    # open_nodes[d] = the latest node opened at depth d
    open_nodes: List[Optional[FormNode]] = [root] + [None] * n_levels
    
    for _, row in df.iterrows():
        # Try to parse measure number; skip if not numeric (e.g., 'Repeat:' rows for now)
        try:
            measure = int(str(row[0]).strip())
        except ValueError:
            # Skip Repeat: and similar special rows for the MVP loader
            continue
        
        # For each form column, check if there's a label
        for col_idx in range(2, df.shape[1]):
            cell = str(row[col_idx]).strip()
            if not cell or cell.lower() == 'nan':
                continue
            
            depth = col_idx - 1  # depth=1 for the leftmost form column
            
            # Close previously-open node at this depth (and deeper) by setting end_measure
            for d in range(depth, len(open_nodes)):
                if open_nodes[d] is not None and open_nodes[d].end_measure is None:
                    open_nodes[d].end_measure = measure - 1
            
            # Reset deeper levels (a new label at depth d invalidates depth d+1, d+2, ...)
            for d in range(depth + 1, len(open_nodes)):
                open_nodes[d] = None
            
            # Create new node
            parent = open_nodes[depth - 1] if depth > 0 else root
            if parent is None:
                # Defensive: if a deeper-than-expected label appears, attach to root
                parent = root
            new_node = FormNode(label=cell, level=depth, start_measure=measure)
            parent.add_child(new_node)
            open_nodes[depth] = new_node
    
    return root


if __name__ == "__main__":
    # Find one CSV file from the Beethoven corpus
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    BEETHOVEN_DIR = os.path.join(REPO_ROOT, "external/Taking-Form/corpus/Beethoven_Sonatas")
    
    csv_files = sorted(f for f in os.listdir(BEETHOVEN_DIR) if f.endswith('.csv'))
    print(f"Found {len(csv_files)} CSV files in Beethoven corpus.")
    
    # Load the first one
    target = csv_files[0]
    print(f"\nLoading: {target}")
    tree = load_taking_form_csv(os.path.join(BEETHOVEN_DIR, target))
    
    print(f"\nTotal nodes: {tree.size()}")
    print(f"\nTree structure:")
    print(tree.pretty_print())