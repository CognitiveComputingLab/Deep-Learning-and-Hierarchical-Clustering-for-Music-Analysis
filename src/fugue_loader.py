"""
Loader for the Algomus fugue dataset.

Source: gitlab.com/algomus.fr/algomus-data, fugues/bach-wtc-i/*-ref.dez

Data format: Dezrann .dez files (JSON). Each file contains a flat list of
labels with types like:
  - S, CS1, CS2: subject / counter-subject occurrences (NOT used for tree
                 structure — they overlap and are not hierarchical)
  - Cadence: section-ending markers, with tags like "vi:PAC", "I:HC"
  - Pedal: pedal-tone passages, with tags like "I" (tonic pedal)
  - Structure: high-level structural section (rarely present in ref files)

Hierarchy derivation:
  ROOT (whole fugue)
  └─ Section [→key:cadence]  (one per Cadence, in time order)
       └─ Pedal [tag]         (if the pedal falls within this section)

Positions in .dez files are in QUARTER NOTES, not measure numbers.
We keep this unit; the resulting trees use quarter-beat positions in
their start/end fields. This means trees from this loader are NOT directly
measure-comparable to trees from taking_form_loader or abc_loader,
but the tree structure itself can still be compared via Zhang-Shasha
edit distance, which is label/topology-based.
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from taking_form_loader import FormNode


def load_fugue_dez(dez_path: str) -> FormNode:
    """
    Parse one Algomus fugue .dez file into a FormNode tree.
    
    Hierarchy:
      ROOT (whole fugue)
      ├── Section [→tag]  (one per Cadence; end_measure = cadence position)
      │   └── Pedal [tag]  (nested if pedal lies within section)
    
    Positions are in quarter notes, not measures.
    """
    with open(dez_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = data.get('labels', [])
    
    # Extract cadences (sorted by start position) — these mark section ends
    cadences = sorted(
        [l for l in labels if l.get('type') == 'Cadence'],
        key=lambda l: l.get('start', 0)
    )
    
    # Extract pedals
    pedals = sorted(
        [l for l in labels if l.get('type') == 'Pedal'],
        key=lambda l: l.get('start', 0)
    )
    
    # Determine the overall extent
    # Use the latest "end" across all labels (S/CS included, to capture the
    # full fugue extent, even if no cadence is at the very end)
    end_position = 0
    for l in labels:
        start = l.get('start', 0)
        dur = l.get('duration', 0) or l.get('actual-duration', 0)
        end_position = max(end_position, start + dur)
    
    # Build root
    fugue_name = os.path.basename(dez_path).replace('.dez', '')
    root = FormNode(
        label=f"Fugue [{fugue_name}]",
        level=0,
        start_measure=0,
        end_measure=int(end_position),
    )
    
    # Build sections from cadences
    # Section i goes from previous cadence (or 0) to cadence i
    section_boundaries = [0] + [int(c.get('start', 0)) for c in cadences]
    # If the last cadence is not at the end, add a final section
    if section_boundaries[-1] < end_position:
        section_boundaries.append(int(end_position))
    
    sections = []
    for i in range(len(section_boundaries) - 1):
        start = section_boundaries[i]
        end = section_boundaries[i + 1]
        # Find the cadence that defines this section's end (if any)
        if i < len(cadences):
            cad_tag = cadences[i].get('tag', '').strip()
            label = f"Section [→{cad_tag}]" if cad_tag else f"Section [{i+1}]"
        else:
            label = f"Section [end]"
        
        section_node = FormNode(
            label=label,
            level=1,
            start_measure=start,
            end_measure=end,
        )
        sections.append((section_node, start, end))
    
    # Attach pedals to their containing section
    for p in pedals:
        p_start = int(p.get('start', 0))
        p_dur = int(p.get('duration', 0) or p.get('actual-duration', 0))
        p_end = p_start + p_dur
        p_tag = p.get('tag', '').strip()
        
        for (section_node, s_start, s_end) in sections:
            # Pedal belongs to section if its start is within section bounds
            if s_start <= p_start < s_end:
                pedal_node = FormNode(
                    label=f"Pedal [{p_tag}]" if p_tag else "Pedal",
                    level=2,
                    start_measure=p_start,
                    end_measure=p_end,
                )
                section_node.add_child(pedal_node)
                break
    
    # Attach sections to root
    for (section_node, _, _) in sections:
        root.add_child(section_node)
    
    return root


if __name__ == "__main__":
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    FUGUE_DIR = os.path.join(REPO_ROOT, "external/algomus-data/fugues/bach-wtc-i")
    
    # Find all ref.dez files
    ref_files = sorted(f for f in os.listdir(FUGUE_DIR) if f.endswith('-ref.dez'))
    print(f"Total reference fugue files: {len(ref_files)}\n")
    
    # Load the example file from our investigation
    target = "07-bwv852-ref.dez"
    if target not in ref_files:
        target = ref_files[0]
    
    print(f"Loading: {target}")
    tree = load_fugue_dez(os.path.join(FUGUE_DIR, target))
    
    print(f"\nTotal nodes: {tree.size()}")
    print(f"\nTree structure:")
    print(tree.pretty_print())