"""
Demo: compute Zhang-Shasha tree edit distance between two real
Beethoven sonata movements from the Taking Form corpus.

This file connects taking_form_loader.py and tree_utils.py.
"""
import os
from zss import simple_distance, Node

from taking_form_loader import load_taking_form_csv, FormNode


def form_node_to_zss(node: FormNode) -> Node:
    """Convert our FormNode tree to a zss.Node tree."""
    zss_node = Node(node.label)
    for child in node.children:
        zss_node.addkid(form_node_to_zss(child))
    return zss_node


def tree_size_form(node: FormNode) -> int:
    """Count nodes in a FormNode tree."""
    return 1 + sum(tree_size_form(c) for c in node.children)


def compare_two_movements(csv_path1: str, csv_path2: str):
    """Load two Taking Form CSVs and compute their tree edit distance."""
    # Load both as FormNode trees
    tree1 = load_taking_form_csv(csv_path1)
    tree2 = load_taking_form_csv(csv_path2)
    
    # Convert to zss format
    zss_tree1 = form_node_to_zss(tree1)
    zss_tree2 = form_node_to_zss(tree2)
    
    # Compute distance
    ted = simple_distance(zss_tree1, zss_tree2)
    
    # Sizes
    size1 = tree_size_form(tree1)
    size2 = tree_size_form(tree2)
    normalized = ted / (size1 + size2)
    
    return {
        "file1": os.path.basename(csv_path1),
        "file2": os.path.basename(csv_path2),
        "size1": size1,
        "size2": size2,
        "raw_ted": ted,
        "normalized_ted": normalized,
    }


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    BEETHOVEN_DIR = os.path.join(REPO_ROOT, "external/Taking-Form/corpus/Beethoven_Sonatas")
    
    csv_files = sorted(f for f in os.listdir(BEETHOVEN_DIR) if f.endswith('.csv'))
    
    # Pick 5 movements to compare pairwise
    sample_files = csv_files[:5]
    print(f"Comparing the following 5 movements pairwise:\n")
    for f in sample_files:
        print(f"  - {f}")
    print()
    
    # Compute pairwise distances
    print(f"{'File 1':<35} {'File 2':<35} {'Size1':>6} {'Size2':>6} {'TED':>6} {'Norm':>8}")
    print("-" * 100)
    
    for i in range(len(sample_files)):
        for j in range(i + 1, len(sample_files)):
            path1 = os.path.join(BEETHOVEN_DIR, sample_files[i])
            path2 = os.path.join(BEETHOVEN_DIR, sample_files[j])
            result = compare_two_movements(path1, path2)
            print(f"{result['file1']:<35} {result['file2']:<35} "
                  f"{result['size1']:>6} {result['size2']:>6} "
                  f"{result['raw_ted']:>6.0f} {result['normalized_ted']:>8.3f}")
    
    # Sanity check: a movement compared to itself should have distance 0
    print()
    print("Sanity check (movement compared to itself):")
    self_result = compare_two_movements(
        os.path.join(BEETHOVEN_DIR, sample_files[0]),
        os.path.join(BEETHOVEN_DIR, sample_files[0])
    )
    print(f"  {self_result['file1']} vs itself: TED = {self_result['raw_ted']} "
          f"(should be 0)")
    
    # Compute baseline: average pairwise distance between different movements
    print()
    print("Baseline statistics:")
    distances = []
    for i in range(len(sample_files)):
        for j in range(i + 1, len(sample_files)):
            path1 = os.path.join(BEETHOVEN_DIR, sample_files[i])
            path2 = os.path.join(BEETHOVEN_DIR, sample_files[j])
            result = compare_two_movements(path1, path2)
            distances.append(result['normalized_ted'])
    
    avg = sum(distances) / len(distances)
    min_d = min(distances)
    max_d = max(distances)
    print(f"  Average pairwise normalized TED: {avg:.3f}")
    print(f"  Min: {min_d:.3f}, Max: {max_d:.3f}")
    print(f"  This is a random baseline to compare with future models.")