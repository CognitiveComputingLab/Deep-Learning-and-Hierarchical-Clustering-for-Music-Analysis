"""
Cross-dataset evaluation: compare tree distances within and across THREE datasets.

Datasets:
  - Taking Form: Beethoven piano sonata movements (Gotham & Ireland 2019)
  - ABC:         Beethoven string quartet movements (Neuwirth et al. 2018)
  - Fugue:       Bach WTC I fugues (Giraud et al. 2015)

Each loader produces FormNode trees with a unified data structure, but
semantically the labels and depths differ. This script quantifies
similarity *within* each dataset (the meaningful baseline) and *across*
datasets (which is expected to be large, since label vocabularies are
disjoint by design).
"""
import os
import sys
from zss import simple_distance, Node

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from taking_form_loader import load_taking_form_csv, FormNode
from abc_loader import load_abc_tsv
from fugue_loader import load_fugue_dez


def form_node_to_zss(node: FormNode) -> Node:
    zss_node = Node(node.label)
    for child in node.children:
        zss_node.addkid(form_node_to_zss(child))
    return zss_node


def tree_size_form(node: FormNode) -> int:
    return 1 + sum(tree_size_form(c) for c in node.children)


def normalized_ted(tree1: FormNode, tree2: FormNode):
    zss1 = form_node_to_zss(tree1)
    zss2 = form_node_to_zss(tree2)
    ted = simple_distance(zss1, zss2)
    size1 = tree_size_form(tree1)
    size2 = tree_size_form(tree2)
    return {
        "raw_ted": ted,
        "size1": size1,
        "size2": size2,
        "normalized": ted / (size1 + size2),
    }


def average_pairwise(trees_dict):
    """Compute average normalized TED across all unique pairs."""
    names = list(trees_dict.keys())
    distances = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = normalized_ted(trees_dict[names[i]], trees_dict[names[j]])
            distances.append(r['normalized'])
    return sum(distances) / len(distances) if distances else 0.0


def average_cross(trees_a, trees_b):
    """Compute average normalized TED across all pairs from two sets."""
    distances = []
    for ta in trees_a.values():
        for tb in trees_b.values():
            r = normalized_ted(ta, tb)
            distances.append(r['normalized'])
    return sum(distances) / len(distances) if distances else 0.0


def main():
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    
    BEETHOVEN_DIR = os.path.join(REPO_ROOT, "external/Taking-Form/corpus/Beethoven_Sonatas")
    ABC_DIR = os.path.join(REPO_ROOT, "external/ABC/harmonies")
    FUGUE_DIR = os.path.join(REPO_ROOT, "external/algomus-data/fugues/bach-wtc-i")
    
    N = 5  # number of movements per dataset to use
    
    # === Load N movements from each dataset ===
    tf_files = sorted(f for f in os.listdir(BEETHOVEN_DIR) if f.endswith('.csv'))[:N]
    abc_files = sorted(f for f in os.listdir(ABC_DIR) if f.endswith('.harmonies.tsv'))[:N]
    fugue_files = sorted(f for f in os.listdir(FUGUE_DIR) if f.endswith('-ref.dez'))[:N]
    
    print("="*60)
    print(f"Loading {N} movements from each dataset")
    print("="*60)
    
    tf_trees = {}
    print("\nTaking Form (Beethoven piano sonatas):")
    for f in tf_files:
        tree = load_taking_form_csv(os.path.join(BEETHOVEN_DIR, f))
        tf_trees[f] = tree
        print(f"  {f}: {tree_size_form(tree)} nodes")
    
    abc_trees = {}
    print("\nABC (Beethoven string quartets):")
    for f in abc_files:
        tree = load_abc_tsv(os.path.join(ABC_DIR, f))
        abc_trees[f] = tree
        print(f"  {f}: {tree_size_form(tree)} nodes")
    
    fugue_trees = {}
    print("\nFugue (Bach WTC I):")
    for f in fugue_files:
        tree = load_fugue_dez(os.path.join(FUGUE_DIR, f))
        fugue_trees[f] = tree
        print(f"  {f}: {tree_size_form(tree)} nodes")
    
    # === Compute averages ===
    print()
    print("="*60)
    print("Within-dataset baselines (avg normalized TED)")
    print("="*60)
    print(f"  Taking Form:  {average_pairwise(tf_trees):.3f}")
    print(f"  ABC:          {average_pairwise(abc_trees):.3f}")
    print(f"  Fugue:        {average_pairwise(fugue_trees):.3f}")
    
    print()
    print("="*60)
    print("Cross-dataset averages")
    print("="*60)
    print(f"  Taking Form vs ABC:    {average_cross(tf_trees, abc_trees):.3f}")
    print(f"  Taking Form vs Fugue:  {average_cross(tf_trees, fugue_trees):.3f}")
    print(f"  ABC vs Fugue:          {average_cross(abc_trees, fugue_trees):.3f}")
    
    print()
    print("Note: Within-dataset distances are the meaningful baseline for")
    print("model evaluation. Cross-dataset distances reflect disjoint label")
    print("vocabularies and are expected to be larger.")


if __name__ == "__main__":
    main()