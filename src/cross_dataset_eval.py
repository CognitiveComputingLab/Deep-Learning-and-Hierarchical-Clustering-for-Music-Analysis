"""
Cross-dataset evaluation: compare tree distances within and across datasets.

Loads:
  - Taking Form: Beethoven piano sonata movements (CSV → tree)
  - ABC:         Beethoven string quartet movements (TSV → tree)

Computes:
  - Within-dataset baseline (Taking Form pairwise, ABC pairwise)
  - Cross-dataset distance (Taking Form vs ABC)
"""
import os
import sys
from zss import simple_distance, Node

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from taking_form_loader import load_taking_form_csv, FormNode
from abc_loader import load_abc_tsv


def form_node_to_zss(node: FormNode) -> Node:
    """Convert FormNode tree to zss.Node tree."""
    zss_node = Node(node.label)
    for child in node.children:
        zss_node.addkid(form_node_to_zss(child))
    return zss_node


def tree_size_form(node: FormNode) -> int:
    return 1 + sum(tree_size_form(c) for c in node.children)


def normalized_ted(tree1: FormNode, tree2: FormNode):
    """Compute normalized TED between two FormNode trees."""
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


def main():
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # === Load 3 movements from each dataset ===
    BEETHOVEN_DIR = os.path.join(REPO_ROOT, "external/Taking-Form/corpus/Beethoven_Sonatas")
    ABC_DIR = os.path.join(REPO_ROOT, "external/ABC/harmonies")
    
    tf_files = sorted(f for f in os.listdir(BEETHOVEN_DIR) if f.endswith('.csv'))[:3]
    abc_files = sorted(f for f in os.listdir(ABC_DIR) if f.endswith('.harmonies.tsv'))[:3]
    
    print("="*60)
    print("Loading Taking Form movements (Beethoven piano sonatas)")
    print("="*60)
    tf_trees = {}
    for f in tf_files:
        tree = load_taking_form_csv(os.path.join(BEETHOVEN_DIR, f))
        tf_trees[f] = tree
        print(f"  {f}: {tree.size()} nodes")
    
    print()
    print("="*60)
    print("Loading ABC movements (Beethoven string quartets)")
    print("="*60)
    abc_trees = {}
    for f in abc_files:
        tree = load_abc_tsv(os.path.join(ABC_DIR, f))
        abc_trees[f] = tree
        print(f"  {f}: {tree.size()} nodes")
    
    # === Within Taking Form ===
    print()
    print("="*60)
    print("Within Taking Form (pairwise)")
    print("="*60)
    tf_names = list(tf_trees.keys())
    within_tf = []
    for i in range(len(tf_names)):
        for j in range(i + 1, len(tf_names)):
            r = normalized_ted(tf_trees[tf_names[i]], tf_trees[tf_names[j]])
            within_tf.append(r['normalized'])
            print(f"  {tf_names[i][:25]} vs {tf_names[j][:25]}: "
                  f"TED={r['raw_ted']:.0f} norm={r['normalized']:.3f}")
    
    # === Within ABC ===
    print()
    print("="*60)
    print("Within ABC (pairwise)")
    print("="*60)
    abc_names = list(abc_trees.keys())
    within_abc = []
    for i in range(len(abc_names)):
        for j in range(i + 1, len(abc_names)):
            r = normalized_ted(abc_trees[abc_names[i]], abc_trees[abc_names[j]])
            within_abc.append(r['normalized'])
            print(f"  {abc_names[i][:25]} vs {abc_names[j][:25]}: "
                  f"TED={r['raw_ted']:.0f} norm={r['normalized']:.3f}")
    
    # === Cross-dataset ===
    print()
    print("="*60)
    print("Cross-dataset (Taking Form vs ABC)")
    print("="*60)
    cross = []
    for tf_name in tf_names:
        for abc_name in abc_names:
            r = normalized_ted(tf_trees[tf_name], abc_trees[abc_name])
            cross.append(r['normalized'])
            print(f"  {tf_name[:25]} vs {abc_name[:25]}: "
                  f"TED={r['raw_ted']:.0f} norm={r['normalized']:.3f}")
    
    # === Summary ===
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    if within_tf:
        print(f"  Within Taking Form (avg norm): {sum(within_tf)/len(within_tf):.3f}")
    if within_abc:
        print(f"  Within ABC (avg norm):         {sum(within_abc)/len(within_abc):.3f}")
    if cross:
        print(f"  Cross-dataset (avg norm):      {sum(cross)/len(cross):.3f}")
    
    print()
    print("If cross-dataset distance is in a similar range as within-dataset,")
    print("the unified format produces comparable measurements across datasets.")


if __name__ == "__main__":
    main()