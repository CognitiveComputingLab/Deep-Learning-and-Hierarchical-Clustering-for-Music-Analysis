"""
Tree utilities for Taking Form analysis.

Two main functions:
1. parse_nbn(text) — parse Nested Bracket Notation into a tree
2. tree_edit_distance(tree1, tree2) — compute Zhang-Shasha edit distance
"""
from zss import simple_distance, Node


def parse_nbn(text):
    """
    Parse Nested Bracket Notation (NBN) string into a zss.Node tree.
    
    Example input:
        [Exposition
          [Subject~I
            [Th.i 4 4]
            [Th.ii 4 4 5]]
          [Transition 4]]
    
    Returns: zss.Node representing the root of the tree.
    
    Convention used by Taking Form:
    - First token after '[' is the node label
    - Subsequent tokens that are integers are "leaf children" (measure counts)
    - Subsequent '[...]' are nested children
    """
    # Tokenize: split on whitespace, but keep '[' and ']' as separate tokens
    tokens = []
    current = []
    for char in text:
        if char in '[]':
            if current:
                tokens.append(''.join(current))
                current = []
            tokens.append(char)
        elif char.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
        else:
            current.append(char)
    if current:
        tokens.append(''.join(current))
    
    # Recursive parse
    pos = [0]  # mutable position pointer
    
    def parse_node():
        # Expect '['
        if tokens[pos[0]] != '[':
            raise ValueError(f"Expected '[' at position {pos[0]}, got '{tokens[pos[0]]}'")
        pos[0] += 1
        
        # Read label
        label = tokens[pos[0]]
        pos[0] += 1
        node = Node(label)
        
        # Read children
        while pos[0] < len(tokens) and tokens[pos[0]] != ']':
            if tokens[pos[0]] == '[':
                child = parse_node()
                node.addkid(child)
            else:
                # Numeric leaf (measure count) or other label
                leaf_label = tokens[pos[0]]
                node.addkid(Node(leaf_label))
                pos[0] += 1
        
        # Consume ']'
        if pos[0] < len(tokens) and tokens[pos[0]] == ']':
            pos[0] += 1
        
        return node
    
    return parse_node()


def tree_to_string(node, indent=0):
    """Pretty-print a zss.Node tree for debugging."""
    result = "  " * indent + str(node.label) + "\n"
    for child in node.children:
        result += tree_to_string(child, indent + 1)
    return result


def tree_size(node):
    """Count total nodes in a tree."""
    return 1 + sum(tree_size(child) for child in node.children)


def tree_edit_distance(tree1, tree2):
    """Compute Zhang-Shasha edit distance between two trees."""
    return simple_distance(tree1, tree2)


def normalized_ted(tree1, tree2):
    """
    Normalized tree edit distance, in [0, 1].
    Divides by max possible distance (sum of sizes).
    """
    ted = tree_edit_distance(tree1, tree2)
    max_dist = tree_size(tree1) + tree_size(tree2)
    return ted / max_dist


if __name__ == "__main__":
    # Quick test
    sample_nbn = """[Exposition
        [Subject~I
            [Th.i 4 4]
            [Th.ii 4 4 5 4 5 5 2]]
        [Transition 4]
        [Subject~II
            [Th.iii 2 4 5 4 6]
            [Th.iv 4 4 2]]
        [Codetta 2 4]]"""
    
    tree = parse_nbn(sample_nbn)
    print("Parsed tree:")
    print(tree_to_string(tree))
    print(f"Tree size: {tree_size(tree)} nodes\n")
    
    # Test edit distance: identical trees should have distance 0
    tree2 = parse_nbn(sample_nbn)
    print(f"Distance to itself: {tree_edit_distance(tree, tree2)} (should be 0)")
    
    # Modified tree: remove a section
    modified_nbn = """[Exposition
        [Subject~I
            [Th.i 4 4]
            [Th.ii 4 4 5 4 5 5 2]]
        [Transition 4]
        [Subject~II
            [Th.iii 2 4 5 4 6]]
        [Codetta 2 4]]"""
    
    tree3 = parse_nbn(modified_nbn)
    print(f"Distance to modified (removed Th.iv): {tree_edit_distance(tree, tree3)}")
    print(f"Normalized: {normalized_ted(tree, tree3):.3f}")