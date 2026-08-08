"""Globally optimal ordered binary clustering by interval dynamic programming.

The implementation deliberately keeps the same assumptions as the greedy stage:

* leaves are fixed-duration, temporally ordered bins;
* every node covers one contiguous interval;
* an interval representation is the sum of its leaf pitch-class vectors;
* a local merge cost compares the left and right child representations;
* the full-tree objective is the sum of all internal-node merge costs.

For a fixed distance function, the returned tree is globally optimal under this
additive objective.  It is not claimed to be globally optimal according to
musical ground truth or Boundary F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
import math
import time
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

DistanceFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class DPDiagnostics:
    """Diagnostics returned alongside an optimal tree."""

    total_cost: float
    n_bins: int
    evaluated_splits: int
    elapsed_seconds: float
    root_split: int
    tie_count: int = 0
    objective_name: str = "additive_child_distance"


def _validate_inputs(
    matrix: np.ndarray,
    bounds: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    bounds = np.asarray(bounds, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D; got shape {matrix.shape}")
    if matrix.shape[0] < 1:
        raise ValueError("matrix must contain at least one bin")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains NaN or infinite values")
    if np.any(matrix < 0):
        raise ValueError("pitch-class duration vectors must be non-negative")
    if bounds.ndim == 2 and bounds.shape == (len(matrix), 2):
        if len(bounds) > 1 and not np.allclose(bounds[:-1, 1], bounds[1:, 0]):
            raise ValueError('interval bounds must be temporally contiguous')
        bounds = np.r_[bounds[0, 0], bounds[:, 1]]
    elif bounds.ndim != 1 or len(bounds) != len(matrix) + 1:
        raise ValueError('bounds must have shape (n_bins,2) or (n_bins+1,)')
    if not np.all(np.isfinite(bounds)):
        raise ValueError("bounds contains NaN or infinite values")
    if np.any(np.diff(bounds) <= 0):
        raise ValueError("bounds must be strictly increasing")

    return matrix, bounds


def _construct_node(
    node_type: type,
    start: float,
    end: float,
    vector: np.ndarray,
    children: Optional[list[Any]] = None,
) -> Any:
    """Construct a project ClusterNode across common constructor variants."""
    children = [] if children is None else children

    attempts = [
        lambda: node_type(start=start, end=end, feature=vector, children=children),
        lambda: node_type(start, end, vector, children=children),
        lambda: node_type(start=start, end=end, vector=vector, children=children),
        lambda: node_type(start=start, end=end, profile=vector, children=children),
        lambda: node_type(start=start, end=end, pc_vector=vector, children=children),
        lambda: node_type(start, end, vector),
        lambda: node_type(start=start, end=end, vector=vector),
    ]

    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            node = attempt()
            if not hasattr(node, "children"):
                setattr(node, "children", children)
            elif children:
                setattr(node, "children", children)

            # Keep a representation available even if the project's class uses
            # another field name.
            if not any(hasattr(node, name) for name in ('feature', 'vector', 'profile', 'pc_vector')):
                setattr(node, 'feature', vector)
            return node
        except (TypeError, AttributeError) as error:
            last_error = error

    raise TypeError(
        "Could not construct ClusterNode. Adapt _construct_node() to the "
        "constructor used in your project."
    ) from last_error


def prefix_sums(matrix: np.ndarray) -> np.ndarray:
    """Return prefix sums so every interval vector is available in O(d)."""
    matrix = np.asarray(matrix, dtype=float)
    prefix = np.zeros((len(matrix) + 1, matrix.shape[1]), dtype=float)
    prefix[1:] = np.cumsum(matrix, axis=0)
    return prefix


def interval_vector(prefix: np.ndarray, i: int, j: int) -> np.ndarray:
    """Pitch-class duration vector for the half-open interval [i, j)."""
    return prefix[j] - prefix[i]


def optimal_adjacent_binary_tree(
    matrix: np.ndarray,
    bounds: Sequence[float],
    distance_fn: DistanceFunction,
    node_type: type,
    *,
    length_weight: Optional[Callable[[int], float]] = None,
    balance_lambda: float = 0.0,
    tie_tolerance: float = 1e-12,
    max_bins: Optional[int] = None,
) -> tuple[Any, DPDiagnostics]:
    """Compute the globally optimal ordered binary tree.

    The recurrence is

        DP[i,j] = min_k DP[i,k] + DP[k,j] + local_cost(i,k,j),

    where local_cost is the distance between aggregate representations of the
    two children.  Optional duration weighting and balance regularisation are
    provided for ablation experiments; leave both at their defaults for the
    strict greedy-versus-DP comparison.

    Parameters
    ----------
    matrix:
        Array of shape (n_bins, n_features).
    bounds:
        Temporal boundaries of length n_bins + 1.
    distance_fn:
        Non-negative distance applied to two aggregate interval vectors.
    node_type:
        The project's ClusterNode class.
    length_weight:
        Optional function of parent length in bins.  Default is 1.
    balance_lambda:
        Optional penalty for uneven child sizes.  Default 0.
    tie_tolerance:
        Numerical tolerance used for deterministic tie-breaking.
    max_bins:
        Optional safety guard against accidental cubic runs.

    Returns
    -------
    root, diagnostics
    """
    matrix, bounds = _validate_inputs(matrix, bounds)
    n = len(matrix)

    if max_bins is not None and n > max_bins:
        raise ValueError(
            f"{n} bins exceeds max_bins={max_bins}. Increase bin size or "
            "raise the guard deliberately."
        )
    if balance_lambda < 0:
        raise ValueError("balance_lambda must be non-negative")

    if length_weight is None:
        length_weight = lambda length: 1.0

    started = time.perf_counter()
    prefix = prefix_sums(matrix)
    dp = np.full((n + 1, n + 1), np.inf, dtype=float)
    split = np.full((n + 1, n + 1), -1, dtype=int)

    interval_repr = None
    component_reprs = None
    transform_batch = getattr(distance_fn, 'transform_batch', None)
    batch_distance = getattr(distance_fn, 'batch_distance', None)
    mixture_functions=getattr(distance_fn,'functions',())
    if mixture_functions and all(getattr(fn,'transform_batch',None) is not None
                                 for fn in mixture_functions):
        component_reprs=[]
        for fn in mixture_functions:
            sample=np.asarray(fn.transform_batch(matrix[:1]),dtype=float)
            component_reprs.append(np.zeros((n+1,n+1,sample.shape[1]),dtype=float))
        for length in range(1,n+1):
            starts=np.arange(0,n-length+1); ends=starts+length
            vectors=prefix[ends]-prefix[starts]
            for target,fn in zip(component_reprs,mixture_functions):
                target[starts,ends]=fn.transform_batch(vectors)
    elif transform_batch is not None:
        sample=np.asarray(transform_batch(matrix[:1]),dtype=float)
        interval_repr=np.zeros((n+1,n+1,sample.shape[1]),dtype=float)
        for length in range(1,n+1):
            starts=np.arange(0,n-length+1); ends=starts+length
            vectors=prefix[ends]-prefix[starts]
            interval_repr[starts,ends]=transform_batch(vectors)

    for i in range(n):
        dp[i, i + 1] = 0.0

    evaluated_splits = 0
    tie_count = 0
    @lru_cache(maxsize=None)
    def local_cost(i: int, k: int, j: int) -> float:
        nonlocal evaluated_splits
        evaluated_splits += 1

        left = interval_vector(prefix, i, k)
        right = interval_vector(prefix, k, j)
        base = float(distance_fn(left, right))
        if not math.isfinite(base):
            raise ValueError(
                f"distance function returned a non-finite value for "
                f"[{i},{k}) and [{k},{j})"
            )
        if base < -tie_tolerance:
            raise ValueError(
                "distance function must be non-negative for this objective; "
                f"got {base}"
            )
        base = max(0.0, base)

        parent_length = j - i
        weighted = float(length_weight(parent_length)) * base

        left_length = k - i
        right_length = j - k
        imbalance = abs(left_length - right_length) / parent_length
        return weighted + balance_lambda * imbalance

    # Bottom-up interval DP.
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            ks=np.arange(i+1,j)
            if component_reprs is not None:
                factors=np.asarray(distance_fn.weights)/np.asarray(distance_fn.scales)
                base=sum(factor*np.linalg.norm(values[i,ks]-values[ks,j],axis=1)
                         for factor,values in zip(factors,component_reprs))
            elif interval_repr is not None:
                base=np.linalg.norm(interval_repr[i,ks]-interval_repr[ks,j],axis=1)
            else:
                base=None
            if component_reprs is None and interval_repr is None and batch_distance is not None:
                left=prefix[ks]-prefix[i]
                right=prefix[j]-prefix[ks]
                base=np.asarray(batch_distance(left,right),dtype=float)
                if base.shape != (len(ks),):
                    raise ValueError('batch_distance must return one value per row')
            if base is not None:
                if not np.all(np.isfinite(base)) or np.any(base < -tie_tolerance):
                    raise ValueError(f'invalid batch distance for interval [{i},{j})')
                base=np.maximum(base,0.0)
                local=float(length_weight(length))*base
                local+=balance_lambda*np.abs((ks-i)-(j-ks))/length
                candidates=dp[i,ks]+dp[ks,j]+local
                evaluated_splits+=len(ks)
            else:
                candidates=np.array([dp[i,k]+dp[k,j]+local_cost(i,k,j) for k in ks])
            if not np.all(np.isfinite(candidates)):
                raise ValueError(f'non-finite candidate cost for interval [{i},{j})')
            best_cost=float(candidates.min())
            tied=np.flatnonzero(candidates<=best_cost+tie_tolerance)
            if len(tied)>1: tie_count+=1
            best_k=int(ks[int(tied[0])])
            best_cost=float(candidates[int(tied[0])])

            if best_k < 0:
                raise RuntimeError(f"No valid split found for interval [{i}, {j})")
            dp[i, j] = best_cost
            split[i, j] = best_k

    merge_order = [0]

    def attach_metadata(node: Any, merge_cost: float, objective: float) -> Any:
        setattr(node, 'merge_cost', float(merge_cost))
        setattr(node, 'subtree_objective', float(objective))
        return node

    def build(i: int, j: int) -> Any:
        vector = interval_vector(prefix, i, j).copy()
        if j == i + 1:
            return attach_metadata(_construct_node(
                node_type,
                float(bounds[i]),
                float(bounds[j]),
                vector,
                children=[],
            ), 0.0, 0.0)

        k = int(split[i, j])
        left = build(i, k)
        right = build(k, j)
        node = _construct_node(
            node_type,
            float(bounds[i]),
            float(bounds[j]),
            vector,
            children=[left, right],
        )
        local=float(dp[i,j]-dp[i,k]-dp[k,j])
        setattr(node,'merge_order',merge_order[0])
        merge_order[0]+=1
        return attach_metadata(node,local,float(dp[i,j]))

    root = build(0, n)
    elapsed = time.perf_counter() - started
    diagnostics = DPDiagnostics(
        total_cost=float(dp[0, n]),
        n_bins=n,
        evaluated_splits=evaluated_splits,
        elapsed_seconds=elapsed,
        root_split=int(split[0, n]) if n > 1 else -1,
        tie_count=tie_count,
    )
    return root, diagnostics


def node_vector(node: Any) -> np.ndarray:
    """Retrieve a node representation across common project field names."""
    for name in ('feature', 'vector', 'profile', 'pc_vector'):
        if hasattr(node, name):
            value = getattr(node, name)
            if value is not None:
                return np.asarray(value, dtype=float)
    raise AttributeError(
        "Tree node has no vector/profile/pc_vector representation. "
        "Store aggregate interval vectors on every node."
    )


def additive_tree_cost(
    root: Any,
    distance_fn: DistanceFunction,
    *,
    length_weight: Optional[Callable[[int], float]] = None,
    balance_lambda: float = 0.0,
    leaf_width: Optional[float] = None,
) -> float:
    """Score any compatible tree under the same additive DP objective.

    Use this to compare the greedy tree's objective with the globally optimal
    DP objective.  If leaf_width is supplied, duration is converted to an
    approximate number of bins for optional length weighting.
    """
    if length_weight is None:
        length_weight = lambda length: 1.0

    def count_leaves(node: Any) -> int:
        children = list(getattr(node, "children", []) or [])
        return 1 if not children else sum(count_leaves(child) for child in children)

    def walk(node: Any) -> float:
        children = list(getattr(node, "children", []) or [])
        if not children:
            return 0.0
        if len(children) != 2:
            raise ValueError(
                "additive_tree_cost expects a binary tree; "
                f"found {len(children)} children"
            )

        left, right = children
        left_n = count_leaves(left)
        right_n = count_leaves(right)
        parent_n = left_n + right_n

        base = float(distance_fn(node_vector(left), node_vector(right)))
        imbalance = abs(left_n - right_n) / parent_n
        local = float(length_weight(parent_n)) * base + balance_lambda * imbalance
        return walk(left) + walk(right) + local

    return float(walk(root))


def assert_valid_ordered_binary_tree(
    root: Any,
    bounds: Sequence[float],
    *,
    atol: float = 1e-8,
) -> None:
    """Raise AssertionError when temporal/tree invariants are violated."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim==2:
        if bounds.shape[1]!=2: raise AssertionError('interval bounds must have two columns')
        bounds=np.r_[bounds[0,0],bounds[:,1]]
    leaves: list[Any] = []

    def walk(node: Any) -> None:
        children = list(getattr(node, "children", []) or [])
        if not children:
            leaves.append(node)
            return
        assert len(children) == 2, "Every internal node must be binary"
        left, right = children
        assert abs(float(left.end) - float(right.start)) <= atol
        assert abs(float(node.start) - float(left.start)) <= atol
        assert abs(float(node.end) - float(right.end)) <= atol
        walk(left)
        walk(right)

    walk(root)
    assert len(leaves) == len(bounds) - 1
    for index, leaf in enumerate(leaves):
        assert abs(float(leaf.start) - float(bounds[index])) <= atol
        assert abs(float(leaf.end) - float(bounds[index + 1])) <= atol
