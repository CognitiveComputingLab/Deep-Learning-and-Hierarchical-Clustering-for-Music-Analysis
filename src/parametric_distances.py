"""Simple interpretable parametric distance functions for the DP stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

DistanceFunction = Callable[[np.ndarray, np.ndarray], float]


def l1_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    total = float(vector.sum())
    if total <= eps:
        return np.zeros_like(vector, dtype=float)
    return vector / total


def softmax(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


@dataclass(frozen=True)
class WeightedDistanceMixture:
    """Non-negative convex mixture of existing handcrafted distances.

    This is recommended as the first trainable model because it has very few
    parameters and each learned weight has a direct musical interpretation.
    """

    names: tuple[str, ...]
    functions: tuple[DistanceFunction, ...]
    weights: np.ndarray
    scales: np.ndarray

    @classmethod
    def from_logits(
        cls,
        named_functions: Mapping[str, DistanceFunction],
        logits: Sequence[float],
    ) -> "WeightedDistanceMixture":
        names = tuple(named_functions)
        functions = tuple(named_functions[name] for name in names)
        weights = softmax(logits)
        if len(weights) != len(functions):
            raise ValueError("Number of logits must match number of distances")
        return cls(names=names, functions=functions, weights=weights,
                   scales=np.ones(len(weights),dtype=float))

    @classmethod
    def from_weights(
        cls,
        named_functions: Mapping[str, DistanceFunction],
        weights: Sequence[float],
        scales: Sequence[float] | None = None,
    ) -> "WeightedDistanceMixture":
        names = tuple(named_functions)
        functions = tuple(named_functions[name] for name in names)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(functions):
            raise ValueError("Number of weights must match number of distances")
        if np.any(weights < 0):
            raise ValueError("Mixture weights must be non-negative")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("At least one mixture weight must be positive")
        if scales is None:
            scales=np.ones(len(functions),dtype=float)
        scales=np.asarray(scales,dtype=float)
        if scales.shape != weights.shape or np.any(~np.isfinite(scales)) or np.any(scales<=0):
            raise ValueError('Scales must be positive and match the distances')
        return cls(names=names, functions=functions, weights=weights / total,
                   scales=scales)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> float:
        values = np.array(
            [fn(left, right) for fn in self.functions],
            dtype=float,
        )
        return float(self.weights @ (values/self.scales))

    def batch_distance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        columns=[]
        for fn in self.functions:
            batch=getattr(fn,'batch_distance',None)
            if batch is not None:
                columns.append(np.asarray(batch(left,right),dtype=float))
            else:
                columns.append(np.asarray([fn(a,b) for a,b in zip(left,right)]))
        return np.column_stack(columns) @ (self.weights/self.scales)

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(weight)
            for name, weight in zip(self.names, self.weights)
        }

    def scale_dict(self) -> dict[str, float]:
        return {name:float(scale) for name,scale in zip(self.names,self.scales)}


def estimate_distance_scales(
    named_functions: Mapping[str, DistanceFunction],
    interval_pairs: Sequence[tuple[np.ndarray,np.ndarray]],
    eps: float = 1e-12,
) -> dict[str,float]:
    '''Estimate non-zero median scales from training pairs only.'''
    if not interval_pairs:
        raise ValueError('At least one training interval pair is required')
    result={}
    for name,fn in named_functions.items():
        values=np.asarray([fn(left,right) for left,right in interval_pairs],dtype=float)
        nonzero=values[np.isfinite(values)&(values>eps)]
        result[name]=float(np.median(nonzero)) if len(nonzero) else 1.0
    return result


@dataclass(frozen=True)
class DiagonalMahalanobisDistance:
    """Trainable weighted Euclidean distance over L1-normalised features."""

    weights: np.ndarray
    eps: float = 1e-12

    @classmethod
    def from_logits(cls, logits: Sequence[float]) -> "DiagonalMahalanobisDistance":
        # Softmax prevents negative weights and fixes arbitrary scale.
        return cls(weights=softmax(logits))

    @classmethod
    def from_weights(cls, weights: Sequence[float]) -> "DiagonalMahalanobisDistance":
        weights = np.asarray(weights, dtype=float)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        return cls(weights=weights / total)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> float:
        left = l1_normalize(left, self.eps)
        right = l1_normalize(right, self.eps)
        difference = left - right
        if difference.shape != self.weights.shape:
            raise ValueError(
                f"Expected feature shape {self.weights.shape}; "
                f"got {difference.shape}"
            )
        return float(np.sqrt(np.sum(self.weights * difference * difference)))

    def batch_distance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left=np.asarray(left,dtype=float); right=np.asarray(right,dtype=float)
        left_total=left.sum(axis=1,keepdims=True)
        right_total=right.sum(axis=1,keepdims=True)
        left=np.divide(left,left_total,out=np.zeros_like(left),where=left_total>self.eps)
        right=np.divide(right,right_total,out=np.zeros_like(right),where=right_total>self.eps)
        if left.shape[1:] != self.weights.shape or right.shape != left.shape:
            raise ValueError('Batch feature shape does not match learned weights')
        return np.sqrt(np.sum(self.weights*(left-right)**2,axis=1))
