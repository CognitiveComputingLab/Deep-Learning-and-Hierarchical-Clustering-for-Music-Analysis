# Methods: representation, tree search, and neural supervision

## Input and shared similarity interface

For a score, timed notes are mapped to consecutive quarterbeat bins. In bin $i$, $x_i\in\mathbb{R}^{12}_{\geq0}$ contains duration-weighted sounding time for each pitch class. Empty bins are collapsed with their time assigned to neighbouring sounding bins; the resulting $n$ leaves retain their chronological order and interval endpoints. The primary ABC experiment uses 8-quarterbeat bins. See `src/greedy_clustering.py:load_pc_bins()` and `scripts/evaluate_neural_distance_stage.py`.

A distance specification supplies `transform_batch(X)` and `batch_distance(left, right)`; the temporal neural model also supplies `transform_contextual_bins(X, radius)` and `boundary_contrast_batch(X, contexts)`. The shared downstream contract is one finite representation vector $z_i=f_\theta(X,i)$ per input bin, plus local contrast evidence $c_k\in[0,1]$ for each bin gap. Handcrafted specifications include normalized pitch-class Euclidean, circle-of-fifths embedding, and 24-dimensional key-profile activation. Learned specifications include a diagonal Mahalanobis transform and a frozen neural encoder. A piece-specific tonic-weighted distance is an additional handcrafted Taking Form comparator. The same representation and affinity for a given experimental cell feed both searches. See `src/greedy_clustering.py:DistanceSpec`, `src/neural_distance.py`, and `src/ordered_affinity.py:pairwise_affinity()`.

For an embedding $z_i$, the pairwise distance and affinity are

\[
d_{ij}=\lVert z_i-z_j\rVert_2,\qquad
A_{ij}=\exp\!\left[-\tfrac12(d_{ij}/\sigma)^2\right].
\]

Thus $A\in[0,1]^{n\times n}$ is symmetric, with larger values meaning greater similarity. A non-zero median distance sets $\sigma$ within a piece for an unsupervised handcrafted baseline; learned-model scales are estimated from training works and then frozen. The code can additionally smooth affinity along diagonals, with the radius recorded in metadata. The neural representation can use an ordered context around a bin; this changes $z_i$, while the RBF conversion stays the same. Local contrast $c_k$ is calibrated separately from the affinity matrix, although it uses the same frozen learned representation in the neural experiment. In the main neural stage, contexts are 1, 2, 4, and 8 bins and the representation context radius is 4.

## Ordered trees and the whole-tree objective

The leaves are the $n$ bins. Every internal node covers an interval $[i,j)$ and splits at $k$, where $i<k<j$. Write $L=[i,k)$, $R=[k,j)$, $C(i,k,j)=\sum_{a=i}^{k-1}\sum_{b=k}^{j-1} A_{ab}$, and $D=(n-2)\sum_{a<b} A_{ab}$ for $n>2$. The final ABC boundary-aware objective is

\[
J(T)=\sum_{(i,k,j)\in I(T)}\left[
(1-\lambda)\frac{(n-j+i)C(i,k,j)}{D}
+\lambda\frac{\log(j-i)}{\log n}\frac{c_{k-1}}{n-1}
-\beta\frac{1}{n-1}\left(\frac{|L|-|R|}{|L|+|R|}\right)^2
\right].
\]

The first term is normalized affinity revenue; the second rewards a strong local contrast when its split is high in the tree; the third penalizes child-size imbalance. For the degenerate $D=0$ case, the affinity term is defined as zero. The implementation separately validates all components and evaluates this **same completed-tree $J$** on either output. The objective is neither boundary F1 nor expert T-measure. See `src/ordered_affinity.py:boundary_aware_tree_objective()`.

Exact interval DP uses $F(i,i+1)=0$ and

\[
F(i,j)=\max_{i<k<j}\{F(i,k)+F(k,j)+\ell(i,k,j)\},
\]

where $\ell$ is the corresponding summand of $J$. Backtracking from $F(0,n)$ produces one complete ordered binary tree. Optimality applies to the fixed bins, frozen affinity and contrast, chosen $\lambda,\beta$, and this additive objective. It does not apply to alternative representations or evaluation metrics. The current implementation evaluates $O(n^3)$ candidate splits and stores $O(n^2)$ interval scores; it enforces a maximum-bin limit. See `src/ordered_affinity.py:optimal_boundary_aware_tree()`.

Greedy starts with one leaf per cluster and repeatedly merges adjacent clusters $L,R$ with the highest priority

\[
P(L,R)=(1-\lambda)\operatorname{mean}_{a\in L,b\in R} A_{ab}
-\lambda c_{\operatorname{gap}(L,R)}
-\beta\left(\frac{|L|-|R|}{|L|+|R|}\right)^2.
\]

It chooses the leftmost maximum when priorities tie and cannot revise an earlier merge. $P$ uses the same affinity, contrast, and weights but is **not** $\ell(i,k,j)$: it uses mean affinity rather than the DP's span-weighted cross-affinity sum, and its balance term lacks DP's tree-wide normalization. Greedy is therefore a local heuristic evaluated under $J$, not a second optimizer of exactly the same objective. For matched inputs, $J(T_{DP})\geq J(T_{Greedy})$ follows from the DP search space. This inequality is verified in saved ABC objective comparisons. See `src/ordered_affinity.py:greedy_boundary_aware_tree()`.

The older fixed-representation Taking Form comparison uses a **different additive merge-distance cost**, which DP **minimizes**. The separately trained cross-composer Taking Form experiment uses the boundary-aware $J$-style objective, which DP **maximizes**. Their objective values should never be combined into one numerical table without this distinction. See `scripts/evaluate_shared_hierarchy_stage.py` and `scripts/evaluate_parametric_hierarchy_stage.py`.

## Neural training and its relation to tree search

ABC expert local-key change times are mapped to their nearest internal bin gaps. For each gap, local left/right pitch-class contexts of 1, 2, 4, or 8 bins form candidate examples. Selected nearest gaps are positives. Other gaps within 8 quarterbeats of a reference are ignored, and the remaining gaps are negatives; no negative subsampling is used. Training examples are weighted so each work has equal total weight and each class present within a work has equal weight. Entire quartet works, not individual movements, define training, inner validation, and held-out outer test groups. See `src/boundary_contrast.py:tolerance_aware_boundary_labels()` and `src/neural_distance.py:build_interval_training_set()`.

The temporal circular-convolution Siamese encoder maps each side to an embedding. Its distance is the Euclidean distance between embeddings, and a learned non-negative gain and intercept convert distance into a boundary logit. For example $r$, with label $y_r$, logit $u_r$, and training weight $w_r$, the default minibatch loss is

\[
\mathcal L_{BCE}=\frac{\sum_r w_r\,\operatorname{BCEWithLogits}(u_r,y_r)}{\sum_r w_r}.
\]

Optional metric and ranking losses exist, but **both weights were zero in the frozen primary run**. Training used AdamW with initial learning rate 0.001, weight decay 0.001, minibatch size 64, cosine learning-rate annealing, and gradient clipping at norm 1. Checkpoints and model/threshold choices are selected on training-side validation, not outer-test labels. The chosen encoder and probability calibration are frozen; a training-work distance scale is then used to form affinities and local boundary evidence for both Greedy and DP. See `src/neural_distance.py:train_distance_checkpoints()` and `results/dissertation_main/neural/metadata.json`.

The neural model was supervised by **flat local-key boundary events**, not by Taking Form hierarchies or complete expert trees. It may learn evidence related to modulation; whether it also captures more general form is an empirical question. The cross-composer Taking Form parametric mixture is a **separate model trained on form labels**, not a transfer of this ABC neural network.

## From a tree to reported ABC boundaries

Each internal split contributes one candidate boundary at the end of its left child. With $L_k$ leaves under the candidate's least common ancestor, its salience is $s_k=c_k\log L_k/\log n$. Every adjacent leaf gap appears once in a complete ordered binary tree; a leaf itself does not decide whether its edge is reported. A work-blocked inner validation selects the salience threshold from $0,0.05,\ldots,1$ along with the other operating settings. Optional local-maximum filtering and minimum-separation suppression precede reporting. This threshold is distinct from the 8-quarterbeat **matching tolerance** used by boundary F1. See `src/greedy_evaluation.py:boundary_salience()` and `collect_salient_splits()`.
