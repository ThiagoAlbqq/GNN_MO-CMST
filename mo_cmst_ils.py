"""
mo_cmst_ils.py
==============
Multi-Objective Capacitated Minimum Spanning Tree (MO-CMST)
Instance Generator + ILS → Training Data for GNN

Two objectives
  f1 : total edge cost              (minimize)
  f2 : max subtree demand / load    (minimize) — proxy for balance

Scalarisation
  scalar(f1, f2, alpha) = alpha * f1_norm + (1-alpha) * f2_norm

Training data format (per sample)
  node_features    : (n, 4)  — [x_norm, y_norm, demand_norm, is_root]
  edge_index       : (E, 2)  — complete undirected graph, upper triangle
  edge_features    : (E, 1)  — [dist_norm]
  edge_labels_hard : (E,)    — 1 if edge appears in ≥1 Pareto solution
  edge_labels_soft : (E,)    — fraction of Pareto solutions using edge

Usage
  python mo_cmst_ils.py                          # default run
  python mo_cmst_ils.py --instances 100 \\
         --min-customers 10 --max-customers 30 \\
         --ils-iter 200 --alphas 11 --output data/

Author   : (your name)
Paper    : MO-CMST with Neural-guided ILS
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import pickle
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DEFAULTS
# (override via CLI flags or run_pipeline() kwargs)
# ─────────────────────────────────────────────────────────────────────────────

CFG: dict = dict(
    n_instances      = 30,
    min_customers    = 8,
    max_customers    = 20,
    # capacity = max(max_demand,  factor * total_demand)
    capacity_factor  = 0.35,
    n_alphas         = 7,      # alpha ∈ linspace(0, 1, n_alphas)
    ils_iter         = 100,    # ILS iterations per alpha
    perturb_strength = 3,      # leaves randomly relocated per perturbation
    output_dir       = "mo_cmst_data",
    seed             = 42,
    n_workers        = max(1, (os.cpu_count() or 1)),  # parallel workers (1 = serial)
    save_every       = 10,     # checkpoint every N completed instances
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Instance:
    """
    A single CMST instance.

    Node 0 is always the root / depot  (demand = 0).
    Nodes 1..n-1 are customers.

    Attributes
    ----------
    n        : total nodes (root + customers)
    capacity : max total demand per subtree
    coords   : (n, 2)  node positions in [0, 100]²
    demands  : (n,)    demands[0] = 0
    dist     : (n, n)  Euclidean distance matrix
    root     : always 0
    """

    n        : int
    capacity : float
    coords   : np.ndarray   # (n, 2)
    demands  : np.ndarray   # (n,)
    dist     : np.ndarray   # (n, n)
    root     : int = 0

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def n_customers(self) -> int:
        return self.n - 1

    # ── factory ──────────────────────────────────────────────────────────────

    @staticmethod
    def random(
        n_customers: int,
        capacity_factor: float = 0.35,
        seed: Optional[int] = None,
    ) -> "Instance":
        """Generate a random Euclidean CMST instance."""
        rng = np.random.default_rng(seed)
        n       = n_customers + 1
        coords  = rng.uniform(0.0, 100.0, (n, 2))
        demands = np.zeros(n)
        demands[1:] = rng.integers(1, 10, n_customers).astype(float)
        dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(2))
        capacity = max(
            float(demands[1:].max()),
            capacity_factor * float(demands.sum()),
        )
        return Instance(
            n=n, capacity=capacity, coords=coords, demands=demands, dist=dist
        )

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "n"       : self.n,
            "capacity": self.capacity,
            "coords"  : self.coords.tolist(),
            "demands" : self.demands.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SOLUTION
# ─────────────────────────────────────────────────────────────────────────────

class Solution:
    """
    Spanning tree stored as a **parent array**.

      parent[root] = root   (self-loop sentinel)
      parent[v]    = u      ⟺  directed edge u ← v in the rooted tree

    Safety invariant
    ----------------
    All local-search moves are **leaf relocations only**.
    Moving a leaf (no children) to any other node is always cycle-free:
    the leaf has no subtree below it, so it cannot create a back-edge.
    This eliminates the need for explicit cycle detection.
    """

    def __init__(self, n: int, root: int = 0):
        self.n      = n
        self.root   = root
        self.parent = list(range(n))   # sentinel: parent[i] = i for all i

    def copy(self) -> "Solution":
        s = Solution(self.n, self.root)
        s.parent = self.parent[:]
        return s

    # ── topology helpers ─────────────────────────────────────────────────────

    def children_of(self, v: int) -> List[int]:
        return [i for i in range(self.n) if self.parent[i] == v and i != v]

    def subtree_of(self, v: int) -> List[int]:
        """BFS: all nodes in the subtree rooted at v (including v)."""
        out, stack = [], [v]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.children_of(cur))
        return out

    def root_adjacent(self) -> List[int]:
        """Direct children of the depot."""
        return [
            i for i in range(self.n)
            if i != self.root and self.parent[i] == self.root
        ]

    def subtrees(self) -> Dict[int, List[int]]:
        """root-adjacent node  →  list of all nodes in its subtree."""
        return {rc: self.subtree_of(rc) for rc in self.root_adjacent()}

    def leaf_nodes(self) -> List[int]:
        """Nodes with no children (excluding root)."""
        cnt = [0] * self.n
        for i in range(self.n):
            if i != self.root:
                cnt[self.parent[i]] += 1
        return [i for i in range(self.n) if i != self.root and cnt[i] == 0]

    # ── export helpers ────────────────────────────────────────────────────────

    def edges(self) -> List[Tuple[int, int]]:
        return [(v, self.parent[v]) for v in range(self.n) if v != self.root]

    def edge_set(self) -> Set[Tuple[int, int]]:
        return {
            (min(v, self.parent[v]), max(v, self.parent[v]))
            for v in range(self.n) if v != self.root
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OBJECTIVES & FEASIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def f_cost(sol: Solution, inst: Instance) -> float:
    """f1: sum of all edge distances in the spanning tree."""
    return float(sum(
        inst.dist[v][sol.parent[v]]
        for v in range(inst.n) if v != inst.root
    ))


def f_balance(sol: Solution, inst: Instance) -> float:
    """f2: max subtree demand (lower → better load balance)."""
    st = sol.subtrees()
    return max(float(inst.demands[nodes].sum()) for nodes in st.values()) if st else 0.0


def objectives(sol: Solution, inst: Instance) -> Tuple[float, float]:
    return f_cost(sol, inst), f_balance(sol, inst)


def scalar_obj(
    c: float, b: float, alpha: float, inst: Instance
) -> float:
    """
    Normalised weighted sum.
      f1 normalised by  max_dist × n_customers  (approx. star at max dist)
      f2 normalised by  capacity
    """
    c_norm = c / (float(inst.dist.max()) * inst.n_customers + 1e-9)
    b_norm = b / (inst.capacity + 1e-9)
    return alpha * c_norm + (1.0 - alpha) * b_norm


def is_feasible(sol: Solution, inst: Instance) -> bool:
    return all(
        float(inst.demands[nodes].sum()) <= inst.capacity + 1e-9
        for nodes in sol.subtrees().values()
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SUBTREE CACHE
# (enables O(1) feasibility and O(n_subtrees) delta-balance in local search)
# ─────────────────────────────────────────────────────────────────────────────

class SubtreeCache:
    """
    Incrementally maintained subtree-demand index.

    sub_dem[rc]  = total demand of the subtree rooted at rc (root-adjacent)
    node2sub[v]  = root-adjacent ancestor of v

    After a *leaf relocation* (v: old_par → new_par), update in O(1)
    by calling move_leaf(...).
    """

    def __init__(self, sol: Solution, inst: Instance):
        self.sub_dem : Dict[int, float] = {}
        self.node2sub: Dict[int, int]   = {}
        for rc, nodes in sol.subtrees().items():
            self.sub_dem[rc] = float(inst.demands[nodes].sum())
            for v in nodes:
                self.node2sub[v] = rc

    # ── queries ───────────────────────────────────────────────────────────────

    def max_load(self) -> float:
        return max(self.sub_dem.values()) if self.sub_dem else 0.0

    def feasible_move(
        self, v: int, new_par: int, sol: Solution, inst: Instance
    ) -> bool:
        """O(1) capacity check for moving leaf v to new_par."""
        d    = float(inst.demands[v])
        root = sol.root
        if new_par == root:
            # v would start a fresh single-node subtree
            return d <= inst.capacity + 1e-9
        new_sub = self.node2sub.get(new_par)
        if new_sub is None:
            return False
        if new_sub == self.node2sub.get(v):
            return True                   # same subtree — demand unchanged
        return self.sub_dem.get(new_sub, 0.0) + d <= inst.capacity + 1e-9

    def delta_balance(
        self, v: int, new_par: int, sol: Solution, inst: Instance
    ) -> float:
        """Max subtree load if we move leaf v to new_par (not committed yet)."""
        old_sub = self.node2sub.get(v)
        root    = sol.root

        if new_par == root:
            new_sub = v
        else:
            new_sub = self.node2sub.get(new_par)

        # Same subtree → no demand change
        if new_sub == old_sub:
            return self.max_load()

        d           = float(inst.demands[v])
        old_sub_dem = (self.sub_dem.get(old_sub, 0.0) - d) if old_sub is not None else 0.0
        new_sub_dem = (d if new_sub not in self.sub_dem
                       else self.sub_dem[new_sub] + d)

        max_load = 0.0
        for sub, dem in self.sub_dem.items():
            if   sub == old_sub: max_load = max(max_load, old_sub_dem)
            elif sub == new_sub: max_load = max(max_load, new_sub_dem)
            else:                max_load = max(max_load, dem)
        if new_sub not in self.sub_dem:          # brand-new subtree
            max_load = max(max_load, new_sub_dem)

        return max_load

    # ── update ────────────────────────────────────────────────────────────────

    def move_leaf(
        self, v: int, old_par: int, new_par: int,
        sol: Solution, inst: Instance
    ):
        """O(1) cache update after committing leaf v: old_par → new_par."""
        d       = float(inst.demands[v])
        root    = sol.root
        old_sub = self.node2sub.get(v)
        new_sub = v if new_par == root else self.node2sub.get(new_par)

        # Remove from old subtree
        if old_sub is not None:
            self.sub_dem[old_sub] -= d
            if self.sub_dem[old_sub] < 1e-9:
                del self.sub_dem[old_sub]

        # Add to new subtree
        if new_sub == v:                          # fresh root-adjacent subtree
            self.sub_dem[v] = d
        else:
            self.sub_dem[new_sub] = self.sub_dem.get(new_sub, 0.0) + d

        self.node2sub[v] = new_sub


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CONSTRUCTION (MODIFIED PRIM)
# ─────────────────────────────────────────────────────────────────────────────

def prim_construct(
    inst: Instance,
    noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Solution:
    """
    Modified Prim's algorithm for CMST.

    At each step add the cheapest feasible edge  (tree node → unvisited node)
    that does not violate the subtree capacity constraint.

    noise > 0  injects uniform noise on edge scores for GRASP-style
    randomisation (scale = noise × max_dist × 0.1).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, root, Q = inst.n, inst.root, inst.capacity
    sol = Solution(n, root)

    in_tree : Set[int]         = {root}
    sub_load: Dict[int, float] = {}   # root-adjacent node → cumulative demand
    node_sub: Dict[int, int]   = {}   # node → its root-adjacent ancestor

    while len(in_tree) < n:
        best_score, best_move = float("inf"), None

        for u in list(in_tree):
            for v in range(n):
                if v in in_tree:
                    continue
                d = float(inst.demands[v])
                if u == root:
                    if d > Q + 1e-9:
                        continue
                    sub_v = v                     # v starts a new subtree
                else:
                    sr = node_sub[u]
                    if sub_load[sr] + d > Q + 1e-9:
                        continue
                    sub_v = sr                    # v joins u's subtree

                score = float(inst.dist[u][v])
                if noise > 0.0:
                    score += noise * rng.uniform(-1.0, 1.0) * float(inst.dist.max()) * 0.1

                if score < best_score:
                    best_score = score
                    best_move  = (u, v, sub_v)

        if best_move is None:
            # Safety fallback: attach remaining nodes directly to root
            for v in range(n):
                if v not in in_tree:
                    sol.parent[v] = root
                    node_sub[v]   = v
                    sub_load[v]   = float(inst.demands[v])
                    in_tree.add(v)
            break

        u, v, sub_v = best_move
        sol.parent[v] = u
        in_tree.add(v)
        node_sub[v]  = sub_v
        if u == root:
            sub_load[sub_v] = float(inst.demands[v])
        else:
            sub_load[sub_v] += float(inst.demands[v])

    return sol


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — LOCAL SEARCH  (best-improvement leaf relocation)
# ─────────────────────────────────────────────────────────────────────────────

def ls_leaf_relocate(
    sol: Solution, inst: Instance, alpha: float
) -> Solution:
    """
    Best-improvement leaf relocation local search.

    For each leaf v (in random order), find the parent p* that minimises
    the scalarised objective; commit the move if it improves the incumbent.
    Repeat until no leaf can be profitably relocated.

    Complexity per pass : O(n_leaves × n × n_subtrees)
                        ≈ O(n²·√n)  in practice
    Uses SubtreeCache for O(1) feasibility and O(n_sub) balance delta.
    """
    best  = sol.copy()
    cache = SubtreeCache(best, inst)

    cur_cost = f_cost(best, inst)
    cur_bal  = cache.max_load()
    max_dist = float(inst.dist.max())
    n_cust   = inst.n_customers
    Q        = inst.capacity

    def obj(c: float, b: float) -> float:
        return (alpha * (c / (max_dist * n_cust + 1e-9)) +
                (1.0 - alpha) * (b / (Q + 1e-9)))

    improved = True
    while improved:
        improved = False
        leaves   = best.leaf_nodes()
        random.shuffle(leaves)

        for v in leaves:
            old_par  = best.parent[v]
            cur_val  = obj(cur_cost, cur_bal)
            old_dist = float(inst.dist[v][old_par])

            best_par, best_val = old_par, cur_val

            for p in range(inst.n):
                if p == v or p == old_par:
                    continue
                if not cache.feasible_move(v, p, best, inst):
                    continue

                new_cost = cur_cost + float(inst.dist[v][p]) - old_dist
                new_bal  = cache.delta_balance(v, p, best, inst)
                val      = obj(new_cost, new_bal)

                if val < best_val - 1e-9:
                    best_val, best_par = val, p

            if best_par != old_par:
                cache.move_leaf(v, old_par, best_par, best, inst)
                best.parent[v] = best_par
                cur_cost += float(inst.dist[v][best_par]) - old_dist
                cur_bal   = cache.max_load()
                improved  = True
                break          # restart scan with updated leaf list

    return best


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PERTURBATION
# ─────────────────────────────────────────────────────────────────────────────

def perturb(
    sol: Solution,
    inst: Instance,
    strength: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Solution:
    """
    Random leaf relocation for `strength` randomly chosen leaves.
    Infeasible moves are rejected (leaf stays put).
    Provides the ILS diversification kick.
    """
    if rng is None:
        rng = np.random.default_rng()

    s     = sol.copy()
    cache = SubtreeCache(s, inst)

    for _ in range(strength):
        leaves = s.leaf_nodes()
        if not leaves:
            break
        v       = int(rng.choice(leaves))
        old_par = s.parent[v]
        cands   = [p for p in range(inst.n) if p != v and p != old_par]
        rng.shuffle(cands)

        for p in cands:
            if cache.feasible_move(v, p, s, inst):
                cache.move_leaf(v, old_par, p, s, inst)
                s.parent[v] = p
                break

    return s


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — ILS  (single alpha, scalarised)
# ─────────────────────────────────────────────────────────────────────────────

def ils(
    inst     : Instance,
    alpha    : float,
    n_iter   : int = 100,
    strength : int = 3,
    seed     : Optional[int] = None,
) -> Solution:
    """
    Iterated Local Search for the scalarised MO-CMST.

    Pipeline per iteration
      1. Perturb current solution (random leaf relocation)
      2. Local search (best-improvement leaf relocation)
      3. Update best if improved  (strict-descent acceptance)
      4. Restart from best

    Parameters
    ----------
    alpha    : weight on f1 (cost);  1-alpha on f2 (balance)
    n_iter   : number of ILS iterations
    strength : perturbation strength (leaves relocated per kick)
    seed     : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    random.seed(int(rng.integers(0, 2**31)))

    # ── initialisation ────────────────────────────────────────────────────
    best     = prim_construct(inst, noise=0.15, rng=rng)
    best     = ls_leaf_relocate(best, inst, alpha)
    best_val = scalar_obj(*objectives(best, inst), alpha, inst)
    current  = best.copy()

    # ── main loop ─────────────────────────────────────────────────────────
    for _ in range(n_iter):
        candidate     = perturb(current, inst, strength=strength, rng=rng)
        candidate     = ls_leaf_relocate(candidate, inst, alpha)
        candidate_val = scalar_obj(*objectives(candidate, inst), alpha, inst)

        if candidate_val < best_val - 1e-9:
            best, best_val = candidate.copy(), candidate_val

        current = best.copy()      # restart from best (strict descent)

    return best


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — PARETO DOMINANCE
# ─────────────────────────────────────────────────────────────────────────────

def dominates(
    a: Tuple[float, float], b: Tuple[float, float]
) -> bool:
    """a dominates b ⟺ a ≤ b component-wise and a < b in ≥1 component."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def pareto_filter(
    pool: List[Tuple[Solution, Tuple[float, float]]]
) -> List[Tuple[Solution, Tuple[float, float]]]:
    """Keep only non-dominated solutions from pool."""
    return [
        (s, o) for s, o in pool
        if not any(dominates(o2, o) for _, o2 in pool)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — MO-ILS  (multi-alpha sweep → Pareto front)
# ─────────────────────────────────────────────────────────────────────────────

def mo_ils(
    inst    : Instance,
    n_alphas: int = 7,
    n_iter  : int = 100,
    strength: int = 3,
    seed    : Optional[int] = None,
) -> List[dict]:
    """
    Run ILS independently for each alpha ∈ linspace(0, 1, n_alphas),
    then return the Pareto-optimal subset.

    Returns
    -------
    List of dicts, one per Pareto-optimal solution:
      {parent, edges, edge_set, cost, balance}
    """
    alphas = np.linspace(0.0, 1.0, n_alphas)
    pool  : List[Tuple[Solution, Tuple[float, float]]] = []

    for i, alpha in enumerate(alphas):
        s   = ils(
            inst, float(alpha), n_iter=n_iter, strength=strength,
            seed=(seed + i * 7) if seed is not None else None,
        )
        pool.append((s, objectives(s, inst)))

    front = pareto_filter(pool)

    return [
        {
            "parent"  : s.parent[:],
            "edges"   : s.edges(),
            "edge_set": s.edge_set(),
            "cost"    : c,
            "balance" : b,
        }
        for s, (c, b) in front
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — TRAINING SAMPLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_sample(inst: Instance, pareto_sols: List[dict]) -> dict:
    """
    Convert one instance + its Pareto solutions into a GNN-ready sample.

    Node features  (n, 4)
      [x / 100,  y / 100,  demand / capacity,  is_root]

    Edge index  (E, 2)
      Complete undirected graph, upper triangle only  (i < j)

    Edge features  (E, 1)
      [dist / max_dist]

    Edge labels
      hard  (E,)  — 1 if edge is in ≥1 Pareto solution
      soft  (E,)  — fraction of Pareto solutions containing the edge
                    (useful for weighted / soft-label training)

    Pareto solutions stored verbatim for later inspection / replay.
    """
    n        = inst.n
    max_dist = float(inst.dist.max())
    n_pareto = len(pareto_sols)

    # ── node features ─────────────────────────────────────────────────────
    is_root = np.zeros(n); is_root[inst.root] = 1.0
    node_features = np.column_stack([
        inst.coords[:, 0] / 100.0,
        inst.coords[:, 1] / 100.0,
        inst.demands       / (inst.capacity + 1e-9),
        is_root,
    ]).tolist()                            # Python list (n, 4)

    # ── union edge index + per-edge count ─────────────────────────────────
    union_edges : Set[Tuple[int, int]] = set()
    edge_count  : Dict[Tuple[int, int], int] = {}

    for s in pareto_sols:
        for e in s["edge_set"]:
            union_edges.add(e)
            edge_count[e] = edge_count.get(e, 0) + 1

    # ── build edge tensors (upper-triangle complete graph) ─────────────────
    edge_index       : List[List[int]]    = []
    edge_features    : List[List[float]]  = []
    edge_labels_hard : List[int]          = []
    edge_labels_soft : List[float]        = []

    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            edge_index.append([i, j])
            edge_features.append([inst.dist[i][j] / max_dist])
            in_p = key in union_edges
            edge_labels_hard.append(1 if in_p else 0)
            edge_labels_soft.append(edge_count.get(key, 0) / (n_pareto + 1e-9))

    return {
        "instance"         : inst.to_dict(),
        "node_features"    : node_features,        # (n, 4)
        "edge_index"       : edge_index,            # (E, 2)
        "edge_features"    : edge_features,         # (E, 1)
        "edge_labels_hard" : edge_labels_hard,      # (E,)  binary
        "edge_labels_soft" : edge_labels_soft,      # (E,)  [0, 1]
        "n_pareto"         : n_pareto,
        "pareto_solutions" : [
            {"parent" : s["parent"],
             "cost"   : s["cost"],
             "balance": s["balance"]}
            for s in pareto_sols
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — SOLUTION VALIDATOR  (optional debug utility)
# ─────────────────────────────────────────────────────────────────────────────

def validate_solution(sol: Solution, inst: Instance) -> bool:
    """
    Verify two invariants:
      1. spanning tree  — BFS from root visits all n nodes
      2. feasibility    — no subtree exceeds capacity
    """
    # Build children index
    children: Dict[int, List[int]] = {i: [] for i in range(inst.n)}
    for v in range(inst.n):
        if v != inst.root:
            children[sol.parent[v]].append(v)

    # BFS from root
    visited = {inst.root}
    queue   = [inst.root]
    while queue:
        cur = queue.pop()
        for c in children[cur]:
            if c in visited:
                return False   # cycle
            visited.add(c)
            queue.append(c)

    if len(visited) != inst.n:
        return False           # not spanning

    return is_feasible(sol, inst)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — PARALLEL WORKER  (top-level so it can be pickled)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict:
    """
    Process a single instance: generate → MO-ILS → build sample → validate.
    Runs in a subprocess when n_workers > 1.

    Parameters  (packed as a tuple for ProcessPoolExecutor.map compatibility)
    ----------
    idx              : instance index (deterministic seed offset)
    n_cust           : number of customers
    capacity_factor  : passed to Instance.random
    n_alphas         : ILS alpha sweep
    ils_iter         : ILS iterations per alpha
    perturb_strength : ILS perturbation strength
    seed             : base seed
    """
    (idx, n_cust, capacity_factor, n_alphas,
     ils_iter, perturb_strength, seed) = args

    inst_seed = seed + idx * 13
    inst      = Instance.random(n_cust, capacity_factor=capacity_factor,
                                seed=inst_seed)
    t0         = time.time()
    pareto_sols = mo_ils(inst, n_alphas=n_alphas, n_iter=ils_iter,
                         strength=perturb_strength, seed=inst_seed)
    elapsed    = time.time() - t0
    sample     = build_sample(inst, pareto_sols)

    # validate every Pareto solution
    for ps in pareto_sols:
        s = Solution(inst.n, inst.root)
        s.parent = ps["parent"]
        if not validate_solution(s, inst):
            raise RuntimeError(f"Invalid solution at instance {idx}")

    # attach metadata used by the pipeline for live stats
    sample["_meta"] = {
        "idx"    : idx,
        "n"      : inst.n,
        "cap"    : inst.capacity,
        "pareto" : len(pareto_sols),
        "time"   : elapsed,
    }
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    samples    : List[dict],
    output_dir : str,
    label      : str,
    lock       : threading.Lock,
) -> None:
    """
    Thread-safe checkpoint save.
    Writes  checkpoints/ckpt_<label>.pkl  and updates  summary.json.
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    pkl_path  = os.path.join(ckpt_dir, f"ckpt_{label}.pkl")
    json_path = os.path.join(output_dir, "summary.json")

    with lock:
        with open(pkl_path, "wb") as fh:
            pickle.dump(samples, fh)

        summary = [
            {
                "id"      : i,
                "n"       : s["instance"]["n"],
                "capacity": s["instance"]["capacity"],
                "n_pareto": s["n_pareto"],
                "costs"   : [p["cost"]    for p in s["pareto_solutions"]],
                "balances": [p["balance"] for p in s["pareto_solutions"]],
            }
            for i, s in enumerate(samples)
        ]
        with open(json_path, "w") as fh:
            json.dump(summary, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(**kw) -> List[dict]:
    """
    Generate a complete MO-CMST dataset with multiprocessing + progress bar.

    Extra keyword arguments (vs previous version)
    ----------------------------------------------
    n_workers  : int   — parallel worker processes (1 = serial, default = cpu_count)
    save_every : int   — write a checkpoint every N completed instances

    Keyword arguments override defaults in CFG.
    Returns list of training samples (also saved as .pkl + .json).
    """
    cfg = {**CFG, **kw}
    os.makedirs(cfg["output_dir"], exist_ok=True)
    rng       = random.Random(cfg["seed"])
    save_lock = threading.Lock()

    n_workers  : int = max(1, int(cfg["n_workers"]))
    save_every : int = max(1, int(cfg["save_every"]))

    # ── pre-generate per-instance parameters (deterministic order) ───────
    task_args: List[tuple] = []
    for idx in range(cfg["n_instances"]):
        n_cust = rng.randint(cfg["min_customers"], cfg["max_customers"])
        task_args.append((
            idx,
            n_cust,
            cfg["capacity_factor"],
            cfg["n_alphas"],
            cfg["ils_iter"],
            cfg["perturb_strength"],
            cfg["seed"],
        ))

    # ── header ────────────────────────────────────────────────────────────
    alphas_str = ", ".join(
        f"{a:.2f}" for a in np.linspace(0, 1, cfg["n_alphas"])
    )
    print(f"\n{'═'*66}")
    print(f"  MO-CMST  ·  ILS Data Generator  ·  Prototype")
    print(f"{'═'*66}")
    col = 24
    for k, v in cfg.items():
        print(f"  {k:<{col}}: {v}")
    print(f"  {'alphas':<{col}}: [{alphas_str}]")
    print(f"{'─'*66}\n")

    # ── run ───────────────────────────────────────────────────────────────
    samples     : List[dict]  = []
    n_total                    = cfg["n_instances"]
    t_start                    = time.time()
    last_checkpoint            = 0          # count of instances at last save

    bar_fmt = (
        "  {l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}]"
    )

    # tqdm writes to stderr by default → compatible with print to stdout
    with tqdm(
        total      = n_total,
        desc       = "  Generating",
        unit       = "inst",
        bar_format = bar_fmt,
        colour     = "cyan",
        dynamic_ncols = True,
    ) as pbar:

        def _on_done(sample: dict) -> None:
            """Called (in main process) each time a worker finishes."""
            nonlocal last_checkpoint
            m = sample.pop("_meta")      # remove internal metadata
            samples.append(sample)

            n_done = len(samples)
            pbar.set_postfix(
                n   = m["n"],
                par = m["pareto"],
                t   = f"{m['time']:.2f}s",
                wkr = n_workers,
            )
            pbar.update(1)

            # ── periodic checkpoint ───────────────────────────────────
            if n_done - last_checkpoint >= save_every:
                label          = f"{n_done:05d}"
                last_checkpoint = n_done
                _save_checkpoint(samples, cfg["output_dir"], label, save_lock)
                tqdm.write(
                    f"  ✓ checkpoint  [{n_done}/{n_total}]  "
                    f"→ checkpoints/ckpt_{label}.pkl"
                )

        if n_workers == 1:
            # ── serial path (easier to debug) ─────────────────────────
            for args in task_args:
                _on_done(_worker(args))
        else:
            # ── parallel path ─────────────────────────────────────────
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker, a): a[0] for a in task_args}
                for fut in as_completed(futures):
                    try:
                        _on_done(fut.result())
                    except Exception as exc:
                        idx = futures[fut]
                        tqdm.write(f"  ✗ instance {idx} failed: {exc}")

    # ── final save ────────────────────────────────────────────────────────
    final_pkl  = os.path.join(cfg["output_dir"], "dataset.pkl")
    final_json = os.path.join(cfg["output_dir"], "summary.json")

    with open(final_pkl, "wb") as fh:
        pickle.dump(samples, fh)

    summary = [
        {
            "id"      : i,
            "n"       : s["instance"]["n"],
            "capacity": s["instance"]["capacity"],
            "n_pareto": s["n_pareto"],
            "costs"   : [p["cost"]    for p in s["pareto_solutions"]],
            "balances": [p["balance"] for p in s["pareto_solutions"]],
        }
        for i, s in enumerate(samples)
    ]
    with open(final_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── footer ────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    n_ok          = len(samples)
    avg_pareto    = sum(s["n_pareto"] for s in samples) / max(n_ok, 1)

    print(f"\n{'─'*66}")
    print(f"  Instances   : {n_ok}/{n_total}")
    print(f"  Avg |Pareto|: {avg_pareto:.2f}")
    print(f"  Total time  : {elapsed_total:.1f}s  "
          f"({elapsed_total/max(n_ok,1):.2f}s / instance)")
    print(f"  Workers     : {n_workers}")
    print(f"  Final pkl   : {final_pkl}")
    print(f"  Final json  : {final_json}")
    print(f"\n  Load data:")
    print(f"    import pickle")
    print(f"    data = pickle.load(open('{final_pkl}', 'rb'))")
    print(f"    # keys: {list(samples[0].keys()) if samples else '[]'}")
    print(f"{'═'*66}\n")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MO-CMST ILS — generate GNN training data"
    )
    p.add_argument("--instances",        type=int,   default=CFG["n_instances"],
                   help="number of instances to generate")
    p.add_argument("--min-customers",    type=int,   default=CFG["min_customers"])
    p.add_argument("--max-customers",    type=int,   default=CFG["max_customers"])
    p.add_argument("--capacity-factor",  type=float, default=CFG["capacity_factor"],
                   help="capacity = max(max_demand, factor × total_demand)")
    p.add_argument("--alphas",           type=int,   default=CFG["n_alphas"],
                   help="number of alpha values in [0,1]")
    p.add_argument("--ils-iter",         type=int,   default=CFG["ils_iter"],
                   help="ILS iterations per alpha")
    p.add_argument("--perturb-strength", type=int,   default=CFG["perturb_strength"],
                   help="leaves relocated per perturbation kick")
    p.add_argument("--output",           type=str,   default=CFG["output_dir"])
    p.add_argument("--seed",             type=int,   default=CFG["seed"])
    p.add_argument("--workers",          type=int,   default=CFG["n_workers"],
                   help="parallel worker processes (1 = serial)")
    p.add_argument("--save-every",       type=int,   default=CFG["save_every"],
                   help="checkpoint every N completed instances")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        n_instances      = args.instances,
        min_customers    = args.min_customers,
        max_customers    = args.max_customers,
        capacity_factor  = args.capacity_factor,
        n_alphas         = args.alphas,
        ils_iter         = args.ils_iter,
        perturb_strength = args.perturb_strength,
        output_dir       = args.output,
        seed             = args.seed,
        n_workers        = args.workers,
        save_every       = args.save_every,
    )