"""
mo_cmst_gnn.py
==============
Multi-Objective Capacitated Minimum Spanning Tree — GNN Module

Architecture
  GraphSAGE (2 layers)  →  Graph Attention (1 layer)  →  Node Embeddings
  Edge Scoring MLP  (embed_i ⊕ embed_j ⊕ dist ⊕ alpha)  →  P(edge ∈ tree)

Training modes
  • Supervised   — BCE against hard/soft labels from ILS dataset
  • Alpha-conditioned — alpha feature injected at edge scoring;
                        model learns the full Pareto spectrum in one pass

Usage
  # Train
  python mo_cmst_gnn.py --mode train --data mo_cmst_data/dataset.pkl

  # Evaluate (GNN Pareto vs ILS Pareto)
  python mo_cmst_gnn.py --mode eval  --data mo_cmst_data/dataset.pkl \\
                         --checkpoint runs/best.pt

  # Inference on one instance index
  python mo_cmst_gnn.py --mode infer --data mo_cmst_data/dataset.pkl \\
                         --checkpoint runs/best.pt --instance-idx 0

Dependencies
  pip install torch torch_geometric tqdm matplotlib numpy
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import math
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                    # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, SAGEConv
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

CFG: dict = dict(
    # data
    data_path       = "mo_cmst_data/dataset.pkl",
    val_split       = 0.15,          # fraction of dataset used for validation
    test_split      = 0.10,

    # model
    node_in_dim     = 4,             # [x, y, demand, is_root]
    edge_in_dim     = 1,             # [dist_norm]
    hidden_dim      = 64,
    n_sage_layers   = 2,
    gat_heads       = 4,             # multi-head attention
    dropout         = 0.1,
    use_soft_labels = True,          # soft (fraction) vs hard (0/1) BCE target

    # training
    epochs          = 100,
    batch_size      = 32,
    lr              = 3e-4,
    weight_decay    = 1e-5,
    pos_weight      = 5.0,           # BCE class imbalance (few edges in tree)
    n_alphas_train  = 5,             # alpha samples per graph during training
    save_every      = 10,            # save checkpoint every N epochs
    runs_dir        = "runs",

    # inference / eval
    top_k_factor    = 3.0,           # keep top (top_k_factor * n_nodes) edges
    n_alphas_eval   = 11,            # alpha sweep for Pareto approximation
    ls_iter         = 80,            # local search passes per ILS iteration
    ils_iter        = 50,            # ILS iterations after GNN warm-start
    ils_strength    = 3,             # leaves relocated per perturbation kick
    n_workers       = max(1, (os.cpu_count() or 1)),  # parallel ILS workers

    seed            = 42,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATASET  (pkl → torch_geometric Data objects)
# ─────────────────────────────────────────────────────────────────────────────

def sample_to_pyg(sample: dict, alpha: Optional[float] = None) -> Data:
    """
    Convert one pipeline sample dict → torch_geometric Data.

    Node features  x      : (n, 4)   [x_norm, y_norm, demand_norm, is_root]
    Edge index     ei     : (2, E)   upper-triangle undirected
    Edge attr      ea     : (E, 2)   [dist_norm, alpha]   ← alpha injected here
    Labels         y_hard : (E,)     binary
                   y_soft : (E,)     [0,1]  fraction of Pareto solutions

    If alpha is None a random value in [0,1] is sampled (training augmentation).
    """
    a = float(np.random.uniform(0.0, 1.0)) if alpha is None else float(alpha)

    x      = torch.tensor(sample["node_features"], dtype=torch.float)   # (n, 4)
    ei_np  = np.array(sample["edge_index"], dtype=np.int64)              # (E, 2)
    ef_np  = np.array(sample["edge_features"], dtype=np.float32)        # (E, 1)
    yh_np  = np.array(sample["edge_labels_hard"], dtype=np.float32)     # (E,)
    ys_np  = np.array(sample["edge_labels_soft"], dtype=np.float32)     # (E,)

    E      = ef_np.shape[0]
    alpha_col = np.full((E, 1), a, dtype=np.float32)
    edge_attr = torch.tensor(
        np.concatenate([ef_np, alpha_col], axis=1), dtype=torch.float
    )                                                                    # (E, 2)

    # PyG convention: edge_index shape (2, E), both directions
    src = torch.tensor(ei_np[:, 0], dtype=torch.long)
    dst = torch.tensor(ei_np[:, 1], dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ], dim=0)                                                            # (2, 2E)

    # For edge scoring we only score upper-triangle edges → keep original E
    # Store undirected edge_index for message passing, upper-tri for scoring
    edge_attr_bi = torch.cat([edge_attr, edge_attr], dim=0)             # (2E, 2)

    data = Data(
        x           = x,
        edge_index  = edge_index,
        edge_attr   = edge_attr_bi,
        y_hard      = torch.tensor(yh_np, dtype=torch.float),
        y_soft      = torch.tensor(ys_np, dtype=torch.float),
        # keep upper-tri indices for scoring head
        edge_index_score = torch.stack([src, dst], dim=0),              # (2, E)
        edge_attr_score  = edge_attr,                                   # (E, 2)
        alpha            = torch.tensor([a], dtype=torch.float),
        n_nodes          = torch.tensor([x.shape[0]], dtype=torch.long),
    )
    return data


class MOCMSTDataset(torch.utils.data.Dataset):
    """
    Wraps the pkl dataset.  Each call to __getitem__ samples a fresh random
    alpha (training augmentation) so the model sees every graph at many
    different trade-off points across epochs.
    """

    def __init__(self, samples: List[dict], random_alpha: bool = True):
        self.samples      = samples
        self.random_alpha = random_alpha

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        alpha = None if self.random_alpha else 0.5
        return sample_to_pyg(self.samples[idx], alpha=alpha)


def load_splits(
    data_path : str,
    val_split : float = 0.15,
    test_split: float = 0.10,
    seed      : int   = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Load pkl and split deterministically into train / val / test."""
    with open(data_path, "rb") as fh:
        samples = pickle.load(fh)

    rng = random.Random(seed)
    rng.shuffle(samples)

    n   = len(samples)
    nv  = max(1, int(n * val_split))
    nt  = max(1, int(n * test_split))

    test  = samples[:nt]
    val   = samples[nt: nt + nv]
    train = samples[nt + nv:]

    print(f"  Dataset  train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL
# ─────────────────────────────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """
    GraphSAGE (2 layers) + GAT (1 layer) → node embeddings.

    Input node features  : (n, node_in_dim)
    Input edge features  : (E, edge_in_dim + 1)  — dist + alpha
    Output               : (n, hidden_dim)

    Architecture rationale
    ----------------------
    • SAGEConv aggregates neighbourhood structure efficiently (mean aggr).
    • GATConv adds attention-weighted refinement — learns which neighbours
      matter most, which is crucial for capacity-constrained problems.
    • Both are well-cited and directly applicable to combinatorial optimisation
      on graphs (cf. Kool et al. 2019, Joshi et al. 2020).
    """

    def __init__(
        self,
        node_in_dim : int,
        hidden_dim  : int,
        n_sage      : int = 2,
        gat_heads   : int = 4,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)

        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        for _ in range(n_sage):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))

        # GAT layer (multi-head, then project back to hidden_dim)
        gat_out = hidden_dim // gat_heads
        assert gat_out * gat_heads == hidden_dim, (
            f"hidden_dim ({hidden_dim}) must be divisible by gat_heads ({gat_heads})"
        )
        self.gat = GATConv(
            hidden_dim, gat_out,
            heads=gat_heads, dropout=dropout, concat=True
        )

        # Layer norms
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_sage + 1)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))

        for i, sage in enumerate(self.sage_layers):
            h = h + F.relu(self.norms[i](sage(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT layer (residual)
        h = h + F.relu(self.norms[-1](self.gat(h, edge_index)))
        return h                                                # (n, hidden_dim)


class EdgeScoringMLP(nn.Module):
    """
    For each candidate edge (i, j) score P(edge ∈ tree | alpha).

    Input  : [embed_i ‖ embed_j ‖ dist_norm ‖ alpha]   dim = 2*H + 2
    Output : scalar logit  (apply sigmoid for probability)

    The alpha feature makes the network Pareto-aware: trained with random
    alpha augmentation it learns to shift its predictions continuously
    across the cost–balance trade-off spectrum.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        in_dim = hidden_dim * 2 + 2      # embed_i + embed_j + dist + alpha
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_embed     : torch.Tensor,   # (n, H)
        edge_index_score: torch.Tensor,  # (2, E)  upper-tri
        edge_attr_score : torch.Tensor,  # (E, 2)  [dist, alpha]
    ) -> torch.Tensor:                   # (E,) logits
        i_idx, j_idx = edge_index_score[0], edge_index_score[1]
        e_in = torch.cat([
            node_embed[i_idx],           # (E, H)
            node_embed[j_idx],           # (E, H)
            edge_attr_score,             # (E, 2)
        ], dim=-1)                       # (E, 2H+2)
        return self.mlp(e_in).squeeze(-1)                       # (E,)


class MOCMSTNet(nn.Module):
    """
    Full GNN: NodeEncoder → EdgeScoringMLP.

    Returns
    -------
    logits : (E,)   raw scores for each upper-triangle edge
    probs  : (E,)   sigmoid(logits)
    """

    def __init__(
        self,
        node_in_dim : int   = 4,
        hidden_dim  : int   = 64,
        n_sage      : int   = 2,
        gat_heads   : int   = 4,
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.encoder = NodeEncoder(
            node_in_dim, hidden_dim, n_sage, gat_heads, dropout
        )
        self.scorer  = EdgeScoringMLP(hidden_dim, dropout)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embed = self.encoder(data.x, data.edge_index)
        logits     = self.scorer(
            node_embed, data.edge_index_score, data.edge_attr_score
        )
        return logits, torch.sigmoid(logits)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — LOSS
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits        : torch.Tensor,    # (E,)
    data          : Data,
    pos_weight    : float = 5.0,
    use_soft      : bool  = True,
) -> torch.Tensor:
    """
    Weighted BCE loss.

    Soft labels (fraction of Pareto solutions using each edge) provide a
    smoother training signal than hard 0/1 labels — edges that appear in
    many solutions get higher target values, edges appearing rarely get
    intermediate values.  This implicitly encodes solution diversity.

    pos_weight > 1 compensates for class imbalance: in a spanning tree of
    n nodes only (n-1) edges are selected out of n(n-1)/2 candidates.
    """
    targets = data.y_soft if use_soft else data.y_hard
    pw      = torch.tensor([pos_weight], device=logits.device)
    loss    = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=pw,
    )
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def edge_metrics(
    probs  : torch.Tensor,
    data   : Data,
    thresh : float = 0.5,
) -> dict:
    """
    Precision / Recall / F1 on edge prediction (hard labels).

    A predicted edge is considered relevant if its probability ≥ thresh.
    """
    pred = (probs >= thresh).float()
    gt   = (data.y_hard >= 0.5).float()

    tp   = (pred * gt).sum().item()
    fp   = (pred * (1 - gt)).sum().item()
    fn   = ((1 - pred) * gt).sum().item()

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TRAIN / VALIDATE ONE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model      : MOCMSTNet,
    loader     : DataLoader,
    optimiser  : torch.optim.Optimizer,
    cfg        : dict,
    device     : torch.device,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for data in loader:
        data = data.to(device)
        optimiser.zero_grad()
        logits, _ = model(data)
        loss = compute_loss(
            logits, data,
            pos_weight = cfg["pos_weight"],
            use_soft   = cfg["use_soft_labels"],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(
    model  : MOCMSTNet,
    loader : DataLoader,
    cfg    : dict,
    device : torch.device,
) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_prec, all_rec, all_f1 = [], [], []
    n_batches = 0

    for data in loader:
        data = data.to(device)
        logits, probs = model(data)
        loss = compute_loss(
            logits, data,
            pos_weight = cfg["pos_weight"],
            use_soft   = cfg["use_soft_labels"],
        )
        total_loss += loss.item()
        n_batches  += 1
        m = edge_metrics(probs, data)
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
        all_f1.append(m["f1"])

    return total_loss / max(n_batches, 1), {
        "precision": float(np.mean(all_prec)),
        "recall"   : float(np.mean(all_rec)),
        "f1"       : float(np.mean(all_f1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FULL TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> MOCMSTNet:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── data ──────────────────────────────────────────────────────────────
    train_s, val_s, _ = load_splits(
        cfg["data_path"], cfg["val_split"], cfg["test_split"], cfg["seed"]
    )
    train_ds = MOCMSTDataset(train_s, random_alpha=True)
    val_ds   = MOCMSTDataset(val_s,   random_alpha=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        follow_batch=["y_hard", "y_soft"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        follow_batch=["y_hard", "y_soft"]
    )

    # ── model ─────────────────────────────────────────────────────────────
    model = MOCMSTNet(
        node_in_dim = cfg["node_in_dim"],
        hidden_dim  = cfg["hidden_dim"],
        n_sage      = cfg["n_sage_layers"],
        gat_heads   = cfg["gat_heads"],
        dropout     = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────
    optimiser = AdamW(
        model.parameters(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg["epochs"], eta_min=1e-6)

    os.makedirs(cfg["runs_dir"], exist_ok=True)
    best_val_loss = float("inf")
    best_path     = os.path.join(cfg["runs_dir"], "best.pt")

    # ── history ───────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "precision": [], "recall": [], "f1": [],
    }

    # ── loop ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Prec':>7}  "
          f"{'Rec':>7}  {'F1':>7}  {'LR':>9}")
    print(f"{'─'*68}")

    bar = tqdm(range(1, cfg["epochs"] + 1), desc="  Training",
               unit="ep", colour="green", leave=False)

    for epoch in bar:
        t_loss = train_epoch(model, train_loader, optimiser, cfg, device)
        v_loss, metrics = val_epoch(model, val_loader, cfg, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1"].append(metrics["f1"])

        lr_now = scheduler.get_last_lr()[0]
        bar.set_postfix(
            tr=f"{t_loss:.4f}", vl=f"{v_loss:.4f}", f1=f"{metrics['f1']:.3f}"
        )

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"  {epoch:>6}  {t_loss:>8.4f}  {v_loss:>8.4f}  "
                f"{metrics['precision']:>7.3f}  {metrics['recall']:>7.3f}  "
                f"{metrics['f1']:>7.3f}  {lr_now:>9.2e}"
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_model(model, cfg, best_path, epoch, metrics)
            tqdm.write(f"  ★ new best  epoch={epoch}  val_loss={v_loss:.4f}")

        if epoch % cfg["save_every"] == 0:
            ckpt = os.path.join(
                cfg["runs_dir"], f"ckpt_ep{epoch:04d}.pt"
            )
            save_model(model, cfg, ckpt, epoch, metrics)

    print(f"{'─'*68}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {best_path}")

    # ── loss curve ────────────────────────────────────────────────────────
    _plot_training(history, cfg["runs_dir"])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE / LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def save_model(
    model   : MOCMSTNet,
    cfg     : dict,
    path    : str,
    epoch   : int   = 0,
    metrics : dict  = None,
) -> None:
    torch.save({
        "epoch"      : epoch,
        "state_dict" : model.state_dict(),
        "cfg"        : cfg,
        "metrics"    : metrics or {},
    }, path)


def load_model(path: str, device: Optional[torch.device] = None) -> MOCMSTNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:                          # torch < 1.13 has no weights_only
        ckpt = torch.load(path, map_location=device)
    cfg   = ckpt["cfg"]
    model = MOCMSTNet(
        node_in_dim = cfg["node_in_dim"],
        hidden_dim  = cfg["hidden_dim"],
        n_sage      = cfg["n_sage_layers"],
        gat_heads   = cfg["gat_heads"],
        dropout     = cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded model from {path}  (epoch {epoch})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — INFERENCE  (instance → ranked edges)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    model   : MOCMSTNet,
    sample  : dict,
    alpha   : float,
    top_k   : Optional[int] = None,
    device  : Optional[torch.device] = None,
) -> List[dict]:
    """
    Run the GNN on one sample and return a ranked list of candidate edges.

    Parameters
    ----------
    sample  : one entry from dataset.pkl
    alpha   : trade-off weight (0 = balance only, 1 = cost only)
    top_k   : keep only top-k edges by probability (None = all)

    Returns
    -------
    List of dicts sorted by descending probability:
      { i, j, probability, dist_norm }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    data   = sample_to_pyg(sample, alpha=alpha).to(device)
    _, probs = model(data)
    probs    = probs.cpu().numpy()

    ei    = np.array(sample["edge_index"])    # (E, 2)
    ef    = np.array(sample["edge_features"]) # (E, 1)

    results = [
        {
            "i"          : int(ei[k, 0]),
            "j"          : int(ei[k, 1]),
            "probability": float(probs[k]),
            "dist_norm"  : float(ef[k, 0]),
        }
        for k in range(len(probs))
    ]
    results.sort(key=lambda r: r["probability"], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def print_ranked_edges(
    ranked : List[dict],
    sample : dict,
    alpha  : float,
    top_n  : int = 20,
) -> None:
    n = sample["instance"]["n"]
    print(f"\n  Instance  n={n}  |  alpha={alpha:.2f}")
    print(f"  {'Rank':>4}  {'(i,j)':>8}  {'P(edge)':>9}  {'dist_norm':>10}  {'in_Pareto':>10}")
    print(f"  {'─'*52}")
    edge_set = set(
        tuple(e) for s in sample["pareto_solutions"]
        for e in _parent_to_edge_set(s["parent"], n)
    )
    for rank, r in enumerate(ranked[:top_n], 1):
        key  = (min(r["i"], r["j"]), max(r["i"], r["j"]))
        flag = "✓" if key in edge_set else " "
        print(
            f"  {rank:>4}  ({r['i']:>2},{r['j']:>2})  "
            f"{r['probability']:>9.4f}  {r['dist_norm']:>10.4f}  {flag:>10}"
        )


def _parent_to_edge_set(parent: List[int], n: int) -> List[Tuple[int, int]]:
    root = 0
    return [
        (min(v, parent[v]), max(v, parent[v]))
        for v in range(n) if v != root
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — PARETO EVALUATION  (GNN vs ILS)
# ─────────────────────────────────────────────────────────────────────────────

def objectives_from_sample(
    parent  : List[int],
    sample  : dict,
) -> Tuple[float, float]:
    """
    Recompute (cost, balance) from a parent array + instance data.
    Avoids importing mo_cmst_ils.py (standalone).
    """
    inst    = sample["instance"]
    n       = inst["n"]
    coords  = np.array(inst["coords"])
    demands = np.array(inst["demands"])
    root    = 0

    # euclidean distance matrix
    diff    = coords[:, None] - coords[None, :]
    dist    = np.sqrt((diff ** 2).sum(2))

    # f1: total edge cost
    cost = sum(dist[v][parent[v]] for v in range(n) if v != root)

    # f2: max subtree demand
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    for v in range(n):
        if v != root:
            children[parent[v]].append(v)

    def subtree_demand(v: int) -> float:
        stack, total = [v], 0.0
        while stack:
            cur = stack.pop()
            total += demands[cur]
            stack.extend(children[cur])
        return total

    root_children = children[root]
    balance = max(subtree_demand(rc) for rc in root_children) if root_children else 0.0

    return float(cost), float(balance)


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL WORKERS  (module-level → picklable by ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_neural_ils(args: tuple) -> Tuple[float, float]:
    """
    Subprocess worker for evaluate_pareto.

    Receives pre-computed ranked edges (GNN inference already done in the
    main process on GPU) so this is pure CPU work — safe for subprocesses.

    args : (ranked, sample, alpha, ils_iter, ls_iter, strength, seed)
    returns : (cost, balance)
    """
    ranked, sample, alpha, ils_iter, ls_iter, strength, seed = args
    parent = _neural_guided_ils(
        ranked, sample, alpha,
        ils_iter = ils_iter,
        ls_iter  = ls_iter,
        strength = strength,
        seed     = seed,
    )
    return objectives_from_sample(parent, sample)


def _worker_pure_ils(args: tuple) -> Tuple[float, float]:
    """
    Subprocess worker for evaluate_baseline.

    Pure Prim warm-start + ILS — no GNN involved at any step.

    args : (sample, alpha, ils_iter, ls_iter, strength, seed)
    returns : (cost, balance)
    """
    sample, alpha, ils_iter, ls_iter, strength, seed = args
    parent = _pure_ils(
        sample, alpha,
        ils_iter = ils_iter,
        ls_iter  = ls_iter,
        strength = strength,
        seed     = seed,
    )
    return objectives_from_sample(parent, sample)


def _pareto_filter(
    pool: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Return non-dominated subset of (cost, balance) pairs."""
    front = []
    for i, (c, b) in enumerate(pool):
        dominated = any(
            c2 <= c and b2 <= b and (c2 < c or b2 < b)
            for j, (c2, b2) in enumerate(pool) if j != i
        )
        if not dominated:
            front.append((c, b))
    return front


def gnn_pareto_front(
    model        : MOCMSTNet,
    sample       : dict,
    n_alphas     : int   = 11,
    top_k_factor : float = 3.0,
    ls_iter      : int   = 80,
    ils_iter     : int   = 50,
    strength     : int   = 3,
    n_workers    : int   = 1,
    device       : Optional[torch.device] = None,
) -> List[Tuple[float, float]]:
    """
    Approximate the Pareto front via Neural-Guided ILS for each alpha.

    Parallelism strategy
    ────────────────────
    GNN inference is GPU-bound and very fast (~ms per alpha).
    ILS is CPU-bound and slow (~seconds per alpha).

    We therefore split the work:
      • Main process  : run all n_alphas GNN forward passes → n_alphas ranked lists
      • Worker pool   : run _neural_guided_ils independently per alpha (pure CPU)

    This gives near-linear speedup on the ILS phase with no GPU contention.

    Returns non-dominated (cost, balance) pairs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alphas = np.linspace(0.0, 1.0, n_alphas)

    # ── Step 1: GNN inference for all alphas (serial, GPU) ────────────────
    all_ranked = []
    for alpha in alphas:
        all_ranked.append(infer(model, sample, float(alpha),
                                top_k=None, device=device))

    # ── Step 2: ILS per alpha (parallel, CPU) ─────────────────────────────
    task_args = [
        (all_ranked[i], sample, float(alphas[i]),
         ils_iter, ls_iter, strength, i * 7)
        for i in range(n_alphas)
    ]

    pool_results: List[Tuple[float, float]] = []

    if n_workers <= 1 or ils_iter == 0:
        for args in task_args:
            pool_results.append(_worker_neural_ils(args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_neural_ils, a): a
                       for a in task_args}
            for fut in as_completed(futures):
                pool_results.append(fut.result())

    return _pareto_filter(pool_results)



def _neural_prim(
    ranked : List[dict],
    sample : dict,
) -> List[int]:
    """
    Neural-Prim: build a spanning tree using GNN edge scores as priorities.

    Why Prim and not Kruskal here
    ─────────────────────────────
    CMST is a *rooted* problem.  Capacity constraints are defined in terms of
    subtrees hanging from the root, so we need to know the root-adjacent
    ancestor of every node at the moment we attach it.  Prim naturally
    maintains this information because the tree always grows outward from an
    already-connected component; at each step the "tree side" of the chosen
    edge is already rooted correctly.

    Kruskal merges arbitrary components and would require an extra pass to
    determine subtree membership — and the previous implementation was
    actually Prim anyway (it required exactly one endpoint to already be in
    the tree), but it iterated over a *fixed* global ranking so edges between
    two not-yet-connected nodes were skipped for the entire run, leaving many
    nodes with no valid attachment and forcing the O(n) root-fallback that
    inflated costs by ~200 %.

    Algorithm
    ─────────
    1.  Build a dict  score[(min(i,j), max(i,j))]  from ranked list.
    2.  Maintain a priority queue (max-heap by score) of *frontier* edges:
        all edges from a tree node to a non-tree node.
    3.  At each step pop the highest-score edge that:
          a) connects a tree node to a non-tree node (not stale),
          b) respects the subtree capacity constraint.
    4.  When no feasible edge remains in the heap, fall back to the cheapest
        feasible edge in the full edge list (guaranteed to terminate).

    Complexity : O(E log E)  per construction call.
    """
    import heapq

    inst     = sample["instance"]
    n        = inst["n"]
    coords   = np.array(inst["coords"])
    demands  = np.array(inst["demands"])
    capacity = inst["capacity"]
    root     = 0

    # ── score lookup ─────────────────────────────────────────────────────
    score: Dict[Tuple[int, int], float] = {}
    for e in ranked:
        key = (min(e["i"], e["j"]), max(e["i"], e["j"]))
        score[key] = e["probability"]

    # ── euclidean distances (for fallback) ────────────────────────────────
    diff = coords[:, None] - coords[None, :]
    dist = np.sqrt((diff ** 2).sum(2))

    def get_score(u: int, v: int) -> float:
        return score.get((min(u, v), max(u, v)), 0.0)

    # ── state ─────────────────────────────────────────────────────────────
    parent    = list(range(n))
    in_tree   = [False] * n
    in_tree[root] = True

    sub_demand: Dict[int, float] = {}   # root-adjacent node → total demand
    node_sub  : Dict[int, int]   = {root: root}

    def can_attach(new_node: int, tree_node: int) -> bool:
        """True if attaching new_node to tree_node respects capacity."""
        d = float(demands[new_node])
        if tree_node == root:
            return d <= capacity + 1e-9
        rc = node_sub.get(tree_node)
        if rc is None:
            return False
        return sub_demand.get(rc, 0.0) + d <= capacity + 1e-9

    def attach(new_node: int, tree_node: int) -> None:
        """Commit the attachment and update tracking structures."""
        d = float(demands[new_node])
        if tree_node == root:
            sub_demand[new_node] = d
            node_sub[new_node]   = new_node
        else:
            rc = node_sub[tree_node]
            sub_demand[rc]      += d
            node_sub[new_node]   = rc
        parent[new_node]   = tree_node
        in_tree[new_node]  = True

    # ── max-heap  (negate score so heappop gives max) ─────────────────────
    # Entry: (-score, tree_node, new_node)
    heap: List[Tuple[float, int, int]] = []
    for v in range(n):
        if v != root:
            heapq.heappush(heap, (-get_score(root, v), root, v))

    n_connected = 1

    while n_connected < n and heap:
        neg_sc, u, v = heapq.heappop(heap)

        # Stale check: u must be in tree, v must NOT be
        if not in_tree[u] or in_tree[v]:
            continue

        if not can_attach(v, u):
            # Capacity violated — try other frontier edges first (stay in heap)
            continue

        attach(v, u)
        n_connected += 1

        # Push all edges from newly added node v to non-tree nodes
        for w in range(n):
            if not in_tree[w]:
                heapq.heappush(heap, (-get_score(v, w), v, w))

    # ── fallback: attach any remaining node to cheapest feasible parent ───
    if n_connected < n:
        for v in range(n):
            if in_tree[v]:
                continue
            # Try tree nodes sorted by distance (cheapest first)
            candidates = sorted(
                [u for u in range(n) if in_tree[u]],
                key=lambda u: dist[u][v],
            )
            attached = False
            for u in candidates:
                if can_attach(v, u):
                    attach(v, u)
                    n_connected += 1
                    attached = True
                    break
            if not attached:
                # Last resort: attach to root regardless of capacity
                attach(v, root)
                n_connected += 1

    return parent


def _local_search_refine(
    parent : List[int],
    sample : dict,
    alpha  : float,
    n_iter : int = 80,
) -> List[int]:
    """
    Best-improvement leaf-relocation local search applied *after* Neural-Prim.

    Why this is necessary
    ─────────────────────
    Neural-Prim (and any greedy construction) optimises edge scores one at a
    time.  Balance (f2 = max subtree demand) is a *global* property: it can
    only be improved by moving entire subtrees, which requires seeing the
    whole tree at once.  Local search does exactly that.

    A single pass over all leaves, trying every possible new parent, takes
    O(n²) — fast enough for the instance sizes used in this paper.

    The scalarised objective used here mirrors the one in mo_cmst_ils.py:
        obj = alpha * (cost / cost_norm) + (1-alpha) * (balance / cap)

    Parameters
    ----------
    parent  : parent array from _neural_prim
    sample  : instance dict (coords, demands, capacity)
    alpha   : current trade-off weight
    n_iter  : max improvement passes (early-exit on no improvement)
    """
    inst     = sample["instance"]
    n        = inst["n"]
    coords   = np.array(inst["coords"])
    demands  = np.array(inst["demands"])
    capacity = inst["capacity"]
    root     = 0

    # ── precompute distance matrix ────────────────────────────────────────
    diff = coords[:, None] - coords[None, :]
    dist = np.sqrt((diff ** 2).sum(2))

    max_dist = float(dist.max())
    n_cust   = n - 1

    def obj(cost: float, bal: float) -> float:
        return (alpha  * (cost / (max_dist * n_cust + 1e-9)) +
                (1 - alpha) * (bal  / (capacity + 1e-9)))

    # ── build children index ──────────────────────────────────────────────
    def build_children(par: List[int]) -> Dict[int, List[int]]:
        ch: Dict[int, List[int]] = {i: [] for i in range(n)}
        for v in range(n):
            if v != root:
                ch[par[v]].append(v)
        return ch

    # ── subtree demand (BFS) ──────────────────────────────────────────────
    def subtree_demand(v: int, ch: Dict[int, List[int]]) -> float:
        stack, total = [v], 0.0
        while stack:
            cur = stack.pop()
            total += float(demands[cur])
            stack.extend(ch[cur])
        return total

    # ── root-adjacent ancestor of every node ─────────────────────────────
    def build_node_sub(par: List[int], ch: Dict[int, List[int]]) -> Dict[int, int]:
        ns: Dict[int, int] = {root: root}
        for rc in ch[root]:
            for v in _bfs(rc, ch):
                ns[v] = rc
        return ns

    def _bfs(start: int, ch: Dict[int, List[int]]) -> List[int]:
        out, stack = [], [start]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(ch[cur])
        return out

    # ── current state ─────────────────────────────────────────────────────
    par      = parent[:]
    ch       = build_children(par)
    ns       = build_node_sub(par, ch)

    cur_cost = sum(dist[v][par[v]] for v in range(n) if v != root)
    sub_dem  = {rc: subtree_demand(rc, ch) for rc in ch[root]}
    cur_bal  = max(sub_dem.values()) if sub_dem else 0.0

    # ── main loop ─────────────────────────────────────────────────────────
    for _ in range(n_iter):
        improved = False

        # collect leaf nodes (no children, not root)
        leaves = [v for v in range(n) if v != root and not ch[v]]

        for v in leaves:
            old_par  = par[v]
            old_dist = dist[v][old_par]
            d        = float(demands[v])
            old_sub  = ns.get(v)

            best_par = old_par
            best_val = obj(cur_cost, cur_bal)

            for p in range(n):
                if p == v or p == old_par:
                    continue

                # capacity check
                if p == root:
                    if d > capacity + 1e-9:
                        continue
                    new_sub = v         # would start fresh subtree
                    new_sub_dem = d
                else:
                    new_sub = ns.get(p)
                    if new_sub is None:
                        continue
                    if new_sub == old_sub:
                        # same subtree — demand unchanged
                        new_sub_dem = sub_dem.get(old_sub, 0.0)
                    else:
                        if sub_dem.get(new_sub, 0.0) + d > capacity + 1e-9:
                            continue
                        new_sub_dem = sub_dem.get(new_sub, 0.0) + d

                # delta cost
                new_cost = cur_cost + dist[v][p] - old_dist

                # delta balance — O(|subtrees|) estimate
                old_sub_dem_after = sub_dem.get(old_sub, 0.0) - d
                new_bal = 0.0
                for sub, dem in sub_dem.items():
                    if   sub == old_sub: new_bal = max(new_bal, old_sub_dem_after)
                    elif sub == new_sub: new_bal = max(new_bal, new_sub_dem)
                    else:                new_bal = max(new_bal, dem)
                if new_sub not in sub_dem:
                    new_bal = max(new_bal, new_sub_dem)

                val = obj(new_cost, new_bal)
                if val < best_val - 1e-9:
                    best_val, best_par = val, p

            if best_par != old_par:
                # ── commit move ───────────────────────────────────────────
                d_val = float(demands[v])

                # update sub_dem
                if old_sub is not None and old_sub in sub_dem:
                    sub_dem[old_sub] -= d_val
                    if sub_dem[old_sub] < 1e-9:
                        del sub_dem[old_sub]

                if best_par == root:
                    new_sub_commit = v
                    sub_dem[v]     = d_val
                else:
                    new_sub_commit = ns[best_par]
                    sub_dem[new_sub_commit] = sub_dem.get(new_sub_commit, 0.0) + d_val

                # update parent / children
                ch[old_par].remove(v)
                ch[best_par].append(v)
                par[v]  = best_par
                ns[v]   = new_sub_commit

                cur_cost += dist[v][best_par] - old_dist
                cur_bal   = max(sub_dem.values()) if sub_dem else 0.0

                improved = True
                break          # restart scan with updated leaf list

        if not improved:
            break

    return par


def _perturb(
    parent  : List[int],
    sample  : dict,
    strength: int,
    rng     : np.random.Generator,
) -> List[int]:
    """
    Relocate `strength` randomly chosen leaf nodes to random feasible parents.
    This is the ILS diversification kick — identical logic to mo_cmst_ils.py.
    """
    inst     = sample["instance"]
    n        = inst["n"]
    demands  = np.array(inst["demands"])
    capacity = inst["capacity"]
    root     = 0

    par = parent[:]

    # Build children index and subtree tracking
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    for v in range(n):
        if v != root:
            children[par[v]].append(v)

    sub_dem : Dict[int, float] = {}
    node_sub: Dict[int, int]   = {root: root}

    def subtree_nodes(v: int) -> List[int]:
        out, stk = [], [v]
        while stk:
            cur = stk.pop(); out.append(cur); stk.extend(children[cur])
        return out

    for rc in children[root]:
        nodes = subtree_nodes(rc)
        sub_dem[rc] = float(demands[nodes].sum())
        for v in nodes:
            node_sub[v] = rc

    for _ in range(strength):
        leaves = [v for v in range(n) if v != root and not children[v]]
        if not leaves:
            break
        v       = int(rng.choice(leaves))
        old_par = par[v]
        d       = float(demands[v])
        old_sub = node_sub.get(v)

        cands = list(range(n))
        rng.shuffle(cands)
        for p in cands:
            if p == v or p == old_par:
                continue
            if p == root:
                if d > capacity + 1e-9:
                    continue
                # feasible — commit
                if old_sub in sub_dem:
                    sub_dem[old_sub] -= d
                    if sub_dem[old_sub] < 1e-9:
                        del sub_dem[old_sub]
                sub_dem[v]   = d
                node_sub[v]  = v
            else:
                ns = node_sub.get(p)
                if ns is None:
                    continue
                if ns == old_sub:
                    pass    # same subtree, demand neutral — still relocate
                else:
                    if sub_dem.get(ns, 0.0) + d > capacity + 1e-9:
                        continue
                    if old_sub in sub_dem:
                        sub_dem[old_sub] -= d
                        if sub_dem[old_sub] < 1e-9:
                            del sub_dem[old_sub]
                    sub_dem[ns] = sub_dem.get(ns, 0.0) + d
                    node_sub[v] = ns

            children[old_par].remove(v)
            children[p].append(v)
            par[v] = p
            break

    return par


def _neural_guided_ils(
    ranked   : List[dict],
    sample   : dict,
    alpha    : float,
    ils_iter : int = 50,
    ls_iter  : int = 80,
    strength : int = 3,
    seed     : Optional[int] = None,
) -> List[int]:
    """
    Neural-Guided ILS — the core contribution of this pipeline.

    Pipeline
    ────────
    1. Neural-Prim   : GNN scores → warm-start spanning tree
                       (replaces random/Prim construction used in vanilla ILS)
    2. Local Search  : best-improvement leaf relocation on scalarised obj
    3. ILS loop      :
         a. Perturb current best (random leaf relocations)
         b. Local Search
         c. Accept if better (strict descent)
         d. Restart from best

    Why the GNN warm start matters
    ───────────────────────────────
    Vanilla ILS starts from a noisy Prim construction every time.
    The GNN has seen hundreds of instances and learned which edges are
    structurally good for the given (instance, alpha) pair.  Starting from
    the GNN solution reduces the LS convergence distance significantly,
    especially for balance (f2) which requires coordinated subtree shapes.

    This is the "learning to initialise" paradigm — cheap at inference time
    (one forward pass), potentially large gain in solution quality.

    Parameters
    ----------
    ranked   : GNN-ranked edge list from infer()
    sample   : instance dict
    alpha    : scalarisation weight (0=balance, 1=cost)
    ils_iter : ILS iterations (perturbation+LS loops)
    ls_iter  : LS passes per ILS iteration
    strength : leaves relocated per perturbation kick
    seed     : RNG seed
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: GNN warm start ────────────────────────────────────────────
    best = _neural_prim(ranked, sample)
    best = _local_search_refine(best, sample, alpha, n_iter=ls_iter)

    best_cost, best_bal = objectives_from_sample(best, sample)
    inst = sample["instance"]
    capacity = inst["capacity"]
    n_cust   = inst["n"] - 1

    # Reconstruct dist max for normalisation
    coords   = np.array(inst["coords"])
    diff     = coords[:, None] - coords[None, :]
    max_dist = float(np.sqrt((diff ** 2).sum(2)).max())

    def obj(c: float, b: float) -> float:
        return (alpha  * (c / (max_dist * n_cust + 1e-9)) +
                (1 - alpha) * (b / (capacity + 1e-9)))

    best_val = obj(best_cost, best_bal)
    current  = best[:]

    # ── Step 2: ILS loop ──────────────────────────────────────────────────
    for _ in range(ils_iter):
        candidate = _perturb(current, sample, strength=strength, rng=rng)
        candidate = _local_search_refine(candidate, sample, alpha, n_iter=ls_iter)
        c_cost, c_bal = objectives_from_sample(candidate, sample)
        c_val = obj(c_cost, c_bal)

        if c_val < best_val - 1e-9:
            best, best_val = candidate[:], c_val

        current = best[:]   # restart from best (strict descent acceptance)

    return best


@torch.no_grad()
def evaluate_pareto(
    model      : MOCMSTNet,
    test_samples: List[dict],
    cfg        : dict,
    device     : Optional[torch.device] = None,
) -> dict:
    """
    Compare GNN Pareto front vs ILS Pareto front across all test instances.

    Metrics
    -------
    hypervolume_ratio : GNN HV / ILS HV   (closer to 1 is better)
    avg_cost_gap      : mean (GNN best cost - ILS best cost) / ILS best cost
    avg_balance_gap   : same for balance
    avg_gnn_pareto_size
    avg_ils_pareto_size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    hv_ratios, cost_gaps, bal_gaps = [], [], []
    gnn_sizes, ils_sizes = [], []

    print(f"\n  Evaluating Pareto fronts on {len(test_samples)} test instances…")
    bar = tqdm(test_samples, desc="  Eval", unit="inst", colour="yellow")

    for sample in bar:
        # ── ILS front (from dataset) ───────────────────────────────────────
        ils_front = [
            (s["cost"], s["balance"])
            for s in sample["pareto_solutions"]
        ]

        # ── GNN front ─────────────────────────────────────────────────────
        gnn_front = gnn_pareto_front(
            model, sample,
            n_alphas     = cfg["n_alphas_eval"],
            top_k_factor = cfg["top_k_factor"],
            ls_iter      = cfg["ls_iter"],
            ils_iter     = cfg["ils_iter"],
            strength     = cfg["ils_strength"],
            n_workers    = cfg["n_workers"],
            device       = device,
        )

        if not gnn_front or not ils_front:
            continue

        # ── Hypervolume (2D, nadir = max + 10% slack) ─────────────────────
        all_pts   = ils_front + gnn_front
        ref_cost  = max(c for c, _ in all_pts) * 1.1
        ref_bal   = max(b for _, b in all_pts) * 1.1

        hv_ils = _hypervolume_2d(ils_front, ref_cost, ref_bal)
        hv_gnn = _hypervolume_2d(gnn_front, ref_cost, ref_bal)
        if hv_ils > 1e-9:
            hv_ratios.append(hv_gnn / hv_ils)

        # ── Best cost / balance gap ────────────────────────────────────────
        best_ils_cost = min(c for c, _ in ils_front)
        best_gnn_cost = min(c for c, _ in gnn_front)
        cost_gaps.append((best_gnn_cost - best_ils_cost) / (best_ils_cost + 1e-9))

        best_ils_bal = min(b for _, b in ils_front)
        best_gnn_bal = min(b for _, b in gnn_front)
        bal_gaps.append((best_gnn_bal - best_ils_bal) / (best_ils_bal + 1e-9))

        gnn_sizes.append(len(gnn_front))
        ils_sizes.append(len(ils_front))

        bar.set_postfix(
            hv_ratio=f"{np.mean(hv_ratios):.3f}",
            cost_gap=f"{np.mean(cost_gaps)*100:.1f}%",
        )

    results = {
        "hypervolume_ratio"      : float(np.mean(hv_ratios)),
        "hypervolume_ratio_std"  : float(np.std(hv_ratios)),
        "avg_cost_gap_pct"       : float(np.mean(cost_gaps) * 100),
        "avg_balance_gap_pct"    : float(np.mean(bal_gaps) * 100),
        "avg_gnn_pareto_size"    : float(np.mean(gnn_sizes)),
        "avg_ils_pareto_size"    : float(np.mean(ils_sizes)),
        "n_instances"            : len(hv_ratios),
    }

    print(f"\n{'─'*54}")
    print(f"  {'Metric':<30}  {'Value':>12}")
    print(f"{'─'*54}")
    for k, v in results.items():
        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<30}  {fmt:>12}")
    print(f"{'─'*54}")

    # ── save a Pareto plot for the first test instance ─────────────────────
    if test_samples:
        _plot_pareto(
            model, test_samples[0], cfg,
            device=device,
            out_path=os.path.join(cfg["runs_dir"], "pareto_example.png"),
        )

    return results


def _prim_construct_baseline(
    sample : dict,
    noise  : float = 0.15,
    rng    : Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Modified Prim construction (no GNN) — identical to mo_cmst_ils.py.
    Used as the warm-start for the pure ILS baseline.

    At each step attaches the cheapest feasible unvisited node to the tree,
    with optional uniform noise for GRASP-style randomisation.
    """
    if rng is None:
        rng = np.random.default_rng()

    inst     = sample["instance"]
    n        = inst["n"]
    coords   = np.array(inst["coords"])
    demands  = np.array(inst["demands"])
    capacity = inst["capacity"]
    root     = 0

    diff = coords[:, None] - coords[None, :]
    dist = np.sqrt((diff ** 2).sum(2))
    max_dist = float(dist.max())

    parent   = list(range(n))
    in_tree  = {root}
    sub_dem  : Dict[int, float] = {}
    node_sub : Dict[int, int]   = {root: root}

    while len(in_tree) < n:
        best_score, best_move = float("inf"), None

        for u in list(in_tree):
            for v in range(n):
                if v in in_tree:
                    continue
                d = float(demands[v])
                if u == root:
                    if d > capacity + 1e-9:
                        continue
                    sub_v = v
                else:
                    sr = node_sub.get(u)
                    if sr is None:
                        continue
                    if sub_dem.get(sr, 0.0) + d > capacity + 1e-9:
                        continue
                    sub_v = sr

                score = float(dist[u][v])
                if noise > 0.0:
                    score += noise * rng.uniform(-1.0, 1.0) * max_dist * 0.1
                if score < best_score:
                    best_score = score
                    best_move  = (u, v, sub_v)

        if best_move is None:
            # fallback: attach remaining nodes to root
            for v in range(n):
                if v not in in_tree:
                    parent[v] = root
                    node_sub[v] = v
                    sub_dem[v]  = float(demands[v])
                    in_tree.add(v)
            break

        u, v, sub_v = best_move
        parent[v] = u
        in_tree.add(v)
        node_sub[v] = sub_v
        if u == root:
            sub_dem[sub_v] = float(demands[v])
        else:
            sub_dem[sub_v] += float(demands[v])

    return parent


def _pure_ils(
    sample   : dict,
    alpha    : float,
    ils_iter : int = 50,
    ls_iter  : int = 80,
    strength : int = 3,
    seed     : Optional[int] = None,
) -> List[int]:
    """
    Pure ILS baseline — identical budget to Neural-Guided ILS but NO GNN.
    Warm-start via noisy Prim instead of Neural-Prim.

    This is the fair comparison: same (ils_iter, ls_iter, strength),
    only the construction heuristic differs.
    """
    rng = np.random.default_rng(seed)

    best = _prim_construct_baseline(sample, noise=0.15, rng=rng)
    best = _local_search_refine(best, sample, alpha, n_iter=ls_iter)

    inst     = sample["instance"]
    capacity = inst["capacity"]
    n_cust   = inst["n"] - 1
    coords   = np.array(inst["coords"])
    diff     = coords[:, None] - coords[None, :]
    max_dist = float(np.sqrt((diff ** 2).sum(2)).max())

    def obj(c: float, b: float) -> float:
        return (alpha  * (c / (max_dist * n_cust + 1e-9)) +
                (1 - alpha) * (b / (capacity + 1e-9)))

    best_val = obj(*objectives_from_sample(best, sample))
    current  = best[:]

    for _ in range(ils_iter):
        candidate = _perturb(current, sample, strength=strength, rng=rng)
        candidate = _local_search_refine(candidate, sample, alpha, n_iter=ls_iter)
        c_val     = obj(*objectives_from_sample(candidate, sample))
        if c_val < best_val - 1e-9:
            best, best_val = candidate[:], c_val
        current = best[:]

    return best


def baseline_pareto_front(
    sample      : dict,
    n_alphas    : int = 11,
    ils_iter    : int = 50,
    ls_iter     : int = 80,
    strength    : int = 3,
    n_workers   : int = 1,
) -> List[Tuple[float, float]]:
    """
    Build the Pareto front using pure ILS (no GNN), parallelised over alphas.
    Same budget as gnn_pareto_front — fair comparison.
    """
    alphas    = np.linspace(0.0, 1.0, n_alphas)
    task_args = [
        (sample, float(alphas[i]), ils_iter, ls_iter, strength, i * 7)
        for i in range(n_alphas)
    ]

    pool_results: List[Tuple[float, float]] = []

    if n_workers <= 1:
        for args in task_args:
            pool_results.append(_worker_pure_ils(args))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_pure_ils, a): a
                       for a in task_args}
            for fut in as_completed(futures):
                pool_results.append(fut.result())

    return _pareto_filter(pool_results)



def evaluate_baseline(
    test_samples : List[dict],
    cfg          : dict,
) -> dict:
    """
    Run pure ILS (no GNN) vs the ILS dataset reference.
    Uses the same unified-reference HV calculation as evaluate_compare
    for consistency.
    """
    hv_ratios, cost_gaps, bal_gaps = [], [], []
    base_sizes, ils_sizes = [], []

    print(f"\n  Running pure ILS baseline on {len(test_samples)} test instances…")
    print(f"  Budget: ils_iter={cfg['ils_iter']}  ls_iter={cfg['ls_iter']}  "
          f"strength={cfg['ils_strength']}  alphas={cfg['n_alphas_eval']}")
    print(f"  (identical budget to Neural-Guided ILS — only warm-start differs)\n")

    bar = tqdm(test_samples, desc="  Baseline", unit="inst", colour="magenta")

    for sample in bar:
        ref_front = [
            (s["cost"], s["balance"])
            for s in sample["pareto_solutions"]
        ]
        # Auto-disable workers for small instances (overhead > gain)
        n_nodes   = sample["instance"]["n"]
        workers   = cfg["n_workers"] if n_nodes >= 30 else 1

        base_front = baseline_pareto_front(
            sample,
            n_alphas  = cfg["n_alphas_eval"],
            ils_iter  = cfg["ils_iter"],
            ls_iter   = cfg["ls_iter"],
            strength  = cfg["ils_strength"],
            n_workers = workers,
        )

        if not base_front or not ref_front:
            continue

        # Unified reference point from all solutions
        all_pts  = ref_front + base_front
        ref_cost = max(c for c, _ in all_pts) * 1.1
        ref_bal  = max(b for _, b in all_pts) * 1.1

        hv_ref  = _hypervolume_2d(ref_front,  ref_cost, ref_bal)
        hv_base = _hypervolume_2d(base_front, ref_cost, ref_bal)
        if hv_ref > 1e-9:
            hv_ratios.append(hv_base / hv_ref)

        best_ref_cost  = min(c for c, _ in ref_front)
        best_base_cost = min(c for c, _ in base_front)
        cost_gaps.append((best_base_cost - best_ref_cost) / (best_ref_cost + 1e-9))

        best_ref_bal  = min(b for _, b in ref_front)
        best_base_bal = min(b for _, b in base_front)
        bal_gaps.append((best_base_bal - best_ref_bal) / (best_ref_bal + 1e-9))

        base_sizes.append(len(base_front))
        ils_sizes.append(len(ref_front))

        bar.set_postfix(
            hv_ratio=f"{np.mean(hv_ratios):.3f}",
            cost_gap=f"{np.mean(cost_gaps)*100:.1f}%",
        )

    results = {
        "hypervolume_ratio"       : float(np.mean(hv_ratios)),
        "hypervolume_ratio_std"   : float(np.std(hv_ratios)),
        "avg_cost_gap_pct"        : float(np.mean(cost_gaps) * 100),
        "avg_balance_gap_pct"     : float(np.mean(bal_gaps) * 100),
        "avg_baseline_pareto_size": float(np.mean(base_sizes)),
        "avg_ref_pareto_size"     : float(np.mean(ils_sizes)),
        "n_instances"             : len(hv_ratios),
    }

    print(f"\n{'─'*54}")
    print(f"  Pure ILS Baseline  (vs ILS reference)")
    print(f"{'─'*54}")
    print(f"  {'Metric':<30}  {'Value':>12}")
    print(f"{'─'*54}")
    for k, v in results.items():
        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<30}  {fmt:>12}")
    print(f"{'─'*54}")
    print(f"\n  → Use --mode eval-compare for side-by-side GNN vs Prim comparison.")

    return results


def evaluate_compare(
    model        : MOCMSTNet,
    test_samples : List[dict],
    cfg          : dict,
    device       : Optional[torch.device] = None,
) -> dict:
    """
    Side-by-side comparison: GNN+ILS vs Prim+ILS vs ILS reference.

    Why unified reference point matters
    ────────────────────────────────────
    The hypervolume indicator is only comparable between methods when it is
    computed against the SAME reference point.  If method A produces a
    bad solution with very high cost, using (A+ref) to set the nadir
    inflates the reference point, artificially boosting A's HV ratio.

    Here we compute one reference point per instance from ALL solutions
    (ref_front ∪ gnn_front ∪ base_front), then evaluate all three methods
    against that common nadir.  This is the standard practice in the
    multi-objective optimisation literature (Zitzler et al., 2003).

    Both GNN+ILS and Prim+ILS use IDENTICAL budgets
      ils_iter, ls_iter, ils_strength, n_alphas_eval
    so any quality difference isolates the warm-start contribution.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Per-instance results stored for summary
    gnn_hvr,  base_hvr            = [], []
    gnn_cost, base_cost           = [], []
    gnn_bal,  base_bal            = [], []
    gnn_sz,   base_sz,  ils_sz    = [], [], []

    print(f"\n  Comparing GNN+ILS vs Prim+ILS on {len(test_samples)} instances…")
    print(f"  Budget (both): ils_iter={cfg['ils_iter']}  ls_iter={cfg['ls_iter']}  "
          f"strength={cfg['ils_strength']}  alphas={cfg['n_alphas_eval']}")
    print(f"  Reference point: unified (ref ∪ gnn ∪ base per instance)\n")

    bar = tqdm(test_samples, desc="  Compare", unit="inst", colour="cyan")

    for sample in bar:
        # ── ILS dataset reference (100 iter gabarito) ─────────────────────
        ref_front = [
            (s["cost"], s["balance"])
            for s in sample["pareto_solutions"]
        ]

        # Auto-disable workers for small instances
        n_nodes = sample["instance"]["n"]
        workers = cfg["n_workers"] if n_nodes >= 30 else 1

        # ── GNN+ILS front ─────────────────────────────────────────────────
        gnn_front = gnn_pareto_front(
            model, sample,
            n_alphas     = cfg["n_alphas_eval"],
            top_k_factor = cfg["top_k_factor"],
            ls_iter      = cfg["ls_iter"],
            ils_iter     = cfg["ils_iter"],
            strength     = cfg["ils_strength"],
            n_workers    = workers,
            device       = device,
        )

        # ── Prim+ILS front (same budget, no GNN) ──────────────────────────
        base_front = baseline_pareto_front(
            sample,
            n_alphas  = cfg["n_alphas_eval"],
            ils_iter  = cfg["ils_iter"],
            ls_iter   = cfg["ls_iter"],
            strength  = cfg["ils_strength"],
            n_workers = workers,
        )

        if not gnn_front or not base_front or not ref_front:
            continue

        # ── Unified reference point ────────────────────────────────────────
        all_pts  = ref_front + gnn_front + base_front
        ref_c    = max(c for c, _ in all_pts) * 1.1
        ref_b    = max(b for _, b in all_pts) * 1.1

        hv_ref  = _hypervolume_2d(ref_front,  ref_c, ref_b)
        hv_gnn  = _hypervolume_2d(gnn_front,  ref_c, ref_b)
        hv_base = _hypervolume_2d(base_front, ref_c, ref_b)

        if hv_ref > 1e-9:
            gnn_hvr.append(hv_gnn  / hv_ref)
            base_hvr.append(hv_base / hv_ref)

        # Cost gaps vs reference
        best_rc = min(c for c, _ in ref_front)
        gnn_cost.append((min(c for c, _ in gnn_front)  - best_rc) / (best_rc + 1e-9))
        base_cost.append((min(c for c, _ in base_front) - best_rc) / (best_rc + 1e-9))

        # Balance gaps vs reference
        best_rb = min(b for _, b in ref_front)
        gnn_bal.append((min(b for _, b in gnn_front)  - best_rb) / (best_rb + 1e-9))
        base_bal.append((min(b for _, b in base_front) - best_rb) / (best_rb + 1e-9))

        gnn_sz.append(len(gnn_front))
        base_sz.append(len(base_front))
        ils_sz.append(len(ref_front))

        bar.set_postfix(
            gnn_hv  = f"{np.mean(gnn_hvr):.3f}",
            base_hv = f"{np.mean(base_hvr):.3f}",
        )

    n = len(gnn_hvr)
    results = {
        "n_instances"                  : n,
        # ── GNN+ILS ──────────────────────────────────────────────────────
        "gnn_hypervolume_ratio"        : float(np.mean(gnn_hvr)),
        "gnn_hypervolume_ratio_std"    : float(np.std(gnn_hvr)),
        "gnn_avg_cost_gap_pct"         : float(np.mean(gnn_cost)  * 100),
        "gnn_avg_balance_gap_pct"      : float(np.mean(gnn_bal)   * 100),
        "gnn_avg_pareto_size"          : float(np.mean(gnn_sz)),
        # ── Prim+ILS ─────────────────────────────────────────────────────
        "base_hypervolume_ratio"       : float(np.mean(base_hvr)),
        "base_hypervolume_ratio_std"   : float(np.std(base_hvr)),
        "base_avg_cost_gap_pct"        : float(np.mean(base_cost) * 100),
        "base_avg_balance_gap_pct"     : float(np.mean(base_bal)  * 100),
        "base_avg_pareto_size"         : float(np.mean(base_sz)),
        # ── reference ────────────────────────────────────────────────────
        "ref_avg_pareto_size"          : float(np.mean(ils_sz)),
    }

    # ── Side-by-side table ────────────────────────────────────────────────
    W = 68
    print(f"\n{'─'*W}")
    print(f"  {'Metric':<32}  {'GNN+ILS':>12}  {'Prim+ILS':>12}  {'Winner':>6}")
    print(f"{'─'*W}")

    comparisons = [
        ("hv_ratio (↑ better)",
         results["gnn_hypervolume_ratio"],
         results["base_hypervolume_ratio"],
         True),
        ("hv_ratio_std (↓ better)",
         results["gnn_hypervolume_ratio_std"],
         results["base_hypervolume_ratio_std"],
         False),
        ("cost_gap_pct (↓ better)",
         results["gnn_avg_cost_gap_pct"],
         results["base_avg_cost_gap_pct"],
         False),
        ("balance_gap_pct (↓ better)",
         results["gnn_avg_balance_gap_pct"],
         results["base_avg_balance_gap_pct"],
         False),
        ("pareto_size (~ neutral)",
         results["gnn_avg_pareto_size"],
         results["base_avg_pareto_size"],
         None),
    ]

    gnn_wins = 0
    for label, gv, bv, higher_is_better in comparisons:
        if higher_is_better is None:
            winner = " —"
        elif higher_is_better:
            winner = "GNN ✓" if gv > bv + 1e-4 else ("BASE ✓" if bv > gv + 1e-4 else "  TIE")
            if winner == "GNN ✓": gnn_wins += 1
        else:
            winner = "GNN ✓" if gv < bv - 1e-4 else ("BASE ✓" if bv < gv - 1e-4 else "  TIE")
            if winner == "GNN ✓": gnn_wins += 1

        print(f"  {label:<32}  {gv:>12.4f}  {bv:>12.4f}  {winner:>6}")

    print(f"{'─'*W}")
    print(f"  GNN wins {gnn_wins}/4 metrics  |  "
          f"ref_pareto_size (gabarito): {results['ref_avg_pareto_size']:.2f}")
    print(f"{'─'*W}")

    # ── Interpretation ────────────────────────────────────────────────────
    print(f"\n  Interpretation")
    print(f"  {'─'*62}")
    if results["gnn_hypervolume_ratio"] > results["base_hypervolume_ratio"] + 0.02:
        print(f"  ✓ GNN warm-start improves Pareto front diversity (HV).")
    else:
        print(f"  ✗ GNN warm-start does NOT improve HV over Prim+ILS.")
        print(f"    → On small instances ILS converges fast from any start.")
        print(f"    → Re-run with larger instances (--max-customers 50+) to")
        print(f"      test whether benefit emerges on harder problems.")

    if results["gnn_avg_cost_gap_pct"] < results["base_avg_cost_gap_pct"] - 1.0:
        print(f"  ✓ GNN warm-start finds cheaper solutions.")
    if results["gnn_avg_balance_gap_pct"] < results["base_avg_balance_gap_pct"] - 1.0:
        print(f"  ✓ GNN warm-start finds better-balanced solutions.")

    # ── Save comparison plot ──────────────────────────────────────────────
    if test_samples:
        _plot_compare(
            model, test_samples[0], cfg, device,
            out_path=os.path.join(cfg["runs_dir"], "compare_example.png"),
        )

    return results


def _hypervolume_2d(
    front    : List[Tuple[float, float]],
    ref_cost : float,
    ref_bal  : float,
) -> float:
    """
    2-D hypervolume indicator (WFG/sweep).
    Front is assumed to contain non-dominated points.
    Both objectives are minimised.
    """
    pts = sorted(set(front), key=lambda p: p[0])
    hv  = 0.0
    prev_bal = ref_bal
    for cost, bal in pts:
        if cost < ref_cost and bal < ref_bal:
            hv       += (ref_cost - cost) * (prev_bal - bal)
            prev_bal  = bal
    return hv


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training(history: dict, runs_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train loss", color="#2563EB")
    ax.plot(history["val_loss"],   label="Val loss",   color="#DC2626", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # Metrics curve
    ax = axes[1]
    ax.plot(history["precision"], label="Precision", color="#16A34A")
    ax.plot(history["recall"],    label="Recall",    color="#CA8A04")
    ax.plot(history["f1"],        label="F1",        color="#7C3AED", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Edge Prediction Metrics (val)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(runs_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Loss curve saved → {path}")


def _plot_pareto(
    model    : MOCMSTNet,
    sample   : dict,
    cfg      : dict,
    device   : torch.device,
    out_path : str,
) -> None:
    gnn_front = gnn_pareto_front(
        model, sample,
        n_alphas     = cfg["n_alphas_eval"],
        top_k_factor = cfg["top_k_factor"],
        ls_iter      = cfg["ls_iter"],
        ils_iter     = cfg["ils_iter"],
        strength     = cfg["ils_strength"],
        n_workers    = cfg["n_workers"],
        device       = device,
    )
    ils_front = [(s["cost"], s["balance"]) for s in sample["pareto_solutions"]]

    fig, ax = plt.subplots(figsize=(7, 5))

    if ils_front:
        cx, cy = zip(*sorted(ils_front))
        ax.plot(cx, cy, "o-", color="#2563EB", label="ILS Pareto", linewidth=2,
                markersize=8, zorder=3)

    if gnn_front:
        gx, gy = zip(*sorted(gnn_front))
        ax.plot(gx, gy, "s--", color="#DC2626", label="GNN Pareto", linewidth=2,
                markersize=8, zorder=3)

    ax.set_xlabel("f₁  (total cost)")
    ax.set_ylabel("f₂  (max subtree demand)")
    ax.set_title(f"Pareto Front Comparison  (n={sample['instance']['n']})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Pareto plot saved → {out_path}")


def _plot_compare(
    model    : MOCMSTNet,
    sample   : dict,
    cfg      : dict,
    device   : torch.device,
    out_path : str,
) -> None:
    """Three-way Pareto plot: ILS reference vs GNN+ILS vs Prim+ILS."""
    n_nodes = sample["instance"]["n"]
    workers = cfg["n_workers"] if n_nodes >= 30 else 1

    gnn_front = gnn_pareto_front(
        model, sample,
        n_alphas=cfg["n_alphas_eval"], top_k_factor=cfg["top_k_factor"],
        ls_iter=cfg["ls_iter"], ils_iter=cfg["ils_iter"],
        strength=cfg["ils_strength"], n_workers=workers, device=device,
    )
    base_front = baseline_pareto_front(
        sample, n_alphas=cfg["n_alphas_eval"],
        ils_iter=cfg["ils_iter"], ls_iter=cfg["ls_iter"],
        strength=cfg["ils_strength"], n_workers=workers,
    )
    ref_front = [(s["cost"], s["balance"]) for s in sample["pareto_solutions"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = [
        (ref_front,  "ILS Reference (gabarito)", "#2563EB", "o-",  10),
        (gnn_front,  "GNN+ILS",                  "#DC2626", "s--",  8),
        (base_front, "Prim+ILS",                 "#16A34A", "^:",   8),
    ]
    for front, label, color, style, ms in styles:
        if front:
            xs, ys = zip(*sorted(front))
            ax.plot(xs, ys, style, color=color, label=label,
                    linewidth=2, markersize=ms, zorder=3)

    ax.set_xlabel("f₁  (total cost)")
    ax.set_ylabel("f₂  (max subtree demand)")
    ax.set_title(f"Pareto Front Comparison  (n={sample['instance']['n']}, "
                 f"ils_iter={cfg['ils_iter']})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Comparison plot saved → {out_path}")



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MO-CMST GNN")
    p.add_argument("--mode", choices=["train", "eval", "eval-baseline", "eval-compare", "infer"],
                   default="train")
    p.add_argument("--data",           default=CFG["data_path"])
    p.add_argument("--checkpoint",     default=None,
                   help="path to .pt file (for eval / infer)")
    p.add_argument("--instance-idx",   type=int, default=0,
                   help="index in dataset (infer mode)")
    p.add_argument("--alpha",          type=float, default=0.5,
                   help="trade-off weight for inference")
    p.add_argument("--epochs",         type=int,   default=CFG["epochs"])
    p.add_argument("--batch-size",     type=int,   default=CFG["batch_size"])
    p.add_argument("--hidden-dim",     type=int,   default=CFG["hidden_dim"])
    p.add_argument("--lr",             type=float, default=CFG["lr"])
    p.add_argument("--runs-dir",       default=CFG["runs_dir"])
    p.add_argument("--seed",           type=int,   default=CFG["seed"])
    p.add_argument("--soft-labels",    action="store_true",
                   default=CFG["use_soft_labels"])
    p.add_argument("--pos-weight",     type=float, default=CFG["pos_weight"])
    p.add_argument("--n-alphas-eval",  type=int,   default=CFG["n_alphas_eval"])
    p.add_argument("--top-k-factor",   type=float, default=CFG["top_k_factor"])
    p.add_argument("--ls-iter",        type=int,   default=CFG["ls_iter"],
                   help="LS passes per ILS iteration (0 = LS disabled)")
    p.add_argument("--ils-iter",       type=int,   default=CFG["ils_iter"],
                   help="ILS iterations after GNN warm-start (0 = Prim+LS only)")
    p.add_argument("--ils-strength",   type=int,   default=CFG["ils_strength"],
                   help="leaves relocated per perturbation kick")
    p.add_argument("--workers",        type=int,   default=CFG["n_workers"],
                   help="parallel ILS workers per instance (1 = serial)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = {**CFG,
           "data_path"      : args.data,
           "epochs"         : args.epochs,
           "batch_size"     : args.batch_size,
           "hidden_dim"     : args.hidden_dim,
           "lr"             : args.lr,
           "runs_dir"       : args.runs_dir,
           "seed"           : args.seed,
           "use_soft_labels": args.soft_labels,
           "pos_weight"     : args.pos_weight,
           "n_alphas_eval"  : args.n_alphas_eval,
           "top_k_factor"   : args.top_k_factor,
           "ls_iter"        : args.ls_iter,
           "ils_iter"       : args.ils_iter,
           "ils_strength"   : args.ils_strength,
           "n_workers"      : args.workers,
           }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        # ── Train ─────────────────────────────────────────────────────────
        print(f"\n{'═'*68}")
        print(f"  MO-CMST  ·  GNN Training  ·  GraphSAGE + GAT + EdgeMLP")
        print(f"{'═'*68}")
        model = train(cfg)

    elif args.mode == "eval":
        # ── Evaluate ──────────────────────────────────────────────────────
        assert args.checkpoint, "--checkpoint required for eval mode"
        model = load_model(args.checkpoint, device)
        _, _, test_s = load_splits(
            cfg["data_path"], cfg["val_split"], cfg["test_split"], cfg["seed"]
        )
        evaluate_pareto(model, test_s, cfg, device)

    elif args.mode == "eval-compare":
        # ── Side-by-side GNN+ILS vs Prim+ILS, unified HV reference ───────
        assert args.checkpoint, "--checkpoint required for eval-compare mode"
        model = load_model(args.checkpoint, device)
        _, _, test_s = load_splits(
            cfg["data_path"], cfg["val_split"], cfg["test_split"], cfg["seed"]
        )
        evaluate_compare(model, test_s, cfg, device)

    elif args.mode == "eval-baseline":
        # ── Pure ILS baseline (no GNN) — fair comparison ──────────────────
        # Uses identical budget (ils_iter, ls_iter, strength, n_alphas)
        # but Prim warm-start instead of GNN warm-start.
        #
        # Compare the two outputs to isolate the GNN's contribution:
        #   eval-baseline  →  what ILS alone achieves with this budget
        #   eval           →  what GNN+ILS achieves with this budget
        print(f"\n{'═'*68}")
        print(f"  MO-CMST  ·  Pure ILS Baseline  ·  (no GNN, fair budget)")
        print(f"{'═'*68}")
        _, _, test_s = load_splits(
            cfg["data_path"], cfg["val_split"], cfg["test_split"], cfg["seed"]
        )
        evaluate_baseline(test_s, cfg)

    elif args.mode == "infer":
        # ── Inference on single instance ──────────────────────────────────
        assert args.checkpoint, "--checkpoint required for infer mode"
        model = load_model(args.checkpoint, device)
        with open(cfg["data_path"], "rb") as fh:
            samples = pickle.load(fh)
        sample = samples[args.instance_idx]
        ranked = infer(model, sample, alpha=args.alpha, device=device)
        print_ranked_edges(ranked, sample, alpha=args.alpha)