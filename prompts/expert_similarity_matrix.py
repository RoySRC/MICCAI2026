#!/usr/bin/env python3
"""
expert_similarity_matrix.py

Builds expert-level cosine-similarity matrices from PLIP prompt embeddings saved as .npz.

You get three E×E matrices:
  - Spp: positive vs positive (expert i pos prototype vs expert j pos prototype)
  - Spn: positive vs hard_negative (expert i pos prototype vs expert j neg prototype)
  - Snn: hard_negative vs hard_negative (expert i neg prototype vs expert j neg prototype)

Default metric is "centroid" (mean of row-normalized embeddings, then L2-normalized).
This is the most stable/cheap way to summarize a bank.

Optionally:
  - sampled_mean: estimates mean cosine between two banks by sampling pairs
  - mean_max: estimates "worst-case overlap" via mean over rows of max similarity
              (computed by sampling, to avoid huge A@B^T).

Works with either:
  - a sidecar JSON (recommended) containing the mapping expert_name -> npz keys, OR
  - keys in the NPZ of the form "<expert>__positives" and "<expert>__hard_negatives".

Usage examples:
  python expert_similarity_matrix.py --npz plip_prompt_embeddings.npz --out_dir sim_out
  python expert_similarity_matrix.py --npz plip_prompt_embeddings.npz --sidecar plip_prompt_embeddings.json --out_dir sim_out
  python expert_similarity_matrix.py --npz plip_prompt_embeddings.npz --metric sampled_mean --pairs 20000 --out_dir sim_out
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def sanitize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n if n > 1e-12 else 1e-12)


def load_keymap(sidecar_path: Optional[str], npz: np.lib.npyio.NpzFile) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Returns:
      expert_names (human-readable)
      keymap: expert_name -> {"positives": npz_key, "hard_negatives": npz_key}
    """
    if sidecar_path:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "keys" not in meta:
            raise ValueError("Sidecar JSON missing 'keys'.")
        keymap = meta["keys"]
        expert_names = list(keymap.keys())
        # Validate
        for ex in expert_names:
            if "positives" not in keymap[ex] or "hard_negatives" not in keymap[ex]:
                raise ValueError(f"Sidecar keys for {ex!r} must include 'positives' and 'hard_negatives'.")
            if keymap[ex]["positives"] not in npz.files or keymap[ex]["hard_negatives"] not in npz.files:
                raise KeyError(f"NPZ missing keys for expert {ex!r}: {keymap[ex]}")
        return expert_names, keymap

    # Infer from NPZ naming convention
    tmp: Dict[str, Dict[str, str]] = {}
    for k in npz.files:
        if k.endswith("__positives"):
            ex = k[:-len("__positives")]
            tmp.setdefault(ex, {})["positives"] = k
        elif k.endswith("__hard_negatives"):
            ex = k[:-len("__hard_negatives")]
            tmp.setdefault(ex, {})["hard_negatives"] = k

    keymap2: Dict[str, Dict[str, str]] = {}
    for ex, d in tmp.items():
        if "positives" in d and "hard_negatives" in d:
            # Human-readable label: replace underscores with spaces
            human = ex.replace("_", " ").strip()
            keymap2[human] = d

    if not keymap2:
        raise ValueError(
            "Could not infer expert keys from NPZ. Expected keys ending with "
            "'__positives' and '__hard_negatives'."
        )

    expert_names = sorted(keymap2.keys())
    return expert_names, keymap2


def compute_centroid_prototype(bank: np.ndarray) -> np.ndarray:
    """
    bank: (N,D) float32/float64
    Returns: (D,) float32, L2-normalized.
    """
    bank = bank.astype(np.float32, copy=False)
    bank = l2_normalize_rows(bank)  # safe even if already normalized
    v = bank.mean(axis=0)
    v = l2_normalize_vec(v).astype(np.float32, copy=False)
    return v


def sampled_mean_cosine(A: np.ndarray, B: np.ndarray, pairs: int, seed: int) -> float:
    """
    Estimates mean cosine similarity between banks A and B by sampling pairs.
    A,B are assumed row-normalized (or will be normalized here).
    """
    A = l2_normalize_rows(A.astype(np.float32, copy=False))
    B = l2_normalize_rows(B.astype(np.float32, copy=False))
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, A.shape[0], size=pairs, dtype=np.int64)
    ib = rng.integers(0, B.shape[0], size=pairs, dtype=np.int64)
    sims = np.sum(A[ia] * B[ib], axis=1)
    return float(np.mean(sims))


def sampled_mean_max_cosine(A: np.ndarray, B: np.ndarray, rows: int, cols_per_row: int, seed: int) -> float:
    """
    Estimates mean over sampled rows of max cosine similarity into B:
      mean_i max_j cos(A_i, B_j)
    where for each selected A_i we sample cols_per_row candidates from B.

    This approximates worst-case overlap without constructing A@B^T.
    """
    A = l2_normalize_rows(A.astype(np.float32, copy=False))
    B = l2_normalize_rows(B.astype(np.float32, copy=False))
    rng = np.random.default_rng(seed)

    nA = A.shape[0]
    nB = B.shape[0]
    rows = min(rows, nA)

    ia = rng.choice(nA, size=rows, replace=False)
    max_sims = np.empty((rows,), dtype=np.float32)

    for idx, i in enumerate(ia):
        jb = rng.integers(0, nB, size=cols_per_row, dtype=np.int64)
        sims = B[jb] @ A[i]  # (cols_per_row,)
        max_sims[idx] = float(np.max(sims))

    return float(np.mean(max_sims))


def load_bank(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    arr = npz[key]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for key {key!r}, got shape {arr.shape}.")
    return arr


def build_matrices(
    npz_path: str,
    sidecar_path: Optional[str],
    metric: str,
    pairs: int,
    mean_max_rows: int,
    mean_max_cols: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: expert_names, Spp, Spn, Snn
    """
    npz = np.load(npz_path, allow_pickle=False)
    expert_names, keymap = load_keymap(sidecar_path, npz)
    E = len(expert_names)

    # Fast path: centroid prototypes -> just dot products
    if metric == "centroid":
        P = np.zeros((E, 0), dtype=np.float32)
        N = np.zeros((E, 0), dtype=np.float32)

        pos_protos: List[np.ndarray] = []
        neg_protos: List[np.ndarray] = []

        for ex in expert_names:
            pk = keymap[ex]["positives"]
            nk = keymap[ex]["hard_negatives"]
            A = load_bank(npz, pk)
            B = load_bank(npz, nk)
            pos_protos.append(compute_centroid_prototype(A))
            neg_protos.append(compute_centroid_prototype(B))

        P = np.vstack(pos_protos)  # (E,D)
        N = np.vstack(neg_protos)  # (E,D)

        Spp = P @ P.T
        Spn = P @ N.T
        Snn = N @ N.T
        return expert_names, Spp.astype(np.float32), Spn.astype(np.float32), Snn.astype(np.float32)

    # General path: estimate each cell by sampling in the prompt banks
    Spp = np.zeros((E, E), dtype=np.float32)
    Spn = np.zeros((E, E), dtype=np.float32)
    Snn = np.zeros((E, E), dtype=np.float32)

    # Cache banks lazily (still may be big; cache only if you want speed over memory)
    banks_pos: Dict[str, np.ndarray] = {}
    banks_neg: Dict[str, np.ndarray] = {}

    def get_pos(ex: str) -> np.ndarray:
        if ex not in banks_pos:
            banks_pos[ex] = load_bank(npz, keymap[ex]["positives"])
        return banks_pos[ex]

    def get_neg(ex: str) -> np.ndarray:
        if ex not in banks_neg:
            banks_neg[ex] = load_bank(npz, keymap[ex]["hard_negatives"])
        return banks_neg[ex]

    for i, ex_i in enumerate(expert_names):
        Apos = get_pos(ex_i)
        Aneg = get_neg(ex_i)
        for j, ex_j in enumerate(expert_names):
            Bpos = get_pos(ex_j)
            Bneg = get_neg(ex_j)

            base_seed = seed + 1000 * i + 17 * j

            if metric == "sampled_mean":
                Spp[i, j] = sampled_mean_cosine(Apos, Bpos, pairs=pairs, seed=base_seed)
                Spn[i, j] = sampled_mean_cosine(Apos, Bneg, pairs=pairs, seed=base_seed + 1)
                Snn[i, j] = sampled_mean_cosine(Aneg, Bneg, pairs=pairs, seed=base_seed + 2)
            elif metric == "mean_max":
                # mean over A rows of max similarity into B (sampled)
                Spp[i, j] = sampled_mean_max_cosine(Apos, Bpos, rows=mean_max_rows, cols_per_row=mean_max_cols, seed=base_seed)
                Spn[i, j] = sampled_mean_max_cosine(Apos, Bneg, rows=mean_max_rows, cols_per_row=mean_max_cols, seed=base_seed + 1)
                Snn[i, j] = sampled_mean_max_cosine(Aneg, Bneg, rows=mean_max_rows, cols_per_row=mean_max_cols, seed=base_seed + 2)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    return expert_names, Spp, Spn, Snn


def save_csv(M: np.ndarray, labels: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(M):
            f.write(labels[i] + "," + ",".join(f"{float(x):.6f}" for x in row) + "\n")


def plot_heatmap(M: np.ndarray, labels: List[str], title: str, out_png: str) -> None:
    plt.figure(figsize=(10, 8))
    im = plt.imshow(M, aspect="auto")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=220)


def print_top_pairs(M: np.ndarray, labels: List[str], kind: str, k: int = 12000, skip_diag: bool = True) -> None:
    E = M.shape[0]
    pairs = []
    for i in range(E):
        for j in range(E):
            if skip_diag and i == j:
                continue
            pairs.append((float(M[i, j]), i, j))
    pairs.sort(reverse=True)
    print(f"\nTop {k} pairs: {kind}")
    for val, i, j in pairs[:k]:
        print(f"  {val:.4f}  |  {labels[i]}  vs  {labels[j]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to plip_prompt_embeddings.npz")
    ap.add_argument("--sidecar", default="", help="Optional JSON sidecar (maps expert names to NPZ keys)")
    ap.add_argument("--out_dir", default="expert_similarity_out", help="Output directory")

    ap.add_argument("--metric", default="centroid", choices=["centroid", "sampled_mean", "mean_max"],
                    help="How to summarize bank-to-bank similarity for each matrix cell.")
    ap.add_argument("--pairs", type=int, default=20000, help="For sampled_mean: number of sampled pairs per cell")
    ap.add_argument("--mean_max_rows", type=int, default=2000, help="For mean_max: sampled rows from A per cell")
    ap.add_argument("--mean_max_cols", type=int, default=256, help="For mean_max: sampled candidates from B per row")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--no_csv", action="store_true")
    args = ap.parse_args()

    expert_names, Spp, Spn, Snn = build_matrices(
        npz_path=args.npz,
        sidecar_path=args.sidecar if args.sidecar else None,
        metric=args.metric,
        pairs=args.pairs,
        mean_max_rows=args.mean_max_rows,
        mean_max_cols=args.mean_max_cols,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_csv:
        save_csv(Spp, expert_names, os.path.join(args.out_dir, "Spp_pos_pos.csv"))
        save_csv(Spn, expert_names, os.path.join(args.out_dir, "Spn_pos_neg.csv"))
        save_csv(Snn, expert_names, os.path.join(args.out_dir, "Snn_neg_neg.csv"))

    if not args.no_plots:
        plot_heatmap(Spp, expert_names, f"Spp pos-pos ({args.metric})", os.path.join(args.out_dir, "Spp_pos_pos.png"))
        plot_heatmap(Spn, expert_names, f"Spn pos-neg ({args.metric})", os.path.join(args.out_dir, "Spn_pos_neg.png"))
        plot_heatmap(Snn, expert_names, f"Snn neg-neg ({args.metric})", os.path.join(args.out_dir, "Snn_neg_neg.png"))
        print(f"[OK] wrote heatmaps to {args.out_dir}")

    print_top_pairs(Spp, expert_names, "pos-pos", k=1200, skip_diag=True)
    print_top_pairs(Spn, expert_names, "pos-neg (row=pos, col=neg)", k=1200, skip_diag=False)
    print_top_pairs(Snn, expert_names, "neg-neg", k=1200, skip_diag=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
