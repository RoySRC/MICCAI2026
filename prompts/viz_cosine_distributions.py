#!/usr/bin/env python3
"""
Visualize (and sanity-check) overlap in PLIP prompt embeddings by plotting
cosine-similarity distributions in the ORIGINAL 512-D space.

Primary use-case (the one you asked for):
  - Compare distribution of cosine similarities for:
      (A) necrosis positives vs necrosis positives
      (B) necrosis positives vs necrosis hard_negatives

Why this helps:
  - If (B) is close to (A), your hard-negatives are not separating from positives
    in text-embedding space, even if your 2D plot looks persuasive.

Input:
  - plip_prompt_embeddings.npz created by your embedding script.
  - Optional JSON sidecar (if you saved it) to map expert names to keys.

Output:
  - Histogram plot (and a text summary) showing both distributions.

Notes:
  - This samples random pairs rather than forming the full O(N^2) matrix.
  - Assumes embeddings are (or will be normalized to) unit length, so dot product
    equals cosine similarity.
"""

import argparse
import json
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def sanitize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-12, None)


def load_npz_arrays(
    npz_path: str,
    expert_name: str,
    sidecar_json: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Returns:
      positives array (N,D), hard_negatives array (M,D), pos_key, neg_key
    """
    z = np.load(npz_path, allow_pickle=False)
    files = set(z.files)

    # If a sidecar JSON exists, prefer its mapping (most robust).
    if sidecar_json:
        with open(sidecar_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "keys" not in meta or expert_name not in meta["keys"]:
            raise ValueError(
                f"Sidecar JSON does not contain a key mapping for expert={expert_name!r}."
            )
        pos_key = meta["keys"][expert_name]["positives"]
        neg_key = meta["keys"][expert_name]["hard_negatives"]
    else:
        # Fall back to sanitized expert key naming convention.
        k = sanitize_key(expert_name)
        pos_key = f"{k}__positives"
        neg_key = f"{k}__hard_negatives"

    if pos_key not in files or neg_key not in files:
        # Helpful listing for debugging
        raise KeyError(
            "Could not find required keys in the .npz.\n"
            f"Requested pos_key={pos_key!r}, neg_key={neg_key!r}\n"
            f"Available keys include (first 40): {sorted(list(files))[:40]}"
        )

    Xp = z[pos_key].astype(np.float32, copy=False)
    Xn = z[neg_key].astype(np.float32, copy=False)

    if Xp.ndim != 2 or Xn.ndim != 2:
        raise ValueError(f"Expected 2D arrays. Got shapes: {Xp.shape}, {Xn.shape}")
    if Xp.shape[1] != Xn.shape[1]:
        raise ValueError(f"Embedding dims differ: {Xp.shape[1]} vs {Xn.shape[1]}")

    return Xp, Xn, pos_key, neg_key


def sample_cosine_pairs(
    A: np.ndarray,
    B: Optional[np.ndarray] = None,
    n_pairs: int = 200_000,
    seed: int = 0,
    batch: int = 8192,
    distinct_when_same: bool = True,
) -> np.ndarray:
    """
    Samples cosine similarities between random pairs.

    If B is None, samples A vs A. If distinct_when_same is True, tries to avoid i=j pairs.
    Computation is batched to keep memory bounded.
    """
    rng = np.random.default_rng(seed)

    if B is None:
        B = A
        same = True
    else:
        same = False

    na = A.shape[0]
    nb = B.shape[0]

    sims = np.empty((n_pairs,), dtype=np.float32)

    done = 0
    while done < n_pairs:
        take = min(batch, n_pairs - done)
        ia = rng.integers(0, na, size=take, dtype=np.int64)
        ib = rng.integers(0, nb, size=take, dtype=np.int64)

        if same and distinct_when_same:
            # Replace any accidental i==j positions (cheap approximate distinctness)
            mask = ia == ib
            if np.any(mask):
                ib[mask] = (ib[mask] + 1) % nb

        # Row-wise dot product (A and B should be normalized)
        sims[done:done + take] = np.sum(A[ia] * B[ib], axis=1)
        done += take

    return sims


def summarize(name: str, sims: np.ndarray) -> None:
    q = np.quantile(sims, [0.01, 0.05, 0.5, 0.95, 0.99])
    print(f"\n{name}")
    print(f"  n={sims.size}")
    print(f"  mean={float(np.mean(sims)):.4f}  std={float(np.std(sims)):.4f}")
    print(f"  quantiles: p01={q[0]:.4f}  p05={q[1]:.4f}  p50={q[2]:.4f}  p95={q[3]:.4f}  p99={q[4]:.4f}")


def plot_hist(
    sims_pospos: np.ndarray,
    sims_posneg: np.ndarray,
    title: str,
    out_png: str = "",
    bins: int = 120,
) -> None:
    plt.figure(figsize=(11, 7))

    # Use "step" hist for clean overlays without manual colors.
    plt.hist(sims_pospos, bins=bins, density=True, histtype="step", linewidth=1.5, label="positives vs positives")
    plt.hist(sims_posneg, bins=bins, density=True, histtype="step", linewidth=1.5, label="positives vs hard_negatives")

    plt.title(title)
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=220)
        print(f"\n[OK] wrote {out_png}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to plip_prompt_embeddings.npz")
    ap.add_argument("--expert", required=True, help="Expert name (matches your prompt pack), e.g. 'necrosis'")
    ap.add_argument("--sidecar_json", default="", help="Optional JSON sidecar produced by your embedding export")
    ap.add_argument("--pairs", type=int, default=250_000, help="Number of random pairs to sample per distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_png", default="", help="If set, saves plot instead of showing it")
    ap.add_argument("--bins", type=int, default=1200)
    args = ap.parse_args()

    Xp, Xn, pos_key, neg_key = load_npz_arrays(
        npz_path=args.npz,
        expert_name=args.expert,
        sidecar_json=(args.sidecar_json or None),
    )

    # Normalize defensively (safe even if already normalized)
    Xp = l2_normalize(Xp)
    Xn = l2_normalize(Xn)

    # Sample similarities
    sims_pospos = sample_cosine_pairs(A=Xp, B=None, n_pairs=args.pairs, seed=args.seed, distinct_when_same=True)
    sims_posneg = sample_cosine_pairs(A=Xp, B=Xn, n_pairs=args.pairs, seed=args.seed + 1, distinct_when_same=False)

    # Print summaries
    print(f"NPZ: {args.npz}")
    print(f"Expert: {args.expert!r}")
    print(f"Keys: positives={pos_key!r} hard_negatives={neg_key!r}")
    summarize("positives vs positives", sims_pospos)
    summarize("positives vs hard_negatives", sims_posneg)

    # Plot
    title = f"{args.expert} cosine similarity distributions (512-D PLIP space)\n{pos_key} vs {neg_key} | sampled pairs={args.pairs}"
    plot_hist(sims_pospos, sims_posneg, title=title, out_png=args.out_png, bins=args.bins)


if __name__ == "__main__":
    main()
