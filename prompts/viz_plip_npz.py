#!/usr/bin/env python3
"""
Visualize PLIP prompt embeddings stored in an .npz file.

Expected .npz keys (from the earlier embedding script):
  "<expert_sanitized>__positives"      -> (N, D)
  "<expert_sanitized>__hard_negatives" -> (M, D)

This script:
  - Loads all arrays from .npz
  - Builds labels: expert name + kind (positives vs hard_negatives)
  - Optionally subsamples for speed
  - Reduces to 2D with PCA (fast) or UMAP (better structure)
  - Plots a scatter with matplotlib
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_npz_keys(npz_keys):
    """
    Returns list of (key, expert, kind) where kind is 'positives' or 'hard_negatives'.
    """
    items = []
    for k in npz_keys:
        if k.endswith("__positives"):
            expert = k[:-len("__positives")]
            items.append((k, expert, "positives"))
        elif k.endswith("__hard_negatives"):
            expert = k[:-len("__hard_negatives")]
            items.append((k, expert, "hard_negatives"))
    return items


def sanitize_to_readable(expert_key: str) -> str:
    # Convert "microvascular_proliferation" -> "microvascular proliferation"
    s = expert_key.strip().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_embeddings(npz_path: str, include_hard_negs: bool = True):
    z = np.load(npz_path, allow_pickle=False)
    keys = list(z.keys())
    items = parse_npz_keys(keys)

    if not items:
        raise ValueError(
            "No keys ending in '__positives' or '__hard_negatives' were found. "
            "Check your .npz contents with: np.load(...).files"
        )

    X_parts = []
    labels_expert = []
    labels_kind = []
    labels_key = []

    for k, expert_key, kind in items:
        if (not include_hard_negs) and (kind == "hard_negatives"):
            continue
        arr = z[k]
        if arr.ndim != 2:
            raise ValueError(f"Key {k} is not 2D. Got shape {arr.shape}")

        X_parts.append(arr.astype(np.float32, copy=False))
        labels_expert.extend([sanitize_to_readable(expert_key)] * arr.shape[0])
        labels_kind.extend([kind] * arr.shape[0])
        labels_key.extend([k] * arr.shape[0])

    X = np.vstack(X_parts)
    labels_expert = np.array(labels_expert)
    labels_kind = np.array(labels_kind)
    labels_key = np.array(labels_key)

    return X, labels_expert, labels_kind, labels_key


def subsample(X, labels_expert, labels_kind, labels_key, max_points: int, seed: int):
    n = X.shape[0]
    if (max_points is None) or (max_points <= 0) or (n <= max_points):
        return X, labels_expert, labels_kind, labels_key

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], labels_expert[idx], labels_kind[idx], labels_key[idx]


def reduce_pca(X: np.ndarray, n_components: int = 2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=0)
    Y = pca.fit_transform(X)
    return Y, pca.explained_variance_ratio_


def reduce_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1):
    # Requires: pip install umap-learn
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=0,
    )
    Y = reducer.fit_transform(X)
    return Y


def plot_scatter(
    Y,
    labels_expert,
    labels_kind,
    title: str,
    out_png: str = "",
    show_legend: bool = True,
):
    """Scatter plot with distinct color+marker per (expert, kind) group.

    Legend is placed outside the Axes (right side) so it never obscures points.
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    # Build groups
    groups = {}
    for i in range(Y.shape[0]):
        g = (labels_expert[i], labels_kind[i])
        groups.setdefault(g, []).append(i)

    # Deterministic ordering so color/marker assignment is stable across runs.
    group_items = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    # Assign a unique color + marker per group.
    cmap = plt.get_cmap("tab20", max(1, len(group_items)))
    marker_cycle = [
        "o", "s", "^", "v", "<", ">",
        "D", "P", "X", "*", "p", "h", "H", "8",
        "1", "2", "3", "4", "+", "x",
        ".", ",",
    ]

    for gi, ((expert, kind), idxs) in enumerate(group_items):
        idxs = np.asarray(idxs, dtype=np.int64)
        color = cmap(gi)
        marker = marker_cycle[gi % len(marker_cycle)]
        ax.scatter(
            Y[idxs, 0],
            Y[idxs, 1],
            s=10,
            alpha=0.65,
            color=color,
            marker=marker,
            label=f"{expert} | {kind}",
        )

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    if show_legend:
        ax.legend(
            markerscale=2,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.75, 1.0))
    else:
        fig.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)
        if show_legend:
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
        else:
            plt.savefig(out_png, dpi=200)
        print(f"[OK] saved: {out_png}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to plip_prompt_embeddings.npz")
    ap.add_argument("--method", default="pca", choices=["pca", "umap"], help="Dimensionality reduction method")
    ap.add_argument("--max_points", type=int, default=20000, help="Subsample for speed (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_hard_negs", action="store_true", help="Only plot positives")

    # UMAP params (only used if method=umap)
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)

    ap.add_argument("--out_png", default="", help="If set, saves plot instead of showing it")
    args = ap.parse_args()

    X, labels_expert, labels_kind, labels_key = load_embeddings(
        args.npz, include_hard_negs=(not args.no_hard_negs)
    )

    X, labels_expert, labels_kind, labels_key = subsample(
        X, labels_expert, labels_kind, labels_key, args.max_points, args.seed
    )

    # Normalize to unit length (safe even if already normalized)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-12, None)

    if args.method == "pca":
        Y, evr = reduce_pca(X, n_components=2)
        title = f"PLIP prompt embeddings (PCA) | points={Y.shape[0]} | EVR={evr[0]:.3f},{evr[1]:.3f}"
    else:
        try:
            Y = reduce_umap(X, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
        except ImportError as e:
            raise SystemExit(
                "UMAP selected but umap-learn is not installed. Install with: pip install umap-learn"
            ) from e
        title = f"PLIP prompt embeddings (UMAP cosine) | points={Y.shape[0]}"

    plot_scatter(Y, labels_expert, labels_kind, title=title, out_png=args.out_png)


if __name__ == "__main__":
    main()
