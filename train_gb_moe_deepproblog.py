#!/usr/bin/env python3
"""
train_gb_moe_deepproblog.py

DeepProbLog PyTorch interface integration for GB-vs-normal slide-level supervision:
- Style A: direct training on DeepProbLog query likelihood (weak supervision)
- Style B: explicit EM with program-derived soft pseudo-targets

Assumptions (verify against your installed deepproblog version):
- deepproblog provides Model, Network, Query and a standard training interface.
- Model.solve(...) (or solve_query) returns query success probabilities.
- Networks are wrapped by deepproblog.network.Network and called with tensor inputs.
If any API names differ, search/replace the few marked sections.

Sources:
- DeepProbLog repo & concept: https://github.com/ML-KULeuven/deepproblog  (installation / design) :contentReference[oaicite:3]{index=3}
- Neural predicate overview: https://dtai.cs.kuleuven.be/projects/nesy/deepproblog.html :contentReference[oaicite:4]{index=4}
- DeepProbLog paper: https://arxiv.org/pdf/1805.10872 :contentReference[oaicite:5]{index=5}
- BiomedCLIP HF model card: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 :contentReference[oaicite:6]{index=6}
"""

import importlib
import argparse
import os, json
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip

# -------- DeepProbLog imports (API names may vary slightly by version) --------
# If you get import errors, locate the symbols in your installed package and update these lines.
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from problog.logic import Term, Constant


# Import Engines
try:
    from deepproblog.engines import ExactEngine, ApproximateEngine
except ImportError:
    from deepproblog.engines.exact_engine import ExactEngine
    from deepproblog.engines.approximate_engine import ApproximateEngine



# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

def sample_random_patches(img: torch.Tensor, patch: int, n: int) -> torch.Tensor:
    """
    img: (3,H,W) float tensor (already normalized for encoder)
    returns: (n,3,patch,patch)
    """
    _, H, W = img.shape
    if H < patch or W < patch:
        raise ValueError(f"Image {H}x{W} too small for patch {patch}")
    out = []
    for _ in range(n):
        y = random.randint(0, H - patch)
        x = random.randint(0, W - patch)
        out.append(img[:, y:y+patch, x:x+patch])
    return torch.stack(out, dim=0)

def logsumexp_sim(v: torch.Tensor, proto: torch.Tensor, tau: float) -> torch.Tensor:
    """
    v: (B,D) normalized
    proto: (K,D) normalized
    returns: (B,) logsumexp(cosine/tau)
    """
    sims = (v @ proto.T) / tau
    return torch.logsumexp(sims, dim=-1)


def concept_grounding_loss(
    *,
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    s_oth: torch.Tensor,
    q_present: torch.Tensor,
    margin_neg: float = 0.50,
    margin_oth: float = 0.50,
    gamma_oth: float = 1.00,
) -> torch.Tensor:
    """Posterior-weighted concept grounding regularizer.

    We only regularize samples proportional to q_present (EM posterior that the
    expert is present). For those samples, we encourage a strict ordering:
      s_pos >= s_neg + margin_neg
      s_pos >= s_oth + margin_oth

    This encourages true concept alignment (v matches the expert's own prompt bank)
    and discourages leakage into hard negatives or other experts' prompt banks.

    Notes:
    - Uses softplus as a smooth hinge.
    - Does NOT force absent samples toward any competing concept bank.
    """
    q = q_present.clamp(0.0, 1.0)
    # (B,)
    l_neg = F.softplus((s_neg + float(margin_neg)) - s_pos)
    l_oth = F.softplus((s_oth + float(margin_oth)) - s_pos)
    return (q * (l_neg + float(gamma_oth) * l_oth)).mean()

import re

_NN_LINE = re.compile(
    r"^\s*nn\(\s*([A-Za-z0-9_]+)\s*,.*?\)\s*::\s*([A-Za-z0-9_]+)\s*\(",
)

def extract_expert_names_from_pl(pl_path: str):
    expert_net_names = []
    expert_pred_names = []

    with open(pl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # ProbLog/Problog comment lines start with "%"
            if s.startswith("%"):
                continue

            m = _NN_LINE.match(line)
            if m:
                net_name = m.group(1)
                pred_name = m.group(2)
                expert_net_names.append(net_name)
                expert_pred_names.append(pred_name)

    # Basic sanity checks
    if len(expert_net_names) == 0:
        raise ValueError(f"No nn(...) expert declarations found in {pl_path}")
    if len(expert_net_names) != len(expert_pred_names):
        raise ValueError("Internal parse error: nets and predicates length mismatch")
    if len(set(expert_net_names)) != len(expert_net_names):
        raise ValueError("Duplicate expert net names found in base_pl")
    if len(set(expert_pred_names)) != len(expert_pred_names):
        raise ValueError("Duplicate expert predicate names found in base_pl")

    return expert_net_names, expert_pred_names


# -------------------------
# Sanity Checking
# -------------------------

import math
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import torch
from deepproblog.query import Query
from problog.logic import Term, Constant

PRESENT = Term("present")
ABSENT = Term("absent")

def q_unary(pred: str, idx: int) -> Query:
    return Query(Term(pred, Constant(idx)))

def q_binary_atom(pred: str, idx: int, atom: Term) -> Query:
    # For neural predicates like microvascular_prolif(I, present)
    return Query(Term(pred, Constant(idx), atom))

def q_evidence(idx: int, ev_name: str) -> Query:
    # evidence(I, microvascular_prolif) has a 0-arity atom as second arg
    return Query(Term("evidence", Constant(idx), Term(ev_name)))

def _extract_query_prob(single_solve_item, q: Query) -> float:
    """
    Compatible with two common DeepProbLog return styles:
    - list of objects with `.result[q.query]`
    - list of floats
    """
    if hasattr(single_solve_item, "result"):
        return float(single_solve_item.result[q.query])
    return float(single_solve_item)

def solve_probs_chunked(model, queries: List[Query], chunk_size: int = 256) -> np.ndarray:
    out: List[float] = []
    for k in range(0, len(queries), chunk_size):
        qs = queries[k : k + chunk_size]
        res = model.solve(qs)

        # Tensor output
        if torch.is_tensor(res):
            out.extend([float(x) for x in res.detach().cpu().flatten().tolist()])
            continue

        # List output (either float-like or result objects)
        if isinstance(res, list):
            for r, q in zip(res, qs):
                out.append(_extract_query_prob(r, q))
            continue

        # Single scalar fallback (rare)
        out.extend([float(res)] * len(qs))
    return np.asarray(out, dtype=np.float64)

def _predicate_available(model, q: Query) -> bool:
    try:
        _ = solve_probs_chunked(model, [q], chunk_size=1)
        return True
    except Exception:
        return False

@torch.no_grad()
def sanity_check_mvp_gating(
    model,
    dataset,
    *,
    max_items: int = 200,
    not_normal_thr: float = 0.80,
    low_ctx_thr: float = 0.20,
    trusted_thr: float = 0.50,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Checks two failure modes:
      1) trusted_mvp is almost identical to microvascular_prolif(present)
      2) evidence(microvascular_prolif) fires on "not-normal but low context" cases

    Returns a small metrics dict you can log.
    """

    print("sanity_check_mvp_gating...")
    N = min(len(dataset), max_items)
    idxs = list(range(N))

    # Optional predicate: strong_tumor_context/1 exists only if you added it in the .pl
    has_strong_ctx = _predicate_available(model, q_unary("strong_tumor_context", 0))

    # Build query pack (one pass through the engine)
    queries: List[Query] = []
    for i in idxs:
        queries.extend([
            q_binary_atom("microvascular_prolif", i, PRESENT),
            q_unary("trusted_mvp", i),
            q_evidence(i, "microvascular_prolif"),
            q_binary_atom("normal_parenchyma", i, ABSENT),
            q_unary("tumor_context", i),
            q_unary("good_quality", i),
        ])
        if has_strong_ctx:
            queries.append(q_unary("strong_tumor_context", i))

    probs = solve_probs_chunked(model, queries, chunk_size=256)

    # Unpack
    step = 7 if has_strong_ctx else 6
    p_mvp = probs[0::step]
    p_trusted = probs[1::step]
    p_evid = probs[2::step]
    p_not_normal = probs[3::step]
    p_tumor_ctx = probs[4::step]
    p_goodq = probs[5::step]
    p_strong_ctx = probs[6::step] if has_strong_ctx else None

    # Basic consistency: evidence should match trusted_mvp by construction
    evid_gap = np.abs(p_evid - p_trusted)

    # Gating effectiveness: trusted_mvp should be meaningfully below raw MVP on average
    gate_gap = p_mvp - p_trusted  # should be mostly positive if gating is doing work
    corr = float(np.corrcoef(p_mvp, p_trusted)[0, 1]) if N >= 2 else float("nan")

    # Suspicious: trusted_mvp high even when "not normal" but low context
    if has_strong_ctx:
        low_ctx = p_strong_ctx < low_ctx_thr
    else:
        low_ctx = p_tumor_ctx < low_ctx_thr

    suspicious_mask = (p_not_normal > not_normal_thr) & low_ctx & (p_trusted > trusted_thr) & (p_goodq > 0.50)
    suspicious = np.where(suspicious_mask)[0]
    suspicious_sorted = suspicious[np.argsort(-p_trusted[suspicious])] if suspicious.size else suspicious

    # Print report
    print("\n[sanity] MVP gating report")
    print(f"[sanity] N={N} strong_tumor_context_available={has_strong_ctx}")
    print(f"[sanity] mean P(mvp_present)={p_mvp.mean():.4f}   mean P(trusted_mvp)={p_trusted.mean():.4f}")
    print(f"[sanity] mean (P(mvp_present)-P(trusted_mvp))={gate_gap.mean():.4f}   median={np.median(gate_gap):.4f}")
    print(f"[sanity] corr(P(mvp_present), P(trusted_mvp))={corr:.4f}")
    print(f"[sanity] max(P(trusted_mvp)-P(mvp_present))={(p_trusted - p_mvp).max():.4f}")
    print(f"[sanity] mean |P(evidence_mvp)-P(trusted_mvp)|={evid_gap.mean():.6f}   max={evid_gap.max():.6f}")

    print(f"[sanity] suspicious_count={int(suspicious_sorted.size)} "
          f"(not_normal>{not_normal_thr}, ctx<{low_ctx_thr}, trusted>{trusted_thr}, goodq>0.5)")

    if suspicious_sorted.size:
        print("[sanity] top suspicious cases:")
        for j in suspicious_sorted[:top_k]:
            i = idxs[int(j)]
            y = int(dataset.labels[i]) if hasattr(dataset, "labels") else None
            ctx_val = float(p_strong_ctx[j]) if has_strong_ctx else float(p_tumor_ctx[j])
            print(
                f"  idx={i} label={y} "
                f"p_mvp={p_mvp[j]:.3f} p_trusted={p_trusted[j]:.3f} "
                f"p_not_normal={p_not_normal[j]:.3f} p_ctx={ctx_val:.3f} p_goodq={p_goodq[j]:.3f}"
            )

    metrics = {
        "mvp_mean": float(p_mvp.mean()),
        "trusted_mvp_mean": float(p_trusted.mean()),
        "gate_gap_mean": float(gate_gap.mean()),
        "gate_gap_median": float(np.median(gate_gap)),
        "mvp_trusted_corr": corr,
        "trusted_minus_mvp_max": float((p_trusted - p_mvp).max()),
        "evidence_trusted_abs_gap_mean": float(evid_gap.mean()),
        "suspicious_count": float(suspicious_sorted.size),
    }
    return metrics


# -------------------------
# Data: image-level GB vs normal
# -------------------------

@dataclass
class Sample:
    idx: int
    label: int  # 1=GB, 0=normal

class SlideDataset(Dataset):
    """
    Stores slides as tensors (3,H,W) already in encoder normalization space.
    If you have WSIs, replace __getitem__ to load per-slide images from disk.
    """
    def __init__(self, images: List[torch.Tensor], labels: List[int]):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> Tuple[int, torch.Tensor, int]:
        return i, self.images[i], self.labels[i]


# -------------------------
# Frozen BiomedCLIP image encoder (ViT-B/16 224)
# -------------------------

class FrozenBiomedCLIP(nn.Module):
    def __init__(self, hf_id: str, device: torch.device):
        super().__init__()
        model, preprocess = open_clip.create_model_from_pretrained(hf_id)
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.preprocess = preprocess  # PIL transform; we will implement tensor normalize separately

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x)


# -------------------------
# Expert head: patches -> expert embedding -> predicate probs [present,absent]
# -------------------------

class ExpertProjector(nn.Module):
    def __init__(self, d_enc: int, d_concept: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_enc, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_concept),
            )
        else:
            self.net = nn.Linear(d_enc, d_concept)

    def forward(self, z_patch: torch.Tensor) -> torch.Tensor:
        """
        z_patch: (B,P,Denc)
        returns: (B,Dc) normalized
        """
        e_patch = self.net(z_patch)         # (B,P,Dc)
        e_sum = e_patch.sum(dim=1)          # (B,Dc)
        return l2norm(e_sum)


class ConceptBank(nn.Module):
    """
    pos[e]: (Kpos,Dc), neg[e]: (Kneg,Dc), already normalized.
    """
    def __init__(self, pos: Dict[str, torch.Tensor], neg: Dict[str, torch.Tensor]):
        super().__init__()
        self.experts = sorted(list(pos.keys()))
        self.other_pos_cache = None
        for e in self.experts:
            self.register_buffer(f"{e}__pos", l2norm(pos[e].float()))
            self.register_buffer(f"{e}__neg", l2norm(neg[e].float()))

    def pos(self, e: str) -> torch.Tensor:
        return getattr(self, f"{e}__pos")

    def neg(self, e: str) -> torch.Tensor:
        return getattr(self, f"{e}__neg")

    def other_pos(self, e: str) -> torch.Tensor:
        if self.other_pos_cache is not None:
            return self.other_pos_cache
        xs = [self.pos(x) for x in self.experts if x != e]
        if not xs:
            return self.pos(e)[:0]
        self.other_pos_cache = torch.cat(xs, dim=0)
        return self.other_pos_cache


class MoEExpertModule(nn.Module):
    """
    A single expert predicate network for DeepProbLog:
    input: indices (B,) long
    output: (B,2) probs [present,absent]
    """
    def __init__(
        self,
        name: str,
        dataset_ref: SlideDataset,
        encoder: FrozenBiomedCLIP,
        head: ExpertProjector,
        bank: ConceptBank,
        device: torch.device,
        n_patches: int = 32,
        patch: int = 224,
        tau: float = 0.07,
        repel_weight: float = 0.3,
    ):
        super().__init__()
        self.name = name
        self.ds = dataset_ref
        self.encoder = encoder
        self.head = head
        self.bank = bank
        self.device = device
        self.n_patches = n_patches
        self.patch = patch
        self.tau = tau
        self.repel_weight = repel_weight

        # Infer encoder dim once
        with torch.no_grad():
            dummy = torch.zeros(1, 3, patch, patch, device=device)
            d_enc = self.encoder.encode_image(dummy).shape[-1]
        self.d_enc = d_enc

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        """Return only predicate probabilities.

        This keeps the DeepProbLog interface stable: DeepProbLog expects the
        network call to return a (B,2) tensor (or (2,) for a single grounding).
        """
        probs, _ = self._forward_impl(idxs, return_stats=False)
        return probs

    def forward_with_stats(self, idxs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return predicate probabilities plus concept-grounding statistics.

        Returns:
          probs: (B,2)
          stats: dict with keys {"v", "s_pos", "s_neg", "s_oth"}
        """
        return self._forward_impl(idxs, return_stats=True)

    def _forward_impl(
        self,
        idxs: torch.Tensor,
        *,
        return_stats: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Internal forward shared by inference and EM M-step."""
        def _to_int(x):
            if isinstance(x, int):
                return int(x)
            # ProbLog Constant-like objects often expose a .value attribute
            if hasattr(x, "value"):
                try:
                    return int(x.value)
                except Exception:
                    pass
            # Fallback: parse string representation (often "0", "1", ...)
            try:
                return int(str(x))
            except Exception as e:
                raise TypeError(f"{self.name}: expected an integer index, got {type(x)}: {x}") from e

        # single grounding from DeepProbLog if idxs is not a list/tuple and is not a batched tensor
        single = (not isinstance(idxs, (list, tuple))) and (not isinstance(idxs, torch.Tensor) or idxs.ndim == 0)

        if isinstance(idxs, torch.Tensor):
            idx_list = idxs.detach().to("cpu").tolist()
            if not isinstance(idx_list, list):
                idx_list = [idx_list]
            idx_list = [_to_int(x) for x in idx_list]
        elif isinstance(idxs, (list, tuple)):
            idx_list = [_to_int(x) for x in idxs]
        else:
            idx_list = [_to_int(idxs)]

        imgs = []
        for i in idx_list:
            img = self.ds.images[i]
            patches = sample_random_patches(img, self.patch, self.n_patches)
            imgs.append(patches)


        patches = torch.stack(imgs, dim=0).to(self.device)  # (B,P,3,224,224)

        B, P, C, H, W = patches.shape
        flat = patches.view(B * P, C, H, W)
        with torch.no_grad():
            z = self.encoder.encode_image(flat)  # (B*P,Denc)
        z = z.view(B, P, -1)

        v = self.head(z)  # (B,Dc)

        s_pos = logsumexp_sim(v, self.bank.pos(self.name), tau=self.tau)
        s_neg = logsumexp_sim(v, self.bank.neg(self.name), tau=self.tau)

        oth = self.bank.other_pos(self.name)
        if oth.numel() > 0:
            s_oth = logsumexp_sim(v, oth, tau=self.tau)
        else:
            s_oth = torch.zeros_like(s_pos)

        logit_present = (s_pos - s_neg) - self.repel_weight * s_oth
        logits = torch.stack([logit_present, -logit_present], dim=-1)
        probs = F.softmax(logits, dim=-1)

        if single:
            probs_out = probs[0]
            if return_stats:
                stats_out = {
                    "v": v[0],
                    "s_pos": s_pos[0],
                    "s_neg": s_neg[0],
                    "s_oth": s_oth[0],
                }
            else:
                stats_out = None
        else:
            probs_out = probs
            if return_stats:
                stats_out = {
                    "v": v,
                    "s_pos": s_pos,
                    "s_neg": s_neg,
                    "s_oth": s_oth,
                }
            else:
                stats_out = None

        return probs_out, stats_out


# -------------------------
# DeepProbLog program helpers (add gb/normal + joint predicates for EM)
# -------------------------

def augment_program_for_training(
    base_pl_path: str,
    out_pl_path: str,
    expert_predicates: List[str],
    image_var: str = "I",
) -> None:
    """
    Writes a new .pl that:
    - preserves your base program
    - adds gb/1 and normal/1 wrappers
    - adds joint_* predicates for EM posterior computation
    """
    with open(base_pl_path, "r", encoding="utf-8") as f:
        base = f.read().rstrip() + "\n"

    # These wrappers assume your prediction/2 is exactly as you posted.
    extra = "\n% ---- Added for training convenience ----\n"
    extra += "gb(I) :- prediction(I, grade4_features_present).\n"
    extra += "normal(I) :- prediction(I, grade4_features_not_detected).\n"

    # Joint predicates for EM posteriors: P(expert_present ∧ label)
    for e in expert_predicates:
        extra += f"joint_{e}_gb(I) :- gb(I), {e}(I, present).\n"
        extra += f"joint_{e}_normal(I) :- normal(I), {e}(I, present).\n"

    with open(out_pl_path, "w", encoding="utf-8") as f:
        f.write(base + extra)


# -------------------------
# Query builders
# -------------------------

def q_label(idx: int, is_gb: bool) -> Query:
    return Query(Term("gb" if is_gb else "normal", Constant(idx)))

def q_joint(idx: int, expert_name: str, is_gb: bool) -> Query:
    # joint_expert_gb(I) or joint_expert_normal(I)
    pred = f"joint_{expert_name}_{'gb' if is_gb else 'normal'}"
    return Query(Term(pred, Constant(idx)))


# -------------------------
# Style A: direct query training
# -------------------------

def train_direct(
    model: Model,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    """
    Directly maximize query likelihood via DeepProbLog's own training.
    We implement the loop explicitly so you can add regularizers easily.
    """

    # Collect PyTorch parameters from all DeepProbLog networks
    torch_params = []
    seen = set()

    for net in model.networks.values():          # dict: name -> deepproblog.network.Network :contentReference[oaicite:0]{index=0}
        for p in net.parameters():              # delegates to underlying torch.nn.Module.parameters() :contentReference[oaicite:1]{index=1}
            if p.requires_grad and id(p) not in seen:
                torch_params.append(p)
                seen.add(id(p))

    opt = torch.optim.Adam(torch_params, lr=lr)


    model.train()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n = 0

        for idxs, _, labels in train_loader:
            idxs = idxs.to(device)
            labels = labels.to(device)

            queries = []
            targets = []
            for i, y in zip(idxs.tolist(), labels.tolist()):
                is_gb = bool(y == 1)
                queries.append(q_label(i, is_gb))
                targets.append(torch.tensor(1.0, device=device))  # want query to succeed

            # --- DeepProbLog probability of each query ---
            # API NOTE: depending on version, this could be model.solve(queries) or model.solve_query(q).
            # We support both patterns.
            # --- DeepProbLog probability of each query ---
            # model.solve(batch) returns List[Result], not floats
            results = model.solve(queries)

            probs_list = []
            for q, r in zip(queries, results):
                res_dict = r.result  # Dict[Term, Union[float, torch.Tensor]]

                # Usually the key is exactly q.query. If not, fall back to the single entry.
                if q.query in res_dict:
                    v = res_dict[q.query]
                elif len(res_dict) == 1:
                    v = next(iter(res_dict.values()))
                else:
                    preview = ", ".join(str(k) for k in list(res_dict.keys())[:5])
                    raise KeyError(
                        f"Could not find query term {q.query} in Result keys. "
                        f"First keys: {preview}"
                    )

                if torch.is_tensor(v):
                    v = v.to(device=device, dtype=torch.float32)
                    if v.numel() != 1:
                        raise ValueError(f"Expected scalar probability, got shape {tuple(v.shape)} for query {q}")
                    v = v.reshape(())  # ensure scalar tensor
                else:
                    v = torch.tensor(float(v), device=device, dtype=torch.float32)

                probs_list.append(v)

            probs = torch.stack(probs_list, dim=0)  # shape (N,)


            # Negative log likelihood: -log P(query=true)
            eps = 1e-8
            loss = -torch.log(probs.clamp_min(eps)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu()) * len(queries)
            n += len(queries)

        print(f"[direct] epoch={ep} nll={total_loss/max(n,1):.6f}")


# -------------------------
# Style B: EM (E-step via program posteriors, M-step supervised on soft targets)
# -------------------------

@torch.no_grad()
def em_e_step(
    model: Model,
    dataset: SlideDataset,
    expert_net_names: List[str],
    expert_pred_names: List[str],
    device: torch.device,
    max_items: Optional[int] = None,
    *,
    chunk_size: int = 5120,
) -> Dict[str, torch.Tensor]:
    """
    Computes soft pseudo-targets q_e(i) = P(expert_present | label(i))
    via q = P(joint)/P(label).

    Key change vs. the original:
    - batches queries and calls model.solve(...) in chunks instead of per-example.
      This is usually the dominant speedup because Model.solve expects a batch and
      internally groups NN calls per net for batched evaluation. :contentReference[oaicite:1]{index=1}

    Returns:
      soft_targets[expert_net_name] = (N,) float tensor on CPU
    """
    N = len(dataset) if max_items is None else min(len(dataset), max_items)
    idxs = list(range(N))

    # ---- Label probabilities: P(gb(I)) or P(normal(I)) ----
    label_queries: List[Query] = [
        q_label(i, is_gb=bool(dataset.labels[i] == 1)) for i in idxs
    ]
    label_probs_np = solve_probs_chunked(model, label_queries, chunk_size=chunk_size)
    label_probs = torch.from_numpy(label_probs_np).to(dtype=torch.float32)  # CPU (N,)
    print(f"{label_probs = }")

    # NOTE on multi-GPU:
    # DeepProbLog's solver runs in a single Python process; it does not natively
    # parallelize inference across multiple GPUs. If you truly need multi-GPU, the
    # standard approach is to shard idxs across *processes*, each process constructs
    # its own Model(+Engine) pinned to one GPU, runs solve() on its shard, then you
    # concatenate results. The batching below is usually the biggest win first.

    soft_targets: Dict[str, torch.Tensor] = {}

    # ---- For each expert: joint probabilities P(label(I) ∧ expert_present(I)) ----
    for net_name, pred_name in zip(expert_net_names, expert_pred_names):
        print(f"computing soft targets for {net_name}...")
        joint_queries: List[Query] = [
            q_joint(i, expert_name=pred_name, is_gb=bool(dataset.labels[i] == 1))
            for i in idxs
        ]
        joint_probs_np = solve_probs_chunked(model, joint_queries, chunk_size=chunk_size)
        joint_probs = torch.from_numpy(joint_probs_np).to(dtype=torch.float32)  # CPU (N,)

        q = joint_probs / label_probs.clamp_min(1e-8)
        soft_targets[net_name] = q.clamp(0.0, 1.0)

    print("Displaying soft targets...") 
    for expert, t in soft_targets.items(): 
        print(expert) 
        print(t)
    return soft_targets


def em_m_step(
    networks: Dict[str, nn.Module],
    dataset: SlideDataset,
    soft_targets: Dict[str, torch.Tensor],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    *,
    lambda_ground: float = 0.0,
    ground_margin_neg: float = 0.50,
    ground_margin_oth: float = 0.50,
    ground_gamma_oth: float = 1.00,
) -> None:
    """
    M-step: train each expert network to match q_e(i) (soft present probability).

    Important:
    - This M-step ignores the full logic graph and just fits neural predicates to EM posteriors.
    - Then you re-run E-step with the updated predicates plugged back into DeepProbLog.
    """
    params = []
    for m in networks.values():
        params += [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    loader = DataLoader(list(range(len(dataset))), batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for idxs in loader:
            idxs = idxs.to(device)

            loss = 0.0
            for e, net in networks.items():
                # For EM M-step, we optionally pull concept-grounding statistics.
                if hasattr(net, "forward_with_stats"):
                    probs, stats = net.forward_with_stats(idxs)  # type: ignore[attr-defined]
                else:
                    probs, stats = net(idxs), None

                p_present = probs[:, 0]  # present prob
                q = soft_targets[e][idxs.detach().to("cpu")].to(device)  # (B,)

                # BCE with soft targets
                loss = loss + F.binary_cross_entropy(p_present, q)

                # Posterior-weighted concept grounding regularizer
                if (stats is not None) and (lambda_ground > 0.0):
                    g = concept_grounding_loss(
                        s_pos=stats["s_pos"],
                        s_neg=stats["s_neg"],
                        s_oth=stats["s_oth"],
                        q_present=q,
                        margin_neg=ground_margin_neg,
                        margin_oth=ground_margin_oth,
                        gamma_oth=ground_gamma_oth,
                    )
                    loss = loss + float(lambda_ground) * g

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu()) * idxs.shape[0]
            n += idxs.shape[0]

        print(f"[em-m] epoch={ep} loss={total/max(n,1):.6f}")


def train_em(
    model: Model,
    networks: Dict[str, nn.Module],
    dataset: SlideDataset,
    expert_net_names: List[str],
    expert_pred_names: List[str],
    device: torch.device,
    em_rounds: int,
    e_max_items: Optional[int],
    m_epochs: int,
    lr: float,
    batch_size: int,
    *,
    em_lambda_ground: float = 0.0,
    em_ground_margin_neg: float = 0.50,
    em_ground_margin_oth: float = 0.50,
    em_ground_gamma_oth: float = 1.00,
) -> None:
    """
    Full EM (Expectation Maximization) outer loop.
    """
    for r in range(1, em_rounds + 1):
        print(f"[em] round {r}/{em_rounds} E-step...")
        soft = em_e_step(
            model, 
            dataset, 
            expert_net_names, 
            expert_pred_names, 
            device=device, 
            max_items=e_max_items
        )

        # _ = sanity_check_mvp_gating(
        #     model=model,
        #     dataset=dataset,
        #     max_items=200,
        #     not_normal_thr=0.80,
        #     low_ctx_thr=0.20,
        #     trusted_thr=0.50,
        #     top_k=20,
        # )

        print(f"[em] round {r}/{em_rounds} M-step...")
        em_m_step(
            networks=networks,
            dataset=dataset,
            soft_targets=soft,
            device=device,
            epochs=m_epochs,
            lr=lr,
            batch_size=batch_size,
            lambda_ground=em_lambda_ground,
            ground_margin_neg=em_ground_margin_neg,
            ground_margin_oth=em_ground_margin_oth,
            ground_gamma_oth=em_ground_gamma_oth,
        )

        # DeepProbLog model already holds references to the torch nets; no extra “refresh” needed
        # unless your Model clones networks internally (rare).


# -------------------------
# Wiring DeepProbLog networks
# -------------------------

import importlib

def _resolve_engine_class(engine_kind: str):
    engine_kind = engine_kind.lower().strip()
    if engine_kind not in {"exact", "approx", "approximate"}:
        raise ValueError(f"Unknown engine kind: {engine_kind}")

    desired = "ExactEngine" if engine_kind == "exact" else "ApproximateEngine"

    # Try a few likely import locations (covers small version differences)
    if desired == "ExactEngine":
        candidates = [
            ("deepproblog.engines", desired),
            ("deepproblog.engines.exact_engine", desired),
        ]
    elif desired == "ApproximateEngine":
        candidates = [
            ("deepproblog.engines", desired),
            ("deepproblog.engines.approximate_engine", desired),
        ]

    last_err = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)
        except Exception as e:
            last_err = e

    raise ImportError(f"Could not import {desired}. Tried: {candidates}. Last error: {last_err}")

def configure_model_engine(model: Model, engine_kind: str, *, cache: bool = False) -> None:
    EngineCls = _resolve_engine_class(engine_kind)
    engine = EngineCls(model)
    model.set_engine(engine, cache=cache)

def build_deepproblog_model(pl_path, expert_nets, *, engine_kind="exact", cache=False) -> Model:
    dpl_networks = [Network(net, name) for name, net in expert_nets.items()]
    model = Model(pl_path, dpl_networks)

    # IMPORTANT: without this, model.solver stays None and model.solve(...) crashes
    configure_model_engine(model, engine_kind, cache=cache)
    return model



# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_pl", type=str, required=True, help="Path to your base moe_glioma_histology.pl")
    ap.add_argument("--work_pl", type=str, default="moe_glioma_histology_train.pl", help="Augmented .pl output path")

    ap.add_argument("--train_mode", type=str, choices=["direct", "em"], required=True)
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=16, help="Direct training epochs (train_mode=direct)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)

    # EM controls
    ap.add_argument("--em_rounds", type=int, default=5)
    ap.add_argument("--em_e_max_items", type=int, default=0, help="0 means use full dataset for E-step")
    ap.add_argument("--em_m_epochs", type=int, default=16)

    # EM M-step: concept grounding regularizer (added on top of BCE)
    ap.add_argument("--em_lambda_ground", type=float, default=0.25)
    ap.add_argument("--em_ground_margin_neg", type=float, default=0.00)
    ap.add_argument("--em_ground_margin_oth", type=float, default=0.50)
    ap.add_argument("--em_ground_gamma_oth", type=float, default=1.00) # how much of other concept sim. loss to incorporate in grounding loss

    # BiomedCLIP / patching
    ap.add_argument("--biomedclip_id", type=str, default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    ap.add_argument("--n_patches", type=int, default=32)
    ap.add_argument("--patch", type=int, default=224)

    # Concept space
    ap.add_argument("--d_concept", type=int, default=512)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--repel_weight", type=float, default=0.3)

    # Add Engine Switch
    ap.add_argument("--dpl_engine", type=str, default="exact", choices=["exact", "approx"])
    ap.add_argument("--dpl_cache", action="store_true", help="")

    # Data
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--data_base_path", type=str, required=True)
    ap.add_argument("--skip_clip_norm", action="store_false", help="Set if your tensors are already CLIP-normalized")



    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")


    # Placeholder: random tensors (REPLACE THIS)
    # Load slides from CSV (expects columns: path,label)
    import csv
    from pathlib import Path
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    # CLIP-style normalization (BiomedCLIP uses CLIP-like image normalization)
    CLIP_MEAN = torch.tensor([0.48145466, 0.45782750, 0.40821073]).view(3, 1, 1)
    CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    csv_path = args.data_csv          # add this arg, or hardcode "/path/to/P1.csv"
    apply_clip_norm = not getattr(args, "skip_clip_norm", False)

    paths, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(f"{args.data_base_path}/{row['path']}")
            labels.append(int(row["label"]))

    images: List[torch.Tensor] = []
    for p in paths:
        pil = Image.open(p).convert("RGB")
        x = pil_to_tensor(pil).float() / 255.0  # (3,H,W), in [0,1]
        if apply_clip_norm:
            x = (x - CLIP_MEAN) / CLIP_STD
        images.append(x)
        print(p)

    print(f"Loaded {len(images)} slides from {csv_path}. Label counts: "
          f"{sum(labels)} positives, {len(labels) - sum(labels)} negatives")

    ds = SlideDataset(images, labels)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8)




    # Concept bank placeholders (REPLACE with your text-prompt embeddings)
    expert_net_names, expert_pred_names = extract_expert_names_from_pl(args.base_pl)
    print(f"{expert_net_names = }")
    print(f"{expert_pred_names = }")

    
    # --- Replace the random ConceptBank initialization with this ---
    concept_npz_path = "biomedclip.npz"
    concept_sidecar_path = "biomedclip.json"

    with open(concept_sidecar_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Sidecar declares embedding dimension (should match args.d_concept)
    sidecar_dim = int(meta["embed_dim"])
    if sidecar_dim != int(args.d_concept):
        raise ValueError(f"Concept dim mismatch: sidecar embed_dim={sidecar_dim} but args.d_concept={args.d_concept}")

    keymap = meta["keys"]  # expert_name (human) -> {"positives": npz_key, "hard_negatives": npz_key}

    npz = np.load(concept_npz_path)

    # Map your network names to the sidecar's human-readable expert names.
    # Note: you have 14 expert nets, but the sidecar has both "edema" and "microcysts" separately.
    # We merge them into edema_net by concatenating their banks.
    net_to_sidecar_experts = {
        "necrosis_net": ["necrosis"],
        "mvp_net": ["microvascular proliferation"],
        "vascularity_net": ["high vascularity"],
        "hemorrhage_net": ["hemorrhage"],
        "thrombosis_net": ["thrombosis"],
        "normal_net": ["normal parenchyma"],
        "artifact_net": ["artifact"],
        "hypercell_net": ["hypercellularity"],
        "pleomorphism_net": ["pleomorphism"],
        "mitosis_net": ["mitoses"],
        "reactive_net": ["reactive gliosis"],
        "edema_net": ["edema", "microcysts"],
        "calcification_net": ["calcification"],
        "infiltration_net": ["infiltration cues"],
    }

    def load_and_concat(expert_names, which: str) -> torch.Tensor:
        # which is either "positives" or "hard_negatives"
        ts = []
        for name in expert_names:
            if name not in keymap:
                raise KeyError(f"Sidecar missing expert {name!r}. Available: {list(keymap.keys())}")
            npz_key = keymap[name][which]
            if npz_key not in npz.files:
                raise KeyError(f"NPZ missing key {npz_key!r} for expert {name!r}")
            arr = npz[npz_key]  # (K, D)
            ts.append(torch.from_numpy(arr).float())
        return torch.cat(ts, dim=0) if len(ts) > 1 else ts[0]

    pos = {}
    neg = {}
    for net_name in expert_net_names:
        sidecar_names = net_to_sidecar_experts[net_name]
        pos[net_name] = (load_and_concat(sidecar_names, "positives"))
        neg[net_name] = (load_and_concat(sidecar_names, "hard_negatives"))

    bank = ConceptBank(pos=pos, neg=neg).to(device)



    # Frozen encoder
    encoder = FrozenBiomedCLIP(args.biomedclip_id, device=device)

    # Infer encoder dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.patch, args.patch, device=device)
        d_enc = encoder.encode_image(dummy).shape[-1]

    # Build per-expert heads + expert predicate networks
    expert_heads = {
        e: ExpertProjector(d_enc=d_enc, d_concept=args.d_concept, hidden=args.head_hidden).to(device)
        for e in expert_net_names
    }
    expert_nets = {
        e: MoEExpertModule(
            name=e,
            dataset_ref=ds,
            encoder=encoder,
            head=expert_heads[e],
            bank=bank,
            device=device,
            n_patches=args.n_patches,
            patch=args.patch,
            tau=args.tau,
            repel_weight=args.repel_weight,
        ).to(device)
        for e in expert_net_names
    }

    # Augment your logic program for training + EM
    augment_program_for_training(
        base_pl_path=args.base_pl,
        out_pl_path=args.work_pl,
        expert_predicates=expert_pred_names,
    )

    # Build DeepProbLog model
    model = build_deepproblog_model(
        args.work_pl,
        expert_nets,
        engine_kind=args.dpl_engine,
        cache=args.dpl_cache,
    )


    # Train
    if args.train_mode == "direct":
        train_direct(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )
    else:
        e_max = None if args.em_e_max_items == 0 else args.em_e_max_items
        train_em(
            model=model,
            networks=expert_nets,
            dataset=ds,
            expert_net_names=expert_net_names,
            expert_pred_names=expert_pred_names,
            device=device,
            em_rounds=args.em_rounds,
            e_max_items=e_max,
            m_epochs=args.em_m_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            em_lambda_ground=args.em_lambda_ground,
            em_ground_margin_neg=args.em_ground_margin_neg,
            em_ground_margin_oth=args.em_ground_margin_oth,
            em_ground_gamma_oth=args.em_ground_gamma_oth,
        )


if __name__ == "__main__":
    main()
