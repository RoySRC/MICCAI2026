#!/usr/bin/env python3
"""
Embed a concept prompt pack into text embeddings.

Supported encoders:
  - plip : CLIP text encoder from a PLIP-compatible CLIPModel (e.g., vinid/plip)
  - mstar: mSTAR-style report text encoder (BioBERT-like) for prompt embedding

Outputs:
  - A single .pt file with:
      {
        "model_id": str,
        "embed_dim": int,
        "meta": {...},
        "experts": {
          expert_name: {
            "positives": {"prompts": [str...], "embeddings": FloatTensor[N,D]},
            "hard_negatives": {"prompts": [str...], "embeddings": FloatTensor[M,D]},
          },
          ...
        }
      }

Optionally also writes:
  - .npz (arrays) + .json (prompt lists + key mapping)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


# -------------------------
# JSON loading + validation
# -------------------------

REQUIRED_TOP_KEYS = {"meta", "blacklists", "experts"}
REQUIRED_META_KEYS = {"modality_phrase", "experts", "note"}
REQUIRED_BLACKLIST_KEYS = {"global", "per_expert", "hard_negative_integrity"}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_prompt_pack(pack: Dict[str, Any]) -> None:
    _assert(isinstance(pack, dict), "Prompt pack must be a JSON object.")
    _assert(REQUIRED_TOP_KEYS.issubset(pack.keys()), f"Missing required top-level keys: {REQUIRED_TOP_KEYS - set(pack.keys())}")

    meta = pack["meta"]
    blacklists = pack["blacklists"]
    experts = pack["experts"]

    _assert(isinstance(meta, dict), "`meta` must be an object.")
    # _assert(REQUIRED_META_KEYS.issubset(meta.keys()), f"Missing required meta keys: {REQUIRED_META_KEYS - set(meta.keys())}")
    # _assert(isinstance(meta["modality_phrase"], str) and len(meta["modality_phrase"]) > 0, "`meta.modality_phrase` must be a non-empty string.")
    _assert(isinstance(meta["experts"], list) and len(meta["experts"]) > 0, "`meta.experts` must be a non-empty list.")
    _assert(all(isinstance(x, str) for x in meta["experts"]), "`meta.experts` entries must be strings.")

    _assert(isinstance(blacklists, dict), "`blacklists` must be an object.")
    _assert(REQUIRED_BLACKLIST_KEYS.issubset(blacklists.keys()),
            f"Missing required blacklist keys: {REQUIRED_BLACKLIST_KEYS - set(blacklists.keys())}")

    _assert(isinstance(blacklists["global"], list) and all(isinstance(x, str) for x in blacklists["global"]),
            "`blacklists.global` must be a list of strings.")
    _assert(isinstance(blacklists["per_expert"], dict), "`blacklists.per_expert` must be an object.")
    _assert(isinstance(blacklists["hard_negative_integrity"], dict), "`blacklists.hard_negative_integrity` must be an object.")

    _assert(isinstance(experts, dict), "`experts` must be an object mapping expert name -> prompt sets.")
    # Ensure meta.experts matches expert keys
    expert_names_meta = set(meta["experts"])
    expert_names_obj = set(experts.keys())
    _assert(expert_names_meta == expert_names_obj,
            f"meta.experts ({sorted(expert_names_meta)}) must match experts keys ({sorted(expert_names_obj)}).")

    for name, payload in experts.items():
        _assert(isinstance(payload, dict), f"experts['{name}'] must be an object.")
        _assert("positives" in payload and "hard_negatives" in payload, f"experts['{name}'] must have 'positives' and 'hard_negatives'.")
        _assert(isinstance(payload["positives"], list) and all(isinstance(x, str) for x in payload["positives"]),
                f"experts['{name}'].positives must be a list of strings.")
        _assert(isinstance(payload["hard_negatives"], list) and all(isinstance(x, str) for x in payload["hard_negatives"]),
                f"experts['{name}'].hard_negatives must be a list of strings.")


# -------------------------
# Prompt building + linting
# -------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_full_prompt(modality_phrase: str, prompt: str, mode: str) -> str:
    """
    mode:
      - "prefix": full = modality_phrase + " " + prompt
      - "ellipsis": replace first occurrence of "..." or "…" in modality_phrase with prompt
    """
    modality_phrase = normalize_ws(modality_phrase)
    prompt = normalize_ws(prompt)

    # If user already included the modality phrase in the prompt, don't duplicate it.
    if prompt.lower().startswith(modality_phrase.lower()):
        return prompt

    if mode == "ellipsis":
        if "..." in modality_phrase:
            return normalize_ws(modality_phrase.replace("...", prompt, 1))
        if "…" in modality_phrase:
            return normalize_ws(modality_phrase.replace("…", prompt, 1))
        # fall back
        return normalize_ws(f"{modality_phrase} {prompt}")

    # default: prefix
    return normalize_ws(f"{modality_phrase} {prompt}")


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = x.strip()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def compile_terms(terms: List[str]) -> List[str]:
    # Normalize terms for simple substring matching (case-insensitive).
    # You can swap this to regex word-boundary matching if you want stricter behavior.
    return [t.strip().lower() for t in terms if isinstance(t, str) and t.strip()]


def find_violations(text: str, banned_terms: List[str]) -> List[str]:
    tl = text.lower()
    hits = []
    for term in banned_terms:
        if term and tl.startswith(term.lower()):
            hits.append(term)
    return hits


def lint_prompts(
    expert_name: str,
    positives: List[str],
    hard_negs: List[str],
    modality_phrase: str,
    global_ban: List[str],
    per_expert_ban_map: Dict[str, List[str]],
    hard_neg_integrity_map: Dict[str, List[str]],
    build_mode: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      cleaned_positives_full, cleaned_hard_negs_full, list_of_error_strings
    """
    errors: List[str] = []

    global_terms = compile_terms(global_ban)
    per_expert_terms = compile_terms(per_expert_ban_map.get(expert_name, []))
    integrity_terms = compile_terms(hard_neg_integrity_map.get(expert_name, []))

    # Build full prompts with modality phrase
    pos_full = [build_full_prompt(modality_phrase, p, build_mode) for p in positives]
    neg_full = [build_full_prompt(modality_phrase, p, build_mode) for p in hard_negs]

    # Dedupe
    pos_full = dedupe_preserve_order(pos_full)
    neg_full = dedupe_preserve_order(neg_full)

    # Lint positives: global + per-expert
    cleaned_pos = []
    for p in pos_full:
        v_global = find_violations(p, global_terms)
        v_per = find_violations(p, per_expert_terms)
        if v_global or v_per:
            errors.append(f"[{expert_name}][positives] rejected: {p!r} | global_hits={v_global} per_expert_hits={v_per}")
        else:
            cleaned_pos.append(p)

    # Lint hard negatives: global + hard_negative_integrity
    cleaned_neg = []
    for p in neg_full:
        v_global = find_violations(p, global_terms)
        v_integrity = find_violations(p, integrity_terms)
        if v_global or v_integrity:
            errors.append(f"[{expert_name}][hard_negatives] rejected: {p!r} | global_hits={v_global} integrity_hits={v_integrity}")
        else:
            cleaned_neg.append(p)

    if len(cleaned_pos) == 0:
        errors.append(f"[{expert_name}] No valid POSITIVE prompts after linting.")
    if len(cleaned_neg) == 0:
        errors.append(f"[{expert_name}] No valid HARD-NEGATIVE prompts after linting.")

    return cleaned_pos, cleaned_neg, errors


# -------------------------
# PLIP embedding (text only)
# -------------------------

@torch.no_grad()
def embed_texts_plip(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device: str,
    batch_size: int,
    fp16: bool,
) -> torch.Tensor:
    """
    Returns L2-normalized embeddings: FloatTensor[N,D] on CPU.
    Works across transformers versions where get_text_features may return
    either a Tensor or a BaseModelOutputWithPooling-like object.
    """
    all_embs: List[torch.Tensor] = []
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_len = getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 77)
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Prefer the canonical CLIP pipeline: text_model -> (pooler_output) -> text_projection
        if hasattr(model, "text_model"):
            text_out = model.text_model(**inputs)
        else:
            # Fallback if someone accidentally loaded a text-only model
            text_out = model(**inputs)

        # Get pooled text representation
        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            pooled = text_out.pooler_output
        else:
            # Tuple fallback: (last_hidden_state, pooled)
            pooled = text_out[1]

        # Project into CLIP joint embedding space if available
        if hasattr(model, "text_projection"):
            feats = model.text_projection(pooled)
        else:
            feats = pooled

        feats = feats.to(dtype=dtype)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        all_embs.append(feats.detach().cpu().to(torch.float32))

    return torch.cat(all_embs, dim=0)


def _load_mstar_text_encoder(
    model_id: str,
    device: str,
    fp16: bool,
    projection_dim: int,
    projection_ckpt: str,
    use_pooler: bool,
) -> Tuple[AutoTokenizer, AutoModel, torch.nn.Module | None, int]:
    """Load an mSTAR-style BioBERT text encoder.

    Notes:
      - The mSTAR paper describes a "Bert-like" encoder following BioBERT-Base-v1.2,
        with an optional linear projection used during contrastive pretraining.
      - If projection_ckpt is not provided, we return the base encoder CLS embedding
        (or pooler_output if available and requested).
    """

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    enc = AutoModel.from_pretrained(model_id)
    enc.eval().to(device)

    # BioBERT hidden size is typically 768.
    hidden = int(getattr(enc.config, "hidden_size", 768))

    proj: torch.nn.Module | None = None
    if projection_ckpt:
        proj = torch.nn.Linear(hidden, projection_dim, bias=True)
        _loaded = _try_load_linear_projection(proj, projection_ckpt)
        if not _loaded:
            raise RuntimeError(
                "Failed to load --mstar_projection_ckpt into a Linear(hidden, proj_dim). "
                "Expected a checkpoint containing a compatible weight (and optional bias)."
            )
        proj.eval().to(device)

    # Mixed precision: keep encoder weights in fp32; autocast handles fp16 math on GPU.
    # (Casting BERT weights to fp16 can be unstable across versions/hardware.)
    _ = fp16  # reserved; behavior handled in embed_texts_mstar via autocast

    out_dim = projection_dim if proj is not None else hidden
    return tok, enc, proj, out_dim


def _try_load_linear_projection(proj: torch.nn.Linear, ckpt_path: str) -> bool:
    """Best-effort loader for a projection head from an arbitrary checkpoint.

    Accepts:
      - state_dict with keys like 'weight'/'bias'
      - nested dicts containing a matching tensor of shape [out_dim, in_dim]
    """

    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        return False

    out_dim, in_dim = proj.weight.shape

    # 1) Direct match
    if "weight" in sd and isinstance(sd["weight"], torch.Tensor) and tuple(sd["weight"].shape) == (out_dim, in_dim):
        proj.weight.data.copy_(sd["weight"].to(proj.weight.dtype))
        if "bias" in sd and isinstance(sd["bias"], torch.Tensor) and tuple(sd["bias"].shape) == (out_dim,):
            proj.bias.data.copy_(sd["bias"].to(proj.bias.dtype))
        else:
            proj.bias.data.zero_()
        return True

    # 2) Search for a tensor with matching shape
    weight_key = None
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and tuple(v.shape) == (out_dim, in_dim):
            weight_key = k
            break
    if weight_key is None:
        return False

    proj.weight.data.copy_(sd[weight_key].to(proj.weight.dtype))

    # bias: prefer a sibling key if present
    bias_key_candidates = []
    if weight_key.endswith(".weight"):
        bias_key_candidates.append(weight_key[:-7] + ".bias")
    bias_key_candidates += ["bias", "proj.bias", "projection.bias", "text_projection.bias"]

    bias_loaded = False
    for bk in bias_key_candidates:
        if bk in sd and isinstance(sd[bk], torch.Tensor) and tuple(sd[bk].shape) == (out_dim,):
            proj.bias.data.copy_(sd[bk].to(proj.bias.dtype))
            bias_loaded = True
            break
    if not bias_loaded:
        proj.bias.data.zero_()

    return True


def embed_texts_mstar(
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    projection: torch.nn.Module | None,
    texts: List[str],
    device: str,
    batch_size: int,
    fp16: bool,
    max_len: int,
    use_pooler: bool,
) -> torch.Tensor:
    """Return L2-normalized embeddings: FloatTensor[N,D] on CPU."""

    all_embs: List[torch.Tensor] = []

    use_amp = bool(fp16 and device == "cuda")
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with amp_ctx:
                out = encoder(**inputs)
                if use_pooler and hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feats = out.pooler_output
                else:
                    # CLS token
                    feats = out.last_hidden_state[:, 0, :]

                if projection is not None:
                    feats = projection(feats)

                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            all_embs.append(feats.detach().cpu().to(torch.float32))

    return torch.cat(all_embs, dim=0)



def sanitize_key(s: str) -> str:
    # NPZ keys must be str without special separators for portability
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_pack", type=str, required=True, help="Path to prompt pack JSON.")
    ap.add_argument(
        "--encoder",
        type=str,
        default="mstar",
        choices=["mstar", "plip"],
        help="Text encoder backend: mstar (BioBERT-like) or plip (CLIP text encoder).",
    )
    ap.add_argument(
        "--model_id",
        "--plip_model_id",
        dest="plip_model_id",
        type=str,
        default="vinid/plip",
        help="Hugging Face model id for PLIP (used when --encoder=plip).",
    )
    ap.add_argument(
        "--mstar_model_id",
        type=str,
        default="dmis-lab/biobert-base-cased-v1.2",
        help="Hugging Face model id for mSTAR-style text encoder (BioBERT-like).",
    )
    ap.add_argument(
        "--mstar_projection_ckpt",
        type=str,
        default="",
        help="Optional torch checkpoint containing a Linear(hidden, proj_dim) projection for mSTAR text embeddings.",
    )
    ap.add_argument(
        "--mstar_projection_dim",
        type=int,
        default=512,
        help="Projection output dimension if --mstar_projection_ckpt is provided.",
    )
    ap.add_argument(
        "--mstar_use_pooler",
        action="store_true",
        help="For mSTAR: use pooler_output when available (otherwise CLS token).",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=0,
        help="For mSTAR: max token length (0 means auto; default auto=512).",
    )
    ap.add_argument("--out_pt", type=str, required=True, help="Output .pt path (torch.save).")
    ap.add_argument("--out_npz", type=str, default="", help="Optional output .npz path.")
    ap.add_argument("--out_json", type=str, default="", help="Optional output .json metadata path (recommended with --out_npz).")

    ap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", help="Use float16 compute on GPU for speed (embeddings are saved float32).")

    ap.add_argument("--build_mode", type=str, default="prefix", choices=["prefix", "ellipsis"],
                    help="How to combine modality_phrase with each prompt.")
    ap.add_argument("--fail_on_lint", action="store_true",
                    help="If set, abort on any lint rejection. Otherwise proceeds with remaining prompts.")
    args = ap.parse_args()

    pack = load_json(args.prompt_pack)
    validate_prompt_pack(pack)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available.")

    # Load requested text encoder
    encoder_kind = args.encoder.lower()
    used_model_id: str

    plip_model: CLIPModel | None = None
    plip_processor: CLIPProcessor | None = None

    mstar_tokenizer: AutoTokenizer | None = None
    mstar_encoder: AutoModel | None = None
    mstar_projection: torch.nn.Module | None = None
    mstar_max_len: int = 0

    if encoder_kind == "plip":
        used_model_id = args.plip_model_id
        plip_model = CLIPModel.from_pretrained(used_model_id).to(device)
        plip_processor = CLIPProcessor.from_pretrained(used_model_id)
        plip_model.eval()
    elif encoder_kind == "mstar":
        used_model_id = args.mstar_model_id
        mstar_max_len = int(args.max_len) if int(args.max_len) > 0 else 512
        mstar_tokenizer, mstar_encoder, mstar_projection, _ = _load_mstar_text_encoder(
            model_id=used_model_id,
            device=device,
            fp16=args.fp16,
            projection_dim=int(args.mstar_projection_dim),
            projection_ckpt=str(args.mstar_projection_ckpt),
            use_pooler=bool(args.mstar_use_pooler),
        )
    else:
        raise ValueError(f"Unknown --encoder: {args.encoder}")

    modality_phrase = str(pack.get("meta", {}).get("modality_phrase", "")).strip()
    global_ban = pack["blacklists"]["global"]
    per_expert_ban_map = pack["blacklists"]["per_expert"]
    hard_neg_integrity_map = pack["blacklists"]["hard_negative_integrity"]

    experts_out: Dict[str, Any] = {}
    lint_errors_all: List[str] = []

    for expert_name, prompt_set in pack["experts"].items():
        positives_raw = prompt_set["positives"]
        hard_negs_raw = prompt_set["hard_negatives"]

        pos_full, neg_full, errs = lint_prompts(
            expert_name=expert_name,
            positives=positives_raw,
            hard_negs=hard_negs_raw,
            modality_phrase=modality_phrase,
            global_ban=global_ban,
            per_expert_ban_map=per_expert_ban_map,
            hard_neg_integrity_map=hard_neg_integrity_map,
            build_mode=args.build_mode,
        )

        lint_errors_all.extend(errs)

        if args.fail_on_lint and errs:
            # Hard stop with detailed lint output
            raise ValueError("Lint errors encountered:\n" + "\n".join(errs))

        # Embed
        if encoder_kind == "plip":
            assert plip_model is not None and plip_processor is not None
            pos_emb = embed_texts_plip(
                plip_model,
                plip_processor,
                pos_full,
                device=device,
                batch_size=args.batch_size,
                fp16=args.fp16,
            )
            neg_emb = embed_texts_plip(
                plip_model,
                plip_processor,
                neg_full,
                device=device,
                batch_size=args.batch_size,
                fp16=args.fp16,
            )
        else:
            assert mstar_tokenizer is not None and mstar_encoder is not None
            pos_emb = embed_texts_mstar(
                mstar_tokenizer,
                mstar_encoder,
                mstar_projection,
                pos_full,
                device=device,
                batch_size=args.batch_size,
                fp16=args.fp16,
                max_len=mstar_max_len,
                use_pooler=bool(args.mstar_use_pooler),
            )
            neg_emb = embed_texts_mstar(
                mstar_tokenizer,
                mstar_encoder,
                mstar_projection,
                neg_full,
                device=device,
                batch_size=args.batch_size,
                fp16=args.fp16,
                max_len=mstar_max_len,
                use_pooler=bool(args.mstar_use_pooler),
            )

        experts_out[expert_name] = {
            "positives": {"prompts": pos_full, "embeddings": pos_emb},
            "hard_negatives": {"prompts": neg_full, "embeddings": neg_emb},
        }

    embed_dim = next(iter(experts_out.values()))["positives"]["embeddings"].shape[-1]

    bundle = {
        "encoder": encoder_kind,
        "model_id": used_model_id,
        "embed_dim": int(embed_dim),
        "meta": pack["meta"],
        "blacklists": pack["blacklists"],
        "lint_errors": lint_errors_all,
        "experts": experts_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)) or ".", exist_ok=True)
    torch.save(bundle, args.out_pt)

    # Optional NPZ export (arrays only) + JSON sidecar (prompts + mapping)
    if args.out_npz:
        npz_dict: Dict[str, np.ndarray] = {}
        meta_json: Dict[str, Any] = {
            "encoder": encoder_kind,
            "model_id": used_model_id,
            "embed_dim": int(embed_dim),
            "meta": pack["meta"],
            "blacklists": pack["blacklists"],
            "lint_errors": lint_errors_all,
            "keys": {},
            "experts": {},
        }

        for expert_name, obj in experts_out.items():
            k = sanitize_key(expert_name)

            kp = f"{k}__positives"
            kn = f"{k}__hard_negatives"

            npz_dict[kp] = obj["positives"]["embeddings"].numpy().astype(np.float32)
            npz_dict[kn] = obj["hard_negatives"]["embeddings"].numpy().astype(np.float32)

            meta_json["keys"][expert_name] = {"positives": kp, "hard_negatives": kn}
            meta_json["experts"][expert_name] = {
                "positives_prompts": obj["positives"]["prompts"],
                "hard_negatives_prompts": obj["hard_negatives"]["prompts"],
                "positives_shape": list(npz_dict[kp].shape),
                "hard_negatives_shape": list(npz_dict[kn].shape),
            }

        os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
        np.savez_compressed(args.out_npz, **npz_dict)

        # Write JSON sidecar (recommended)
        out_json = args.out_json if args.out_json else os.path.splitext(args.out_npz)[0] + ".json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"[OK] Saved torch bundle: {args.out_pt}")
    if args.out_npz:
        print(f"[OK] Saved npz embeddings: {args.out_npz}")
        print(f"[OK] Saved npz metadata: {args.out_json if args.out_json else os.path.splitext(args.out_npz)[0] + '.json'}")
    if lint_errors_all:
        print(f"[WARN] Lint rejections: {len(lint_errors_all)} (see bundle['lint_errors'])")


if __name__ == "__main__":
    main()
