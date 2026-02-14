#!/usr/bin/env python3
"""embed_prompt_pack_biomedclip.py

Embed an HSI concept prompt pack into BiomedCLIP text embeddings.

This script is a drop-in replacement for the previous PLIP-based version.
It uses OpenCLIP's HF-hub integration, as recommended in the BiomedCLIP
model card.

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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from open_clip import create_model_from_pretrained, get_tokenizer
except Exception as e:  # pragma: no cover
    raise ImportError(
        "BiomedCLIP embedding requires open_clip_torch. "
        "Install with: pip install open_clip_torch\n"
        "Original import error: " + str(e)
    )


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


"""Notes on BiomedCLIP

BiomedCLIP is distributed in OpenCLIP format on the Hugging Face Hub.
The model card demonstrates loading via:

  model, preprocess = create_model_from_pretrained('hf-hub:microsoft/...')
  tokenizer = get_tokenizer('hf-hub:microsoft/...')

and tokenization with a context length of 256.
"""


# ------------------------------
# BiomedCLIP embedding (text only)
# ------------------------------

@torch.no_grad()
def embed_texts_biomedclip(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
    fp16: bool,
    context_length: int,
) -> torch.Tensor:
    """
    Returns L2-normalized embeddings: FloatTensor[N,D] on CPU.

    BiomedCLIP uses a PubMedBERT-based text tower with a 256 token context
    length (per the model card). Tokenization is performed by OpenCLIP's
    tokenizer returned by `get_tokenizer()`.
    """
    all_embs: List[torch.Tensor] = []
    use_amp = bool(fp16 and device == "cuda")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, context_length=context_length).to(device)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = model.encode_text(tokens)
        else:
            feats = model.encode_text(tokens)

        # Normalize in float32 for numerical stability
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        all_embs.append(feats.detach().cpu())

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
        "--model_id",
        type=str,
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        help="OpenCLIP model id (HF hub).",
    )
    ap.add_argument("--out_pt", type=str, required=True, help="Output .pt path (torch.save).")
    ap.add_argument("--out_npz", type=str, default="", help="Optional output .npz path.")
    ap.add_argument("--out_json", type=str, default="", help="Optional output .json metadata path (recommended with --out_npz).")

    ap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", help="Use float16 compute on GPU for speed (embeddings are saved float32).")
    ap.add_argument(
        "--context_length",
        type=int,
        default=256,
        help="Tokenizer context length (BiomedCLIP uses 256 per model card).",
    )

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

    # Load BiomedCLIP via OpenCLIP
    model, _preprocess = create_model_from_pretrained(args.model_id)
    tokenizer = get_tokenizer(args.model_id)
    model = model.to(device)
    model.eval()

    modality_phrase = ""
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
        pos_emb = embed_texts_biomedclip(
            model,
            tokenizer,
            pos_full,
            device=device,
            batch_size=args.batch_size,
            fp16=args.fp16,
            context_length=args.context_length,
        )
        neg_emb = embed_texts_biomedclip(
            model,
            tokenizer,
            neg_full,
            device=device,
            batch_size=args.batch_size,
            fp16=args.fp16,
            context_length=args.context_length,
        )

        experts_out[expert_name] = {
            "positives": {"prompts": pos_full, "embeddings": pos_emb},
            "hard_negatives": {"prompts": neg_full, "embeddings": neg_emb},
        }

    embed_dim = next(iter(experts_out.values()))["positives"]["embeddings"].shape[-1]

    bundle = {
        "model_id": args.model_id,
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
            "model_id": args.model_id,
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
