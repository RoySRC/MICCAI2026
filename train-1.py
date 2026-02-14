import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms.functional import to_tensor

import cv2
from skimage.filters import frangi

# ============================================================
# 1) Utilities
# ============================================================

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def resize_to(img: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def sobel_mag(gray01: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return safe_normalize01(mag)


def local_variance(gray01: np.ndarray, k: int = 15) -> np.ndarray:
    # Var = E[x^2] - E[x]^2
    k = int(k)
    if k % 2 == 0:
        k += 1
    mean = cv2.GaussianBlur(gray01, (k, k), 0)
    mean2 = cv2.GaussianBlur(gray01 * gray01, (k, k), 0)
    var = np.maximum(mean2 - mean * mean, 0.0)
    return safe_normalize01(var)


def patchify_view(view_chw: torch.Tensor, patch: int) -> torch.Tensor:
    # view_chw is C,H,W
    c, h, w = view_chw.shape
    if h % patch != 0 or w % patch != 0:
        new_h = (h // patch) * patch
        new_w = (w // patch) * patch
        view_chw = view_chw[:, :new_h, :new_w]
        c, h, w = view_chw.shape
    # N patches
    view = view_chw.unsqueeze(0)  # 1,C,H,W
    patches = view.unfold(2, patch, patch).unfold(3, patch, patch)  # 1,C,Hp,Wp,ph,pw
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # 1,Hp,Wp,C,ph,pw
    patches = patches.view(-1, c, patch, patch)  # N,C,ph,pw
    return patches


import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def overlay_attention_on_rgb(rgb_uint8: np.ndarray, att_1hw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    rgb_uint8: H,W,3 in RGB
    att_1hw:  H,W attention in [0,1]
    returns: H,W,3 RGB uint8 overlay
    """
    att = np.clip(att_1hw, 0.0, 1.0)
    att_u8 = (att * 255).astype(np.uint8)

    # Use a standard heatmap colormap (OpenCV outputs BGR)
    heat_bgr = cv2.applyColorMap(att_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heat_rgb.astype(np.float32) + (1.0 - alpha) * rgb_uint8.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)

@torch.no_grad()
def save_expert_attention_overlays(
    ckpt_path: str,
    patch_path: str,
    out_dir: str = "attn_vis",
    out_hw=(256, 256),
    alpha: float = 0.55,
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = MultiExpertMoE().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load RGB patch
    bgr = cv2.imread(patch_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(patch_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Route views (same as training)
    router = ViewRouter(out_hw=out_hw)
    routed = router.route(rgb)

    views = {
        "necrosis": routed.necrosis.unsqueeze(0).to(device),     # 1,3,H,W
        "mvp": routed.mvp.unsqueeze(0).to(device),
        "mitoses": routed.mitoses.unsqueeze(0).to(device),
        "thrombosis": routed.thrombosis.unsqueeze(0).to(device),
    }

    outs = model(views)

    # Save the resized input RGB for reference
    rgb_r = cv2.resize(rgb, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, "input_rgb.png"), cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR))

    # For each expert, upsample attention to out_hw and overlay
    for name in ["necrosis", "mvp", "mitoses", "thrombosis"]:
        att = outs[name]["att"]  # 1,1,h,w
        att_up = F.interpolate(att, size=out_hw, mode="bilinear", align_corners=False)  # 1,1,H,W
        att_np = att_up.squeeze(0).squeeze(0).detach().cpu().numpy()  # H,W

        overlay = overlay_attention_on_rgb(rgb_r, att_np, alpha=alpha)

        # Also save raw attention as grayscale
        att_u8 = (np.clip(att_np, 0, 1) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"{name}_att_raw.png"), att_u8)
        cv2.imwrite(os.path.join(out_dir, f"{name}_att_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Saved overlays to: {out_dir}")



# ============================================================
# 2) View Router (hard input routing)
# ============================================================

@dataclass
class RoutedViews:
    necrosis: torch.Tensor     # C,H,W
    mvp: torch.Tensor          # C,H,W
    mitoses: torch.Tensor      # C,H,W
    thrombosis: torch.Tensor   # C,H,W
    meta: Dict[str, torch.Tensor]  # extra masks for counterfactuals


class ViewRouter:
    """
    Produces feature-specific views from an RGB histology patch.

    You can replace these heuristics with stronger preprocessors (nuclei segmenter,
    vessel segmenter). The important invariant is that each expert sees only its view.
    """

    def __init__(self, out_hw: Tuple[int, int] = (256, 256)):
        self.out_hw = out_hw

    def _to_gray01(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray01 = safe_normalize01(gray)
        return gray01

    def _vesselness01(self, rgb: np.ndarray) -> np.ndarray:
        # Frangi expects 2D float image; we use inverted green channel as a crude proxy
        g = rgb[:, :, 1].astype(np.float32)
        g01 = safe_normalize01(g)
        v = frangi(1.0 - g01)  # vessels often appear darker in green; dataset-dependent
        v01 = safe_normalize01(v.astype(np.float32))
        return v01

    def route(self, rgb_uint8: np.ndarray) -> RoutedViews:
        rgb = resize_to(rgb_uint8, self.out_hw).astype(np.uint8)
        gray01 = self._to_gray01(rgb)

        # Texture and edge proxies
        tex01 = local_variance(gray01, k=21)
        edge01 = sobel_mag(gray01)

        # Vesselness proxy
        vess01 = self._vesselness01(rgb)

        # Necrosis candidate proxy:
        # low texture and low edges tend to capture bland regions; can be refined with nuclei density if available.
        necrosis_mask = ((tex01 < 0.25) & (edge01 < 0.25)).astype(np.float32)
        necrosis_mask = blur(necrosis_mask, 11)
        necrosis_mask = safe_normalize01(necrosis_mask)

        # MVP candidate proxy:
        # vesselness-driven emphasis
        mvp_mask = blur(vess01, 7)

        # Mitoses candidate proxy:
        # high edge and texture as a crude "hypercellular / mitotically active" proxy
        mit_mask = ((tex01 > 0.55) & (edge01 > 0.45)).astype(np.float32)
        mit_mask = blur(mit_mask, 7)
        mit_mask = safe_normalize01(mit_mask)

        # Thrombosis proxy:
        # intraluminal occlusion is hard; we gate by vesselness plus dark content
        dark01 = 1.0 - gray01
        thr_mask = (mvp_mask * safe_normalize01(dark01)).astype(np.float32)
        thr_mask = blur(thr_mask, 7)
        thr_mask = safe_normalize01(thr_mask)

        # Build routed views as masked RGB (3-channel)
        rgb01 = safe_normalize01(rgb.astype(np.float32))

        nec_view = (rgb01 * necrosis_mask[..., None]).astype(np.float32)
        mvp_view = (rgb01 * mvp_mask[..., None]).astype(np.float32)
        mit_view = (rgb01 * mit_mask[..., None]).astype(np.float32)
        thr_view = (rgb01 * thr_mask[..., None]).astype(np.float32)

        # Convert to torch CHW
        nec_t = torch.from_numpy(nec_view).permute(2, 0, 1)
        mvp_t = torch.from_numpy(mvp_view).permute(2, 0, 1)
        mit_t = torch.from_numpy(mit_view).permute(2, 0, 1)
        thr_t = torch.from_numpy(thr_view).permute(2, 0, 1)

        meta = {
            "necrosis_mask": torch.from_numpy(necrosis_mask).unsqueeze(0),  # 1,H,W
            "mvp_mask": torch.from_numpy(mvp_mask).unsqueeze(0),
            "mitoses_mask": torch.from_numpy(mit_mask).unsqueeze(0),
            "thrombosis_mask": torch.from_numpy(thr_mask).unsqueeze(0),
        }

        return RoutedViews(
            necrosis=nec_t,
            mvp=mvp_t,
            mitoses=mit_t,
            thrombosis=thr_t,
            meta=meta,
        )


# ============================================================
# 3) Expert architecture with attention map
# ============================================================

class ConvBackbone(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(inplace=True),

            nn.Conv2d(2 * width, 4 * width, 3, padding=1),
            nn.BatchNorm2d(4 * width),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,3,H,W
        return self.net(x)  # B,C,H',W'


class AttentionHead(nn.Module):
    def __init__(self, feat_ch: int, embed_dim: int = 128):
        super().__init__()
        self.att = nn.Conv2d(feat_ch, 1, kernel_size=1)
        self.proj = nn.Conv2d(feat_ch, embed_dim, kernel_size=1)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # feat: B,C,h,w
        att_logits = self.att(feat)             # B,1,h,w
        att = torch.sigmoid(att_logits)         # B,1,h,w

        emb_map = self.proj(feat)               # B,E,h,w
        # weighted global pooling
        wsum = (emb_map * att).sum(dim=(2, 3))
        norm = att.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = wsum / norm                    # B,E

        logit = self.classifier(pooled).squeeze(1)  # B
        return logit, pooled, att


class ExpertNet(nn.Module):
    def __init__(self, name: str, width: int = 32, embed_dim: int = 128):
        super().__init__()
        self.name = name
        self.backbone = ConvBackbone(in_ch=3, width=width)
        # determine feat_ch
        feat_ch = 4 * width
        self.head = AttentionHead(feat_ch=feat_ch, embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        logit, emb, att = self.head(feat)
        prob = torch.sigmoid(logit)
        return {"logit": logit, "prob": prob, "emb": emb, "att": att}


class MultiExpertMoE(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate backbones, no shared trunk
        self.necrosis = ExpertNet("necrosis_net")
        self.mvp = ExpertNet("mvp_net")
        self.mitoses = ExpertNet("mitosis_net")
        self.thrombosis = ExpertNet("thrombosis_net")

    def forward(self, views: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        out = {}
        out["necrosis"] = self.necrosis(views["necrosis"])
        out["mvp"] = self.mvp(views["mvp"])
        out["mitoses"] = self.mitoses(views["mitoses"])
        out["thrombosis"] = self.thrombosis(views["thrombosis"])
        return out


# ============================================================
# 4) Gradient reversal + adversarial anti-leakage
# ============================================================

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverseFn.apply(x, lambd)


class LeakageAdversary(nn.Module):
    """
    Tries to predict other experts' logits from one expert's embedding.
    With gradient reversal, the expert is trained to remove that information.
    """
    def __init__(self, emb_dim: int = 128, hidden: int = 128, out_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(emb)  # B,out_dim


# ============================================================
# 5) Logic aggregator (soft logic matching the rules)
# ============================================================

def gbm_soft_logic(
    p_nec: torch.Tensor,
    p_mvp: torch.Tensor,
    p_mit: torch.Tensor,
    p_thr: torch.Tensor,
    support_weight: float = 0.25,
) -> torch.Tensor:
    """
    Rule match:
      gbm fires if necrosis OR mvp
      support evidence (mitoses, thrombosis) strengthens confidence when a hallmark is present

    We compute:
      base = 1 - (1 - p_nec) * (1 - p_mvp)  (noisy OR)
      support = clamp(p_mit + p_thr, 0, 1)
      gbm = clamp(base + support_weight * support * base, 0, 1)
    """
    base = 1.0 - (1.0 - p_nec) * (1.0 - p_mvp)
    support = torch.clamp(p_mit + p_thr, 0.0, 1.0)
    gbm = torch.clamp(base + support_weight * support * base, 0.0, 1.0)
    return gbm


# ============================================================
# 6) Push and Pull regularizers
# ============================================================

def entropy_bernoulli(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def push_confidence_sparsity(p: torch.Tensor, w_entropy: float = 1.0, w_l1: float = 0.25) -> torch.Tensor:
    # Encourage confident outputs (low entropy) and sparsity (small average p)
    ent = entropy_bernoulli(p).mean()
    l1 = p.mean()
    return w_entropy * ent + w_l1 * l1


def usage_range_penalty(p: torch.Tensor, lo: float = 0.02, hi: float = 0.35) -> torch.Tensor:
    # Prevent "always off" and "always on" collapse without concept labels
    m = p.mean()
    pen_lo = F.relu(torch.tensor(lo, device=p.device) - m) ** 2
    pen_hi = F.relu(m - torch.tensor(hi, device=p.device)) ** 2
    return pen_lo + pen_hi


def embedding_orthogonality(embs: List[torch.Tensor]) -> torch.Tensor:
    # embs is list of B,E
    loss = 0.0
    n = len(embs)
    for i in range(n):
        ei = F.normalize(embs[i], dim=1)
        for j in range(i + 1, n):
            ej = F.normalize(embs[j], dim=1)
            cos = (ei * ej).sum(dim=1)  # B
            loss = loss + (cos ** 2).mean()
    return loss


def attention_decorrelation(atts: List[torch.Tensor]) -> torch.Tensor:
    """
    Penalize overlap between attention maps.
    atts: list of B,1,h,w
    """
    loss = 0.0
    n = len(atts)
    # normalize each attention map to sum to 1
    normed = []
    for a in atts:
        a = a.clamp_min(0.0)
        s = a.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        normed.append(a / s)
    for i in range(n):
        ai = normed[i]
        for j in range(i + 1, n):
            aj = normed[j]
            overlap = (ai * aj).sum(dim=(2, 3))  # B,1
            loss = loss + (overlap.squeeze(1) ** 2).mean()
    return loss


# ============================================================
# 7) Counterfactual constraints
# ============================================================

def counterfactual_mask_out(view: torch.Tensor, mask_1hw: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """
    view: B,3,H,W
    mask_1hw: B,1,H,W with values in [0,1]
    We mask out the feature region by multiplying by (1 - strength*mask).
    """
    m = torch.clamp(mask_1hw, 0.0, 1.0)
    keep = torch.clamp(1.0 - strength * m, 0.0, 1.0)
    return view * keep


def counterfactual_losses(
    views_b: Dict[str, torch.Tensor],
    meta_b: Dict[str, torch.Tensor],
    model: MultiExpertMoE,
    margin_drop: float = 0.15,
    w_selective: float = 1.0,
    w_invariance: float = 0.25,
) -> torch.Tensor:
    """
    Build counterfactuals:
      for necrosis: mask out necrosis candidate region only in necrosis view
      for mvp: mask out mvp candidate region only in mvp view

    Enforce:
      necrosis prob should drop by at least margin_drop on necrosis counterfactual
      other experts remain close (invariance)
    """
    device = next(model.parameters()).device

    # Original forward
    out0 = model(views_b)
    p0 = {k: out0[k]["prob"] for k in out0}

    # Necrosis CF
    views_cf_nec = dict(views_b)
    views_cf_nec["necrosis"] = counterfactual_mask_out(
        views_b["necrosis"], meta_b["necrosis_mask"], strength=1.0
    )
    out_nec = model(views_cf_nec)
    p_nec = {k: out_nec[k]["prob"] for k in out_nec}

    # MVP CF
    views_cf_mvp = dict(views_b)
    views_cf_mvp["mvp"] = counterfactual_mask_out(
        views_b["mvp"], meta_b["mvp_mask"], strength=1.0
    )
    out_mvp = model(views_cf_mvp)
    p_mvp = {k: out_mvp[k]["prob"] for k in out_mvp}

    # Selective drop: target expert must decrease
    drop_nec = F.relu((p_nec["necrosis"] - p0["necrosis"]) + margin_drop).mean()
    drop_mvp = F.relu((p_mvp["mvp"] - p0["mvp"]) + margin_drop).mean()

    # Invariance: other experts should not change much
    inv_nec = 0.0
    inv_mvp = 0.0
    for k in ["mvp", "mitoses", "thrombosis"]:
        inv_nec = inv_nec + (p_nec[k] - p0[k]).abs().mean()
    for k in ["necrosis", "mitoses", "thrombosis"]:
        inv_mvp = inv_mvp + (p_mvp[k] - p0[k]).abs().mean()

    return w_selective * (drop_nec + drop_mvp) + w_invariance * (inv_nec + inv_mvp)


# ============================================================
# 8) Dataset stub (replace with your loader)
# ============================================================

class HistologyPatchDataset(torch.utils.data.Dataset):
    """
    Expected:
      - You provide a list of (path, label) where label is 0 or 1 for GBM.
      - Each path points to an RGB image patch (PNG, JPG).
    """

    def __init__(self, items: List[Tuple[str, int]], out_hw: Tuple[int, int] = (256, 256)):
        self.items = items
        self.router = ViewRouter(out_hw=out_hw)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        routed = self.router.route(rgb)

        sample = {
            "necrosis": routed.necrosis,
            "mvp": routed.mvp,
            "mitoses": routed.mitoses,
            "thrombosis": routed.thrombosis,
            "meta": routed.meta,
            "y": torch.tensor(float(y)),
            "path": path,
        }
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Stack views
    def stack_view(key: str) -> torch.Tensor:
        return torch.stack([b[key] for b in batch], dim=0)  # B,3,H,W

    views = {
        "necrosis": stack_view("necrosis"),
        "mvp": stack_view("mvp"),
        "mitoses": stack_view("mitoses"),
        "thrombosis": stack_view("thrombosis"),
    }

    # Stack meta masks
    meta = {}
    for mk in batch[0]["meta"].keys():
        meta[mk] = torch.stack([b["meta"][mk] for b in batch], dim=0)  # B,1,H,W

    y = torch.stack([b["y"] for b in batch], dim=0)  # B
    return {"views": views, "meta": meta, "y": y}


# ============================================================
# 9) Training loop with all constraints
# ============================================================

class TrainerConfig:
    def __init__(
        self,
        lr: float = 1e-4,
        epochs: int = 128,
        batch_size: int = 8,
        w_logic: float = 1.0,
        w_push: float = 0.25,
        w_usage: float = 0.25,
        w_ortho: float = 0.10,
        w_attn: float = 0.10,
        w_adv: float = 0.25,
        grl_lambda: float = 0.5,
        w_cf: float = 0.25,
        support_weight: float = 0.25,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w_logic = w_logic
        self.w_push = w_push
        self.w_usage = w_usage
        self.w_ortho = w_ortho
        self.w_attn = w_attn
        self.w_adv = w_adv
        self.grl_lambda = grl_lambda
        self.w_cf = w_cf
        self.support_weight = support_weight
        self.device = device


def build_adversaries(emb_dim: int = 128) -> Dict[str, LeakageAdversary]:
    # For each expert, predict the other three logits (out_dim = 3)
    return {
        "necrosis": LeakageAdversary(emb_dim=emb_dim, out_dim=3),
        "mvp": LeakageAdversary(emb_dim=emb_dim, out_dim=3),
        "mitoses": LeakageAdversary(emb_dim=emb_dim, out_dim=3),
        "thrombosis": LeakageAdversary(emb_dim=emb_dim, out_dim=3),
    }


def adversarial_anti_leakage_loss(
    outs: Dict[str, Dict[str, torch.Tensor]],
    adversaries: Dict[str, LeakageAdversary],
    grl_lambda: float,
) -> torch.Tensor:
    """
    For each expert k:
      adv tries to predict logits of the other experts from GRL(emb_k)
      minimize MSE(adv(GRL(emb_k)), target_other_logits_detached)
    """
    keys = ["necrosis", "mvp", "mitoses", "thrombosis"]
    logits = {k: outs[k]["logit"] for k in keys}  # B

    loss = 0.0
    for k in keys:
        others = [o for o in keys if o != k]
        target = torch.stack([logits[o].detach() for o in others], dim=1)  # B,3
        emb = outs[k]["emb"]
        emb_rev = grad_reverse(emb, grl_lambda)
        pred = adversaries[k](emb_rev)  # B,3
        loss = loss + F.mse_loss(pred, target)
    return loss


def train_one_epoch(
    model: MultiExpertMoE,
    adversaries: Dict[str, LeakageAdversary],
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    cfg: TrainerConfig,
) -> Dict[str, float]:
    model.train()
    for a in adversaries.values():
        a.train()

    meters = {
        "loss_total": 0.0,
        "loss_logic": 0.0,
        "loss_push": 0.0,
        "loss_usage": 0.0,
        "loss_ortho": 0.0,
        "loss_attn": 0.0,
        "loss_adv": 0.0,
        "loss_cf": 0.0,
    }
    n = 0

    for batch in loader:
        views = {k: v.to(cfg.device) for k, v in batch["views"].items()}
        meta = {k: v.to(cfg.device) for k, v in batch["meta"].items()}
        y = batch["y"].to(cfg.device)

        outs = model(views)
        p_nec = outs["necrosis"]["prob"]
        p_mvp = outs["mvp"]["prob"]
        p_mit = outs["mitoses"]["prob"]
        p_thr = outs["thrombosis"]["prob"]

        # Main logic loss: BCE between gbm_prob and image-level label
        gbm_prob = gbm_soft_logic(p_nec, p_mvp, p_mit, p_thr, support_weight=cfg.support_weight)
        loss_logic = F.binary_cross_entropy(gbm_prob, y)

        # Push terms per expert
        loss_push = (
            push_confidence_sparsity(p_nec) +
            push_confidence_sparsity(p_mvp) +
            push_confidence_sparsity(p_mit) +
            push_confidence_sparsity(p_thr)
        ) / 4.0

        # Usage range to prevent collapse
        loss_usage = (
            usage_range_penalty(p_nec) +
            usage_range_penalty(p_mvp) +
            usage_range_penalty(p_mit) +
            usage_range_penalty(p_thr)
        ) / 4.0

        # Pull terms
        embs = [outs[k]["emb"] for k in ["necrosis", "mvp", "mitoses", "thrombosis"]]
        atts = [outs[k]["att"] for k in ["necrosis", "mvp", "mitoses", "thrombosis"]]
        loss_ortho = embedding_orthogonality(embs)
        loss_attn = attention_decorrelation(atts)

        # Adversarial anti-leakage
        loss_adv = adversarial_anti_leakage_loss(outs, adversaries, grl_lambda=cfg.grl_lambda)

        # Counterfactual constraints
        loss_cf = counterfactual_losses(views, meta, model)

        loss_total = (
            cfg.w_logic * loss_logic +
            cfg.w_push * loss_push +
            cfg.w_usage * loss_usage +
            cfg.w_ortho * loss_ortho +
            cfg.w_attn * loss_attn +
            cfg.w_adv * loss_adv +
            cfg.w_cf * loss_cf
        )

        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        opt.step()

        bs = y.shape[0]
        n += bs
        meters["loss_total"] += float(loss_total.detach()) * bs
        meters["loss_logic"] += float(loss_logic.detach()) * bs
        meters["loss_push"] += float(loss_push.detach()) * bs
        meters["loss_usage"] += float(loss_usage.detach()) * bs
        meters["loss_ortho"] += float(loss_ortho.detach()) * bs
        meters["loss_attn"] += float(loss_attn.detach()) * bs
        meters["loss_adv"] += float(loss_adv.detach()) * bs
        meters["loss_cf"] += float(loss_cf.detach()) * bs

    for k in meters:
        meters[k] /= max(n, 1)
    return meters


@torch.no_grad()
def evaluate(
    model: MultiExpertMoE,
    loader: torch.utils.data.DataLoader,
    cfg: TrainerConfig,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    n = 0
    for batch in loader:
        views = {k: v.to(cfg.device) for k, v in batch["views"].items()}
        y = batch["y"].to(cfg.device)
        outs = model(views)
        gbm_prob = gbm_soft_logic(
            outs["necrosis"]["prob"],
            outs["mvp"]["prob"],
            outs["mitoses"]["prob"],
            outs["thrombosis"]["prob"],
            support_weight=cfg.support_weight,
        )
        loss = F.binary_cross_entropy(gbm_prob, y)
        bs = y.shape[0]
        n += bs
        loss_sum += float(loss) * bs
    return {"val_bce": loss_sum / max(n, 1)}


def main_train(
    train_items: List[Tuple[str, int]],
    val_items: List[Tuple[str, int]],
    cfg: TrainerConfig,
) -> None:
    set_seed(0)

    train_ds = HistologyPatchDataset(train_items, out_hw=(256, 256))
    val_ds = HistologyPatchDataset(val_items, out_hw=(256, 256))

    train_ld = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_ld = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = MultiExpertMoE().to(cfg.device)
    adversaries = build_adversaries(emb_dim=128)
    for a in adversaries.values():
        a.to(cfg.device)

    params = list(model.parameters())
    for a in adversaries.values():
        params += list(a.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)

    best = float("inf")
    for ep in range(cfg.epochs):
        tr = train_one_epoch(model, adversaries, train_ld, opt, cfg)
        va = evaluate(model, val_ld, cfg)

        print(f"epoch={ep} train={tr} val={va}")

        if va["val_bce"] < best:
            best = va["val_bce"]
            ckpt = {
                "model": model.state_dict(),
                "adversaries": {k: v.state_dict() for k, v in adversaries.items()},
                "cfg": cfg.__dict__,
            }
            torch.save(ckpt, "best_gbm_moe.pt")
            print("saved best_gbm_moe.pt")

            save_expert_attention_overlays(
                ckpt_path="best_gbm_moe.pt",
                patch_path='/Volumes/External SSD1/HSI-GBM/P3/ROI_01_C06_T/rgb.png',
                out_dir="attn_vis_example",
                out_hw=(256, 256),
                alpha=0.55,
            )



if __name__ == "__main__":
    import csv

    # CSV must have columns: path,label
    csv_path = "P7.csv"  # or an absolute path to your CSV

    items = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append((row["path"], int(row["label"])))

    # Deterministic 80/20 split
    rng = random.Random(0)
    rng.shuffle(items)
    split = int(0.8 * len(items))

    train_items = items[:split]
    val_items = items[split:]

    cfg = TrainerConfig(
        lr=1e-4,
        epochs=128,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    main_train(train_items, val_items, cfg)


