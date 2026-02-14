# competitive_gating.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# NEW: torchvision resnet
from torchvision import models


@dataclass
class AuxLossWeights:
    load_balance: float = 0.01
    decorrelate: float = 0.001


def _resnet_out_channels(name: str) -> int:
    name = name.lower()
    if name in ("resnet18", "resnet34"):
        return 512
    if name in ("resnet50", "resnet101", "resnet152"):
        return 2048
    raise ValueError(f"Unsupported resnet backbone: {name}")


def _make_resnet_backbone(
    name: str = "resnet18",
    in_channels: int = 3,
    weights: Optional[str] = None,
) -> nn.Module:
    """
    Returns a module that maps (B,in_channels,H,W) -> (B,C,H',W') feature map,
    where C depends on resnet variant. This removes avgpool/fc.
    """
    name = name.lower()

    # Handle torchvision API differences:
    # - Newer: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # - Older: models.resnet18(pretrained=True)
    # We support both by trying the newer path first.
    ctor = getattr(models, name)

    resnet = None
    if weights is None:
        resnet = ctor(weights=None)
    else:
        # weights can be "DEFAULT" (string) or a weights enum instance.
        # We'll interpret "DEFAULT" for common variants.
        try:
            if isinstance(weights, str):
                w = weights.upper()
                if name == "resnet18":
                    we = models.ResNet18_Weights.DEFAULT if w == "DEFAULT" else None
                elif name == "resnet34":
                    we = models.ResNet34_Weights.DEFAULT if w == "DEFAULT" else None
                elif name == "resnet50":
                    we = models.ResNet50_Weights.DEFAULT if w == "DEFAULT" else None
                elif name == "resnet101":
                    we = models.ResNet101_Weights.DEFAULT if w == "DEFAULT" else None
                elif name == "resnet152":
                    we = models.ResNet152_Weights.DEFAULT if w == "DEFAULT" else None
                else:
                    we = None
                resnet = ctor(weights=we)
            else:
                resnet = ctor(weights=weights)
        except TypeError:
            # Fallback for older torchvision
            resnet = ctor(pretrained=True)

    # Replace conv1 if in_channels != 3
    if in_channels != 3:
        old = resnet.conv1
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )

        # If pretrained and single-channel input, adapt weights by averaging RGB filters.
        with torch.no_grad():
            if old.weight.shape[1] == 3:
                if in_channels == 1:
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                else:
                    # For other channel counts, do a simple repeat/truncate initialization.
                    # (You can replace this with a better init if desired.)
                    rep = (in_channels + 2) // 3
                    w = old.weight.repeat(1, rep, 1, 1)[:, :in_channels, :, :]
                    new.weight.copy_(w / rep)

        resnet.conv1 = new

    # Trunk up to layer4 (no avgpool, no fc) as a feature-map backbone
    backbone = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )
    return backbone


class CompetitiveSpatialGatingNet(nn.Module):
    """
    Shared backbone produces feature map F.
    Gate produces per-location routing probabilities over experts:
      {MVP, NECROSIS, NONE} with softmax, so they are mutually exclusive per pixel.
    Each concept head pools only the features routed to its expert.
    """

    def __init__(
        self,
        in_channels: int = 3,
        # CHANGED: resnet controls feat_channels; keep arg if you want but it won't be used.
        feat_channels: int = 64,
        rep_dim: int = 128,
        aux_w: AuxLossWeights = AuxLossWeights(),
        # NEW:
        backbone_name: str = "resnet18",
        backbone_weights: Optional[str] = None,  # set "DEFAULT" for pretrained (when available)
    ):
        super().__init__()
        self.aux_w = aux_w

        # NEW: ResNet backbone producing feature map (B,C,H',W')
        self.backbone = _make_resnet_backbone(
            name=backbone_name,
            in_channels=in_channels,
            weights=backbone_weights,
        )
        feat_dim = _resnet_out_channels(backbone_name)

        # Gate logits for 3 experts: mvp, necrosis, none
        self.gate_head = nn.Conv2d(feat_dim, 3, kernel_size=1)

        # Expert projection (shared dimensionality for decorrelation term)
        self.mvp_proj = nn.Sequential(
            nn.Linear(feat_dim, rep_dim),
            nn.ReLU(inplace=True),
        )
        self.nec_proj = nn.Sequential(
            nn.Linear(feat_dim, rep_dim),
            nn.ReLU(inplace=True),
        )

        # Binary concept classifiers (0 absent, 1 present)
        self.mvp_cls = nn.Linear(rep_dim, 2)
        self.nec_cls = nn.Linear(rep_dim, 2)

        # --- state for regularizers / caching ---
        self._aux_loss: torch.Tensor = torch.tensor(0.0)
        self._cache_key: Optional[Tuple[int, ...]] = None
        self._cache_out: Optional[Dict[str, torch.Tensor]] = None

    def reset_aux(self) -> None:
        self._aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)

    def consume_aux(self) -> torch.Tensor:
        out = self._aux_loss
        self.reset_aux()
        return out

    def reset_cache(self) -> None:
        self._cache_key = None
        self._cache_out = None

    def _compute_aux_losses(self, gates: torch.Tensor, z_mvp: torch.Tensor, z_nec: torch.Tensor) -> torch.Tensor:
        gate_mean = gates.mean(dim=(0, 2, 3))  # (3,)
        target = torch.full_like(gate_mean, 1.0 / gate_mean.numel())
        lb_loss = torch.sum((gate_mean - target) ** 2)

        eps = 1e-6
        z1 = (z_mvp - z_mvp.mean(dim=0, keepdim=True)) / (z_mvp.std(dim=0, keepdim=True, unbiased=False) + eps)
        z2 = (z_nec - z_nec.mean(dim=0, keepdim=True)) / (z_nec.std(dim=0, keepdim=True, unbiased=False) + eps)
        corr = (z1.T @ z2) / z1.shape[0]  # (D,D)
        dec_loss = (corr ** 2).mean()

        return self.aux_w.load_balance * lb_loss + self.aux_w.decorrelate * dec_loss

    def forward(self, x: torch.Tensor, cache_key: Optional[Tuple[int, ...]] = None) -> Dict[str, torch.Tensor]:
        if cache_key is not None and self._cache_key == cache_key and self._cache_out is not None:
            return self._cache_out

        feat = self.backbone(x)  # (B,C,H',W') where C is 512 or 2048
        gate_logits = self.gate_head(feat)  # (B,3,H',W')
        gates = F.softmax(gate_logits, dim=1)

        gate_mvp = gates[:, 0:1, :, :]
        gate_nec = gates[:, 1:2, :, :]

        eps = 1e-6
        mvp_mass = gate_mvp.mean(dim=(2, 3)) + eps  # (B,1)
        nec_mass = gate_nec.mean(dim=(2, 3)) + eps  # (B,1)

        mvp_feat = (feat * gate_mvp).mean(dim=(2, 3)) / mvp_mass  # (B,C)
        nec_feat = (feat * gate_nec).mean(dim=(2, 3)) / nec_mass  # (B,C)

        z_mvp = self.mvp_proj(mvp_feat)  # (B,D)
        z_nec = self.nec_proj(nec_feat)  # (B,D)

        mvp_logits = self.mvp_cls(z_mvp)
        nec_logits = self.nec_cls(z_nec)

        mvp_probs = F.softmax(mvp_logits, dim=1)
        nec_probs = F.softmax(nec_logits, dim=1)

        aux = self._compute_aux_losses(gates, z_mvp, z_nec)
        self._aux_loss = self._aux_loss + aux

        out = {
            "mvp_probs": mvp_probs,
            "nec_probs": nec_probs,
            "gates": gates,
            "z_mvp": z_mvp,
            "z_nec": z_nec,
        }

        if cache_key is not None:
            self._cache_key = cache_key
            self._cache_out = out
        return out
