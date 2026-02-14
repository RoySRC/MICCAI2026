import os
import torch
import torch.nn.functional as F

def visualize_gates(model, x, y, pred_prob, save_path="visualize/gates.png"):
    """
    Saves a single PNG containing a grid:
    For each batch item: [Input | MVP | Nec | None | Hard]
    Overlays per-tile probabilities (spatial mean of each gate map) on the
    corresponding tile.

    Additionally overlays predicted and target labels on the Input tile.

    model: returns dict with out["gates"] of shape (B,3,h,w)
           prediction is inferred from common keys if present (e.g., logits).
    x:     tensor (B,C,H,W) on the correct device
    y:     target labels (B,) or one-hot / logits-like (B,K)
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Lazy imports (no matplotlib)
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    from PIL import ImageDraw, ImageFont

    def _infer_pred_label():
        """
        Try common output keys. Returns LongTensor (B,) or None if unavailable.
        Supported:
          - logits-like: (B,K) or (B,K,...) -> argmax over dim=1 after flattening spatial dims
          - pred-like:   (B,) already labels
        """
        if not isinstance(out_dict, dict):
            return None

        # Common keys to try, in order
        for k in ("logits", "pred", "y_pred", "outputs", "cls_logits", "class_logits"):
            if k not in out_dict:
                continue
            t = out_dict[k]
            if not torch.is_tensor(t):
                continue

            # If already label indices (B,)
            if t.ndim == 1:
                return t.long()

            # If logits/probs (B,K) or (B,K,H,W,...) reduce to (B,K,*) then argmax over K
            if t.ndim >= 2:
                # Make (B,K, -1) then mean over last dim if spatial exists
                if t.ndim > 2:
                    t2 = t.flatten(start_dim=2).mean(dim=2)
                else:
                    t2 = t
                return torch.argmax(t2, dim=1).long()

        return None

    def _infer_target_label(y_tensor):
        """
        Accepts:
          - (B,) label indices
          - (B,K) one-hot / logits -> argmax over dim=1
        """
        if y_tensor is None or (not torch.is_tensor(y_tensor)):
            return None
        if y_tensor.ndim == 1:
            return y_tensor.long()
        if y_tensor.ndim >= 2:
            return torch.argmax(y_tensor, dim=1).long()
        return None

    model.eval()
    with torch.no_grad():
        out = model(x)
        gates = out["gates"]  # (B,3,h,w)
    
    pred_labels = pred_prob

    tgt_labels = _infer_target_label(y)

    B, _, H, W = x.shape

    # Upsample gates to input resolution: (B,3,H,W)
    g_up = F.interpolate(gates, size=(H, W), mode="bilinear", align_corners=False)

    # Input image: take first 3 channels, normalize per-image to [0,1]
    img = x[:, :3]  # (B,3,H,W)
    img_min = img.amin(dim=(1, 2, 3), keepdim=True)
    img_max = img.amax(dim=(1, 2, 3), keepdim=True)
    img = (img - img_min) / (img_max - img_min + 1e-6)

    # Probability maps to 3-channel grayscale for saving: (B,3,H,W) each
    p_mvp  = g_up[:, 0:1].repeat(1, 3, 1, 1).clamp(0, 1)
    p_nec  = g_up[:, 1:2].repeat(1, 3, 1, 1).clamp(0, 1)
    p_none = g_up[:, 2:3].repeat(1, 3, 1, 1).clamp(0, 1)

    # Hard routing: argmax in {0,1,2}; scale to [0,1] so it renders in PNG
    hard_idx = torch.argmax(g_up, dim=1, keepdim=True)  # (B,1,H,W), values 0/1/2
    hard = hard_idx.float() / 2.0
    hard = hard.repeat(1, 3, 1, 1)

    # Compute per-sample scalar probabilities to write on tiles (spatial mean)
    # Shape: (B, 3) corresponding to MVP, Nec, None
    gate_means = g_up.clamp(0, 1).mean(dim=(2, 3))

    # Stack per-sample tiles: (B,5,3,H,W) then flatten to (B*5,3,H,W)
    tiles = torch.stack([img, p_mvp, p_nec, p_none, hard], dim=1)
    tiles = tiles.view(B * 5, 3, H, W).cpu()

    # Make grid with 5 columns (one row per batch item)
    pad = 2
    ncol = 5
    grid = make_grid(tiles, nrow=ncol, padding=pad)  # (3, gridH, gridW)

    # Convert to PIL for drawing text overlays
    grid_pil = to_pil_image(grid.clamp(0, 1))
    draw = ImageDraw.Draw(grid_pil)

    # Font (default PIL bitmap font, no external dependency)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def draw_label(x0, y0, text):
        # Draw a small black box behind white text for readability
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = draw.textsize(text, font=font)

        margin = 2
        rect = [x0 - margin, y0 - margin, x0 + tw + margin, y0 + th + margin]
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text((x0, y0), text, fill=(255, 255, 255), font=font)

    # Prepare labels on CPU for safe indexing
    if pred_labels is not None:
        pred_labels = pred_labels.detach().cpu()
    if tgt_labels is not None:
        tgt_labels = tgt_labels.detach().cpu()

    # Overlay text on each tile
    # Tile top-left for (row=b, col=c):
    # x = pad + c*(W + pad), y = pad + b*(H + pad)
    for b in range(B):
        pm = float(gate_means[b, 0].item())
        pn = float(gate_means[b, 1].item())
        p0 = float(gate_means[b, 2].item())

        majority = int(torch.bincount(hard_idx[b, 0].view(-1).cpu(), minlength=3).argmax().item())

        # Build pred/tgt text (only for input tile)
        if pred_labels is None:
            pred_txt = "pred=?"
        else:
            pred_txt = f"pred={float(pred_labels[b].item()):.2f}"

        if tgt_labels is None:
            tgt_txt = "tgt=?"
        else:
            tgt_txt = f"tgt={int(tgt_labels[b].item())}"

        texts = {
            0: f"Input  {pred_txt}  {tgt_txt}",
            1: f"MVP  p={pm:.3f}",
            2: f"Nec  p={pn:.3f}",
            3: f"None p={p0:.3f}",
            4: f"Hard cls={majority}",
        }

        for c in range(ncol):
            x0 = pad + c * (W + pad) + 4
            y0 = pad + b * (H + pad) + 4
            draw_label(x0, y0, texts[c])

    # Save final image
    grid_pil.save(save_path)
