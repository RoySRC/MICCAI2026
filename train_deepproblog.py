# train_deepproblog.py
from __future__ import annotations

import csv, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from itertools import chain
import torchvision.transforms as T

from problog.logic import Constant, Term

from pathlib import Path
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.solver import Solver

# Engines can be imported differently depending on version.
# Try the common ones.
try:
    from deepproblog.engines.exact_engine import ExactEngine
except Exception:
    from deepproblog.engines import ExactEngine  # type: ignore

from problog.program import PrologFile
from deepproblog.query import Query  # Query(query: Term, ..., p: float=1.0, ...)   [oai_citation:3‡Departement Computerwetenschappen](https://dtai.cs.kuleuven.be/projects/nesy/deepproblog_api.html)

from competitive_gating import CompetitiveSpatialGatingNet, AuxLossWeights
from visualizer import visualize_gates

os.environ["SWI_HOME_DIR"] = "/opt/homebrew/opt/swi-prolog/lib/swipl"
os.environ["LIBSWIPL_PATH"] = "/opt/homebrew/opt/swi-prolog/lib/swipl/lib/arm64-darwin/libswipl.10.0.0.dylib"

@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ImageLabelDataset(Dataset):
    def __init__(self, csv_path: str | Path):
        self.items: List[Tuple[str, int]] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"]
                label = int(row["label"])
                self.items.append((path, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.items[idx]


class ConceptNetWrapper(nn.Module):
    """
    Wraps the shared CompetitiveSpatialGatingNet so DeepProbLog can call it as a neural predicate.
    DeepProbLog will pass a list of terms (inputs) when batching=True.

    We load images from paths (string Constants), transform to tensors, run shared model,
    and return a (B,2) probability tensor for the selected head.
    """
    def __init__(self, shared: CompetitiveSpatialGatingNet, head: str, transform: T.Compose):
        super().__init__()
        assert head in ("mvp", "necrosis")
        self.shared = shared
        self.head = head
        self.transform = transform

    def _term_to_path(self, t) -> str:
        # Often a problog.logic.Constant with functor holding the string
        if hasattr(t, "functor"):
            return str(t.functor)
        return str(t)

    def forward(self, inputs: Sequence) -> torch.Tensor:
        paths = [self._term_to_path(t) for t in inputs]
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB")
            imgs.append(self.transform(im))
        x = torch.stack(imgs, dim=0).to(next(self.shared.parameters()).device)

        # Cache key so MVP and Necrosis calls for the same batch reuse forward pass
        # (DeepProbLog often calls both predicates for the same images during a solve)
        cache_key = tuple(hash(p) for p in paths)
        out = self.shared(x, cache_key=cache_key)

        if self.head == "mvp":
            return out["mvp_probs"]
        return out["nec_probs"]


def bce_on_query_prob(p_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    p_hat: (B,) predicted probability of gbm(I)
    y:     (B,) targets in {0,1}
    """
    p_hat = torch.clamp(p_hat, eps, 1.0 - eps)
    return -(y * torch.log(p_hat) + (1.0 - y) * torch.log(1.0 - p_hat)).mean()


def build_queries(batch_paths: List[str], batch_labels: List[int]) -> List[Query]:
    queries: List[Query] = []
    for p, y in zip(batch_paths, batch_labels):
        q_term = Term("gbm", Constant(p))
        queries.append(Query(q_term, p=float(y)))
        # print(f"{Query(Term('gbm', Constant(p)), p=float(y)) = }")
        # print(f"{Query(Term('mvp', Constant(p)), p=float(y)) = }")
        # print(f"{Query(Term('necrosis', Constant(p)), p=float(y)) = }")
        # print()
    return queries


def extract_single_prob(result) -> torch.Tensor:
    """
    DeepProbLog Result.result is a dict {Term: prob}. We assume a single query term per Query.
    """
    # Take the first (and typically only) probability tensor
    v = next(iter(result.result.values()))
    if isinstance(v, float):
        return torch.tensor(v)
    return v


def main(csv_path: str, prolog_path: str = "gbm.pl", cfg: TrainConfig = TrainConfig()) -> None:
    device = torch.device(cfg.device)

    # Shared model
    shared = CompetitiveSpatialGatingNet(
        in_channels=3,
        feat_channels=64,
        rep_dim=128,
        aux_w=AuxLossWeights(load_balance=0.01, decorrelate=0.001),
        backbone_name="resnet50",
        backbone_weights="DEFAULT",
    ).to(device)

    # Image transform (edit as needed)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Two DeepProbLog-visible networks that share the same underlying parameters.
    mvp_module = ConceptNetWrapper(shared, head="mvp", transform=transform)
    nec_module = ConceptNetWrapper(shared, head="necrosis", transform=transform)

    # Important: we manage optimization ourselves (single optimizer on shared parameters),
    # so we pass optimizer=None to DeepProbLog Network wrappers to avoid double-stepping.
    net_mvp = Network(mvp_module, name="mvp_net", optimizer=None, batching=True)
    net_nec = Network(nec_module, name="necrosis_net", optimizer=None, batching=True)

    model = Model(Path(prolog_path), [net_mvp, net_nec])
    engine = ExactEngine(model)
    model.set_engine(engine)

    solver = Solver(model, engine)

    opt = torch.optim.Adam(shared.parameters(), lr=cfg.lr)

    ds = ImageLabelDataset(csv_path)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)

    shared.train()
    lowest_epoch_loss = 10000
    for epoch in range(cfg.epochs):
        total = 0.0
        n = 0
        
        # Visualize
        x = []
        y = []
        predicted_probs = []
        for batch in dl:
            paths, labels = batch
            paths = list(paths)
            labels = labels.to(dtype=torch.float32, device=device)

            for p, l in zip(paths, labels):
                x.append(transform(Image.open(p).convert("RGB")))
                y.append(l)

            # Reset per-step aux state/caches
            shared.reset_aux()
            shared.reset_cache()

            queries = build_queries(paths, labels.tolist())
            results = solver.solve(queries)

            # Predicted gbm probabilities for each query
            p_hats = torch.stack([extract_single_prob(r) for r in results]).to(device).view(-1)
            predicted_probs.append(list(p_hats.detach()))

            logic_loss = bce_on_query_prob(p_hats, labels)
            aux_loss = shared.consume_aux()  # includes load-balance + decorrelation (once per unique forward)

            loss = logic_loss + aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            n += 1
        
        current_epoch_loss = total/max(n,1)
        x = torch.stack(x, dim=0).to(device=device)
        y = torch.stack(y, dim=0).to(device=device)
        predicted_probs = torch.stack(list(chain.from_iterable(predicted_probs)))
        if current_epoch_loss < lowest_epoch_loss:
            visualize_gates(shared, x, y, predicted_probs)
            lowest_epoch_loss = current_epoch_loss

        print(f"epoch={epoch} loss={current_epoch_loss:.4f}")

    # Save shared model params
    torch.save(shared.state_dict(), "shared_competitive_gating.pt")
    print("Saved: shared_competitive_gating.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with columns: path,label (label in {0,1})")
    parser.add_argument("--pl", default="gbm.pl", help="DeepProbLog program file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.bs, lr=args.lr)
    main(args.csv, args.pl, cfg)
