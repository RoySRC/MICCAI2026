from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pandas as pd


# P1.csv ... P13.csv must exist in base_dir
PARTITIONS: Dict[int, Dict[str, List[str]]] = {
    1: {
        "train": ["P2", "P3", "P4", "P5", "P8", "P9", "P10", "P12", "P13"],
        "val":   ["P6"],
        "test":  ["P1", "P7", "P11"],
    },
    2: {
        "train": ["P1", "P2", "P5", "P7", "P8", "P9", "P10", "P11", "P12"],
        "val":   ["P3"],
        "test":  ["P4", "P6", "P13"],
    },
    3: {
        "train": ["P1", "P3", "P4", "P6", "P8", "P10", "P11", "P12", "P13"],
        "val":   ["P7"],
        "test":  ["P2", "P5", "P9"],
    },
    4: {
        "train": ["P2", "P4", "P5", "P6", "P7", "P9", "P11", "P13"],
        "val":   ["P1"],
        "test":  ["P3", "P8", "P10", "P12"],
    },
}


def _pid_num(pid: str) -> int:
    return int(pid[1:])


def _validate_partition(k: int, spec: Dict[str, List[str]]) -> None:
    train, val, test = set(spec["train"]), set(spec["val"]), set(spec["test"])
    if (train & val) or (train & test) or (val & test):
        raise ValueError(
            f"Overlap in fold {k}: "
            f"train∩val={train & val}, train∩test={train & test}, val∩test={val & test}"
        )

    all_ids = {f"P{i}" for i in range(1, 14)}
    seen = train | val | test
    missing = all_ids - seen
    extra = seen - all_ids
    if missing or extra:
        raise ValueError(f"Fold {k} mismatch. missing={sorted(missing)}, extra={sorted(extra)}")


def _load_and_concat(base_dir: Path, pids: List[str]) -> pd.DataFrame:
    frames = []
    for pid in sorted(pids, key=_pid_num):
        fp = base_dir / f"{pid}.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing expected file: {fp}")
        df = pd.read_csv(fp)
        df.insert(0, "source_partition", pid)  # helps trace rows back to P#
        frames.append(df)

    # sort=False keeps column order stable when all files share the same schema
    return pd.concat(frames, ignore_index=True, sort=False)


def write_fold_csvs(base_dir: str | Path, out_dir: str | Path) -> None:
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, spec in PARTITIONS.items():
        _validate_partition(fold_id, spec)

        fold_dir = out_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            df = _load_and_concat(base_dir, spec[split])
            out_path = fold_dir / f"{split}.csv"
            df.to_csv(out_path, index=False)
            print(f"Wrote {out_path}  rows={len(df)}  cols={len(df.columns)}")


if __name__ == "__main__":
    # Example:
    #   base_dir = "/mnt/data"   (contains P1.csv ... P13.csv)
    #   out_dir  = "/mnt/data/folds_out"
    write_fold_csvs(
        base_dir="/home/sajeeb/external_disk_1/NSU_Thesis/HistologyHSI-GB/", 
        out_dir="/home/sajeeb/external_disk_1/NSU_Thesis/HistologyHSI-GB/"
    )
