from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class RunData:
    path: Path
    arr: Dict[str, np.ndarray]
    meta: Dict[str, Any]


def load_npz(path: str | Path) -> RunData:
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    arr = {k: data[k] for k in data.files}
    meta: Dict[str, Any] = {}

    if "meta_json" in arr:
        raw = arr["meta_json"]
        try:
            if raw.shape == ():
                raw_val = raw.item()
            else:
                raw_val = raw.flatten()[0].item()
            if isinstance(raw_val, bytes):
                raw_val = raw_val.decode("utf-8")
            if isinstance(raw_val, str) and raw_val.strip():
                meta = json.loads(raw_val)
        except Exception:
            meta = {}

    return RunData(path=p, arr=arr, meta=meta)


def find_npz_files(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for s in paths:
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.npz")))
        elif p.is_file() and p.suffix.lower() == ".npz":
            out.append(p)
    return sorted(set(out))