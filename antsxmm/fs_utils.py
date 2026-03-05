from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Union


PathLike = Union[str, os.PathLike, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: PathLike, obj: object) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def as_path_list(items: Union[None, PathLike, Iterable[PathLike]]) -> List[Path]:
    if items is None:
        return []
    if isinstance(items, (str, os.PathLike, Path)):
        return [Path(items)]
    return [Path(x) for x in items]
