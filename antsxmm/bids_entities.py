from __future__ import annotations

from pathlib import Path


def parse_entities(path: str) -> dict[str, str]:
    """Parse BIDS-style entities from a filename.

    Returns keys like sub/ses/run/task/dir and always includes suffix.
    """
    name = Path(path).name
    if name.endswith('.nii.gz'):
        stem = name[:-7]
    else:
        stem = Path(name).stem
    tokens = stem.split('_')
    entities: dict[str, str] = {}
    for token in tokens[:-1]:
        if '-' in token:
            k, v = token.split('-', 1)
            if k and v:
                entities[k] = v
    if tokens:
        entities['suffix'] = tokens[-1]
    return entities
