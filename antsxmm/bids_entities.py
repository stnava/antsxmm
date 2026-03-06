from __future__ import annotations

import re
from pathlib import Path


def parse_entities(path: str) -> dict[str, str]:
    """Parse BIDS-style entities from a filename.

    Supports standard entities like ``run-01`` and legacy run tokens like
    ``r0001`` that occur in older exported datasets.
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
                continue
        m = re.fullmatch(r'r(\d+)', token, flags=re.IGNORECASE)
        if m:
            entities['run'] = m.group(1)
    if tokens:
        suffix_token = tokens[-1]
        entities['suffix'] = suffix_token.split('-')[-1] if '-' in suffix_token else suffix_token
    return entities
