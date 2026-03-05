
import os
import json
from hashlib import sha256

from .inputs import plan_session_inputs, _sidecar_paths_for_nifti

def compute_input_fingerprint(session_data, *, t1_run_match: str | None = None) -> dict:
    """Compute a stable fingerprint of the session inputs (NIfTI + sidecars)."""
    plan = plan_session_inputs(session_data, t1_run_match=t1_run_match)
    if not plan.get('processable', False):
        return {
            'algo': 'sha256',
            'hash': None,
            'files': [],
            'processable': False,
            'reason': plan.get('reason', 'unprocessible'),
        }

    files: list[dict] = []
    all_paths: list[str] = []
    for p in plan['nifti_inputs']:
        all_paths.append(os.path.realpath(p))
        all_paths.extend(_sidecar_paths_for_nifti(p))
    all_paths = sorted(set(all_paths))

    for p in all_paths:
        try:
            st = os.stat(p)
            files.append({
                'path': os.path.realpath(p),
                'size': int(st.st_size),
                'mtime_ns': int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))),
            })
        except FileNotFoundError:
            # Treat missing as part of the fingerprint (forces rerun)
            files.append({'path': os.path.realpath(p), 'size': None, 'mtime_ns': None})

    payload = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    h = sha256(payload).hexdigest()
    return {
        'algo': 'sha256',
        'hash': h,
        'files': files,
        'processable': True,
    }
