
import os
import json
from datetime import datetime, timezone

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def write_session_status(
    session_out_dir: str,
    *,
    project_id: str,
    subject_id: str,
    session_id: str,
    success: bool,
    input_fingerprint: dict,
    args: dict,
    error: str | None = None,
) -> str:
    """Write the per-session status file used for resume/force planning."""
    _ensure_dir(session_out_dir)
    status = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'project_id': project_id,
        'subjectID': subject_id,
        'sessionID': session_id,
        'success': bool(success),
        'error': error,
        'input_fingerprint': input_fingerprint,
        'args': args,
    }
    path = os.path.join(session_out_dir, '.antsxmm_status.json')
    _write_json(path, status)
    return path
