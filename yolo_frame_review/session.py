"""Session persistence: saved frame indices and notes per video.

Shape: {"files": [rel_path, ...],          # videos with >=1 saved frame
        "notes": {rel_path: str},
        rel_path: [frame_idx, ...]}
"""
import json
import os

SESSION_FILE = "extraction_session.json"


def load_session(path=SESSION_FILE):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"files": []}


def save_session(data, path=SESSION_FILE):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
