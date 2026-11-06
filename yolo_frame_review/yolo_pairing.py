"""Pair source videos with their YOLO-rendered twins by filename prefix.

A YOLO twin is only usable when it has the same frame count and fps as the
source — the viewer maps frame n of one to frame n of the other.
"""
import os
from pathlib import Path

from .video import VIDEO_EXTS, probe


def yolo_key(rel_path):
    """'cam1 2025-01-01 08_00_00_raw.mp4' -> 'cam1 2025-01-01 08_00_00'."""
    stem = Path(rel_path).stem
    return stem[:-4] if stem.lower().endswith("_raw") else stem


def prefix_matches(key, stem):
    """Case-insensitive prefix match that must end at a word boundary,
    so 'cam1' doesn't grab 'cam10_yolo'."""
    k, s = key.lower(), stem.lower()
    return s.startswith(k) and (len(s) == len(k) or not s[len(k)].isalnum())


def mismatch(orig_total, orig_fps, yolo_total, yolo_fps):
    """Problem lines explaining why two videos can't be index-mapped; [] if OK."""
    problems = []
    if yolo_total != orig_total:
        problems.append(f"Frame count: original {orig_total}, YOLO {yolo_total}")
    if abs(yolo_fps - orig_fps) > 0.01:
        problems.append(f"FPS: original {orig_fps:.3f}, YOLO {yolo_fps:.3f}")
    if problems:
        orig_dur = orig_total / orig_fps if orig_fps > 0 else 0
        yolo_dur = yolo_total / yolo_fps if yolo_fps > 0 else 0
        problems.append(f"Duration: original {orig_dur:.2f}s, YOLO {yolo_dur:.2f}s")
    return problems


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def is_under(path, folder):
    """True if `path` is `folder` itself or anywhere inside it."""
    try:
        return os.path.commonpath([_norm(path), _norm(folder)]) == _norm(folder)
    except ValueError:  # different drives
        return False


def find_videos(folder, exclude=()):
    """All video files under `folder` (recursive), minus paths in `exclude`."""
    excluded = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    found = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        for f in filenames:
            p = os.path.join(dirpath, f)
            if (f.lower().endswith(VIDEO_EXTS)
                    and os.path.normcase(os.path.abspath(p)) not in excluded):
                found.append(p)
    return found


def pair_folder(videos, video_folder, yolo_files, progress=None):
    """Match every source video (relative to `video_folder`) to one YOLO file.

    Returns (mapping {rel_path: yolo_path}, failures [str]). A video pairs
    only if exactly one prefix-matching candidate passes the frame/fps check.
    `progress(i, n)` is called before each video, if given.
    """
    mapping, failures = {}, []
    n = len(videos)
    for i, rel in enumerate(videos, 1):
        if progress:
            progress(i, n)

        key = yolo_key(rel)
        src_path = _norm(os.path.join(video_folder, rel))
        # A video never pairs with itself (YOLO and source may share a folder)
        cands = [p for p in yolo_files
                 if prefix_matches(key, Path(p).stem) and _norm(p) != src_path]
        if not cands:
            failures.append(f"{rel}\n    no YOLO video starting with '{key}'")
            continue
        orig = probe(os.path.join(video_folder, rel), original=True)
        if orig is None:
            failures.append(f"{rel}\n    cannot open original video")
            continue

        good, bad = [], []
        for c in cands:
            y = probe(c)
            if y is None:
                bad.append(f"{Path(c).name}: cannot open")
                continue
            problems = mismatch(*orig, *y)
            if problems:
                bad.append(f"{Path(c).name}: " + "; ".join(problems))
            else:
                good.append(c)

        if len(good) == 1:
            mapping[rel] = good[0]
        elif len(good) > 1:
            failures.append(f"{rel}\n    ambiguous, several valid matches: "
                            + ", ".join(Path(g).name for g in good))
        else:
            failures.append(f"{rel}\n    " + "\n    ".join(bad))
    return mapping, failures
