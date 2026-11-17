"""Video I/O helpers: cheap seeking, metadata probing, background scrub decoding."""
import bisect
import threading

import cv2

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

# Forward hops up to this many frames use sequential grab() instead of a seek.
GRAB_FORWARD_MAX = 40
# Seek-and-correct rounds before giving up on an exact landing.
SEEK_ATTEMPTS = 4


def build_frame_index(path, progress=None):
    """Timestamp (ms) of every real frame, from one grab-only pass.

    Container metadata can lie (e.g. MKV header says 30 fps, camera recorded
    ~27): frame count is then wrong and cap.set(POS_FRAMES) lands on the
    wrong frame. The per-frame timestamps let seeks verify where they landed.
    Returns None if the file can't be opened or timestamps are unusable
    (not strictly increasing). `progress(n_frames)` is called every 2000.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        ts = []
        while cap.grab():
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            if progress and len(ts) % 2000 == 0:
                progress(len(ts))
    finally:
        cap.release()
    if not ts or any(b <= a for a, b in zip(ts, ts[1:])):
        return None
    return ts


def _index_at(ts, msec):
    """Index of the frame whose timestamp is nearest `msec`."""
    i = bisect.bisect_left(ts, msec)
    if i == 0:
        return 0
    if i == len(ts):
        return len(ts) - 1
    return i if ts[i] - msec < msec - ts[i - 1] else i - 1


def _next_pos(cap, ts):
    """Index the next read() will return."""
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if ts is None or pos == 0:
        return pos
    # POS_FRAMES drifts after a seek on bad-metadata files; the timestamp
    # of the last decoded frame doesn't
    return _index_at(ts, cap.get(cv2.CAP_PROP_POS_MSEC)) + 1


def _grab(cap, n):
    for _ in range(n):
        if not cap.grab():
            return False
    return True


def _exact_seek(cap, frame_idx, ts):
    """Seek, check the landed frame's timestamp, correct until exact.

    Aims to land on frame_idx-1 so the next read() is frame_idx. On files
    whose seeks are already right this costs the same as a plain seek
    (one seek + one grab).
    """
    if frame_idx == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return True
    k = frame_idx - 1
    for _ in range(SEEK_ATTEMPTS):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(k, 0))
        if not cap.grab():
            k -= GRAB_FORWARD_MAX  # asked past the real end — back off
            continue
        gap = frame_idx - 1 - _index_at(ts, cap.get(cv2.CAP_PROP_POS_MSEC))
        if 0 <= gap <= GRAB_FORWARD_MAX:
            return _grab(cap, gap)
        k += gap  # shift the request by the measured error and retry
        if gap < 0:
            k -= GRAB_FORWARD_MAX // 2  # overshot: land a little before
    return False


def position_cap(cap, frame_idx, ts=None):
    """Position `cap` so the next read() returns `frame_idx`.

    Short forward hops use sequential grab() (decode-only, keeps decoder
    state) — far cheaper than cap.set(), which always restarts from the
    nearest keyframe even for +1. With `ts` (from build_frame_index) seeks
    are verified and corrected; without it they trust the container.
    """
    if ts is not None and frame_idx >= len(ts):
        return False
    pos = _next_pos(cap, ts)
    if frame_idx == pos:
        return True
    if 0 < frame_idx - pos <= GRAB_FORWARD_MAX:
        return _grab(cap, frame_idx - pos)
    if ts is not None:
        return _exact_seek(cap, frame_idx, ts)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    return True


def read_frame(cap, frame_idx, ts=None):
    """Decode `frame_idx` from `cap` as RGB, or None on failure."""
    if not position_cap(cap, frame_idx, ts):
        return None
    ret, bgr = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def probe(path, original=False):
    """(frame_count, fps) from container metadata, or None if unopenable.

    original=True mirrors how the app loads a source video: falls back to
    seeking to the end when the container reports no frame count, and
    assumes 25 fps when fps is missing. YOLO videos are probed raw.
    """
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not original:
            return total, fps or 0.0
        if total <= 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            total = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        return max(total, 1), fps or 25.0
    finally:
        cap.release()


class ScrubWorker:
    """Decodes slider-drag preview frames off the main thread.

    Single-slot request (latest wins): while the user drags, older requests
    are overwritten rather than queued, so the worker always decodes the
    newest position. Owns its own VideoCapture so it never races the main
    thread's cap.

    `on_frame(rgb, frame_idx, gen)` is called from the worker thread.
    """

    def __init__(self, on_frame):
        self._on_frame = on_frame
        self._cond = threading.Condition()
        self._request = None
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, path, frame_idx, gen, ts=None):
        """`ts`: frame index of `path` (build_frame_index) or None."""
        with self._cond:
            self._request = (path, frame_idx, gen, ts)
            self._cond.notify()

    def _run(self):
        cap, cap_path = None, None
        while True:
            with self._cond:
                while self._request is None:
                    self._cond.wait()
                path, idx, gen, ts = self._request
                self._request = None
            if path != cap_path:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(path)
                cap_path = path
            if not cap.isOpened():
                continue
            rgb = read_frame(cap, idx, ts)
            if rgb is None:
                continue
            try:
                self._on_frame(rgb, idx, gen)
            except RuntimeError:
                return  # GUI mainloop gone, app closing
