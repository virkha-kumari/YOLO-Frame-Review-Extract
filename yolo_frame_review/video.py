"""Video I/O helpers: cheap seeking, metadata probing, background scrub decoding."""
import threading

import cv2

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

# Forward hops up to this many frames use sequential grab() instead of a seek.
GRAB_FORWARD_MAX = 40


def position_cap(cap, frame_idx):
    """Position `cap` so the next read() returns `frame_idx`.

    Short forward hops use sequential grab() (decode-only, keeps decoder
    state) — far cheaper than cap.set(), which always restarts from the
    nearest keyframe even for +1.
    """
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_idx == pos:
        return True
    if 0 < frame_idx - pos <= GRAB_FORWARD_MAX:
        for _ in range(frame_idx - pos):
            if not cap.grab():
                return False
        return True
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    return True


def read_frame(cap, frame_idx):
    """Decode `frame_idx` from `cap` as RGB, or None on failure."""
    if not position_cap(cap, frame_idx):
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

    def request(self, path, frame_idx, gen):
        with self._cond:
            self._request = (path, frame_idx, gen)
            self._cond.notify()

    def _run(self):
        cap, cap_path = None, None
        while True:
            with self._cond:
                while self._request is None:
                    self._cond.wait()
                path, idx, gen = self._request
                self._request = None
            if path != cap_path:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(path)
                cap_path = path
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = cap.read()
            if not ret:
                continue
            try:
                self._on_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), idx, gen)
            except RuntimeError:
                return  # GUI mainloop gone, app closing
