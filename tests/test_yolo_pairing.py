"""Run: python -m unittest discover tests"""
import os
import tempfile
import unittest

import cv2
import numpy as np

from yolo_frame_review import yolo_pairing as yp


def write_video(path, frames, fps=25.0, size=(64, 48)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
    w.release()


class TestNaming(unittest.TestCase):
    def test_key_strips_raw(self):
        self.assertEqual(yp.yolo_key("sub/cam1 2025-01-01 08_00_00_raw.avi"),
                         "cam1 2025-01-01 08_00_00")
        self.assertEqual(yp.yolo_key("CLIP_RAW.mp4"), "CLIP")
        self.assertEqual(yp.yolo_key("clip.mp4"), "clip")

    def test_prefix_word_boundary(self):
        key = "cam1 2025-01-01 08_00_00"
        self.assertTrue(yp.prefix_matches(key, key))
        self.assertTrue(yp.prefix_matches(key, key + "_raw_yolo"))
        self.assertTrue(yp.prefix_matches(key, key.upper() + "-pred"))
        self.assertFalse(yp.prefix_matches(key, key + "0"))
        self.assertFalse(yp.prefix_matches("cam1", "cam10_yolo"))
        self.assertFalse(yp.prefix_matches(key, key[:-1]))

    def test_mismatch(self):
        self.assertEqual(yp.mismatch(100, 25.0, 100, 25.0), [])
        probs = yp.mismatch(100, 25.0, 99, 25.0)
        self.assertEqual(len(probs), 2)  # frame count + duration line
        self.assertIn("Frame count", probs[0])
        self.assertTrue(any("FPS" in p for p in yp.mismatch(100, 25.0, 100, 30.0)))


class TestPairFolder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.src = os.path.join(root, "src")
        self.yolo = os.path.join(root, "yolo")
        os.makedirs(self.src)
        os.makedirs(os.path.join(self.yolo, "nested"))

        write_video(os.path.join(self.src, "a_raw.avi"), 30)
        write_video(os.path.join(self.src, "b_raw.avi"), 30)
        write_video(os.path.join(self.src, "c_raw.avi"), 30)
        write_video(os.path.join(self.src, "d_raw.avi"), 30)

        write_video(os.path.join(self.yolo, "nested", "a_yolo.avi"), 30)  # good
        write_video(os.path.join(self.yolo, "b_yolo.avi"), 20)            # wrong length
        # c: none; d: two valid twins -> ambiguous
        write_video(os.path.join(self.yolo, "d_v1.avi"), 30)
        write_video(os.path.join(self.yolo, "d_v2.avi"), 30)
        write_video(os.path.join(self.yolo, "a0_yolo.avi"), 30)           # must not match 'a'

        self.videos = sorted(os.listdir(self.src))

    def tearDown(self):
        self._tmp.cleanup()

    def test_pairing(self):
        files = yp.find_videos(self.yolo)
        self.assertEqual(len(files), 5)
        mapping, failures = yp.pair_folder(self.videos, self.src, files)

        self.assertEqual(list(mapping), ["a_raw.avi"])
        self.assertEqual(os.path.basename(mapping["a_raw.avi"]), "a_yolo.avi")
        text = "\n".join(failures)
        self.assertEqual(len(failures), 3)
        self.assertIn("b_raw.avi", text)
        self.assertIn("Frame count", text)
        self.assertIn("no YOLO video starting with 'c'", text)
        self.assertIn("ambiguous", text)

    def test_nested_yolo_folder(self):
        # videos/ + videos/yolo_output/ — the layout a recursive scan trips on
        root = os.path.join(self._tmp.name, "nested_case")
        out = os.path.join(root, "yolo_output")
        os.makedirs(out)
        write_video(os.path.join(root, "x 08_00_00.avi"), 30)
        write_video(os.path.join(out, "x 08_00_00_yolo.avi"), 30)
        self.assertTrue(yp.is_under(out, root))
        self.assertFalse(yp.is_under(root, out))

        all_videos = sorted(os.path.relpath(p, root) for p in yp.find_videos(root))
        sources = [v for v in all_videos
                   if not yp.is_under(os.path.join(root, v), out)]
        mapping, failures = yp.pair_folder(sources, root, yp.find_videos(out))
        self.assertEqual(failures, [])
        self.assertEqual(os.path.basename(mapping["x 08_00_00.avi"]),
                         "x 08_00_00_yolo.avi")

    def test_same_folder_never_self_pairs(self):
        root = os.path.join(self._tmp.name, "same_case")
        os.makedirs(root)
        write_video(os.path.join(root, "y.avi"), 30)
        write_video(os.path.join(root, "y_yolo.avi"), 30)
        mapping, _ = yp.pair_folder(["y.avi"], root, yp.find_videos(root))
        self.assertEqual(os.path.basename(mapping["y.avi"]), "y_yolo.avi")

    def test_find_videos_excludes_originals(self):
        found = yp.find_videos(self.src,
                               exclude=[os.path.join(self.src, "a_raw.avi")])
        names = sorted(os.path.basename(p) for p in found)
        self.assertEqual(names, ["b_raw.avi", "c_raw.avi", "d_raw.avi"])


if __name__ == "__main__":
    unittest.main()
