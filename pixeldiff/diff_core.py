from __future__ import annotations
import cv2
import numpy as np
from .config import DiffConfig

class DiffComputer:
    def __init__(self, cfg: DiffConfig):
        self.cfg = cfg

    def _ensure_same_size(self, a, b):
        if a.shape[:2] != b.shape[:2]:
            if self.cfg.resize_b_to_a:
                b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                raise RuntimeError(f"Frame size mismatch: A={a.shape[:2]} vs B={b.shape[:2]} (use --resize).")
        return a, b

    def color_diff(self, a_bgr, b_bgr):
        """
        Per-channel absolute difference in BGR (uint8).
        """
        a_bgr, b_bgr = self._ensure_same_size(a_bgr, b_bgr)
        return cv2.absdiff(a_bgr, b_bgr)

    def binary_diff(self, a_bgr, b_bgr):
        """
        Binary mask per pixel using max channel deviation:
          255 = different (max(|ΔB|,|ΔG|,|ΔR|) > threshold)
          0   = identical (<= threshold)
        Optionally inverted to make white=identical.
        """
        a_bgr, b_bgr = self._ensure_same_size(a_bgr, b_bgr)
        d = cv2.absdiff(a_bgr, b_bgr)          # HxWx3
        maxdiff = d.max(axis=2)                 # HxW
        if self.cfg.threshold <= 0:
            mask = (maxdiff != 0).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(maxdiff, self.cfg.threshold, 255, cv2.THRESH_BINARY)
        if self.cfg.invert:
            mask = cv2.bitwise_not(mask)
        return mask

    def make_matte(self, a_bgr, b_bgr):
        if self.cfg.mode == "color":
            return self.color_diff(a_bgr, b_bgr), True   # is_color=True
        elif self.cfg.mode == "binary":
            return self.binary_diff(a_bgr, b_bgr), False
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
