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

    # ---------- Color comparison ----------

    def _color_absdiff(self, a_bgr, b_bgr):
        a_bgr, b_bgr = self._ensure_same_size(a_bgr, b_bgr)
        return cv2.absdiff(a_bgr, b_bgr)

    def color_diff(self, a_bgr, b_bgr, scale: float = 1.0):
        """
        Per-channel absolute difference in BGR (uint8).
        Optionally multiplies by 'scale' for visibility (clipped to 255).
        """
        d = self._color_absdiff(a_bgr, b_bgr)
        if scale and scale != 1.0:
            d = np.clip(d.astype(np.float32) * float(scale), 0, 255).astype(np.uint8)
        return d

    def color_binary(self, a_bgr, b_bgr):
        """
        Binary mask using per-pixel max channel deviation:
          255 = different (max(|ΔB|,|ΔG|,|ΔR|) > threshold)
          0   = identical (<= threshold)
        Optionally inverted to make white=identical.
        """
        d = self._color_absdiff(a_bgr, b_bgr)  # HxWx3
        maxdiff = d.max(axis=2)                # HxW
        if self.cfg.threshold <= 0:
            mask = (maxdiff != 0).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(maxdiff, self.cfg.threshold, 255, cv2.THRESH_BINARY)
        if self.cfg.invert:
            mask = cv2.bitwise_not(mask)
        return mask

    # ---------- Gray comparison (matches OLD script semantics) ----------

    def _to_gray(self, bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _gray_pair(self, a_bgr, b_bgr):
        a_bgr, b_bgr = self._ensure_same_size(a_bgr, b_bgr)
        return self._to_gray(a_bgr), self._to_gray(b_bgr)

    def gray_diff(self, a_bgr, b_bgr, scale: float = 1.0):
        ga, gb = self._gray_pair(a_bgr, b_bgr)
        d = cv2.absdiff(ga, gb)  # HxW, uint8
        if scale and scale != 1.0:
            d = np.clip(d.astype(np.float32) * float(scale), 0, 255).astype(np.uint8)
        return d  # single-channel

    def gray_binary(self, a_bgr, b_bgr):
        ga, gb = self._gray_pair(a_bgr, b_bgr)
        diff = cv2.absdiff(ga, gb)
        if self.cfg.threshold <= 0:
            mask = (diff != 0).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(diff, self.cfg.threshold, 255, cv2.THRESH_BINARY)
        if self.cfg.invert:
            mask = cv2.bitwise_not(mask)
        return mask

    # ---------- Unified entrypoint ----------

    def make_matte(self, a_bgr, b_bgr, color_scale: float = 1.0):
        """
        Returns (frame, is_color)
          - If is_color=False we’re writing a single-channel grayscale frame.
        """
        if self.cfg.compare == "gray":
            if self.cfg.mode == "binary":
                return self.gray_binary(a_bgr, b_bgr), False
            else:
                return self.gray_diff(a_bgr, b_bgr, scale=color_scale), False
        else:  # compare == "color"
            if self.cfg.mode == "binary":
                return self.color_binary(a_bgr, b_bgr), False
            else:
                return self.color_diff(a_bgr, b_bgr, scale=color_scale), True

def compute_stats(a_bgr, b_bgr, threshold: int):
    """
    Stats to help diagnose “no difference” situations.
    Returns:
      max_bgr: tuple(int,int,int)
      mean_bgr: tuple(float,float,float)
      pct_diff_over_threshold: float (0..100)
      pct_any_diff: float (0..100)
    """
    d = cv2.absdiff(a_bgr, b_bgr).astype(np.uint8)
    max_b, max_g, max_r = int(d[:, :, 0].max()), int(d[:, :, 1].max()), int(d[:, :, 2].max())
    mean_b = float(d[:, :, 0].mean()); mean_g = float(d[:, :, 1].mean()); mean_r = float(d[:, :, 2].mean())
    maxdiff = d.max(axis=2)
    if threshold <= 0:
        pct_any = 100.0 * float((maxdiff != 0).sum()) / maxdiff.size
        pct_thr = pct_any
    else:
        pct_thr = 100.0 * float((maxdiff > threshold).sum()) / maxdiff.size
        pct_any = 100.0 * float((maxdiff != 0).sum()) / maxdiff.size
    return {
        "max_bgr": (max_b, max_g, max_r),
        "mean_bgr": (mean_b, mean_g, mean_r),
        "pct_diff_over_threshold": pct_thr,
        "pct_any_diff": pct_any,
    }
