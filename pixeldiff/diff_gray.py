from __future__ import annotations
import cv2
import numpy as np

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sequential_read_gray_frame_at(path, frame_idx):
    """Always decode sequentially from 0 to frame_idx (exact, backend-agnostic)."""
    frame_idx = max(0, int(frame_idx))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    gray = None
    for i in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok or f is None:
            cap.release()
            raise RuntimeError(f"Failed to read exact frame {frame_idx} (stopped at {i}).")
        if i == frame_idx:
            gray = to_gray(f)
    cap.release()
    return gray

def sequential_pair_reader_two_starts(path_a, path_b, start_a=0, start_b=0,
                                      max_pairs=None, resize_to_a=False):
    cap_a, cap_b = cv2.VideoCapture(path_a), cv2.VideoCapture(path_b)
    if not cap_a.isOpened(): raise RuntimeError(f"Could not open {path_a}")
    if not cap_b.isOpened(): raise RuntimeError(f"Could not open {path_b}")

    # Advance sequentially to start indices
    for cap, start, label in [(cap_a, start_a, "A"), (cap_b, start_b, "B")]:
        for i in range(start):
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to reach start {start} for {label} (stopped at {i}).")

    pair_idx = 0
    try:
        while True:
            if max_pairs is not None and pair_idx >= max_pairs:
                break
            ok_a, fa = cap_a.read()
            ok_b, fb = cap_b.read()
            if not ok_a or fa is None or not ok_b or fb is None:
                break

            if resize_to_a and fa.shape[:2] != fb.shape[:2]:
                fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]), interpolation=cv2.INTER_AREA)
            if fa.shape[:2] != fb.shape[:2]:
                raise RuntimeError(
                    f"Size mismatch at pair {pair_idx}: A={fa.shape[:2]} vs B={fb.shape[:2]} (use --resize)."
                )

            yield pair_idx, to_gray(fa), to_gray(fb)
            pair_idx += 1
    finally:
        cap_a.release(); cap_b.release()

def make_binary_diff(gray_a, gray_b, threshold=0, invert=False):
    diff = cv2.absdiff(gray_a, gray_b)
    if threshold <= 0:
        mask = (diff != 0).astype(np.uint8) * 255
    else:
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(mask) if invert else mask
