from __future__ import annotations
from pathlib import Path
import cv2

class VideoReader:
    """Sequential frame access for reliable positioning on all codecs."""
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {path}")

    def seek_sequential(self, frame_idx: int) -> None:
        frame_idx = max(0, int(frame_idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(frame_idx):
            ok, _ = self.cap.read()
            if not ok:
                raise RuntimeError(f"[{self.path}] Failed to reach frame {frame_idx} (stopped at {i}).")

    def read(self):
        ok, f = self.cap.read()
        return f if ok and f is not None else None

    def get_fps(self) -> float:
        v = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        return float(v)

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass

class VideoWriterGray:
    """MP4 writer for single-channel (grayscale) frames."""
    def __init__(self, out_path: Path, fps: float, size_wh: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.vw = cv2.VideoWriter(str(out_path), fourcc, fps, size_wh, isColor=False)
        if not self.vw.isOpened():
            raise RuntimeError(f"Error opening writer for {out_path}. Try an .mp4 filename.")

    def write(self, gray_frame):
        self.vw.write(gray_frame)

    def close(self):
        self.vw.release()

def get_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return float(fps)
