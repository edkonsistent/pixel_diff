#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2

from .config import IOConfig, RangeConfig, Offsets, DiffConfig
from .io_video import VideoReader, VideoWriter
from .diff_core import DiffComputer

class PixelDiffCLI:
    def __init__(self, io: IOConfig, rng: RangeConfig, off: Offsets, diff: DiffConfig):
        self.io = io
        self.rng = rng
        self.off = off
        self.diffc = DiffComputer(diff)

    def _determine_fps(self, fps_override: float | None) -> float:
        if fps_override and fps_override > 0:
            return fps_override
        ra = VideoReader(self.io.file_a)
        rb = VideoReader(self.io.file_b)
        try:
            fps_a = ra.get_fps()
            fps_b = rb.get_fps()
        finally:
            ra.release(); rb.release()
        if fps_a > 0 and fps_b > 0:
            return min(fps_a, fps_b)
        if fps_a > 0: return fps_a
        if fps_b > 0: return fps_b
        return 30.0

    def run_single_frame(self):
        idx_a = max(0, self.rng.frame + self.off.a)
        idx_b = max(0, self.rng.frame + self.off.b)

        ra = VideoReader(self.io.file_a)
        rb = VideoReader(self.io.file_b)
        try:
            ra.seek_sequential(idx_a)
            rb.seek_sequential(idx_b)
            fa = ra.read()
            fb = rb.read()
            if fa is None or fb is None:
                raise RuntimeError("Could not read requested frames (out of range?).")

            matte, is_color = self.diffc.make_matte(fa, fb)

            if self.io.show:
                title = f"Matte ({'color' if is_color else 'binary'})  A[{idx_a}] vs B[{idx_b}]"
                cv2.imshow(title, matte)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if self.io.save_frame:
                self.io.save_frame.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(self.io.save_frame), matte)
                if not ok:
                    print(f"Warning: failed to save {self.io.save_frame}", file=sys.stderr)

            if self.io.out_path:
                h, w = matte.shape[:2]
                writer = VideoWriter(self.io.out_path, self._determine_fps(self.io.fps), (w, h), is_color=is_color)
                writer.write(matte)
                writer.close()
        finally:
            ra.release(); rb.release()

    def run_video(self):
        start_a = max(0, self.rng.start + self.off.a)
        start_b = max(0, self.rng.start + self.off.b)

        ra = VideoReader(self.io.file_a)
        rb = VideoReader(self.io.file_b)
        writer = None
        pairs_written = 0
        try:
            ra.seek_sequential(start_a)
            rb.seek_sequential(start_b)

            fa = ra.read(); fb = rb.read()
            if fa is None or fb is None:
                raise RuntimeError("No frames available at chosen starts.")

            first_matte, is_color = self.diffc.make_matte(fa, fb)
            h, w = first_matte.shape[:2]
            fps = self._determine_fps(self.io.fps)
            self.io.out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = VideoWriter(self.io.out_path, fps, (w, h), is_color=is_color)
            writer.write(first_matte)
            pairs_written += 1

            if self.io.show:
                cv2.imshow("Difference Matte", first_matte)
                if cv2.waitKey(self.io.wait_ms) == 27:
                    writer.close()
                    cv2.destroyAllWindows()
                    return

            while True:
                if self.rng.length is not None and pairs_written >= self.rng.length:
                    break
                fa = ra.read(); fb = rb.read()
                if fa is None or fb is None:
                    break

                matte, _ = self.diffc.make_matte(fa, fb)
                writer.write(matte)
                pairs_written += 1

                if self.io.show:
                    cv2.imshow("Difference Matte", matte)
                    if cv2.waitKey(self.io.wait_ms) == 27:
                        break
        finally:
            if writer: writer.close()
            ra.release(); rb.release()
            if self.io.show:
                cv2.destroyAllWindows()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pixel_diff",
        description="Pixel-by-pixel video difference. Modes: color absdiff or binary mask."
    )
    # Inputs / outputs
    p.add_argument("filea", help="Path to first video.")
    p.add_argument("fileb", help="Path to second video.")
    p.add_argument("--out", type=Path, help="Output video path (e.g. out/matte.mp4).")
    p.add_argument("--fps", type=float, default=None, help="Output FPS (default=min input FPS or 30).")
    p.add_argument("--save-frame", type=Path, help="Save the single-frame result image here.")
    p.add_argument("--show", action="store_true", help="Preview window.")
    p.add_argument("--wait", type=int, default=1, help="ms delay while previewing video.")
    # Ranging
    p.add_argument("--frame", type=int, help="Single frame index (0-based). If set, --out writes a 1-frame video.")
    p.add_argument("--start", type=int, default=0, help="Video mode: base start index for both videos.")
    p.add_argument("--length", type=int, default=None, help="Video mode: number of pairs to write.")
    # Offsets
    p.add_argument("--offset-a", type=int, default=0, help="Frame offset for A (positive skips frames).")
    p.add_argument("--offset-b", type=int, default=0, help="Frame offset for B (positive skips frames).")
    # Diff config
    p.add_argument("--mode", choices=["color", "binary"], default="color",
                   help="Difference output: 'color' = per-channel absdiff, 'binary' = mask.")
    p.add_argument("--threshold", type=int, default=0,
                   help="Tolerance (0=bit-exact). For 'binary', uses per-pixel max channel diff.")
    p.add_argument("--invert", action="store_true",
                   help="Binary only: invert so white=identical, black=different.")
    p.add_argument("--resize", action="store_true", help="Resize B to match A before diff.")
    return p

def main():
    args = build_parser().parse_args()

    io = IOConfig(
        file_a=args.filea,
        file_b=args.fileb,
        out_path=args.out,
        fps=args.fps,
        show=args.show,
        wait_ms=args.wait,
        save_frame=args.save_frame
    )
    rng = RangeConfig(frame=args.frame, start=args.start, length=args.length)
    off = Offsets(a=args.offset_a, b=args.offset_b)
    diff = DiffConfig(
        mode=args.mode,
        threshold=args.threshold,
        invert=args.invert,
        resize_b_to_a=args.resize
    )

    app = PixelDiffCLI(io, rng, off, diff)
    if rng.frame is not None:
        app.run_single_frame()
    else:
        if io.out_path is None:
            print("In video mode you must provide --out", file=sys.stderr)
            sys.exit(2)
        app.run_video()

if __name__ == "__main__":
    main()
