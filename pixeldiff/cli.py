#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2

from .io_video import VideoWriterGray, get_fps
from .diff_gray import (
    sequential_read_gray_frame_at,
    sequential_pair_reader_two_starts,
    make_binary_diff,
)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pixeldiff",
        description="Pixel-by-pixel grayscale difference matte (black=same, white=different)."
    )
    p.add_argument("filea")
    p.add_argument("fileb")
    p.add_argument("--offset-a", type=int, default=0)
    p.add_argument("--offset-b", type=int, default=0)
    p.add_argument("--frame", type=int)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--out", type=Path)
    p.add_argument("--resize", action="store_true")
    p.add_argument("--threshold", type=int, default=0)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--wait", type=int, default=1)
    p.add_argument("--save-frame", type=Path)
    p.add_argument("--fps", type=float, default=None)
    return p

def main():
    args = build_parser().parse_args()

    # --- Single frame mode ---
    if args.frame is not None:
        fa_idx = max(0, args.frame + args.offset_a)
        fb_idx = max(0, args.frame + args.offset_b)
        gray_a = sequential_read_gray_frame_at(args.filea, fa_idx)
        gray_b = sequential_read_gray_frame_at(args.fileb, fb_idx)

        if gray_a.shape != gray_b.shape:
            if args.resize:
                gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                print("Frame size mismatch; use --resize.", file=sys.stderr)
                sys.exit(1)

        mask = make_binary_diff(gray_a, gray_b, args.threshold, args.invert)

        if args.show:
            cv2.imshow(f"Matte (A[{fa_idx}] vs B[{fb_idx}])", mask)
            cv2.waitKey(0); cv2.destroyAllWindows()

        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_frame), mask)

        if args.out:
            h, w = mask.shape
            fps = args.fps or 30.0
            vw = VideoWriterGray(args.out, fps, (w, h))
            vw.write(mask)
            vw.close()
        return

    # --- Video mode ---
    if not args.out:
        print("Provide --frame or --out.", file=sys.stderr)
        sys.exit(2)

    start_a = max(0, args.start + args.offset_a)
    start_b = max(0, args.start + args.offset_b)

    fps_a, fps_b = get_fps(args.filea), get_fps(args.fileb)
    fps = args.fps or (min(f for f in [fps_a, fps_b] if f > 0) if any([fps_a, fps_b]) else 30.0)

    reader = sequential_pair_reader_two_starts(
        args.filea, args.fileb,
        start_a=start_a, start_b=start_b,
        max_pairs=args.end,
        resize_to_a=args.resize
    )

    try:
        pair0, ga, gb = next(reader)
    except StopIteration:
        print("No frames available.", file=sys.stderr)
        sys.exit(1)

    mask0 = make_binary_diff(ga, gb, args.threshold, args.invert)
    h, w = mask0.shape
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = VideoWriterGray(args.out, fps, (w, h))
    vw.write(mask0)

    if args.show:
        cv2.imshow(f"Start A[{start_a}] vs B[{start_b}]", mask0)
        if cv2.waitKey(args.wait) == 27:
            vw.close(); cv2.destroyAllWindows(); sys.exit(0)

    for pair_idx, ga, gb in reader:
        mask = make_binary_diff(ga, gb, args.threshold, args.invert)
        vw.write(mask)
        if args.show:
            cv2.imshow("Matte", mask)
            if cv2.waitKey(args.wait) == 27:
                break

    vw.close()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
