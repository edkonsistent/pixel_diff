#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import cv2, numpy as np

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

def sequential_pair_reader_two_starts(path_a, path_b, start_a=0, start_b=0, max_pairs=None, resize_to_a=False):
    cap_a, cap_b = cv2.VideoCapture(path_a), cv2.VideoCapture(path_b)
    if not cap_a.isOpened(): raise RuntimeError(f"Could not open {path_a}")
    if not cap_b.isOpened(): raise RuntimeError(f"Could not open {path_b}")

    # Robustly advance sequentially to the starting indices
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

def get_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps

def main():
    p = argparse.ArgumentParser(
        description="Pixel-by-pixel grayscale difference matte (black=same, white=different), with per-video offsets."
    )
    p.add_argument("filea"); p.add_argument("fileb")
    p.add_argument("--offset-a", type=int, default=0, help="Frame offset for A (positive skips frames; negative earlier).")
    p.add_argument("--offset-b", type=int, default=0, help="Frame offset for B (positive skips frames; negative earlier).")
    p.add_argument("--frame", type=int, help="Compare a single frame index (0-based, offsets applied).")
    p.add_argument("--start", type=int, default=0, help="Base start index (applied to both, then offsets added).")
    p.add_argument("--end", type=int, default=None, help="Number of pairs to write (if provided).")
    p.add_argument("--out", type=Path, help="Output video path (e.g. out/matte.mp4).")
    p.add_argument("--resize", action="store_true", help="Resize B to match A.")
    p.add_argument("--threshold", type=int, default=0, help="Tolerance for tiny differences (0=exact match).")
    p.add_argument("--invert", action="store_true", help="Invert colors so white=identical, black=different.")
    p.add_argument("--show", action="store_true"); p.add_argument("--wait", type=int, default=1)
    p.add_argument("--save-frame", type=Path); p.add_argument("--fps", type=float, default=None)
    args = p.parse_args()

    # --- Single frame mode (exact, sequential) ---
    if args.frame is not None:
        fa_idx = args.frame + args.offset_a
        fb_idx = args.frame + args.offset_b
        gray_a = sequential_read_gray_frame_at(args.filea, fa_idx)
        gray_b = sequential_read_gray_frame_at(args.fileb, fb_idx)

        if gray_a.shape != gray_b.shape:
            if args.resize:
                gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                print("Frame size mismatch; use --resize.", file=sys.stderr); sys.exit(1)

        mask = make_binary_diff(gray_a, gray_b, args.threshold, args.invert)

        if args.show:
            cv2.imshow(f"Difference Matte (A[{fa_idx}] vs B[{fb_idx}])", mask)
            cv2.waitKey(0); cv2.destroyAllWindows()

        if args.save_frame:
            args.save_frame.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_frame), mask)

        if args.out:
            h, w = mask.shape
            fps = args.fps or 30.0
            vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
            if not vw.isOpened():
                print("Error opening output writer.", file=sys.stderr); sys.exit(1)
            vw.write(mask); vw.release()
        return

    # --- Video mode (offset starts, sequential, lock-step) ---
    if not args.out:
        print("Provide --frame or --out.", file=sys.stderr); sys.exit(2)

    start_a = max(0, args.start + args.offset_a)
    start_b = max(0, args.start + args.offset_b)

    fps_a, fps_b = get_fps(args.filea), get_fps(args.fileb)
    fps = args.fps or (min(f for f in [fps_a, fps_b] if f > 0) if any([fps_a, fps_b]) else 30.0)

    max_pairs = args.end if args.end is not None else None
    reader = sequential_pair_reader_two_starts(
        args.filea, args.fileb,
        start_a=start_a, start_b=start_b,
        max_pairs=max_pairs,
        resize_to_a=args.resize
    )

    try:
        pair0, ga, gb = next(reader)
    except StopIteration:
        print("No frames available at the chosen starts.", file=sys.stderr); sys.exit(1)

    mask0 = make_binary_diff(ga, gb, args.threshold, args.invert)
    h, w = mask0.shape
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    if not vw.isOpened():
        print("Error opening output writer.", file=sys.stderr); sys.exit(1)
    vw.write(mask0)

    if args.show:
        cv2.imshow(f"Difference Matte (start A[{start_a}] vs B[{start_b}])", mask0)
        if cv2.waitKey(args.wait) == 27:
            vw.release(); cv2.destroyAllWindows(); sys.exit(0)

    for pair_idx, ga, gb in reader:
        mask = make_binary_diff(ga, gb, args.threshold, args.invert)
        vw.write(mask)
        if args.show:
            cv2.imshow("Difference Matte", mask)
            if cv2.waitKey(args.wait) == 27:
                break

    vw.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
