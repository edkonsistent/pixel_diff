from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Offsets:
    a: int = 0
    b: int = 0

@dataclass
class IOConfig:
    file_a: str
    file_b: str
    out_path: Path | None = None
    fps: float | None = None
    show: bool = False
    wait_ms: int = 1
    save_frame: Path | None = None

@dataclass
class RangeConfig:
    frame: int | None = None     # exact frame index (0-based)
    start: int = 0               # base start when exporting video
    length: int | None = None    # number of pairs to write (video mode)

@dataclass
class DiffConfig:
    mode: str = "color"          # "color" or "binary"
    threshold: int = 0           # tolerance (0 = exact)
    invert: bool = False         # only for binary
    resize_b_to_a: bool = False  # resize B to match A
