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
    frame: int | None = None
    start: int = 0
    end: int | None = None   # matches your --end flag

@dataclass
class DiffConfig:
    threshold: int = 0
    invert: bool = False
    resize_b_to_a: bool = False
