from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    index: int
    t_ms: float
    image: np.ndarray  # BGR, uint8


class VideoSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


class OpenCVVideoSource:
    """Читает файл покадрово. t_ms берётся из контейнера (CAP_PROP_POS_MSEC),
    а не считается как index / fps: у скринкастов часто переменная частота кадров."""

    def __init__(self, path: str | Path, stride: int = 1) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.stride = max(1, stride)

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть {self.path}")
        try:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if index % self.stride == 0:
                    yield Frame(index, cap.get(cv2.CAP_PROP_POS_MSEC), image)
                index += 1
        finally:
            cap.release()


def sample_frames(path: str | Path, n: int = 8) -> list[np.ndarray]:
    """n кадров, равномерно по записи (без самого начала и конца), для калибровки.
    Перемотка по номеру кадра в некоторых кодеках неточна; для калибровки это неважно."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть {p}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[np.ndarray] = []
        if total <= 0:                         # контейнер не знает длину: берём подряд с шагом
            i = 0
            while len(frames) < n:
                ok, img = cap.read()
                if not ok:
                    break
                if i % 30 == 0:
                    frames.append(img)
                i += 1
            return frames
        for idx in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            if ok:
                frames.append(img)
        return frames
    finally:
        cap.release()
