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