"""Схемы раскладки окна терминала. Новый тип раскладки = новый класс + @register_layout."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field


class Roi(BaseModel):
    x: int
    y: int
    w: int
    h: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def scaled(self, sx: float, sy: float) -> Roi:
        return Roi(x=round(self.x * sx), y=round(self.y * sy),
                   w=round(self.w * sx), h=round(self.h * sy))


class BaseLayout(BaseModel):
    kind: str
    terminal: str = "unknown"
    reference_size: tuple[int, int]            # (ширина, высота) кадра калибровки

    def scaled_to(self, size: tuple[int, int]) -> BaseLayout:
        raise NotImplementedError

    def draw(self, image: np.ndarray) -> np.ndarray:
        """Копия кадра с наложенной раскладкой (для визуальной проверки)."""
        raise NotImplementedError


LAYOUT_KINDS: dict[str, type[BaseLayout]] = {}


def register_layout(cls: type[BaseLayout]) -> type[BaseLayout]:
    LAYOUT_KINDS[cls.model_fields["kind"].default] = cls
    return cls


@register_layout
class LadderLayout(BaseLayout):
    """Одна непрерывная лестница цен: строки с равным шагом, колонки размера и цены."""
    kind: Literal["ladder"] = "ladder"
    roi: Roi                                   # от верха первой строки до низа последней
    rows: int = Field(gt=0)
    price_col: tuple[float, float]             # доли ширины roi
    size_col: tuple[float, float]

    def scaled_to(self, size: tuple[int, int]) -> LadderLayout:
        sx, sy = size[0] / self.reference_size[0], size[1] / self.reference_size[1]
        return self.model_copy(update={"roi": self.roi.scaled(sx, sy), "reference_size": size})

    def cells(self, image: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Для каждой строки сверху вниз: (ячейка размера, ячейка цены)."""
        ladder = self.roi.crop(image)
        h, w = ladder.shape[:2]
        for i in range(self.rows):
            row = ladder[round(i * h / self.rows) : round((i + 1) * h / self.rows)]
            yield (row[:, round(self.size_col[0] * w) : round(self.size_col[1] * w)],
                   row[:, round(self.price_col[0] * w) : round(self.price_col[1] * w)])

    def draw(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        r = self.roi
        cv2.rectangle(img, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 2)
        for i in range(1, self.rows):
            y = r.y + round(i * r.h / self.rows)
            cv2.line(img, (r.x, y), (r.x + r.w, y), (0, 0, 255), 1)
        for name, (a, b), color in (("size", self.size_col, (255, 0, 0)),
                                    ("price", self.price_col, (0, 160, 0))):
            for frac in (a, b):
                x = r.x + round(frac * r.w)
                cv2.line(img, (x, r.y), (x, r.y + r.h), color, 2)
            cv2.putText(img, name, (r.x + round(a * r.w), max(15, r.y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img


def parse_layout(data: dict[str, Any]) -> BaseLayout:
    kind = data.get("kind", "ladder")
    if kind not in LAYOUT_KINDS:
        raise ValueError(f"неизвестный тип раскладки {kind!r}; есть: {sorted(LAYOUT_KINDS)}")
    return LAYOUT_KINDS[kind].model_validate(data)


def load_layout(path: str | Path) -> BaseLayout:
    return parse_layout(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def save_layout(layout: BaseLayout, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(layout.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8")
