from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from lob.layout import BaseLayout, Roi


@dataclass(frozen=True)
class CalibrationReport:
    quality: float                               # 0..1, см. LadderGeometry.quality
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


class CalibrationError(RuntimeError):
    def __init__(self, report: CalibrationReport) -> None:
        super().__init__("; ".join(report.warnings) or "калибровка не удалась")
        self.report = report


class LayoutResolver(Protocol):
    """Кадры-образцы (+ грубая рамка пользователя) -> layout и отчёт о качестве.
    Полуавтомат, профиль терминала и полный автопоиск -- реализации одного интерфейса."""

    def resolve(self, frames: Sequence[np.ndarray], hint: Roi | None = None
                ) -> tuple[BaseLayout, CalibrationReport]: ...


CALIBRATORS: dict[str, Callable[..., LayoutResolver]] = {}


def register_calibrator(kind: str) -> Callable[[Callable[..., LayoutResolver]], Callable[..., LayoutResolver]]:
    def deco(factory: Callable[..., LayoutResolver]) -> Callable[..., LayoutResolver]:
        CALIBRATORS[kind] = factory
        return factory
    return deco


def make_resolver(terminal: str) -> LayoutResolver:
    from lob.terminals import get_terminal

    profile = get_terminal(terminal)
    if profile.layout_kind not in CALIBRATORS:
        raise KeyError(f"для раскладки {profile.layout_kind!r} нет калибратора; есть: {sorted(CALIBRATORS)}")
    params: dict[str, Any] = dict(profile.calibration)
    return CALIBRATORS[profile.layout_kind](terminal=profile.name, **params)
