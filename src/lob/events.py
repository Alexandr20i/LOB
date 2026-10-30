from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Side(StrEnum):
    BID = "bid"
    ASK = "ask"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class RawRow(_Frozen):
    """Одна строка лестницы цен.

    size: None -- не прочитано; Decimal(0) -- ячейка пуста.
    tint: цвет фона строки. Сторону пустых строк определяет реконструктор
    по границе bid/ask, а не распознаватель.
    """
    price: Decimal | None
    size: Decimal | None
    tint: Side | None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class RawTrade(_Frozen):
    price: Decimal | None
    size: Decimal | None
    side: Side | None          # по цвету строки, если терминал так показывает
    time_label: str | None     # время строки ленты как на экране (HH:MM:SS)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class RawObservation(_Frozen):
    frame_index: int
    t_ms: float
    rows: tuple[RawRow, ...]            # сверху вниз, как на экране
    trades: tuple[RawTrade, ...] = ()
    price_agreement: float = Field(ge=0.0, le=1.0, default=1.0)  # доля строк, согласных с прогрессией цен