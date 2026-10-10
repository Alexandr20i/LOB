from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TerminalProfile:
    """Всё, что известно заранее о конкретном терминале. Данные, не код:
    новый терминал с лестницей цен добавляется файлом из десяти строк."""
    name: str
    layout_kind: str                                  # ключ в LAYOUT_KINDS и CALIBRATORS
    calibration: dict[str, Any] = field(default_factory=dict)   # параметры калибратора
    notes: str = ""                                   # какие настройки терминала требуются


TERMINALS: dict[str, TerminalProfile] = {}


def register_terminal(profile: TerminalProfile) -> TerminalProfile:
    TERMINALS[profile.name] = profile
    return profile


def get_terminal(name: str) -> TerminalProfile:
    if name not in TERMINALS:
        raise KeyError(f"неизвестный терминал {name!r}; есть: {sorted(TERMINALS)}")
    return TERMINALS[name]
