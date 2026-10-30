from __future__ import annotations

from decimal import Decimal

import numpy as np

from lob.events import RawObservation, RawRow, Side
from lob.layout import LadderLayout
from lob.recognition.glyphs import TemplateSet, read_ladder


class LadderRecognizer:
    """Реализация Recognizer для раскладки «лестница цен». Состояния между кадрами нет."""

    def __init__(self, layout: LadderLayout, templates: TemplateSet, tick: Decimal) -> None:
        self.layout, self.templates, self.tick = layout, templates, tick

    def process(self, image: np.ndarray, frame_index: int, t_ms: float) -> RawObservation:
        h, w = image.shape[:2]
        layout = self.layout
        if (w, h) != tuple(layout.reference_size):
            layout = layout.scaled_to((w, h))
        rows, agreement = read_ladder(layout.cells(image), self.templates, self.tick)
        return RawObservation(
            frame_index=frame_index, t_ms=t_ms, price_agreement=agreement,
            rows=tuple(RawRow(price=r.price, size=r.size,
                              tint=Side(r.tint) if r.tint else None,
                              confidence=min(1.0, max(0.0, r.confidence)))
                       for r in rows))