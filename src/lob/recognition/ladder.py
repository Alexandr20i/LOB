from __future__ import annotations

from decimal import Decimal

import numpy as np

from lob.events import RawObservation, RawRow, Side
from lob.layout import LadderLayout
from lob.recognition.glyphs import RowCache, TemplateSet, read_ladder


class LadderRecognizer:
    """Реализация Recognizer для раскладки «лестница цен».

    Результат кадра не зависит от истории: кэш строк (RowCache) только пропускает
    повторную работу над ячейками, совпавшими с уже разобранными с точностью до шума
    сжатия. use_cache=False отключает кэш."""

    def __init__(self, layout: LadderLayout, templates: TemplateSet, tick: Decimal,
                 use_cache: bool = True) -> None:
        self.layout, self.templates, self.tick = layout, templates, tick
        self._scaled: dict[tuple[int, int], LadderLayout] = {}
        self._cache = RowCache() if use_cache else None
        self.frames_total = 0
        self.frames_unchanged = 0  # кадры, где все строки взяты из кэша

    def layout_for(self, size: tuple[int, int]) -> LadderLayout:
        """Раскладка, пересчитанная под размер кадра (считается один раз на размер)."""
        if size not in self._scaled:
            same = size == tuple(self.layout.reference_size)
            self._scaled[size] = self.layout if same else self.layout.scaled_to(size)
        return self._scaled[size]

    @property
    def cache_stats(self) -> tuple[int, int, int]:
        """(строк из кэша, строк всего, из них взято при сдвиге лестницы)."""
        c = self._cache
        return (c.hits, c.hits + c.misses, c.shift_hits) if c is not None else (0, 0, 0)

    def process(self, image: np.ndarray, frame_index: int, t_ms: float) -> RawObservation:
        h, w = image.shape[:2]
        layout = self.layout_for((w, h))
        misses_before = self._cache.misses if self._cache is not None else 0
        read, agreement = read_ladder(layout.cells(image), self.templates, self.tick, self._cache)
        self.frames_total += 1
        if self._cache is not None and self._cache.misses == misses_before:
            self.frames_unchanged += 1
        rows = tuple(RawRow(price=r.price, size=r.size,
                            tint=Side(r.tint) if r.tint else None,
                            confidence=min(1.0, max(0.0, r.confidence)))
                     for r in read)
        return RawObservation(frame_index=frame_index, t_ms=t_ms, rows=rows,
                              price_agreement=agreement)