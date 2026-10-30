from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import cv2
import numpy as np

CELL_H = 32                      # высота ячейки после нормировки, px
WIN = 24                         # ширина окна вокруг центра символа, px
ALPHABET = "0123456789."


def soft_mask(cell: np.ndarray, min_contrast: float = 40.0) -> np.ndarray | None:
    """Ячейка (BGR) -> карта «текстовости» 0..1 высотой CELL_H, не зависящая от цвета,
    контраста и масштаба. None -- в ячейке нет текста."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dev = np.abs(gray - np.median(gray))                 # фон = медиана ячейки
    peak0 = float(np.percentile(dev, 99.5))
    # Кромка чужой заливки (граница строки чуть смещена): отличается от фона вся строка
    # пикселей, даже 10% самых близких к фону. В строке с текстом такие пиксели есть
    # всегда. Медиану строки здесь брать нельзя: на базовой линии текст занимает почти
    # половину ширины и стирался бы вместе с точкой.
    dev[np.percentile(dev, 10, axis=1) > max(8.0, 0.1 * peak0)] = 0.0
    peak = float(np.percentile(dev, 99.5))
    if peak < min_contrast:
        return None
    m = np.clip(dev / peak, 0.0, 1.0)
    return cv2.resize(m, (max(1, round(m.shape[1] * CELL_H / m.shape[0])), CELL_H),
                      interpolation=cv2.INTER_LINEAR)


def segment(mask: np.ndarray, lo: float = 0.25, hi: float = 0.5, min_width: int = 2
            ) -> list[tuple[int, int]]:
    """Символы как полосы столбцов с текстом: [(x0, x1), ...]. Границы по порогу lo,
    но полоса принимается, только если где-то в ней текст достигает hi (отсекает
    шум сжатия и бледные артефакты рядом с символами)."""
    colmax = mask.max(axis=0)
    d = np.diff(np.concatenate([[0], (colmax > lo).astype(np.int8), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
            if b - a >= min_width and colmax[a:b].max() >= hi]


def glyph_vector(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Окно фиксированной ширины вокруг центроида символа (с субпиксельным сдвигом).
    Так положение символа относительно пиксельной сетки не влияет на вектор."""
    lo, hi = max(0, x0 - 1), min(mask.shape[1], x1 + 1)
    col = mask[:, lo:hi].sum(axis=0)
    cx = lo + float((col * (np.arange(hi - lo) + 0.5)).sum() / max(col.sum(), 1e-6))
    xs = cx - WIN / 2 + np.arange(WIN)
    i0 = np.floor(xs).astype(int)
    f = (xs - i0).astype(np.float32)
    own = np.zeros_like(mask)
    own[:, lo:hi] = mask[:, lo:hi]                      # соседние символы в окно не пускаем
    pad = np.pad(own, ((0, 0), (WIN, WIN)))
    return (pad[:, i0 + WIN] * (1 - f) + pad[:, i0 + WIN + 1] * f).ravel()


@dataclass
class TemplateSet:
    labels: list[str]
    vectors: np.ndarray        # (n, D)
    width_max: float           # макс. ширина цифры в обучении, px
    max_dist: float            # дальше этого расстояния символ считается нечитаемым

    def classify(self, v: np.ndarray) -> tuple[str, float]:
        d = ((self.vectors - v) ** 2).mean(axis=1)
        k = int(np.argmin(d))
        return self.labels[k], float(d[k])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, labels=np.array(self.labels), vectors=self.vectors,
                 width_max=self.width_max, max_dist=self.max_dist)

    @classmethod
    def load(cls, path: str | Path) -> TemplateSet:
        z = np.load(path)
        return cls([str(s) for s in z["labels"]], z["vectors"],
                   float(z["width_max"]), float(z["max_dist"]))


def train_templates(pairs: Iterable[tuple[np.ndarray | None, str]]) -> tuple[TemplateSet, dict]:
    """pairs: (soft_mask ячейки цены, ожидаемый текст, например '2 330.77').
    Ячейка берётся, только если число найденных символов равно ожидаемому."""
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    widths: list[int] = []
    used = skipped = 0
    for mask, text in pairs:
        syms = [c for c in text if c != " "]
        runs = segment(mask) if mask is not None else []
        if len(runs) != len(syms):
            skipped += 1
            continue
        used += 1
        for (a, b), c in zip(runs, syms):
            samples[c].append(glyph_vector(mask, a, b))
            if c.isdigit():
                widths.append(b - a)
    labels = sorted(samples)
    templates = np.stack([np.mean(samples[c], axis=0) for c in labels])
    within = [float(((v - templates[i]) ** 2).mean()) for i, c in enumerate(labels) for v in samples[c]]
    p99 = float(np.percentile(within, 99))
    pair_d = [float(((templates[i] - templates[j]) ** 2).mean())
              for i in range(len(labels)) for j in range(i + 1, len(labels))]
    d_min = min(pair_d) if pair_d else 1.0
    ts = TemplateSet(labels, templates, float(max(widths)), min(max(3 * p99, 0.01), 0.8 * d_min))
    info = {"cells_used": used, "cells_skipped": skipped,
            "missing": [c for c in ALPHABET if c not in samples],
            "samples": {c: len(samples[c]) for c in labels},
            "within_p99": p99, "min_between": d_min,
            "separation": d_min / p99 if p99 > 0 else float("inf")}
    return ts, info


def read_text(mask: np.ndarray, ts: TemplateSet) -> tuple[str | None, float]:
    """(текст, уверенность 0..1). None, если хоть один символ не опознан или слипся."""
    runs = segment(mask)
    if not runs:
        return None, 0.0
    out, conf = [], 1.0
    for a, b in runs:
        if b - a > 1.5 * ts.width_max:                 # слипшиеся символы
            return None, 0.0
        label, dist = ts.classify(glyph_vector(mask, a, b))
        if dist > ts.max_dist:
            return None, 0.0
        out.append(label)
        conf = min(conf, 1.0 - dist / ts.max_dist)
    return "".join(out), conf


def parse_decimal(text: str | None, decimals: int | None = None) -> Decimal | None:
    if text is None:
        return None
    pattern = r"\d+\.\d{%d}" % decimals if decimals is not None else r"\d+(\.\d+)?"
    if not re.fullmatch(pattern, text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def expected_texts(top: Decimal, tick: Decimal, rows: int) -> list[str]:
    """Что должно быть написано в ячейках цены, если верхняя строка равна top."""
    dec = max(0, -tick.as_tuple().exponent)
    return [f"{top - i * tick:,.{dec}f}".replace(",", " ") for i in range(rows)]


def row_tint(cell: np.ndarray, delta: int = 12) -> str | None:
    """'ask' (красноватый фон), 'bid' (зеленоватый) или None (серый, пустой уровень)."""
    b, g, r = (float(np.median(cell[..., k])) for k in range(3))
    if r - g >= delta:
        return "ask"
    if g - r >= delta:
        return "bid"
    return None


@dataclass
class RowRead:
    price: Decimal | None
    size: Decimal | None        # Decimal(0): ячейка пуста; None: текст есть, но не прочитан
    tint: str | None
    confidence: float


def reconcile_prices(prices: list[Decimal | None], tick: Decimal, min_agreement: float = 0.6
                     ) -> tuple[list[Decimal | None], float]:
    """Цены лестницы идут строго с шагом tick, поэтому прочитанные значения голосуют
    за цену верхней строки. Если согласны >= min_agreement строк, все цены берутся из
    прогрессии (ошибки OCR исправляются, нечитаемые заполняются). Иначе кадр
    сомнительный (прокрутка, смена окна), цены остаются как прочитаны."""
    votes: dict[Decimal, int] = defaultdict(int)
    for i, p in enumerate(prices):
        if p is not None:
            votes[p + i * tick] += 1
    if not votes:
        return prices, 0.0
    top, n = max(votes.items(), key=lambda kv: kv[1])
    agreement = n / len(prices)
    if agreement < min_agreement:
        return prices, agreement
    return [top - i * tick for i in range(len(prices))], agreement


def read_ladder(cells: Iterator[tuple[np.ndarray, np.ndarray]], ts: TemplateSet, tick: Decimal
                ) -> tuple[list[RowRead], float]:
    dec = max(0, -tick.as_tuple().exponent)
    prices: list[Decimal | None] = []
    partial: list[tuple[Decimal | None, str | None, float]] = []
    for size_cell, price_cell in cells:
        pm = soft_mask(price_cell)
        ptxt, pconf = read_text(pm, ts) if pm is not None else (None, 0.0)
        prices.append(parse_decimal(ptxt, dec))
        sm = soft_mask(size_cell)
        if sm is None:
            size, sconf = Decimal(0), 1.0
        else:
            stxt, sconf = read_text(sm, ts)
            size = parse_decimal(stxt)
            if size is None:
                sconf = 0.0
        partial.append((size, row_tint(size_cell), min(pconf, sconf)))
    fixed, agreement = reconcile_prices(prices, tick)
    rows = [RowRead(fixed[i], s, t, c if prices[i] is not None else 0.0)
            for i, (s, t, c) in enumerate(partial)]
    return rows, agreement