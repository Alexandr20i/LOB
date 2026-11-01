from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import cv2
import numpy as np

CELL_H = 32  # высота холста маски после нормировки, px
TEXT_H = 22  # высота текста (цифр) на холсте, px
MIN_TEXT_ROWS = 6  # ниже этого размаха (в строках холста) текста в ячейке нет
FULL_HEIGHT_FRAC = 0.85  # столбец, занятый на такую долю высоты, это линия или поле, не символ
WIN = 24  # ширина окна вокруг центра символа, px
ALPHABET = "0123456789."


def _pct(a: np.ndarray, q: float) -> float:
    """np.percentile(a, q) (линейная интерполяция) через np.partition: порядковые
    статистики находятся без полной сортировки, на мелких ячейках это в разы быстрее."""
    flat = a.ravel()
    pos = (flat.size - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, flat.size - 1)
    part = np.partition(flat, (lo, hi))
    return float(part[lo] + (part[hi] - part[lo]) * (pos - lo))


def _row_pct10(dev: np.ndarray) -> np.ndarray:
    """np.percentile(dev, 10, axis=1) через np.partition (то же значение, быстрее)."""
    n = dev.shape[1]
    pos = (n - 1) * 0.10
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    part = np.partition(dev, (lo, hi), axis=1)
    return part[:, lo] + (part[:, hi] - part[:, lo]) * (pos - lo)


def soft_mask(cell: np.ndarray, min_contrast: float = 40.0) -> np.ndarray | None:
    """Ячейка (BGR) -> карта «текстовости» 0..1 высотой CELL_H, не зависящая от цвета
    фона, контраста, кегля и сдвига строки на пиксель. None -- в ячейке нет текста.

    Яркость берётся как V = max(B, G, R), а не как серая шкала: цветная полоса объёма
    (оранжевая, на светлом фоне) по V почти не отличается от фона, а чёрный или светлый
    текст остаётся контрастным. Ограничение: текст чистого синего цвета на белом фоне по V
    не виден; для терминалов с цветным текстом нужен другой канал.
    Вертикальный размах текста растягивается до TEXT_H и ставится в центр холста: так
    сдвиг фазы строки (ошибка калибровки на 1 px) и небольшая разница кегля между
    колонками цены и объёма не меняют вектор символа."""
    v = cell.max(axis=2).astype(np.float32)
    dev = np.abs(v - np.median(v))  # фон = медиана ячейки
    peak0 = _pct(dev, 99.5)
    # Кромка чужой заливки (граница строки чуть смещена): отличается от фона вся строка
    # пикселей, даже 10% самых близких к фону. В строке с текстом такие пиксели есть
    # всегда. Медиану строки здесь брать нельзя: на базовой линии текст занимает почти
    # половину ширины и стирался бы вместе с точкой.
    bad = _row_pct10(dev) > max(8.0, 0.1 * peak0)
    dev[bad] = 0.0
    peak = _pct(dev, 99.5)
    if peak < min_contrast:
        return None
    m = np.clip(dev / peak, 0.0, 1.0)
    # Вертикальные линии, границы колонок и тёмные поля занимают почти всю высоту ячейки
    # (считая только строки, не отброшенные как чужая заливка); цифры около 60%.
    valid = max(int((~bad).sum()), 1)
    m[:, ((m > 0.5) & ~bad[:, None]).sum(axis=0) / valid > FULL_HEIGHT_FRAC] = 0.0
    m = cv2.resize(m, (max(1, round(m.shape[1] * CELL_H / m.shape[0])), CELL_H),
                   interpolation=cv2.INTER_LINEAR)
    sums = m.sum(axis=1)
    # Размах текста: строки с чётким ink (>0.5), объединённые в вертикальные фрагменты;
    # берётся самый «массивный». Хвосты соседних строк, заходящие на край ячейки при
    # сдвиге калибровки, отделены от текста пустыми строками и в размах не попадают.
    d = np.diff(np.concatenate([[0], (m.max(axis=1) > 0.5).astype(np.int8), [0]]))
    spans = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    if not spans:
        return None
    top, bot = max(spans, key=lambda ab: float(sums[ab[0] : ab[1]].sum()))
    rows = np.arange(top, bot)
    if len(rows) < MIN_TEXT_ROWS:  # слишком низкий текст: шум или одиночная точка
        return None
    crop = m[rows[0] : rows[-1] + 1]
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * TEXT_H / crop.shape[0])), TEXT_H),
                      interpolation=cv2.INTER_LINEAR)
    out = np.zeros((CELL_H, crop.shape[1]), np.float32)
    y0 = (CELL_H - TEXT_H) // 2
    out[y0 : y0 + TEXT_H] = crop
    return out


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
    Так положение символа относительно пиксельной сетки не влияет на вектор.
    Соседние символы в окно не попадают: за пределами [x0 - 1, x1 + 1) нули."""
    lo, hi = max(0, x0 - 1), min(mask.shape[1], x1 + 1)
    seg = mask[:, lo:hi]
    col = seg.sum(axis=0)
    cx = lo + float((col * (np.arange(hi - lo) + 0.5)).sum() / max(col.sum(), 1e-6))
    xs = cx - WIN / 2 + np.arange(WIN)
    i0 = np.floor(xs).astype(int)
    f = (xs - i0).astype(np.float32)
    w = hi - lo

    def take(idx: np.ndarray) -> np.ndarray:
        valid = (idx >= 0) & (idx < w)
        return seg[:, np.clip(idx, 0, w - 1)] * valid

    return (take(i0 - lo) * (1 - f) + take(i0 - lo + 1) * f).ravel()


@dataclass
class TemplateSet:
    labels: list[str]
    vectors: np.ndarray  # (n, D)
    width_max: float  # макс. ширина цифры в обучении, px
    max_dist: float  # дальше этого расстояния символ считается нечитаемым
    size: TemplateSet | None = None  # шаблоны колонки объёма (если шрифт отличается от цены)

    def classify(self, v: np.ndarray) -> tuple[str, float]:
        d = ((self.vectors - v) ** 2).mean(axis=1)
        k = int(np.argmin(d))
        return self.labels[k], float(d[k])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.size is not None:
            extra = {"size_labels": np.array(self.size.labels), "size_vectors": self.size.vectors,
                     "size_width_max": self.size.width_max, "size_max_dist": self.size.max_dist}
        np.savez(path, labels=np.array(self.labels), vectors=self.vectors,
                 width_max=self.width_max, max_dist=self.max_dist, **extra)

    @classmethod
    def load(cls, path: str | Path) -> TemplateSet:
        z = np.load(path)
        size = None
        if "size_labels" in z.files:
            size = cls([str(s) for s in z["size_labels"]], z["size_vectors"],
                       float(z["size_width_max"]), float(z["size_max_dist"]))
        return cls([str(s) for s in z["labels"]], z["vectors"],
                   float(z["width_max"]), float(z["max_dist"]), size)


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


def bootstrap_templates(masks: Iterable[np.ndarray | None], base: TemplateSet,
                        min_samples: int = 3, margin: float = 2.0) -> tuple[TemplateSet, dict]:
    """Шаблоны для колонки объёма. Значения объёмов заранее неизвестны, поэтому символ
    метится ближайшим классом базовых шаблонов (цена), но принимается только если он
    однозначен: расстояние до лучшего класса не больше 0.6 расстояния между ближайшими
    классами базы, а второй класс дальше лучшего минимум в margin раз. Ячейка берётся,
    только если однозначны все её символы. Класс с числом образцов меньше min_samples
    остаётся шаблоном базы. Так учитывается, что кегль или начертание объёма отличаются от
    цены, а посторонние знаки (K, M, запятые) отсеиваются."""
    d_min_base = min(float(((base.vectors[i] - base.vectors[j]) ** 2).mean())
                     for i in range(len(base.labels)) for j in range(i + 1, len(base.labels)))
    cap = 0.6 * d_min_base
    samples: dict[str, list[np.ndarray]] = defaultdict(list)
    widths: list[int] = []
    used = skipped = 0
    for mask in masks:
        if mask is None:
            continue
        runs = segment(mask)
        labelled: list[tuple[str, np.ndarray, int]] = []
        ok = bool(runs)
        for a, b in runs:
            if b - a > 1.5 * base.width_max:
                ok = False
                break
            v = glyph_vector(mask, a, b)
            d = ((base.vectors - v) ** 2).mean(axis=1)
            order = np.argsort(d)
            d1, d2 = float(d[order[0]]), float(d[order[1]])
            if d1 > cap or d2 < margin * d1:
                ok = False
                break
            labelled.append((base.labels[int(order[0])], v, b - a))
        if not ok:
            skipped += 1
            continue
        used += 1
        for lab, v, w in labelled:
            samples[lab].append(v)
            if lab.isdigit():
                widths.append(w)
    vectors, own = [], []
    for k, lab in enumerate(base.labels):
        if len(samples[lab]) >= min_samples:
            vectors.append(np.mean(samples[lab], axis=0))
            own.append(lab)
        else:
            vectors.append(base.vectors[k])
    templates = np.stack(vectors)
    within = [float(((v - templates[base.labels.index(c)]) ** 2).mean())
              for c in own for v in samples[c]]
    p99 = float(np.percentile(within, 99)) if within else 0.0
    d_min = min(float(((templates[i] - templates[j]) ** 2).mean())
                for i in range(len(templates)) for j in range(i + 1, len(templates)))
    ts = TemplateSet(list(base.labels), templates,
                     float(max(widths)) if widths else base.width_max,
                     min(max(3 * p99, 0.01), 0.8 * d_min))
    info = {"cells_used": used, "cells_skipped": skipped,
            "from_base": [c for c in base.labels if c not in own],
            "samples": {c: len(samples[c]) for c in base.labels},
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
        if b - a > 1.5 * ts.width_max:  # слипшиеся символы
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
    size: Decimal | None  # Decimal(0): ячейка пуста; None: текст есть, но не прочитан
    tint: str | None
    confidence: float


# (цена, размер, цвет, уверенность) одной строки до согласования цен по прогрессии
RowResult = tuple["Decimal | None", "Decimal | None", "str | None", float]


class RowCache:
    """Результаты разбора строк между соседними кадрами.

    Экран терминала при 30 кадр/с меняется в малой части строк, поэтому строка,
    ячейки которой совпали с уже разобранными, не разбирается заново. Совпадением
    считается не более max_bad значений пикселей, отличающихся больше чем на tol
    уровней яркости: шум сжатия даёт единицы таких пикселей, смена цифры даёт сотни
    (измерено на записи 2880x1800: шум до 6 пикселей > 30, смена цифры от 154 пикселей
    > 60). Допуск по максимальной разнице брать нельзя: шум достигает 40 уровней.
    Сравнение идёт с кадром, по которому строка была разобрана, а не с предыдущим,
    так что медленный дрейф не накапливается.

    Автопрокрутка: при сдвиге цены строка i выглядит как бывшая строка i + shift.
    Если на кадре меняется много строк, ищется один общий сдвиг (до max_shift строк
    в обе стороны) и результаты берутся у соседей. Корректность от угаданного сдвига
    не зависит: результат переиспользуется, только когда пиксели строки совпали с
    разобранной ранее, а сдвиг лишь ограничивает перебор."""

    SHIFT_MIN_MISSES = 8  # меньше изменённых строк: сдвиг не ищем
    PROBES = 8  # по скольким строкам проверяется гипотеза сдвига

    def __init__(self, tol: int = 30, max_bad: int = 30, max_shift: int = 8) -> None:
        self.tol = tol
        self.max_bad = max_bad
        self.max_shift = max_shift
        self.items: dict[int, tuple[np.ndarray, np.ndarray, RowResult]] = {}
        self.hits = 0
        self.misses = 0
        self.shift_hits = 0

    def _same(self, a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and (
            a.size == 0 or int(np.count_nonzero(cv2.absdiff(a, b) > self.tol)) <= self.max_bad)

    def matches(self, item: tuple[np.ndarray, np.ndarray, RowResult],
                size_cell: np.ndarray, price_cell: np.ndarray) -> bool:
        return self._same(item[0], size_cell) and self._same(item[1], price_cell)

    def find_shift(self, cells: list[tuple[np.ndarray, np.ndarray]], miss: list[int]) -> int:
        """Сдвиг (в строках), при котором новые строки совпадают со старыми; 0, если нет."""
        probes = miss[:: max(1, len(miss) // self.PROBES)][: self.PROBES]
        best, best_n = 0, 1  # нужно хотя бы два совпадения
        for k in range(1, self.max_shift + 1):
            for s in (k, -k):
                n = sum(1 for i in probes
                        if (it := self.items.get(i + s)) is not None and self.matches(it, *cells[i]))
                if n > best_n:
                    best, best_n = s, n
        return best

    def clear(self) -> None:
        self.items.clear()


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


def read_row(size_cell: np.ndarray, price_cell: np.ndarray, ts: TemplateSet, decimals: int
             ) -> RowResult:
    """Разбор одной строки лестницы: (цена, размер, цвет фона, уверенность)."""
    pm = soft_mask(price_cell)
    ptxt, pconf = read_text(pm, ts) if pm is not None else (None, 0.0)
    price = parse_decimal(ptxt, decimals)
    sm = soft_mask(size_cell)
    if sm is None:
        size, sconf = Decimal(0), 1.0
    else:
        stxt, sconf = read_text(sm, ts.size or ts)
        size = parse_decimal(stxt)
        if size is None:
            sconf = 0.0
    return price, size, row_tint(size_cell), min(pconf, sconf)


def read_ladder(cells: Iterator[tuple[np.ndarray, np.ndarray]], ts: TemplateSet, tick: Decimal,
                cache: RowCache | None = None) -> tuple[list[RowRead], float]:
    dec = max(0, -tick.as_tuple().exponent)
    cell_list = list(cells)
    n = len(cell_list)
    results: list[RowResult | None] = [None] * n
    keep: dict[int, tuple[np.ndarray, np.ndarray, RowResult]] = {}

    if cache is not None and cache.items:
        old = cache.items
        for i, (sc, pc) in enumerate(cell_list):  # 1) строка на том же месте не изменилась
            item = old.get(i)
            if item is not None and cache.matches(item, sc, pc):
                results[i], keep[i] = item[2], item
                cache.hits += 1
        miss = [i for i in range(n) if results[i] is None]
        if len(miss) >= cache.SHIFT_MIN_MISSES:  # 2) лестница прокрутилась
            shift = cache.find_shift(cell_list, miss)
            if shift:
                for i in miss:
                    item = old.get(i + shift)
                    if item is not None and cache.matches(item, *cell_list[i]):
                        results[i], keep[i] = item[2], item
                        cache.hits += 1
                        cache.shift_hits += 1

    for i, (sc, pc) in enumerate(cell_list):  # 3) остальное разбираем
        if results[i] is None:
            res = read_row(sc, pc, ts, dec)
            results[i] = res
            if cache is not None:
                keep[i] = (sc.copy(), pc.copy(), res)
                cache.misses += 1
    if cache is not None:
        cache.items = keep

    done: list[RowResult] = [r for r in results if r is not None]
    fixed, agreement = reconcile_prices([r[0] for r in done], tick)
    rows = [RowRead(fixed[i], size, tint, conf if done[i][0] is not None else 0.0)
            for i, (_, size, tint, conf) in enumerate(done)]
    return rows, agreement