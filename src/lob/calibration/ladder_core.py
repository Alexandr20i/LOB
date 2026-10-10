"""Геометрия «лестницы цен» по нескольким кадрам и грубой рамке пользователя.

Чистые функции на NumPy/OpenCV, без pydantic и без знания о терминалах.
Что делает: находит шаг строки (с субпиксельной точностью), фазу границ строк,
число строк и колонки (цена, размер). Цифры здесь не читаются: проверка идёт
по структуре (периодичность текста, колонка, где текст есть в каждой строке).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class LadderParams:
    min_pitch: float = 10.0        # допустимая высота строки, px
    max_pitch: float = 80.0
    ink_threshold: int = 70        # отклонение яркости от фона строки, считающееся «текстом»
    min_periodicity: float = 0.35  # ниже -- период строки не найден
    price_coverage: float = 0.85   # доля строк, где в колонке цены есть текст
    min_rows: int = 5


@dataclass
class LadderGeometry:
    x: int                      # абсолютные координаты кадра
    y: int
    w: int
    h: int
    rows: int
    pitch: float
    price_col: tuple[float, float]   # доли ширины рамки
    size_col: tuple[float, float]
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def quality(self) -> float:
        if self.warnings and self.rows == 0:
            return 0.0
        return float(min(self.metrics.get("periodicity", 0.0),
                         self.metrics.get("price_coverage", 0.0)))


def _ink_mask(bgr: np.ndarray, thr: int) -> np.ndarray:
    """Пиксели, резко отличающиеся от фона своей строки пикселей.
    Фон = медиана по горизонтали, поэтому работает для светлой и тёмной темы."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg = np.median(gray, axis=1, keepdims=True)
    return np.abs(gray - bg) > thr


def _autocorr(p: np.ndarray) -> np.ndarray:
    p = p - p.mean()
    n = len(p)
    f = np.fft.rfft(p, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / np.arange(n, 0, -1)          # несмещённая оценка
    return ac / ac[0] if ac[0] > 1e-12 else np.zeros(n)


def _peak(ac: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Локальный максимум в [lo, hi] с параболической интерполяцией."""
    a, b = max(1, int(np.floor(lo))), min(len(ac) - 2, int(np.ceil(hi)))
    if b <= a:
        return 0.0, 0.0
    k = a + int(np.argmax(ac[a : b + 1]))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return k + float(np.clip(off, -1, 1)), float(y1)


def estimate_pitch(profile: np.ndarray, params: LadderParams) -> tuple[float, float]:
    """(шаг строки в px, высота пика автокорреляции).

    Первый пик даёт грубую оценку (ошибка до долей пикселя). Дальше уточняем по
    k-му кратному пику: ошибка делится на k, а на 50 строках 0.3 px на строку
    уже дало бы 15 px накопленного сдвига."""
    ac = _autocorr(profile)
    n = len(ac)
    hi = min(params.max_pitch, n / 2 - 2)
    if hi <= params.min_pitch:
        return 0.0, 0.0
    # основной период: первый локальный максимум, не сильно уступающий глобальному
    cands = [k for k in range(int(params.min_pitch), int(hi) + 1)
             if ac[k] >= ac[k - 1] and ac[k] > ac[k + 1] and ac[k] > 0]
    if not cands:
        return 0.0, 0.0
    best = max(ac[k] for k in cands)
    k0 = next(k for k in cands if ac[k] >= 0.8 * best)
    pitch, height = _peak(ac, k0 - 1, k0 + 1)
    k = 2
    while k * pitch <= n / 2:
        lag, _ = _peak(ac, k * pitch - 1.0, k * pitch + 1.0)
        pitch = lag / k
        k += 1
    return pitch, height


def estimate_phase(profile: np.ndarray, pitch: float) -> float:
    """Смещение границы строки (px от верха рамки, в [0, pitch)).
    Граница ставится в центр минимума текста между строками."""
    bins = max(8, int(round(pitch)))
    idx = ((np.arange(len(profile)) % pitch) / pitch * bins).astype(int) % bins
    fold = np.bincount(idx, weights=profile, minlength=bins) / np.maximum(
        np.bincount(idx, minlength=bins), 1)
    ext = np.concatenate([fold[-2:], fold, fold[:2]])
    sm = np.convolve(ext, np.ones(5) / 5, mode="same")[2:-2]
    return (int(np.argmin(sm)) + 0.5) / bins * pitch


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Полуинтервалы [a, b) подряд идущих True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def calibrate_ladder(
    frames: Sequence[np.ndarray],
    roi: tuple[int, int, int, int] | None = None,
    params: LadderParams = LadderParams(),
) -> LadderGeometry:
    """frames: BGR-кадры одного размера (чем разнообразнее по содержимому, тем лучше).
    roi: грубая рамка пользователя (x, y, w, h); без неё берётся весь кадр."""
    if not frames:
        raise ValueError("нужен хотя бы один кадр")
    H, W = frames[0].shape[:2]
    x0, y0, w0, h0 = roi if roi else (0, 0, W, H)
    x0, y0 = max(0, x0), max(0, y0)
    w0, h0 = min(w0, W - x0), min(h0, H - y0)

    masks = [_ink_mask(f[y0 : y0 + h0, x0 : x0 + w0], params.ink_threshold) for f in frames]
    prof = np.mean([m.mean(axis=1) for m in masks], axis=0)

    geo = LadderGeometry(x0, y0, w0, h0, 0, 0.0, (0.0, 0.0), (0.0, 0.0))
    pitch, periodicity = estimate_pitch(prof, params)
    geo.metrics["periodicity"] = periodicity
    if pitch <= 0 or periodicity < params.min_periodicity:
        geo.warnings.append(
            f"период строки не найден (периодичность {periodicity:.2f}); "
            "обведите только стакан и проверьте, что в нём есть текст")
        return geo

    top = estimate_phase(prof, pitch)
    rows = int((h0 - top) / pitch + 0.05)
    if rows < params.min_rows:
        geo.warnings.append(f"найдено всего {rows} строк")
        return geo
    # построчные маски текста: кадры x строки x ширина (средняя часть строки, без границ)
    def per_row_masks(first: int, n_rows: int) -> np.ndarray:
        out = np.zeros((len(masks), n_rows, w0), np.uint8)
        for fi, m in enumerate(masks):
            for i in range(n_rows):
                a = int(round(top + (first + i + 0.2) * pitch))
                b = int(round(top + (first + i + 0.8) * pitch))
                out[fi, i] = m[a:b].any(axis=0)
        return out

    stack = per_row_masks(0, rows)
    has_ink = stack.any(axis=(0, 2))                    # есть ли в строке хоть что-то
    if not has_ink.any():
        geo.warnings.append("в рамке нет текста")
        return geo
    first, last = int(np.argmax(has_ink)), int(rows - 1 - np.argmax(has_ink[::-1]))
    rows = last - first + 1                              # пустые края рамки отбрасываем
    if rows < params.min_rows:
        geo.warnings.append(f"найдено всего {rows} строк")
        return geo
    stack = stack[:, first : last + 1]
    ys = y0 + int(round(top + first * pitch))
    hh = int(round(rows * pitch))
    geo.y, geo.h, geo.rows, geo.pitch = ys, hh, rows, pitch
    geo.metrics["pitch"] = pitch

    # колонки: для каждого x -- в какой доле строк там есть текст
    k = max(3, int(round(pitch)))                       # склеиваем символы цены (пробел, точка)
    kernel = np.ones((1, k), np.uint8)
    cov = np.zeros(w0)
    for fi in range(len(masks)):
        closed = cv2.morphologyEx(stack[fi], cv2.MORPH_CLOSE, kernel)
        cov += closed.mean(axis=0)
    cov /= len(masks)

    runs = [(a, b) for a, b in _runs(cov >= params.price_coverage) if b - a >= 2 * pitch]
    if not runs:
        geo.warnings.append("не найдена колонка, где текст есть почти в каждой строке (цена)")
        geo.metrics["price_coverage"] = float(cov.max())
        return geo
    pa, pb = runs[-1]                                   # правая из таких колонок
    geo.metrics["price_coverage"] = float(cov[pa:pb].mean())
    if geo.metrics["price_coverage"] < params.price_coverage:
        geo.warnings.append("в колонке цены текст есть не во всех строках")

    pad = max(1, int(round(0.15 * pitch)))
    left_ink = np.flatnonzero(cov[:pa] >= 0.02)
    sa = max(0, int(left_ink[0]) - pad) if len(left_ink) else 0
    gap = np.flatnonzero(cov[:pa] < 0.02)
    sb = (int(gap[-1]) + pa) // 2 if len(gap) and gap[-1] < pa - 1 else pa - 1
    sb = max(sb, sa + 1)
    pl = max(0, pa - pad)
    sb = max(min(sb, pl), sa + 1)
    geo.price_col = (pl / w0, min(w0, pb + pad) / w0)
    geo.size_col = (sa / w0, sb / w0)
    geo.metrics["size_coverage_max"] = float(cov[:pa].max()) if pa else 0.0
    return geo
