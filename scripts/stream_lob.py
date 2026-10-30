#!/usr/bin/env python3
"""
Стриминг стакана из видео с автоматическим определением спреда.

Особенности:
- Автоматически находит спред по переходу цвета (ask → bid)
- Показывает стакан в реальном времени
- Сохраняет данные в Parquet для анализа

Примеры:
  uv run python scripts/stream_lob.py ~/Downloads/MetaScalp.mp4 --terminal metascalp --tick 1
  uv run python scripts/stream_lob.py ~/Downloads/MetaScalp.mp4 --terminal metascalp --tick 0.01 --stride 5
  uv run python scripts/stream_lob.py ~/Downloads/MetaScalp.mp4 --terminal metascalp --tick 1 --save lob_data.parquet --show-video
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from lob.layout import load_layout
from lob.recognition.glyphs import TemplateSet
from lob.recognition.ladder import LadderRecognizer
from lob.terminals import TERMINALS
from lob.video import OpenCVVideoSource


def find_spread_by_color(rows_data: list[tuple[int, str | None, np.ndarray]]) -> int | None:
    """
    Находит строку спреда по переходу от ask (красный) к bid (зеленый).
    
    Args:
        rows_data: [(row_index, tint, cell_image), ...]
    
    Returns:
        Индекс строки спреда или None
    """
    last_ask_idx = None
    first_bid_idx = None
    
    for idx, tint, cell in rows_data:
        if tint == "ask":
            last_ask_idx = idx
        elif tint == "bid" and first_bid_idx is None:
            first_bid_idx = idx
    
    if last_ask_idx is not None and first_bid_idx is not None:
        # Спред между последним ask и первым bid
        return (last_ask_idx + first_bid_idx) // 2
    
    return None


def detect_tint_from_cell(cell: np.ndarray, delta: int = 15) -> str | None:
    """
    Определяет цвет ячейки (ask/bid/neutral).
    
    Args:
        cell: BGR изображение ячейки
        delta: порог разницы между каналами
    
    Returns:
        'ask', 'bid', или None
    """
    if cell.size == 0:
        return None
    
    b, g, r = (float(np.median(cell[..., k])) for k in range(3))
    
    # Красный (ask): R значительно больше G и B
    if r - g >= delta and r - b >= delta:
        return "ask"
    
    # Зеленый (bid): G значительно больше R и B
    if g - r >= delta and g - b >= delta:
        return "bid"
    
    return None


def format_lob_table(
    rows: list[tuple[Decimal | None, Decimal | None, str | None, float, np.ndarray]],
    frame_index: int,
    t_ms: float
) -> str:
    """
    Форматирует стакан с автоматическим определением спреда.
    
    Args:
        rows: [(price, size, tint, confidence, cell_image), ...]
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Кадр: {frame_index:5d} | Время: {t_ms/1000:8.2f}s")
    lines.append(f"{'='*70}")
    
    # Находим спред
    rows_with_cells = [(i, tint, cell) for i, (_, _, tint, _, cell) in enumerate(rows)]
    spread_idx = find_spread_by_color(rows_with_cells)
    
    # Разделяем на аски и биды
    asks = []
    bids = []
    
    for idx, (price, size, tint, conf, cell) in enumerate(rows):
        if price is None:
            continue
        
        # Если tint не определен, используем позицию относительно спреда
        effective_tint = tint
        if effective_tint is None and spread_idx is not None:
            effective_tint = "ask" if idx < spread_idx else "bid"
        
        if effective_tint == "ask":
            asks.append((price, size, conf, idx))
        elif effective_tint == "bid":
            bids.append((price, size, conf, idx))
    
    # Сортируем
    asks.sort(key=lambda x: x[0], reverse=True)  # От большей к меньшей (сверху вниз)
    bids.sort(key=lambda x: x[0], reverse=True)  # От большей к меньшей (сверху вниз)
    
    # ASKS
    lines.append("  ASKS (продажи):")
    lines.append(f"  {'Цена':>12s} | {'Объем':>12s} | {'Увер.':>6s} | {'#':>3s} | Визуализация")
    lines.append(f"  {'-'*12}-+-{'-'*12}-+-{'-'*6}-+-{'-'*3}-+-{'-'*20}")
    
    if asks:
        max_vol = max((float(s) if s else 0 for s, _, _, _ in asks), default=1)
        for price, size, conf, idx in asks[:12]:
            vol = float(size) if size else 0
            bar_len = int(vol / max(max_vol, 1) * 20)
            bar = '█' * bar_len
            size_str = f"{size:>12}" if size is not None else "         N/A"
            lines.append(f"  {price:>12} | {size_str} | {conf:>5.1%} | {idx:>3d} | {bar}")
    else:
        lines.append("  (не распознано)")
    
    # SPREAD
    lines.append(f"  {'─'*60}")
    if spread_idx is not None:
        lines.append(f"  {'SPREAD (строка ' + str(spread_idx) + ')':^60s}")
    else:
        lines.append(f"  {'SPREAD':^60s}")
    lines.append(f"  {'─'*60}")
    
    # BIDS
    lines.append("  BIDS (покупки):")
    lines.append(f"  {'Цена':>12s} | {'Объем':>12s} | {'Увер.':>6s} | {'#':>3s} | Визуализация")
    lines.append(f"  {'-'*12}-+-{'-'*12}-+-{'-'*6}-+-{'-'*3}-+-{'-'*20}")
    
    if bids:
        max_vol = max((float(s) if s else 0 for s, _, _, _ in bids), default=1)
        for price, size, conf, idx in bids[:12]:
            vol = float(size) if size else 0
            bar_len = int(vol / max(max_vol, 1) * 20)
            bar = '█' * bar_len
            size_str = f"{size:>12}" if size is not None else "         N/A"
            lines.append(f"  {price:>12} | {size_str} | {conf:>5.1%} | {idx:>3d} | {bar}")
    else:
        lines.append("  (не распознано)")
    
    lines.append(f"{'='*70}\n")
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Путь к видеофайлу")
    ap.add_argument("--terminal", default="metascalp", choices=sorted(TERMINALS),
                    help="Тип терминала (default: metascalp)")
    ap.add_argument("--tick", required=True, type=Decimal,
                    help="Шаг цены (например, 1 для BTC, 0.01 для ETH)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Обрабатывать каждый N-й кадр (default: 1)")
    ap.add_argument("--save", help="Сохранить данные в Parquet файл")
    ap.add_argument("--show-video", action="store_true",
                    help="Показывать видео с наложенной разметкой")
    ap.add_argument("--debug", action="store_true",
                    help="Показывать отладочную информацию")
    args = ap.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Видеофайл не найден: {video_path}")
        return 1
    
    # Загружаем layout
    layout_path = Path(f"configs/layouts/{args.terminal}.yaml")
    if not layout_path.exists():
        print(f"❌ Layout не найден: {layout_path}")
        print(f"   Сначала выполните калибровку: uv run python scripts/calibrate.py {video_path} --terminal {args.terminal}")
        return 1
    
    print(f"🔧 Загрузка layout: {layout_path}")
    layout = load_layout(layout_path)
    
    # Загружаем templates
    templates_path = Path(f"configs/glyphs/{args.terminal}.npz")
    if not templates_path.exists():
        print(f"❌ Templates не найдены: {templates_path}")
        print(f"   Сначала обучите templates: uv run python scripts/train_glyphs.py {video_path} --terminal {args.terminal}")
        return 1
    
    print(f"🔍 Загрузка templates: {templates_path}")
    templates = TemplateSet.load(templates_path)
    
    # Создаем распознаватель
    recognizer = LadderRecognizer(layout, templates, args.tick)
    
    print(f"\n🎬 Видео: {video_path}")
    print(f"   Tick: {args.tick}")
    print(f"   Stride: {args.stride}")
    print(f"\n▶️  Начало обработки...\n")
    
    video_source = OpenCVVideoSource(video_path, stride=args.stride)
    
    prev_rows = None
    records = []
    frame_count = 0
    changes_count = 0
    
    try:
        for frame in video_source:
            frame_count += 1
            
            # Распознаем стакан
            obs = recognizer.process(frame.image, frame.index, frame.t_ms)
            
            # Преобразуем в список с ячейками для определения цвета
            current_rows = []
            for i, r in enumerate(obs.rows):
                # Берем ячейку размера для определения цвета
                h, w = frame.image.shape[:2]
                lay = layout.scaled_to((w, h)) if (w, h) != tuple(layout.reference_size) else layout
                
                cell_img = None
                try:
                    cells = list(lay.cells(frame.image))
                    if i < len(cells):
                        size_cell, _ = cells[i]
                        cell_img = size_cell
                except:
                    pass
                
                current_rows.append((r.price, r.size, r.tint.value if r.tint else None, r.confidence, cell_img))
            
            # Проверяем изменения
            rows_for_compare = [(p, s, t, c) for p, s, t, c, _ in current_rows]
            
            if prev_rows is None or rows_for_compare != prev_rows:
                changes_count += 1
                
                # Выводим
                output = format_lob_table(current_rows, frame.index, frame.t_ms)
                print(output)
                
                # Отладка
                if args.debug:
                    print(f"  [DEBUG] Распознано строк: {len(current_rows)}")
                    print(f"  [DEBUG] Спред найден: {find_spread_by_color([(i, t, c) for i, (_, _, t, _, c) in enumerate(current_rows)])}")
                
                # Сохраняем
                if args.save:
                    records.append({
                        'frame_index': frame.index,
                        't_ms': frame.t_ms,
                        'rows': [
                            {
                                'price': str(p) if p is not None else None,
                                'size': str(s) if s is not None else None,
                                'tint': t,
                                'confidence': c
                            }
                            for p, s, t, c, _ in current_rows
                        ]
                    })
                
                prev_rows = rows_for_compare
            
            # Показываем видео
            if args.show_video:
                img = layout.draw(frame.image)
                cv2.imshow(f"LOB Stream - {args.terminal}", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Остановлено пользователем")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️  Остановлено пользователем")
    
    finally:
        if args.show_video:
            cv2.destroyAllWindows()
    
    print(f"\n✅ Обработка завершена")
    print(f"   Кадров обработано: {frame_count}")
    print(f"   Изменений стакана: {changes_count}")
    
    if args.save and records:
        print(f"\n💾 Сохранение в {args.save}...")
        df = pl.DataFrame(records)
        df.write_parquet(args.save)
        print(f"   Сохранено {len(records)} записей")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())