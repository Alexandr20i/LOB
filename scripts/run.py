#!/usr/bin/env python3
"""
Полный pipeline: калибровка → обучение глифов → стриминг стакана.

Workflow:
  1. Открывается кадр из видео
  2. Мышью выделяешь область стакана (цены + объемы вместе)
  3. Enter → калибровка layout
  4. Обучение/загрузка глифов
  5. Стриминг стакана в консоль

Примеры:
  uv run python scripts/run.py ~/Downloads/MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1
  uv run python scripts/run.py ~/Downloads/MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --show-video
  uv run python scripts/run.py ~/Downloads/MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --stride 5
  uv run python scripts/run.py ~/Downloads/MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --save lob_data.parquet
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from lob.layout import load_layout, save_layout, LadderLayout, Roi
from lob.recognition.glyphs import TemplateSet
from lob.recognition.ladder import LadderRecognizer
from lob.terminals import TERMINALS
from lob.video import OpenCVVideoSource, sample_frames


def pick_roi(image: np.ndarray, title: str = "Выдели область стакана, Enter = ok, c = отмена") -> Roi | None:
    """Ручной выбор ROI мышью через cv2.selectROI."""
    scale = min(1.0, 1000 / image.shape[0], 1800 / image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    
    # showCrosshair=False убирает крестик, если он мешает
    x, y, w, h = cv2.selectROI(title, shown, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    
    if w == 0 or h == 0:
        return None
    
    return Roi(
        x=round(x / scale),
        y=round(y / scale),
        w=round(w / scale),
        h=round(h / scale)
    )


def calibrate(video_path: Path, terminal_name: str, roi: Roi | None = None, frames_count: int = 8) -> LadderLayout:
    """
    Калибровка layout по видео.
    Если roi не передан — открывается окно для ручного выбора мышью.
    """
    print("\n" + "="*70)
    print("  ШАГ 1: КАЛИБРОВКА РАСКЛАДКИ")
    print("="*70)
    
    frames = sample_frames(video_path, frames_count)
    if not frames:
        raise RuntimeError("Не удалось прочитать кадры из видео")
    
    h, w = frames[0].shape[:2]
    print(f"📹 Видео: {w}x{h}, кадров для калибровки: {len(frames)}")
    
    # Ручной выбор ROI если не передан
    if roi is None:
        print("👉 Выдели мышью область стакана (цены + объемы) и нажми Enter")
        print("   💡 Крестик в центре — это нормально, просто помошник для точности")
        roi = pick_roi(frames[0])
        if roi is None:
            raise RuntimeError("ROI не выбран — калибровка отменена")
    
    print(f"📐 ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}")
    
    # Определяем количество строк по высоте
    # Стандартная высота строки ~30px, но можно настроить
    row_height_estimate = 30  # пикселей на строку
    rows = max(10, min(100, roi.h // row_height_estimate))
    print(f"📊 Предполагаемое количество строк: {rows} (высота строки ~{roi.h/rows:.1f}px)")
    
    # Создаем layout с дефолтными пропорциями колонок
    layout = LadderLayout(
        terminal=terminal_name,
        reference_size=(w, h),
        roi=roi,
        rows=rows,
        price_col=(0.72, 1.0),
        size_col=(0.0, 0.72)
    )
    
    # Сохраняем
    layout_path = Path(f"configs/layouts/{terminal_name}.yaml")
    save_layout(layout, layout_path)
    print(f"✅ Layout сохранен: {layout_path}")
    
    # Показываем overlay для проверки
    overlay_path = layout_path.with_name(layout_path.stem + "_overlay.png")
    cv2.imwrite(str(overlay_path), layout.draw(frames[0]))
    print(f"🖼️  Overlay сохранен: {overlay_path}")
    print(f"   ⚠️  Проверьте глазами: красные линии = size, зеленые = price")
    
    return layout


def load_or_train_templates(video_path: Path, terminal_name: str, layout: LadderLayout, tick: Decimal) -> TemplateSet:
    """Загружает существующие templates или обучает новые."""
    print("\n" + "="*70)
    print("  ШАГ 2: ГЛИФЫ (РАСПОЗНАВАНИЕ СИМВОЛОВ)")
    print("="*70)
    
    templates_path = Path(f"configs/glyphs/{terminal_name}.npz")
    
    if templates_path.exists():
        print(f"📦 Загрузка существующих templates: {templates_path}")
        return TemplateSet.load(templates_path)
    
    print(f"⚠️  Templates не найдены: {templates_path}")
    print(f"   Сначала обучите их командой:")
    print(f"   uv run python scripts/train_glyphs.py {video_path} --terminal {terminal_name}")
    raise RuntimeError(f"Templates не найдены для терминала '{terminal_name}'")


def find_spread_row(tints: list[str | None]) -> int | None:
    """Находит индекс строки спреда по переходу ask → bid."""
    last_ask = None
    first_bid = None
    
    for i, tint in enumerate(tints):
        if tint == "ask":
            last_ask = i
        elif tint == "bid" and first_bid is None:
            first_bid = i
    
    if last_ask is not None and first_bid is not None:
        return (last_ask + first_bid) // 2
    return None


def format_lob_table(
    rows: list[tuple[Decimal | None, Decimal | None, str | None, float]],
    frame_index: int,
    t_ms: float,
    spread_idx: int | None = None
) -> str:
    """Форматирует стакан с разделением на ASKS/BIDS по спред индексу."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  Кадр: {frame_index:5d} | Время: {t_ms/1000:8.2f}s")
    lines.append(f"{'='*70}")
    
    asks = []
    bids = []
    spread_found = False
    
    for idx, (price, size, tint, conf) in enumerate(rows):
        if price is None:
            continue
        
        # Вставляем разделитель SPREAD в нужный момент
        if spread_idx is not None and idx == spread_idx and not spread_found:
            lines.append(f"  {'─'*60}")
            lines.append(f"  {'SPREAD (строка ' + str(spread_idx) + ')':^60s}")
            lines.append(f"  {'─'*60}")
            spread_found = True
        
        # Определяем сторону: по цвету или по позиции относительно спреда
        effective_tint = tint
        if effective_tint is None and spread_idx is not None:
            effective_tint = "ask" if idx < spread_idx else "bid"
        
        if effective_tint == "ask":
            asks.append((price, size, conf, idx))
        elif effective_tint == "bid":
            bids.append((price, size, conf, idx))
    
    # Сортируем: asks от большей цены к меньшей (сверху вниз на экране)
    asks.sort(key=lambda x: x[0], reverse=True)
    bids.sort(key=lambda x: x[0], reverse=True)
    
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
    
    if not spread_found:
        lines.append(f"  {'─'*60}")
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


def stream_lob(
    video_path: Path,
    terminal_name: str,
    layout: LadderLayout,
    templates: TemplateSet,
    tick: Decimal,
    stride: int = 1,
    save_path: Path | None = None,
    show_video: bool = False,
) -> int:
    """Стриминг стакана из видео."""
    print("\n" + "="*70)
    print("  ШАГ 3: СТРИМИНГ СТАКАНА")
    print("="*70)
    
    recognizer = LadderRecognizer(layout, templates, tick)
    
    print(f"🎬 Видео: {video_path}")
    print(f"   Tick: {tick}")
    print(f"   Stride: {stride}")
    print(f"\n▶️  Начало обработки...\n")
    
    video_source = OpenCVVideoSource(video_path, stride=stride)
    
    prev_rows = None
    records = []
    frame_count = 0
    changes_count = 0
    recognition_stats = {'total': 0, 'with_price': 0, 'with_size': 0, 'with_tint': 0}
    
    try:
        for frame in video_source:
            frame_count += 1
            
            obs = recognizer.process(frame.image, frame.index, frame.t_ms)
            
            current_rows = [
                (r.price, r.size, r.tint.value if r.tint else None, r.confidence)
                for r in obs.rows
            ]
            
            # Статистика распознавания
            for price, size, tint, conf in current_rows:
                recognition_stats['total'] += 1
                if price is not None:
                    recognition_stats['with_price'] += 1
                if size is not None and size > 0:
                    recognition_stats['with_size'] += 1
                if tint is not None:
                    recognition_stats['with_tint'] += 1
            
            # Находим спред
            tints = [r.tint.value if r.tint else None for r in obs.rows]
            spread_idx = find_spread_row(tints)
            
            # Выводим только при изменении
            if prev_rows is None or current_rows != prev_rows:
                changes_count += 1
                output = format_lob_table(current_rows, frame.index, frame.t_ms, spread_idx)
                print(output)
                
                if save_path:
                    records.append({
                        'frame_index': frame.index,
                        't_ms': frame.t_ms,
                        'spread_row': spread_idx,
                        'rows': [
                            {
                                'price': str(p) if p is not None else None,
                                'size': str(s) if s is not None else None,
                                'tint': t,
                                'confidence': c
                            }
                            for p, s, t, c in current_rows
                        ]
                    })
                
                prev_rows = current_rows
            
            # Показываем видео с разметкой
            if show_video:
                img = layout.draw(frame.image)
                cv2.imshow(f"LOB Stream - {terminal_name}", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Остановлено пользователем (нажата q)")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️  Остановлено пользователем (Ctrl+C)")
    
    finally:
        if show_video:
            cv2.destroyAllWindows()
    
    print(f"\n✅ Обработка завершена")
    print(f"   Кадров обработано: {frame_count}")
    print(f"   Изменений стакана: {changes_count}")
    
    # Статистика распознавания
    if recognition_stats['total'] > 0:
        print(f"\n📊 Статистика распознавания:")
        print(f"   Всего строк: {recognition_stats['total']}")
        print(f"   С ценой: {recognition_stats['with_price']} ({100*recognition_stats['with_price']/recognition_stats['total']:.1f}%)")
        print(f"   С объемом: {recognition_stats['with_size']} ({100*recognition_stats['with_size']/recognition_stats['total']:.1f}%)")
        print(f"   С цветом: {recognition_stats['with_tint']} ({100*recognition_stats['with_tint']/recognition_stats['total']:.1f}%)")
        
        if recognition_stats['with_price'] == 0:
            print(f"\n⚠️  ЦЕНЫ НЕ РАСПОЗНАНЫ! Возможные причины:")
            print(f"   1. Неправильный tick (должен быть {tick}?)")
            print(f"   2. Неправильное количество строк (rows={layout.rows})")
            print(f"   3. Templates не подходят для этого шрифта")
            print(f"   4. ROI захватывает не ту область")
    
    if save_path and records:
        print(f"\n💾 Сохранение в {save_path}...")
        df = pl.DataFrame(records)
        df.write_parquet(save_path)
        print(f"   Сохранено {len(records)} записей")
    
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Путь к видеофайлу")
    ap.add_argument("--terminal", default="metascalp", choices=sorted(TERMINALS),
                    help="Тип терминала (default: metascalp)")
    ap.add_argument("--tick", required=True, type=Decimal,
                    help="Шаг цены (например, 0.1 для ETH, 1 для BTC)")
    ap.add_argument("--step", choices=["calibrate", "train", "stream", "all"], default="all",
                    help="Какой шаг выполнить (default: all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Обрабатывать каждый N-й кадр (default: 1)")
    ap.add_argument("--save", help="Сохранить данные в Parquet файл")
    ap.add_argument("--show-video", action="store_true",
                    help="Показывать видео с наложенной разметкой")
    ap.add_argument("--roi", help="x,y,w,h вместо ручного выбора мышью")
    args = ap.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Видеофайл не найден: {video_path}")
        return 1
    
    layout = None
    templates = None
    
    # === ШАГ 1: Калибровка ===
    if args.step in ["calibrate", "all"]:
        try:
            roi = None
            if args.roi:
                x, y, rw, rh = (int(v) for v in args.roi.split(","))
                roi = Roi(x=x, y=y, w=rw, h=rh)
            
            layout = calibrate(video_path, args.terminal, roi=roi)
        except Exception as e:
            print(f"❌ Ошибка калибровки: {e}")
            return 1
    
    # Загружаем layout если не калибровали
    if layout is None:
        layout_path = Path(f"configs/layouts/{args.terminal}.yaml")
        if not layout_path.exists():
            print(f"❌ Layout не найден: {layout_path}")
            print(f"   Сначала выполните калибровку: --step calibrate")
            return 1
        layout = load_layout(layout_path)
    
    # === ШАГ 2: Глифы ===
    if args.step in ["train", "all"]:
        try:
            templates = load_or_train_templates(video_path, args.terminal, layout, args.tick)
        except Exception as e:
            print(f"❌ Ошибка загрузки глифов: {e}")
            return 1
    
    # Загружаем templates если не на этом шаге
    if templates is None:
        templates_path = Path(f"configs/glyphs/{args.terminal}.npz")
        if not templates_path.exists():
            print(f"❌ Templates не найдены: {templates_path}")
            print(f"   Сначала выполните: uv run python scripts/train_glyphs.py {video_path} --terminal {args.terminal}")
            return 1
        templates = TemplateSet.load(templates_path)
    
    # === ШАГ 3: Стриминг ===
    if args.step in ["stream", "all"]:
        save_path = Path(args.save) if args.save else None
        return stream_lob(
            video_path, args.terminal, layout, templates, args.tick,
            stride=args.stride, save_path=save_path, show_video=args.show_video,
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())