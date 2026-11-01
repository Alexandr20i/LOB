"""Обучение шаблонов цифр по одному кадру записи.

uv run python scripts/train_glyphs.py ~/Downloads/rec.mp4 --terminal metascalp --tick 0.01

Скрипт сохранит configs/glyphs/train_frame.png (область стакана) и спросит цену верхней
строки на нём (или возьмёт --top-price). Шаблоны: configs/glyphs/<terminal>.npz.
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import cv2

from _common import glyphs_path, layout_path, video_arg
from lob.pipeline import load_ladder_layout
from lob.recognition.training import (MIN_SEPARATION, TrainingError, fit_layout, fit_size_templates,
                                      fit_templates, verify_templates)
from lob.terminals import TERMINALS
from lob.video import sample_frames


def train(video: Path, terminal: str, tick: Decimal, layout_file: Path, out: Path,
          top_price: Decimal | None = None, n_frames: int = 8, force: bool = False) -> bool:
    layout = load_ladder_layout(layout_file)
    frames = sample_frames(video, n_frames)
    if not frames:
        print("Не удалось прочитать кадры из видео")
        return False

    train_png = out.with_name("train_frame.png")
    train_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(train_png), fit_layout(layout, frames[0]).roi.crop(frames[0]))

    if top_price is None:
        print(f"Откройте {train_png} и посмотрите цену в самой верхней строке.")
        raw = input("Цена верхней строки (например 2330.77): ")
        try:
            top_price = Decimal(raw.replace(" ", "").replace(",", "."))
        except InvalidOperation:
            print(f"Не число: {raw!r}")
            return False

    try:
        ts, info = fit_templates(frames[0], layout, tick, top_price, force=force)
    except TrainingError as e:
        print("ОБУЧЕНИЕ НЕ УДАЛОСЬ:", e)
        return False
    print(f"Ячеек цены использовано {info['cells_used']} из {layout.rows}")
    print("Образцов по символам:", info["samples"])
    print(f"Разделимость классов: {info['separation']:.1f} (нужно >= {MIN_SEPARATION})")
    ts, sinfo = fit_size_templates(frames, layout, ts)
    if ts.size is not None:
        print(f"Колонка объёма: своих шаблонов по {sinfo['cells_used']} ячейкам "
              f"(пропущено неоднозначных {sinfo['cells_skipped']}); "
              f"классы из шаблонов цены (мало образцов): {sinfo['from_base'] or 'нет'}")
    else:
        print(f"Колонка объёма: однозначных ячеек всего {sinfo['cells_used']}, объём читается "
              "шаблонами цены. Если size_unread ниже велик, проверьте формат объёма (K/M).")
    ts.save(out)
    print("Шаблоны сохранены:", out)

    print("\nПроверка на остальных кадрах (согласие цен с прогрессией, строк с нечитаемым размером):")
    for c in verify_templates(frames, layout, ts, tick):
        print(f"  {c.frame}: agreement={c.agreement:.2f} size_unread={c.size_unread}/{c.rows} "
              f"top={c.top_price}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=video_arg)
    ap.add_argument("--terminal", default="metascalp", choices=sorted(TERMINALS))
    ap.add_argument("--tick", required=True, type=Decimal, help="шаг цены, например 0.01")
    ap.add_argument("--layout", type=Path, help="по умолчанию configs/layouts/<terminal>.yaml")
    ap.add_argument("--out", type=Path, help="по умолчанию configs/glyphs/<terminal>.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--top-price", type=Decimal, help="цена верхней строки на train_frame.png (иначе спросит)")
    ap.add_argument("--force", action="store_true", help="сохранить шаблоны при слабой разделимости классов")
    args = ap.parse_args(argv)
    ok = train(args.video, args.terminal, args.tick, args.layout or layout_path(args.terminal),
               args.out or glyphs_path(args.terminal), args.top_price, args.frames, args.force)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())