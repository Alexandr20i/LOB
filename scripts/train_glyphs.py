"""Обучение шаблонов цифр по одному кадру. Запуск из корня репозитория:

  uv run python scripts/train_glyphs.py ~/Downloads/rec.mp4 --tick 0.01

Скрипт сохранит configs/glyphs/train_frame.png и спросит цену верхней строки на нём.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import cv2

from lob.layout import load_layout
from lob.recognition.glyphs import (expected_texts, read_ladder, soft_mask, train_templates)
from lob.video import sample_frames

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--layout", default="configs/layouts/metascalp.yaml")
ap.add_argument("--tick", default="0.01")
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--out", default="configs/glyphs/metascalp.npz")
ap.add_argument("--top-price", help="цена верхней строки на train_frame.png (иначе спросит)")
args = ap.parse_args()

layout = load_layout(args.layout)
tick = Decimal(args.tick)
frames = sample_frames(Path(args.video).expanduser(), args.frames)
train_png = Path(args.out).with_name("train_frame.png")
train_png.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(train_png), layout.roi.crop(frames[0]))
print(f"Откройте {train_png} и посмотрите цену в самой верхней строке.")
raw = args.top_price or input("Цена верхней строки (например 2330.77): ")
top = Decimal(raw.replace(" ", "").replace(",", "."))

cells = list(layout.cells(frames[0]))
ts, info = train_templates(
    (soft_mask(pc), exp) for (_, pc), exp in zip(cells, expected_texts(top, tick, layout.rows)))
print(f"Ячеек цены использовано {info['cells_used']} из {layout.rows}")
print("Образцов по символам:", info["samples"])
print(f"Разделимость классов: {info['separation']:.1f} (нужно >= 3)")
if info["missing"] or info["cells_used"] < 20:
    raise SystemExit(f"Мало данных: нет символов {info['missing']}. "
                     "Проверьте введённую цену и разрешение кадра.")
ts.save(args.out)
print("Шаблоны сохранены:", args.out)

print("\nПроверка на остальных кадрах (кадр, согласие цен с прогрессией, строк с нечитаемым размером):")
for i, img in enumerate(frames):
    rows, agreement = read_ladder(layout.cells(img), ts, tick)
    bad_size = sum(r.size is None for r in rows)
    top_read = rows[0].price
    print(f"  {i}: agreement={agreement:.2f}  size_unread={bad_size}/{len(rows)}  top={top_read}")