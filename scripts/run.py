"""Полный цикл: калибровка -> обучение глифов -> стриминг стакана.

uv run python scripts/run.py ~/Downloads/MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1

Калибровка выполняется, только если layout ещё нет (или с --recalibrate), обучение
глифов -- если нет шаблонов (или с --retrain). Иначе используются сохранённые файлы,
окно выбора рамки на повторных запусках не открывается.
--step calibrate | train | stream выполняет один шаг принудительно.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import glyphs_path, layout_path, video_arg
from calibrate import calibrate
from lob.layout import Roi
from stream_lob import add_stream_args, stream
from train_glyphs import train


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=video_arg)
    add_stream_args(ap)
    ap.add_argument("--step", choices=["calibrate", "train", "stream", "all"], default="all")
    ap.add_argument("--roi", help="x,y,w,h вместо выбора мышью")
    ap.add_argument("--recalibrate", action="store_true", help="пересчитать layout, даже если он есть")
    ap.add_argument("--retrain", action="store_true", help="переобучить глифы, даже если они есть")
    ap.add_argument("--top-price", help="цена верхней строки на кадре обучения (иначе спросит)")
    ap.add_argument("--force", action="store_true", help="сохранить глифы при слабой разделимости классов")
    args = ap.parse_args(argv)

    video: Path = args.video
    if not video.exists():
        print(f"Видеофайл не найден: {video}")
        return 1
    layout_file, glyph_file = layout_path(args.terminal), glyphs_path(args.terminal)
    only = args.step != "all"

    if args.step in ("calibrate", "all") and (only or args.recalibrate or not layout_file.exists()):
        print("=" * 70, "\n ШАГ 1: КАЛИБРОВКА РАСКЛАДКИ\n", "=" * 70, sep="")
        roi = None
        if args.roi:
            x, y, w, h = (int(v) for v in args.roi.split(","))
            roi = Roi(x=x, y=y, w=w, h=h)
        _, ok = calibrate(video, args.terminal, layout_file, roi)
        if not ok:
            print("Калибровка не принята. Проверьте overlay и повторите.")
            return 1

    if args.step in ("train", "all") and (only or args.retrain or not glyph_file.exists()):
        print("=" * 70, "\n ШАГ 2: ГЛИФЫ\n", "=" * 70, sep="")
        from decimal import Decimal
        top = Decimal(args.top_price.replace(",", ".")) if args.top_price else None
        if not train(video, args.terminal, args.tick, layout_file, glyph_file, top, force=args.force):
            return 1

    if args.step in ("stream", "all"):
        print("=" * 70, "\n ШАГ 3: СТРИМИНГ СТАКАНА\n", "=" * 70, sep="")
        return stream(video, args.terminal, args.tick, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())