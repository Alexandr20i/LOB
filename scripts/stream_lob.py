"""Стриминг стакана из видео: распознавание, вывод в консоль, запись в Parquet.

uv run python scripts/stream_lob.py ~/Downloads/rec.mp4 --terminal metascalp --tick 0.01

uv run python scripts/stream_lob.py rec.mp4 --tick 0.01 --save out/session1 --quiet

Расчёт ведётся под запись 30 кадр/с и кадр 2880x1800 (раскладка из configs/layouts).
Кадры с низким согласием цен и перекрещенные стаканы отбрасываются и считаются.
Данные: каталог part-*.parquet (+ meta.json); читать pl.read_parquet("out/session1/part-*.parquet").
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path

from _common import glyphs_path, layout_path, video_arg
from lob.pipeline import TARGET_FPS, load_ladder_layout, stream_video
from lob.recognition.glyphs import TemplateSet
from lob.terminals import TERMINALS


def add_stream_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--terminal", default="metascalp", choices=sorted(TERMINALS))
    ap.add_argument("--tick", required=True, type=Decimal, help="шаг цены (0.01, 0.1, 1 ...)")
    ap.add_argument("--stride", type=int, default=1, help="обрабатывать каждый N-й кадр (по умолчанию 1: все)")
    ap.add_argument("--save", type=Path, help="каталог для Parquet-файлов")
    ap.add_argument("--show-video", action="store_true", help="показывать видео с разметкой (q -- выход)")
    ap.add_argument("--quiet", action="store_true", help="не печатать стакан, только итоговую статистику")
    ap.add_argument("--print-interval-ms", type=float, default=250.0,
                    help="печатать стакан не чаще, чем раз в N мс записи")
    ap.add_argument("--depth", type=int, default=12, help="сколько уровней каждой стороны печатать")
    ap.add_argument("--min-agreement", type=float, default=0.6,
                    help="кадры с меньшим согласием цен с прогрессией отбрасываются")
    ap.add_argument("--fps", type=float, default=TARGET_FPS, help="частота записи для бюджета времени на кадр")


def stream(video: Path, terminal: str, tick: Decimal, args: argparse.Namespace) -> int:
    if not video.exists():
        print(f"Видеофайл не найден: {video}")
        return 1
    layout_file, glyph_file = layout_path(terminal), glyphs_path(terminal)
    if not layout_file.exists():
        print(f"Layout не найден: {layout_file}\n"
              f"Калибровка: uv run python scripts/calibrate.py {video} --terminal {terminal}")
        return 1
    if not glyph_file.exists():
        print(f"Шаблоны не найдены: {glyph_file}\n"
              f"Обучение: uv run python scripts/train_glyphs.py {video} --terminal {terminal} --tick {tick}")
        return 1
    layout = load_ladder_layout(layout_file)
    templates = TemplateSet.load(glyph_file)
    print(f"Layout: {layout_file}\nШаблоны: {glyph_file}\nVideo: {video}  tick={tick}  stride={args.stride}\n")
    stats = stream_video(video, layout, templates, tick, terminal=terminal, stride=args.stride,
                         min_agreement=args.min_agreement, save_dir=args.save,
                         show_video=args.show_video, quiet=args.quiet,
                         print_interval_ms=args.print_interval_ms, depth=args.depth,
                         target_fps=args.fps)
    print("\n" + stats.summary(args.fps))
    if args.save:
        print(f"Данные: {args.save}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=video_arg)
    add_stream_args(ap)
    args = ap.parse_args(argv)
    return stream(args.video, args.terminal, args.tick, args)


if __name__ == "__main__":
    sys.exit(main())