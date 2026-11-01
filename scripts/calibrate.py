"""Полуавтоматическая калибровка раскладки по видео.

uv run python scripts/calibrate.py data/rec.mp4 --terminal metascalp

uv run python scripts/calibrate.py data/rec.mp4 --terminal metascalp --roi 2444,92,382,1687  # без окна

Открывается кадр: обведите стакан мышью (цены и размеры, грубо, с запасом) и нажмите Enter.
Результат: configs/layouts/<terminal>.yaml и <terminal>_overlay.png для проверки глазами.
Размер кадра записи сохраняется в reference_size (для MetaScalp на Retina: 2880x1800).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from _common import layout_path, video_arg
from lob.calibration import CalibrationError, resolve_layout
from lob.gui import pick_roi
from lob.layout import LadderLayout, Roi, save_layout
from lob.terminals import TERMINALS
from lob.video import sample_frames


def calibrate(video: Path, terminal: str, out: Path, roi: Roi | None = None,
              n_frames: int = 8) -> tuple[LadderLayout | None, bool]:
    """Калибрует и сохраняет layout. Возвращает (layout, качество приемлемо)."""
    frames = sample_frames(video, n_frames)
    if not frames:
        print("Не удалось прочитать кадры из видео")
        return None, False
    h, w = frames[0].shape[:2]
    print(f"Кадров: {len(frames)}, размер {w}x{h}")

    if roi is None:
        roi = pick_roi(frames[0])
        if roi is None:
            print("Рамка не выбрана")
            return None, False

    try:
        layout, report = resolve_layout(terminal, frames, roi)
    except CalibrationError as e:
        print("КАЛИБРОВКА НЕ УДАЛАСЬ:", e)
        for k, v in e.report.metrics.items():
            print(f"  {k}: {v:.3f}")
        return None, False

    assert isinstance(layout, LadderLayout)
    save_layout(layout, out)
    overlay = out.with_name(out.stem + "_overlay.png")
    cv2.imwrite(str(overlay), layout.draw(frames[0]))
    print(f"Строк: {layout.rows}, высота строки {layout.roi.h / layout.rows:.2f} px, "
          f"качество {report.quality:.2f}")
    for k, v in report.metrics.items():
        print(f"  {k}: {v:.3f}")
    for msg in report.warnings:
        print("  ВНИМАНИЕ:", msg)
    print(f"Layout: {out}\nПроверьте глазами: {overlay}")
    return layout, report.ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=video_arg)
    ap.add_argument("--terminal", default="metascalp", choices=sorted(TERMINALS))
    ap.add_argument("--roi", help="x,y,w,h вместо выбора мышью")
    ap.add_argument("--frames", type=int, default=8, help="сколько кадров использовать")
    ap.add_argument("--out", type=Path, help="куда сохранить layout (по умолчанию configs/layouts/<terminal>.yaml)")
    args = ap.parse_args(argv)

    roi = None
    if args.roi:
        x, y, rw, rh = (int(v) for v in args.roi.split(","))
        roi = Roi(x=x, y=y, w=rw, h=rh)
    _, ok = calibrate(args.video, args.terminal, args.out or layout_path(args.terminal), roi, args.frames)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())