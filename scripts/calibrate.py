"""Полуавтоматическая калибровка раскладки по видео.

  uv run python scripts/calibrate.py data/rec.mp4 --terminal metascalp
  uv run python scripts/calibrate.py data/rec.mp4 --roi 917,2,405,1700     # без окна

Открывается кадр: обведите стакан мышью (цены и размеры, грубо, с запасом) и нажмите Enter.
Результат: configs/layouts/<terminal>.yaml и <terminal>_overlay.png для проверки глазами.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from lob.calibration import CalibrationError, resolve_layout
from lob.layout import Roi, save_layout
from lob.terminals import TERMINALS
from lob.video import sample_frames


def pick_roi(image: np.ndarray) -> Roi | None:
    scale = min(1.0, 1000 / image.shape[0], 1800 / image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    x, y, w, h = cv2.selectROI("draw the order book, Enter = ok, c = cancel", shown, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        return None
    return Roi(x=round(x / scale), y=round(y / scale), w=round(w / scale), h=round(h / scale))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--terminal", default="generic", choices=sorted(TERMINALS))
    ap.add_argument("--roi", help="x,y,w,h вместо выбора мышью")
    ap.add_argument("--frames", type=int, default=8, help="сколько кадров использовать")
    ap.add_argument("--out", help="куда сохранить layout (по умолчанию configs/layouts/<terminal>.yaml)")
    args = ap.parse_args()

    frames = sample_frames(args.video, args.frames)
    if not frames:
        print("Не удалось прочитать кадры из видео")
        return 2
    h, w = frames[0].shape[:2]
    print(f"Кадров: {len(frames)}, размер {w}x{h}")

    if args.roi:
        x, y, rw, rh = (int(v) for v in args.roi.split(","))
        roi: Roi | None = Roi(x=x, y=y, w=rw, h=rh)
    else:
        roi = pick_roi(frames[0])
        if roi is None:
            print("Рамка не выбрана")
            return 2

    try:
        layout, report = resolve_layout(args.terminal, frames, roi)
    except CalibrationError as e:
        print("КАЛИБРОВКА НЕ УДАЛАСЬ:", e)
        for k, v in e.report.metrics.items():
            print(f"  {k}: {v:.3f}")
        return 1

    out = Path(args.out or f"configs/layouts/{args.terminal}.yaml")
    save_layout(layout, out)
    overlay = out.with_name(out.stem + "_overlay.png")
    cv2.imwrite(str(overlay), layout.draw(frames[0]))

    print(f"Строк: {layout.rows}, высота строки {layout.roi.h / layout.rows:.2f} px, качество {report.quality:.2f}")
    for k, v in report.metrics.items():
        print(f"  {k}: {v:.3f}")
    for wmsg in report.warnings:
        print("  ВНИМАНИЕ:", wmsg)
    print(f"Layout: {out}\nПроверьте глазами: {overlay}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())