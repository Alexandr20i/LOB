"""Сохраняет области стакана с нескольких кадров, чтобы проверить layout глазами.

uv run python scripts/save_crops.py ~/Downloads/MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp
"""
# uv run python scripts/run.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --recalibrate
# uv run python scripts/stream_lob.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --print-interval-ms 0
# uv run python scripts/stream_lob.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --show-video
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from _common import ROOT, layout_path, video_arg
from lob.pipeline import load_ladder_layout
from lob.recognition.training import fit_layout
from lob.terminals import TERMINALS
from lob.video import sample_frames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=video_arg)
    ap.add_argument("--terminal", default="metascalp", choices=sorted(TERMINALS))
    ap.add_argument("-n", type=int, default=3, help="сколько кадров")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "configs" / "layouts")
    args = ap.parse_args(argv)

    layout = load_ladder_layout(layout_path(args.terminal))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(sample_frames(args.video, args.n)):
        out = args.out_dir / f"crop_{i}.png"
        cv2.imwrite(str(out), fit_layout(layout, img).roi.crop(img))
        print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())