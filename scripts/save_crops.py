import sys
from pathlib import Path

import cv2

from lob.layout import load_layout
from lob.video import sample_frames

video = Path(sys.argv[1]).expanduser()
layout = load_layout("configs/layouts/metascalp.yaml")

for i, img in enumerate(sample_frames(video, 3)):
    out = f"configs/layouts/crop_{i}.png"
    cv2.imwrite(out, layout.roi.crop(img))
    print("saved", out)