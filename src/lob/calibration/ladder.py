from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lob.calibration.base import CalibrationError, CalibrationReport, register_calibrator
from lob.calibration.ladder_core import LadderParams, calibrate_ladder
from lob.layout import LadderLayout, Roi


@register_calibrator("ladder")
class LadderResolver:
    def __init__(self, terminal: str = "generic", **params: float) -> None:
        self.terminal = terminal
        self.params = LadderParams(**params)

    def resolve(self, frames: Sequence[np.ndarray], hint: Roi | None = None
                ) -> tuple[LadderLayout, CalibrationReport]:
        geo = calibrate_ladder(frames, hint.as_tuple() if hint else None, self.params)
        report = CalibrationReport(geo.quality, dict(geo.metrics), list(geo.warnings))
        if geo.rows == 0 or geo.price_col[1] <= geo.price_col[0]:
            raise CalibrationError(report)
        h, w = frames[0].shape[:2]
        layout = LadderLayout(
            terminal=self.terminal, reference_size=(w, h),
            roi=Roi(x=geo.x, y=geo.y, w=geo.w, h=geo.h), rows=geo.rows,
            price_col=geo.price_col, size_col=geo.size_col)
        return layout, report
