from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lob.calibration import ladder as _ladder  # noqa: F401  (регистрирует калибратор)
from lob.calibration.base import (CALIBRATORS, CalibrationError, CalibrationReport,
                                  LayoutResolver, make_resolver, register_calibrator)
from lob.layout import BaseLayout, Roi


def resolve_layout(terminal: str, frames: Sequence[np.ndarray], hint: Roi | None = None
                   ) -> tuple[BaseLayout, CalibrationReport]:
    return make_resolver(terminal).resolve(frames, hint)


__all__ = ["CALIBRATORS", "CalibrationError", "CalibrationReport", "LayoutResolver",
           "make_resolver", "register_calibrator", "resolve_layout"]
