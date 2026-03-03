from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MeasurementResult:
    ok: bool
    contour_length: float
    details: Dict[str, Any]


class MeasurementEngine:
    """Measurement module kept stable while input changed to PreProcess output."""

    def measure(self, preprocessed: Dict[str, Any]) -> MeasurementResult:
        frame = preprocessed["frame"]
        length = float(len(str(frame)))
        return MeasurementResult(
            ok=True,
            contour_length=length,
            details={"input_meta": preprocessed.get("meta", {})},
        )
