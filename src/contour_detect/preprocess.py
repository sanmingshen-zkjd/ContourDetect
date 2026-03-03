from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PreProcessConfig:
    """Replaces camera calibration stage for monocular preprocessing."""

    undistort: bool = True
    crop_border: int = 0
    denoise: bool = False


class PreProcessPipeline:
    """PreProcess stage (single camera) before measurement."""

    def __init__(self, config: PreProcessConfig | None = None) -> None:
        self.config = config or PreProcessConfig()

    def run(self, frame: Any) -> Dict[str, Any]:
        # Keep frame intact for now; output metadata for downstream modules.
        return {
            "frame": frame,
            "meta": {
                "undistort": self.config.undistort,
                "crop_border": self.config.crop_border,
                "denoise": self.config.denoise,
            },
        }
