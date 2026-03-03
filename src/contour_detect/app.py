from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .measurement import MeasurementEngine, MeasurementResult
from .player import ListFrameSource, PlayerController
from .preprocess import PreProcessConfig, PreProcessPipeline


@dataclass
class AppConfig:
    fps: float = 30.0
    preprocess: PreProcessConfig = field(default_factory=PreProcessConfig)


class MonocularContourApp:
    """Top-level app composed as: Player -> PreProcess -> Measurement."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.player = PlayerController()
        self.preprocess = PreProcessPipeline(self.config.preprocess)
        self.measurement = MeasurementEngine()

    def load_frames(self, frames: list[object]) -> None:
        self.player.load(ListFrameSource(frames), fps=self.config.fps)

    def process_current(self) -> MeasurementResult:
        frame = self.player.current_frame()
        preprocessed = self.preprocess.run(frame)
        return self.measurement.measure(preprocessed)
