from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol


class LoopMode(str, Enum):
    OFF = "off"
    ALL = "all"


class FrameSource(Protocol):
    """Single-camera frame source."""

    def frame_count(self) -> int: ...

    def read(self, frame_index: int) -> object: ...


@dataclass
class PlaybackState:
    frame_index: int = 0
    fps: float = 30.0
    speed: float = 1.0
    is_playing: bool = False
    loop_mode: LoopMode = LoopMode.OFF


class PlayerController:
    """Player module migrated from multi-cam pattern to monocular playback.

    Feature parity target with original player area:
    - play / pause
    - frame-accurate seek and step
    - speed control
    - loop mode
    - playlist-like source swapping
    """

    def __init__(self, source: Optional[FrameSource] = None) -> None:
        self._source = source
        self._state = PlaybackState()

    @property
    def state(self) -> PlaybackState:
        return self._state

    def load(self, source: FrameSource, fps: float = 30.0) -> None:
        self._source = source
        self._state = PlaybackState(frame_index=0, fps=fps)

    def play(self) -> None:
        self._ensure_source()
        self._state.is_playing = True

    def pause(self) -> None:
        self._state.is_playing = False

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self._state.speed = speed

    def set_loop_mode(self, mode: LoopMode) -> None:
        self._state.loop_mode = mode

    def seek(self, frame_index: int) -> object:
        source = self._ensure_source()
        max_index = source.frame_count() - 1
        if max_index < 0:
            raise ValueError("empty source")
        self._state.frame_index = min(max(frame_index, 0), max_index)
        return source.read(self._state.frame_index)

    def step(self, delta: int = 1) -> object:
        return self.seek(self._state.frame_index + delta)

    def tick(self) -> Optional[object]:
        """Advance one frame if playing, return frame payload."""
        if not self._state.is_playing:
            return None

        source = self._ensure_source()
        next_index = self._state.frame_index + 1
        if next_index >= source.frame_count():
            if self._state.loop_mode == LoopMode.ALL:
                next_index = 0
            else:
                self.pause()
                next_index = source.frame_count() - 1

        self._state.frame_index = next_index
        return source.read(next_index)

    def current_frame(self) -> object:
        source = self._ensure_source()
        return source.read(self._state.frame_index)

    def _ensure_source(self) -> FrameSource:
        if self._source is None:
            raise RuntimeError("frame source not loaded")
        return self._source


class ListFrameSource:
    """Simple in-memory source for tests and CLI demos."""

    def __init__(self, frames: List[object]) -> None:
        self._frames = frames

    def frame_count(self) -> int:
        return len(self._frames)

    def read(self, frame_index: int) -> object:
        return self._frames[frame_index]
