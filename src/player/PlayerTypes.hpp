#pragma once

#include <QString>

namespace contour {

enum class LoopMode {
    Off,
    All,
};

struct PlaybackState {
    int frameIndex {0};
    double fps {30.0};
    double speed {1.0};
    bool isPlaying {false};
    LoopMode loopMode {LoopMode::Off};
};

struct FrameData {
    int index {0};
    QString payload;
};

} // namespace contour
