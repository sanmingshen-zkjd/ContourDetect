#pragma once

#include "player/PlayerTypes.hpp"

namespace contour {

struct PreProcessConfig {
    bool undistort {true};
    int cropBorder {0};
    bool denoise {false};
};

struct PreProcessResult {
    FrameData frame;
    PreProcessConfig applied;
};

class PreProcessPipeline {
public:
    explicit PreProcessPipeline(PreProcessConfig config = {});

    PreProcessResult run(const FrameData& input) const;
    const PreProcessConfig& config() const { return m_config; }

private:
    PreProcessConfig m_config;
};

} // namespace contour
