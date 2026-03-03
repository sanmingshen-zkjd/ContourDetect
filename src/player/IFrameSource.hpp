#pragma once

#include "player/PlayerTypes.hpp"

namespace contour {

class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    virtual int frameCount() const = 0;
    virtual FrameData read(int frameIndex) const = 0;
};

} // namespace contour
