#pragma once

#include "player/IFrameSource.hpp"

#include <QVector>

namespace contour {

class VectorFrameSource final : public IFrameSource {
public:
    explicit VectorFrameSource(QVector<QString> frames)
        : m_frames(std::move(frames)) {}

    int frameCount() const override { return m_frames.size(); }

    FrameData read(int frameIndex) const override {
        return FrameData {frameIndex, m_frames.at(frameIndex)};
    }

private:
    QVector<QString> m_frames;
};

} // namespace contour
