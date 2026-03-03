#include "player/PlayerController.hpp"

#include <QtGlobal>
#include <stdexcept>

namespace contour {

PlayerController::PlayerController(QObject* parent)
    : QObject(parent) {}

void PlayerController::load(const QSharedPointer<IFrameSource>& source, double fps) {
    m_source = source;
    m_state = PlaybackState {};
    m_state.fps = fps;
    emit playbackStateChanged(m_state);
    emit frameChanged(currentFrame());
}

void PlayerController::play() {
    ensureSource();
    m_state.isPlaying = true;
    emit playbackStateChanged(m_state);
}

void PlayerController::pause() {
    m_state.isPlaying = false;
    emit playbackStateChanged(m_state);
}

void PlayerController::setSpeed(double speed) {
    if (speed <= 0.0) {
        throw std::invalid_argument("speed must be positive");
    }
    m_state.speed = speed;
    emit playbackStateChanged(m_state);
}

void PlayerController::setLoopMode(LoopMode mode) {
    m_state.loopMode = mode;
    emit playbackStateChanged(m_state);
}

FrameData PlayerController::seek(int frameIndex) {
    const auto& source = ensureSource();
    if (source.frameCount() <= 0) {
        throw std::runtime_error("empty source");
    }

    m_state.frameIndex = clampFrameIndex(frameIndex);
    auto frame = source.read(m_state.frameIndex);
    emit frameChanged(frame);
    return frame;
}

FrameData PlayerController::step(int delta) {
    return seek(m_state.frameIndex + delta);
}

FrameData PlayerController::tick() {
    const auto& source = ensureSource();
    if (!m_state.isPlaying) {
        return source.read(m_state.frameIndex);
    }

    auto next = m_state.frameIndex + 1;
    if (next >= source.frameCount()) {
        if (m_state.loopMode == LoopMode::All) {
            next = 0;
        } else {
            pause();
            next = source.frameCount() - 1;
        }
    }

    m_state.frameIndex = next;
    auto frame = source.read(next);
    emit frameChanged(frame);
    return frame;
}

FrameData PlayerController::currentFrame() const {
    return ensureSource().read(m_state.frameIndex);
}

const IFrameSource& PlayerController::ensureSource() const {
    if (m_source.isNull()) {
        throw std::runtime_error("frame source not loaded");
    }
    return *m_source;
}

int PlayerController::clampFrameIndex(int frameIndex) const {
    const auto maxIndex = ensureSource().frameCount() - 1;
    return qBound(0, frameIndex, maxIndex);
}

} // namespace contour
