#pragma once

#include "player/IFrameSource.hpp"

#include <QObject>
#include <QSharedPointer>

namespace contour {

class PlayerController final : public QObject {
    Q_OBJECT
public:
    explicit PlayerController(QObject* parent = nullptr);

    void load(const QSharedPointer<IFrameSource>& source, double fps = 30.0);
    void play();
    void pause();
    void setSpeed(double speed);
    void setLoopMode(LoopMode mode);

    FrameData seek(int frameIndex);
    FrameData step(int delta = 1);
    FrameData tick();
    FrameData currentFrame() const;

    const PlaybackState& state() const { return m_state; }

signals:
    void frameChanged(const contour::FrameData& frame);
    void playbackStateChanged(const contour::PlaybackState& state);

private:
    const IFrameSource& ensureSource() const;
    int clampFrameIndex(int frameIndex) const;

    QSharedPointer<IFrameSource> m_source;
    PlaybackState m_state;
};

} // namespace contour
