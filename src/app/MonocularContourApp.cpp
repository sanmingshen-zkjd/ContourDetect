#include "app/MonocularContourApp.hpp"

namespace contour {

MonocularContourApp::MonocularContourApp(PreProcessConfig preProcessConfig)
    : m_preProcess(preProcessConfig) {}

void MonocularContourApp::loadFrames(const QVector<QString>& frames, double fps) {
    m_player.load(QSharedPointer<VectorFrameSource>::create(frames), fps);
}

MeasurementResult MonocularContourApp::processCurrent() const {
    const auto frame = m_player.currentFrame();
    const auto preprocessed = m_preProcess.run(frame);
    return m_measurement.measure(preprocessed);
}

} // namespace contour
