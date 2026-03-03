#pragma once

#include "measurement/MeasurementEngine.hpp"
#include "player/PlayerController.hpp"
#include "player/VectorFrameSource.hpp"
#include "preprocess/PreProcessPipeline.hpp"

#include <QVector>

namespace contour {

class MonocularContourApp {
public:
    explicit MonocularContourApp(PreProcessConfig preProcessConfig = {});

    void loadFrames(const QVector<QString>& frames, double fps = 30.0);
    MeasurementResult processCurrent() const;

    PlayerController& player() { return m_player; }
    const PlayerController& player() const { return m_player; }

private:
    PlayerController m_player;
    PreProcessPipeline m_preProcess;
    MeasurementEngine m_measurement;
};

} // namespace contour
