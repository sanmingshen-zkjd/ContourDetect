#pragma once

#include "preprocess/PreProcessPipeline.hpp"

#include <QString>

namespace contour {

struct MeasurementResult {
    bool ok {false};
    double contourLength {0.0};
    QString details;
};

class MeasurementEngine {
public:
    MeasurementResult measure(const PreProcessResult& input) const;
};

} // namespace contour
