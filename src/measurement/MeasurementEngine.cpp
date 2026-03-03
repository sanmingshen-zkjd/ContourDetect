#include "measurement/MeasurementEngine.hpp"

namespace contour {

MeasurementResult MeasurementEngine::measure(const PreProcessResult& input) const {
    MeasurementResult result;
    result.ok = true;
    result.contourLength = static_cast<double>(input.frame.payload.size());
    result.details = QString("undistort=%1 cropBorder=%2 denoise=%3")
                         .arg(input.applied.undistort)
                         .arg(input.applied.cropBorder)
                         .arg(input.applied.denoise);
    return result;
}

} // namespace contour
