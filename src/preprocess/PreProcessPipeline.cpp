#include "preprocess/PreProcessPipeline.hpp"

namespace contour {

PreProcessPipeline::PreProcessPipeline(PreProcessConfig config)
    : m_config(config) {}

PreProcessResult PreProcessPipeline::run(const FrameData& input) const {
    PreProcessResult result;
    result.frame = input;
    result.applied = m_config;
    return result;
}

} // namespace contour
