#pragma once

#include <QString>

struct ClassicalInferenceOptions {
  QString modelPath;
  QString inputPath;
  QString outputPath;
  int probabilityClass = 0;
};

struct OnnxInferenceOptions {
  QString modelPath;
  QString inputPath;
  QString outputPath;
};

namespace InferencePipeline {

bool runClassicalInference(const ClassicalInferenceOptions& options, QString* error = nullptr);
bool runOnnxInference(const OnnxInferenceOptions& options, QString* error = nullptr);

}  // namespace InferencePipeline
