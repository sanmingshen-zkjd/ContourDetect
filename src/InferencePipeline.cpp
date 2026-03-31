#include "InferencePipeline.h"

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>

#include "DeepModelRunner.h"
#include "ImageIOUtils.h"
#include "SegmentationEngine.h"

namespace InferencePipeline {

bool runClassicalInference(const ClassicalInferenceOptions& options, QString* error) {
  SegmentationClassifier model;
  std::vector<SegmentationClassInfo> classes;
  SegmentationFeatureSettings featureSettings;
  SegmentationClassifierSettings classifierSettings;
  SegmentationTrainingStats stats;
  if (!model.load(options.modelPath, &classes, &featureSettings, &classifierSettings, &stats)) {
    if (error) *error = QStringLiteral("Failed to load classifier: %1").arg(options.modelPath);
    return false;
  }

  QStringList inputs;
  QFileInfo inputInfo(options.inputPath);
  if (inputInfo.isDir()) {
    QDirIterator it(options.inputPath, QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.tif" << "*.tiff", QDir::Files);
    while (it.hasNext()) inputs << it.next();
  } else {
    inputs << options.inputPath;
  }
  if (inputs.isEmpty()) {
    if (error) *error = QStringLiteral("No input images found.");
    return false;
  }
  if (inputs.size() > 1) {
    QDir().mkpath(options.outputPath);
  }

  for (const QString& imagePath : inputs) {
    std::vector<cv::Mat> slices;
    QString localError;
    if (!ImageIOUtils::loadImageVolume(imagePath, &slices, &localError) || slices.empty()) {
      if (error) *error = localError;
      return false;
    }
    const QString baseName = QFileInfo(imagePath).completeBaseName();
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(slices.size()); ++sliceIndex) {
      const cv::Mat featureStack = SegmentationEngine::computeFeatureStack(slices[sliceIndex], featureSettings);
      const cv::Mat labels = SegmentationEngine::applyModelLabels(model, featureStack, slices[sliceIndex].rows, slices[sliceIndex].cols);
      if (labels.empty()) {
        if (error) *error = QStringLiteral("Failed to infer: %1").arg(imagePath);
        return false;
      }
      cv::Mat exportMask;
      labels.convertTo(exportMask, CV_8U);
      QString targetPath;
      if (inputs.size() == 1 && slices.size() == 1) {
        targetPath = options.outputPath;
      } else {
        targetPath = QDir(options.outputPath).filePath(QString("%1_slice_%2.png").arg(baseName).arg(sliceIndex, 3, 10, QLatin1Char('0')));
      }
      if (!cv::imwrite(targetPath.toStdString(), exportMask)) {
        if (error) *error = QStringLiteral("Failed to write: %1").arg(targetPath);
        return false;
      }
      if (model.supportsProbability() && options.probabilityClass >= 0 && options.probabilityClass < static_cast<int>(classes.size())) {
        const cv::Mat probs = SegmentationEngine::applyModelProbabilities(model, featureStack, slices[sliceIndex].rows, slices[sliceIndex].cols, options.probabilityClass);
        if (!probs.empty()) {
          cv::Mat prob8;
          probs.convertTo(prob8, CV_8U, 255.0);
          const QString probPath = targetPath.left(targetPath.lastIndexOf('.')) + QString("_prob_class_%1.png").arg(options.probabilityClass);
          cv::imwrite(probPath.toStdString(), prob8);
        }
      }
    }
  }
  return true;
}

bool runOnnxInference(const OnnxInferenceOptions& options, QString* error) {
  OnnxSegmentationModel model;
  if (!model.load(options.modelPath, error)) {
    return false;
  }

  QStringList inputs;
  QFileInfo inputInfo(options.inputPath);
  if (inputInfo.isDir()) {
    QDirIterator it(options.inputPath, QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.tif" << "*.tiff", QDir::Files);
    while (it.hasNext()) inputs << it.next();
  } else {
    inputs << options.inputPath;
  }
  if (inputs.isEmpty()) {
    if (error) *error = QStringLiteral("No input images found.");
    return false;
  }
  if (inputs.size() > 1) {
    QDir().mkpath(options.outputPath);
  }

  for (const QString& imagePath : inputs) {
    std::vector<cv::Mat> slices;
    QString localError;
    if (!ImageIOUtils::loadImageVolume(imagePath, &slices, &localError) || slices.empty()) {
      if (error) *error = localError;
      return false;
    }
    const QString baseName = QFileInfo(imagePath).completeBaseName();
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(slices.size()); ++sliceIndex) {
      const cv::Mat labels = model.predictLabelMap(slices[sliceIndex], &localError);
      if (labels.empty()) {
        if (error) *error = localError;
        return false;
      }
      cv::Mat exportMask;
      labels.convertTo(exportMask, CV_8U);
      QString targetPath;
      if (inputs.size() == 1 && slices.size() == 1) {
        targetPath = options.outputPath;
      } else {
        targetPath = QDir(options.outputPath).filePath(QString("%1_slice_%2.png").arg(baseName).arg(sliceIndex, 3, 10, QLatin1Char('0')));
      }
      if (!cv::imwrite(targetPath.toStdString(), exportMask)) {
        if (error) *error = QStringLiteral("Failed to write: %1").arg(targetPath);
        return false;
      }
    }
  }
  return true;
}

}  // namespace InferencePipeline
