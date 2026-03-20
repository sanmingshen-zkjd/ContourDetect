#include "DeepModelRunner.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>

bool OnnxSegmentationModel::load(const QString& path, QString* error) {
  clear();
  try {
    net_ = cv::dnn::readNetFromONNX(path.toStdString());
  } catch (const cv::Exception& ex) {
    if (error) *error = QString::fromStdString(ex.what());
    clear();
    return false;
  }
  if (net_.empty()) {
    if (error) *error = QStringLiteral("Failed to load ONNX model: %1").arg(path);
    return false;
  }
  return true;
}

bool OnnxSegmentationModel::isValid() const {
  return !net_.empty();
}

void OnnxSegmentationModel::clear() {
  net_ = cv::dnn::Net();
}

cv::Mat OnnxSegmentationModel::predictLabelMap(const cv::Mat& image, QString* error) const {
  if (!isValid() || image.empty()) {
    if (error) *error = QStringLiteral("ONNX model is not loaded or image is empty.");
    return cv::Mat();
  }

  cv::Mat resized;
  cv::resize(image, resized, inputSize_);
  cv::Mat blob = cv::dnn::blobFromImage(resized, scale_, inputSize_, mean_, swapRB_, false);

  cv::dnn::Net workingNet = net_;
  workingNet.setInput(blob);
  cv::Mat output;
  try {
    output = workingNet.forward();
  } catch (const cv::Exception& ex) {
    if (error) *error = QString::fromStdString(ex.what());
    return cv::Mat();
  }
  if (output.empty()) {
    if (error) *error = QStringLiteral("ONNX forward() returned empty output.");
    return cv::Mat();
  }

  cv::Mat labelMap;
  if (output.dims == 4 && output.size[1] > 1) {
    const int classCount = output.size[1];
    const int rows = output.size[2];
    const int cols = output.size[3];
    labelMap = cv::Mat(rows, cols, CV_32S, cv::Scalar(0));
    for (int y = 0; y < rows; ++y) {
      int* dst = labelMap.ptr<int>(y);
      for (int x = 0; x < cols; ++x) {
        int bestClass = 0;
        float bestValue = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < classCount; ++c) {
          const int idx[4] = {0, c, y, x};
          const float value = output.at<float>(idx);
          if (value > bestValue) {
            bestValue = value;
            bestClass = c;
          }
        }
        dst[x] = bestClass;
      }
    }
  } else if (output.dims == 4 && output.size[1] == 1) {
    const int rows = output.size[2];
    const int cols = output.size[3];
    labelMap = cv::Mat(rows, cols, CV_32S, cv::Scalar(0));
    for (int y = 0; y < rows; ++y) {
      int* dst = labelMap.ptr<int>(y);
      for (int x = 0; x < cols; ++x) {
        const int idx[4] = {0, 0, y, x};
        dst[x] = output.at<float>(idx) > 0.5f ? 1 : 0;
      }
    }
  } else if (output.dims == 2) {
    if (error) *error = QStringLiteral("Vector outputs are not supported for segmentation inference.");
    return cv::Mat();
  } else {
    if (error) *error = QStringLiteral("Unsupported ONNX output tensor rank: %1").arg(output.dims);
    return cv::Mat();
  }

  if (labelMap.size() != image.size()) {
    cv::Mat labelMapFloat;
    labelMap.convertTo(labelMapFloat, CV_32F);
    cv::Mat resizedLabelsFloat;
    cv::resize(labelMapFloat, resizedLabelsFloat, image.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat resizedLabels;
    resizedLabelsFloat.convertTo(resizedLabels, CV_32S);
    return resizedLabels;
  }
  return labelMap;
}
