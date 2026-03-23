#pragma once

#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class OnnxSegmentationModel {
public:
  bool load(const QString& path, QString* error = nullptr);
  bool isValid() const;
  void clear();

  void setInputSize(const cv::Size& size) { inputSize_ = size; }
  void setScale(double scale) { scale_ = scale; }
  void setSwapRB(bool swap) { swapRB_ = swap; }
  void setMean(const cv::Scalar& mean) { mean_ = mean; }

  cv::Mat predictLabelMap(const cv::Mat& image, QString* error = nullptr) const;

private:
  cv::dnn::Net net_;
  cv::Size inputSize_{256, 256};
  double scale_ = 1.0 / 255.0;
  bool swapRB_ = true;
  cv::Scalar mean_{0, 0, 0};
};
