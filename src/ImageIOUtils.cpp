#include "ImageIOUtils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ImageIOUtils {

cv::Mat normalizeInputImage(const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }

  cv::Mat normalized = image;
  if (normalized.channels() == 4) {
    cv::cvtColor(normalized, normalized, cv::COLOR_BGRA2BGR);
  }

  if (normalized.depth() == CV_8U) {
    return normalized.clone();
  }

  cv::Mat converted;
  double minValue = 0.0;
  double maxValue = 0.0;
  cv::minMaxLoc(normalized.reshape(1), &minValue, &maxValue);
  const double scale = (maxValue > minValue) ? 255.0 / (maxValue - minValue) : 1.0;
  normalized.convertTo(converted, CV_MAKETYPE(CV_8U, normalized.channels()), scale, -minValue * scale);
  return converted;
}

bool loadImageVolume(const QString& path, std::vector<cv::Mat>* slices, QString* error) {
  if (!slices) {
    if (error) *error = QStringLiteral("Output slice container is null.");
    return false;
  }

  slices->clear();
  std::vector<cv::Mat> loaded;
  if (cv::imreadmulti(path.toStdString(), loaded, cv::IMREAD_UNCHANGED) && !loaded.empty()) {
    slices->reserve(loaded.size());
    for (const cv::Mat& slice : loaded) {
      slices->push_back(normalizeInputImage(slice));
    }
    return true;
  }

  cv::Mat single = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
  if (single.empty()) {
    if (error) *error = QStringLiteral("Unable to load image: %1").arg(path);
    return false;
  }

  slices->push_back(normalizeInputImage(single));
  return true;
}

}  // namespace ImageIOUtils
