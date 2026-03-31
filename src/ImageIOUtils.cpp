#include "ImageIOUtils.h"

#include <QFile>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ImageIOUtils {
namespace {
std::string cvPath(const QString& path) {
  return QFile::encodeName(path).toStdString();
}
}

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
  if (cv::imreadmulti(cvPath(path), loaded, cv::IMREAD_UNCHANGED) && !loaded.empty()) {
    slices->reserve(loaded.size());
    for (const cv::Mat& slice : loaded) {
      slices->push_back(normalizeInputImage(slice));
    }
    return true;
  }

  cv::Mat single = cv::imread(cvPath(path), cv::IMREAD_UNCHANGED);
  if (single.empty()) {
    QFile file(path);
    if (file.open(QIODevice::ReadOnly)) {
      const QByteArray bytes = file.readAll();
      if (!bytes.isEmpty()) {
        cv::Mat raw(1, bytes.size(), CV_8U, const_cast<char*>(bytes.constData()));
        single = cv::imdecode(raw, cv::IMREAD_UNCHANGED);
      }
    }
  }
  if (single.empty()) {
    if (error) *error = QStringLiteral("Unable to load image: %1").arg(path);
    return false;
  }

  slices->push_back(normalizeInputImage(single));
  return true;
}

}  // namespace ImageIOUtils
