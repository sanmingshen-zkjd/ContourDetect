#pragma once

#include <QString>
#include <opencv2/core.hpp>

#include <vector>

namespace ImageIOUtils {

cv::Mat normalizeInputImage(const cv::Mat& image);
bool loadImageVolume(const QString& path, std::vector<cv::Mat>* slices, QString* error = nullptr);

}  // namespace ImageIOUtils
