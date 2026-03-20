#include "SegmentationEngine.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/persistence.hpp>

#include <numeric>
#include <random>
#include <cmath>
#include <limits>

namespace {
constexpr double kMinVariance = 1e-6;
constexpr double kLog2Pi = 1.8378770664093453;

cv::Scalar toBgr(const QColor& color) {
  return cv::Scalar(color.blue(), color.green(), color.red());
}

void appendFeature(const cv::Mat& feature, std::vector<cv::Mat>& channels) {
  cv::Mat normalized;
  if (feature.type() != CV_32F) {
    feature.convertTo(normalized, CV_32F);
  } else {
    normalized = feature;
  }
  channels.push_back(normalized);
}
}

bool GaussianNaiveBayesModel::isValid() const {
  return classCount_ > 0 && featureCount_ > 0 && !means_.empty() && !variances_.empty() && !logPriors_.empty();
}

void GaussianNaiveBayesModel::clear() {
  classCount_ = 0;
  featureCount_ = 0;
  means_.release();
  variances_.release();
  logPriors_.release();
}

bool GaussianNaiveBayesModel::train(const cv::Mat& features,
                                    const cv::Mat& labels,
                                    int classCount,
                                    SegmentationTrainingStats* stats) {
  clear();
  if (features.empty() || labels.empty() || features.rows != labels.rows || classCount <= 0) {
    return false;
  }

  classCount_ = classCount;
  featureCount_ = features.cols;
  means_ = cv::Mat::zeros(classCount_, featureCount_, CV_64F);
  variances_ = cv::Mat::ones(classCount_, featureCount_, CV_64F);
  logPriors_ = cv::Mat::zeros(classCount_, 1, CV_64F);

  std::vector<int> counts(classCount_, 0);
  for (int row = 0; row < labels.rows; ++row) {
    const int cls = labels.at<int>(row, 0);
    if (cls >= 0 && cls < classCount_) {
      counts[cls] += 1;
      for (int col = 0; col < featureCount_; ++col) {
        means_.at<double>(cls, col) += features.at<float>(row, col);
      }
    }
  }

  const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
  if (total <= 0.0) {
    clear();
    return false;
  }

  for (int cls = 0; cls < classCount_; ++cls) {
    if (counts[cls] == 0) {
      clear();
      return false;
    }
    for (int col = 0; col < featureCount_; ++col) {
      means_.at<double>(cls, col) /= static_cast<double>(counts[cls]);
    }
    logPriors_.at<double>(cls, 0) = std::log(static_cast<double>(counts[cls]) / total);
  }

  for (int row = 0; row < labels.rows; ++row) {
    const int cls = labels.at<int>(row, 0);
    for (int col = 0; col < featureCount_; ++col) {
      const double diff = static_cast<double>(features.at<float>(row, col)) - means_.at<double>(cls, col);
      variances_.at<double>(cls, col) += diff * diff;
    }
  }

  for (int cls = 0; cls < classCount_; ++cls) {
    for (int col = 0; col < featureCount_; ++col) {
      double variance = variances_.at<double>(cls, col) / std::max(1, counts[cls] - 1);
      variances_.at<double>(cls, col) = std::max(variance, kMinVariance);
    }
  }

  if (stats) {
    stats->classCount = classCount_;
    stats->sampleCount = static_cast<int>(total);
    cv::Mat predicted = predictLabels(features);
    int correct = 0;
    for (int row = 0; row < labels.rows; ++row) {
      if (predicted.at<int>(row, 0) == labels.at<int>(row, 0)) {
        ++correct;
      }
    }
    stats->trainingAccuracy = labels.rows > 0 ? static_cast<double>(correct) / labels.rows : 0.0;
  }

  return true;
}

cv::Mat GaussianNaiveBayesModel::predictProbabilities(const cv::Mat& features) const {
  cv::Mat probs;
  if (!isValid() || features.empty() || features.cols != featureCount_) {
    return probs;
  }

  probs = cv::Mat::zeros(features.rows, classCount_, CV_32F);
  std::vector<double> logLikelihood(classCount_, 0.0);

  for (int row = 0; row < features.rows; ++row) {
    double maxLog = -std::numeric_limits<double>::infinity();
    for (int cls = 0; cls < classCount_; ++cls) {
      double score = logPriors_.at<double>(cls, 0);
      for (int col = 0; col < featureCount_; ++col) {
        const double variance = variances_.at<double>(cls, col);
        const double diff = static_cast<double>(features.at<float>(row, col)) - means_.at<double>(cls, col);
        score += -0.5 * (kLog2Pi + std::log(variance) + (diff * diff) / variance);
      }
      logLikelihood[cls] = score;
      maxLog = std::max(maxLog, score);
    }

    double sum = 0.0;
    for (int cls = 0; cls < classCount_; ++cls) {
      const double value = std::exp(logLikelihood[cls] - maxLog);
      probs.at<float>(row, cls) = static_cast<float>(value);
      sum += value;
    }
    const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int cls = 0; cls < classCount_; ++cls) {
      probs.at<float>(row, cls) *= inv;
    }
  }

  return probs;
}

cv::Mat GaussianNaiveBayesModel::predictLabels(const cv::Mat& features) const {
  cv::Mat labels;
  if (!isValid()) {
    return labels;
  }
  const cv::Mat probs = predictProbabilities(features);
  if (probs.empty()) {
    return labels;
  }
  labels = cv::Mat::zeros(probs.rows, 1, CV_32S);
  for (int row = 0; row < probs.rows; ++row) {
    cv::Point maxIdx;
    double maxVal = 0.0;
    cv::minMaxLoc(probs.row(row), nullptr, &maxVal, nullptr, &maxIdx);
    labels.at<int>(row, 0) = maxIdx.x;
  }
  return labels;
}

bool GaussianNaiveBayesModel::save(const QString& path,
                                   const std::vector<SegmentationClassInfo>& classes,
                                   const SegmentationFeatureSettings& settings,
                                   const SegmentationTrainingStats& stats) const {
  if (!isValid()) {
    return false;
  }
  cv::FileStorage fs(path.toStdString(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "class_count" << classCount_;
  fs << "feature_count" << featureCount_;
  fs << "means" << means_;
  fs << "variances" << variances_;
  fs << "log_priors" << logPriors_;
  fs << "training_accuracy" << stats.trainingAccuracy;
  fs << "sample_count" << stats.sampleCount;

  fs << "settings" << "{";
  fs << "intensity" << static_cast<int>(settings.intensity);
  fs << "gaussian3" << static_cast<int>(settings.gaussian3);
  fs << "gaussian7" << static_cast<int>(settings.gaussian7);
  fs << "gradient" << static_cast<int>(settings.gradient);
  fs << "laplacian" << static_cast<int>(settings.laplacian);
  fs << "localMean" << static_cast<int>(settings.localMean);
  fs << "localStd" << static_cast<int>(settings.localStd);
  fs << "xPosition" << static_cast<int>(settings.xPosition);
  fs << "yPosition" << static_cast<int>(settings.yPosition);
  fs << "}";

  fs << "classes" << "[";
  for (const auto& cls : classes) {
    fs << "{";
    fs << "name" << cls.name.toStdString();
    fs << "color_r" << cls.color.red();
    fs << "color_g" << cls.color.green();
    fs << "color_b" << cls.color.blue();
    fs << "enabled" << static_cast<int>(cls.enabled);
    fs << "}";
  }
  fs << "]";
  return true;
}

bool GaussianNaiveBayesModel::load(const QString& path,
                                   std::vector<SegmentationClassInfo>* classes,
                                   SegmentationFeatureSettings* settings,
                                   SegmentationTrainingStats* stats) {
  clear();
  cv::FileStorage fs(path.toStdString(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  fs["class_count"] >> classCount_;
  fs["feature_count"] >> featureCount_;
  fs["means"] >> means_;
  fs["variances"] >> variances_;
  fs["log_priors"] >> logPriors_;
  if (!isValid()) {
    clear();
    return false;
  }

  if (stats) {
    stats->classCount = classCount_;
    fs["training_accuracy"] >> stats->trainingAccuracy;
    fs["sample_count"] >> stats->sampleCount;
  }

  if (settings) {
    const cv::FileNode node = fs["settings"];
    if (!node.empty()) {
      settings->intensity = static_cast<int>(node["intensity"]) != 0;
      settings->gaussian3 = static_cast<int>(node["gaussian3"]) != 0;
      settings->gaussian7 = static_cast<int>(node["gaussian7"]) != 0;
      settings->gradient = static_cast<int>(node["gradient"]) != 0;
      settings->laplacian = static_cast<int>(node["laplacian"]) != 0;
      settings->localMean = static_cast<int>(node["localMean"]) != 0;
      settings->localStd = static_cast<int>(node["localStd"]) != 0;
      settings->xPosition = static_cast<int>(node["xPosition"]) != 0;
      settings->yPosition = static_cast<int>(node["yPosition"]) != 0;
    }
  }

  if (classes) {
    classes->clear();
    for (const auto& item : fs["classes"]) {
      SegmentationClassInfo info;
      info.name = QString::fromStdString(static_cast<std::string>(item["name"]));
      const int r = static_cast<int>(item["color_r"]);
      const int g = static_cast<int>(item["color_g"]);
      const int b = static_cast<int>(item["color_b"]);
      info.color = QColor(r, g, b);
      info.enabled = static_cast<int>(item["enabled"]) != 0;
      classes->push_back(info);
    }
  }

  return true;
}

cv::Mat SegmentationEngine::ensureGrayFloat(const cv::Mat& image) {
  cv::Mat gray;
  if (image.empty()) {
    return gray;
  }
  if (image.channels() == 1) {
    gray = image.clone();
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat grayFloat;
  gray.convertTo(grayFloat, CV_32F, 1.0 / 255.0);
  return grayFloat;
}

cv::Mat SegmentationEngine::computeFeatureStack(const cv::Mat& image,
                                                const SegmentationFeatureSettings& settings) {
  const cv::Mat gray = ensureGrayFloat(image);
  if (gray.empty()) {
    return cv::Mat();
  }

  std::vector<cv::Mat> channels;
  if (settings.intensity) appendFeature(gray, channels);

  if (settings.gaussian3) {
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0.8);
    appendFeature(blur, channels);
  }
  if (settings.gaussian7) {
    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(7, 7), 1.8);
    appendFeature(blur, channels);
  }
  if (settings.gradient) {
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    appendFeature(mag, channels);
  }
  if (settings.laplacian) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F, 3);
    appendFeature(cv::abs(lap), channels);
  }
  if (settings.localMean || settings.localStd) {
    cv::Mat mean, sqMean, variance;
    cv::blur(gray, mean, cv::Size(9, 9));
    cv::blur(gray.mul(gray), sqMean, cv::Size(9, 9));
    variance = sqMean - mean.mul(mean);
    cv::max(variance, 0.0, variance);
    cv::sqrt(variance, variance);
    if (settings.localMean) appendFeature(mean, channels);
    if (settings.localStd) appendFeature(variance, channels);
  }
  if (settings.xPosition || settings.yPosition) {
    cv::Mat x(gray.size(), CV_32F), y(gray.size(), CV_32F);
    for (int row = 0; row < gray.rows; ++row) {
      float* xp = x.ptr<float>(row);
      float* yp = y.ptr<float>(row);
      const float yn = gray.rows > 1 ? static_cast<float>(row) / static_cast<float>(gray.rows - 1) : 0.0f;
      for (int col = 0; col < gray.cols; ++col) {
        xp[col] = gray.cols > 1 ? static_cast<float>(col) / static_cast<float>(gray.cols - 1) : 0.0f;
        yp[col] = yn;
      }
    }
    if (settings.xPosition) appendFeature(x, channels);
    if (settings.yPosition) appendFeature(y, channels);
  }

  if (channels.empty()) {
    appendFeature(gray, channels);
  }

  cv::Mat merged;
  cv::merge(channels, merged);
  return merged.reshape(1, gray.rows * gray.cols).clone();
}

cv::Mat SegmentationEngine::gatherSamples(const cv::Mat& featureStack,
                                          const std::vector<cv::Mat>& classMasks,
                                          cv::Mat* labels,
                                          int maxSamplesPerClass) {
  if (featureStack.empty() || classMasks.empty() || !labels) {
    return cv::Mat();
  }

  std::vector<cv::Mat> rows;
  std::vector<int> outputLabels;
  std::mt19937 rng(12345);

  for (int cls = 0; cls < static_cast<int>(classMasks.size()); ++cls) {
    const cv::Mat& mask = classMasks[cls];
    if (mask.empty()) continue;

    std::vector<int> indices;
    indices.reserve(mask.rows * mask.cols / 8);
    for (int row = 0; row < mask.rows; ++row) {
      const uchar* ptr = mask.ptr<uchar>(row);
      for (int col = 0; col < mask.cols; ++col) {
        if (ptr[col] > 0) {
          indices.push_back(row * mask.cols + col);
        }
      }
    }
    if (indices.empty()) continue;
    std::shuffle(indices.begin(), indices.end(), rng);
    if (static_cast<int>(indices.size()) > maxSamplesPerClass) {
      indices.resize(maxSamplesPerClass);
    }
    for (int idx : indices) {
      rows.push_back(featureStack.row(idx));
      outputLabels.push_back(cls);
    }
  }

  if (rows.empty()) {
    labels->release();
    return cv::Mat();
  }

  cv::Mat samples;
  cv::vconcat(rows, samples);
  *labels = cv::Mat(static_cast<int>(outputLabels.size()), 1, CV_32S);
  for (int i = 0; i < static_cast<int>(outputLabels.size()); ++i) {
    labels->at<int>(i, 0) = outputLabels[i];
  }
  return samples;
}

cv::Mat SegmentationEngine::applyModelLabels(const GaussianNaiveBayesModel& model,
                                             const cv::Mat& featureStack,
                                             int rows,
                                             int cols) {
  cv::Mat labels = model.predictLabels(featureStack);
  if (labels.empty()) {
    return cv::Mat();
  }
  return labels.reshape(1, rows).clone();
}

cv::Mat SegmentationEngine::applyModelProbabilities(const GaussianNaiveBayesModel& model,
                                                    const cv::Mat& featureStack,
                                                    int rows,
                                                    int cols,
                                                    int classIndex) {
  cv::Mat probs = model.predictProbabilities(featureStack);
  if (probs.empty() || classIndex < 0 || classIndex >= probs.cols) {
    return cv::Mat();
  }
  cv::Mat channel(rows, cols, CV_32F);
  for (int row = 0; row < rows; ++row) {
    float* out = channel.ptr<float>(row);
    for (int col = 0; col < cols; ++col) {
      out[col] = probs.at<float>(row * cols + col, classIndex);
    }
  }
  return channel;
}

cv::Mat SegmentationEngine::makeOverlay(const cv::Mat& image,
                                        const cv::Mat& labelMask,
                                        const std::vector<SegmentationClassInfo>& classes,
                                        double alpha,
                                        bool showContours) {
  if (image.empty()) {
    return cv::Mat();
  }
  cv::Mat base;
  if (image.channels() == 1) {
    cv::cvtColor(image, base, cv::COLOR_GRAY2BGR);
  } else {
    base = image.clone();
  }
  if (labelMask.empty()) {
    return base;
  }

  cv::Mat colored = cv::Mat::zeros(base.size(), CV_8UC3);
  for (int row = 0; row < labelMask.rows; ++row) {
    const int* labelPtr = labelMask.ptr<int>(row);
    cv::Vec3b* colorPtr = colored.ptr<cv::Vec3b>(row);
    for (int col = 0; col < labelMask.cols; ++col) {
      const int cls = labelPtr[col];
      if (cls >= 0 && cls < static_cast<int>(classes.size())) {
        const QColor qc = classes[cls].color;
        colorPtr[col] = cv::Vec3b(qc.blue(), qc.green(), qc.red());
      }
    }
  }

  cv::Mat blended;
  cv::addWeighted(base, 1.0 - alpha, colored, alpha, 0.0, blended);

  if (showContours) {
    for (int cls = 0; cls < static_cast<int>(classes.size()); ++cls) {
      cv::Mat mask = (labelMask == cls);
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(blended, contours, -1, toBgr(classes[cls].color).mul(cv::Scalar(0.75, 0.75, 0.75)), 1, cv::LINE_AA);
    }
  }

  return blended;
}
