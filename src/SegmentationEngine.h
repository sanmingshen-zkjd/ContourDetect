#pragma once

#include <QString>
#include <QColor>
#include <QVector>
#include <opencv2/core.hpp>

#include <vector>

struct SegmentationFeatureSettings {
  bool intensity = true;
  bool gaussian3 = true;
  bool gaussian7 = true;
  bool gradient = true;
  bool laplacian = true;
  bool localMean = true;
  bool localStd = true;
  bool xPosition = true;
  bool yPosition = true;
};

struct SegmentationClassInfo {
  QString name;
  QColor color;
  bool enabled = true;
};

struct SegmentationTrainingStats {
  int classCount = 0;
  int sampleCount = 0;
  double trainingAccuracy = 0.0;
};

class GaussianNaiveBayesModel {
public:
  bool isValid() const;
  void clear();

  bool train(const cv::Mat& features,
             const cv::Mat& labels,
             int classCount,
             SegmentationTrainingStats* stats = nullptr);

  cv::Mat predictLabels(const cv::Mat& features) const;
  cv::Mat predictProbabilities(const cv::Mat& features) const;

  bool save(const QString& path,
            const std::vector<SegmentationClassInfo>& classes,
            const SegmentationFeatureSettings& settings,
            const SegmentationTrainingStats& stats) const;

  bool load(const QString& path,
            std::vector<SegmentationClassInfo>* classes,
            SegmentationFeatureSettings* settings,
            SegmentationTrainingStats* stats);

  int classCount() const { return classCount_; }

private:
  int classCount_ = 0;
  int featureCount_ = 0;
  cv::Mat means_;
  cv::Mat variances_;
  cv::Mat logPriors_;
};

class SegmentationEngine {
public:
  static cv::Mat ensureGrayFloat(const cv::Mat& image);

  static cv::Mat computeFeatureStack(const cv::Mat& image,
                                     const SegmentationFeatureSettings& settings);

  static cv::Mat gatherSamples(const cv::Mat& featureStack,
                               const std::vector<cv::Mat>& classMasks,
                               cv::Mat* labels,
                               int maxSamplesPerClass = 12000);

  static cv::Mat applyModelLabels(const GaussianNaiveBayesModel& model,
                                  const cv::Mat& featureStack,
                                  int rows,
                                  int cols);

  static cv::Mat applyModelProbabilities(const GaussianNaiveBayesModel& model,
                                         const cv::Mat& featureStack,
                                         int rows,
                                         int cols,
                                         int classIndex);

  static cv::Mat makeOverlay(const cv::Mat& image,
                             const cv::Mat& labelMask,
                             const std::vector<SegmentationClassInfo>& classes,
                             double alpha,
                             bool showContours);
};
