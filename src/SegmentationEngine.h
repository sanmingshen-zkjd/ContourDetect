#pragma once

#include <QString>
#include <QColor>
#include <QVector>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

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

struct SegmentationClassifierSettings {
  enum Kind {
    GaussianNaiveBayes = 0,
    RandomForest = 1,
    SupportVectorMachine = 2,
  };

  Kind kind = GaussianNaiveBayes;
  int randomForestTrees = 200;
  int randomForestMaxDepth = 20;
  double svmC = 2.0;
  double svmGamma = 0.5;
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

  bool write(cv::FileStorage& fs) const;
  bool read(const cv::FileNode& node);

private:
  int classCount_ = 0;
  int featureCount_ = 0;
  cv::Mat means_;
  cv::Mat variances_;
  cv::Mat logPriors_;
};

class SegmentationClassifier {
public:
  bool isValid() const;
  void clear();

  bool train(const cv::Mat& features,
             const cv::Mat& labels,
             int classCount,
             const SegmentationClassifierSettings& settings,
             SegmentationTrainingStats* stats = nullptr);

  cv::Mat predictLabels(const cv::Mat& features) const;
  cv::Mat predictProbabilities(const cv::Mat& features) const;
  bool supportsProbability() const;

  bool save(const QString& path,
            const std::vector<SegmentationClassInfo>& classes,
            const SegmentationFeatureSettings& featureSettings,
            const SegmentationClassifierSettings& classifierSettings,
            const SegmentationTrainingStats& stats) const;

  bool load(const QString& path,
            std::vector<SegmentationClassInfo>* classes,
            SegmentationFeatureSettings* featureSettings,
            SegmentationClassifierSettings* classifierSettings,
            SegmentationTrainingStats* stats);

  QString classifierName() const;
  SegmentationClassifierSettings::Kind kind() const { return kind_; }

private:
  static QString sidecarModelPath(const QString& metadataPath);
  static bool computeAccuracy(const cv::Mat& labels, const cv::Mat& predicted, SegmentationTrainingStats* stats, int classCount);

  SegmentationClassifierSettings::Kind kind_ = SegmentationClassifierSettings::GaussianNaiveBayes;
  GaussianNaiveBayesModel gnbModel_;
  cv::Ptr<cv::ml::RTrees> randomForest_;
  cv::Ptr<cv::ml::SVM> svm_;
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

  static cv::Mat applyModelLabels(const SegmentationClassifier& model,
                                  const cv::Mat& featureStack,
                                  int rows,
                                  int cols);

  static cv::Mat applyModelProbabilities(const SegmentationClassifier& model,
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
