#include "SegmentationEngine.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/persistence.hpp>

#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

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

cv::Mat rotateKernel(const cv::Mat& kernel, double angleDegrees) {
  cv::Point2f center((kernel.cols - 1) * 0.5f, (kernel.rows - 1) * 0.5f);
  cv::Mat rotation = cv::getRotationMatrix2D(center, angleDegrees, 1.0);
  cv::Mat rotated;
  cv::warpAffine(kernel, rotated, rotation, kernel.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  return rotated;
}

cv::Mat computeLocalEntropy(const cv::Mat& gray, int radius = 4, int bins = 16) {
  cv::Mat entropy(gray.size(), CV_32F, cv::Scalar(0));
  for (int row = 0; row < gray.rows; ++row) {
    float* out = entropy.ptr<float>(row);
    for (int col = 0; col < gray.cols; ++col) {
      std::vector<int> histogram(bins, 0);
      int count = 0;
      for (int y = std::max(0, row - radius); y <= std::min(gray.rows - 1, row + radius); ++y) {
        const float* src = gray.ptr<float>(y);
        for (int x = std::max(0, col - radius); x <= std::min(gray.cols - 1, col + radius); ++x) {
          const int bin = std::clamp(static_cast<int>(src[x] * bins), 0, bins - 1);
          histogram[bin] += 1;
          ++count;
        }
      }
      double value = 0.0;
      for (int frequency : histogram) {
        if (frequency == 0) continue;
        const double probability = static_cast<double>(frequency) / static_cast<double>(count);
        value -= probability * std::log2(probability);
      }
      out[col] = static_cast<float>(value);
    }
  }
  return entropy;
}

cv::Mat computeGaborResponse(const cv::Mat& gray) {
  cv::Mat response(gray.size(), CV_32F, cv::Scalar(0));
  for (double theta : {0.0, CV_PI * 0.25, CV_PI * 0.5, CV_PI * 0.75}) {
    cv::Mat kernel = cv::getGaborKernel(cv::Size(15, 15), 3.0, theta, 8.0, 0.6, 0.0, CV_32F);
    cv::Mat filtered, magnitude;
    cv::filter2D(gray, filtered, CV_32F, kernel);
    magnitude = cv::abs(filtered);
    cv::max(response, magnitude, response);
  }
  return response;
}

cv::Mat computeMembraneResponse(const cv::Mat& gray) {
  cv::Mat baseKernel = cv::Mat::zeros(15, 15, CV_32F);
  baseKernel.row(baseKernel.rows / 2).setTo(1.0f);
  baseKernel /= cv::sum(baseKernel)[0];

  cv::Mat response(gray.size(), CV_32F, cv::Scalar(0));
  for (double angle : {0.0, 30.0, 60.0, 90.0, 120.0, 150.0}) {
    cv::Mat rotatedKernel = rotateKernel(baseKernel, angle);
    cv::Mat filtered, magnitude;
    cv::filter2D(gray, filtered, CV_32F, rotatedKernel);
    magnitude = cv::abs(filtered);
    cv::max(response, magnitude, response);
  }
  return response;
}

QString classifierKindToString(SegmentationClassifierSettings::Kind kind) {
  switch (kind) {
    case SegmentationClassifierSettings::GaussianNaiveBayes: return QStringLiteral("gaussian_naive_bayes");
    case SegmentationClassifierSettings::RandomForest: return QStringLiteral("random_forest");
    case SegmentationClassifierSettings::SupportVectorMachine: return QStringLiteral("svm");
  }
  return QStringLiteral("gaussian_naive_bayes");
}

SegmentationClassifierSettings::Kind classifierKindFromString(const QString& name) {
  if (name == QStringLiteral("random_forest")) return SegmentationClassifierSettings::RandomForest;
  if (name == QStringLiteral("svm")) return SegmentationClassifierSettings::SupportVectorMachine;
  return SegmentationClassifierSettings::GaussianNaiveBayes;
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

bool GaussianNaiveBayesModel::write(cv::FileStorage& fs) const {
  if (!isValid()) {
    return false;
  }
  fs << "class_count" << classCount_;
  fs << "feature_count" << featureCount_;
  fs << "means" << means_;
  fs << "variances" << variances_;
  fs << "log_priors" << logPriors_;
  return true;
}

bool GaussianNaiveBayesModel::read(const cv::FileNode& node) {
  clear();
  if (node.empty()) {
    return false;
  }
  node["class_count"] >> classCount_;
  node["feature_count"] >> featureCount_;
  node["means"] >> means_;
  node["variances"] >> variances_;
  node["log_priors"] >> logPriors_;
  if (!isValid()) {
    clear();
    return false;
  }
  return true;
}

bool SegmentationClassifier::isValid() const {
  switch (kind_) {
    case SegmentationClassifierSettings::GaussianNaiveBayes:
      return gnbModel_.isValid();
    case SegmentationClassifierSettings::RandomForest:
      return randomForest_ && !randomForest_->empty();
    case SegmentationClassifierSettings::SupportVectorMachine:
      return svm_ && !svm_->empty();
  }
  return false;
}

void SegmentationClassifier::clear() {
  kind_ = SegmentationClassifierSettings::GaussianNaiveBayes;
  gnbModel_.clear();
  randomForest_.release();
  svm_.release();
}

bool SegmentationClassifier::computeAccuracy(const cv::Mat& labels,
                                             const cv::Mat& predicted,
                                             SegmentationTrainingStats* stats,
                                             int classCount) {
  if (!stats || labels.empty() || predicted.empty() || labels.rows != predicted.rows) {
    return true;
  }
  int correct = 0;
  for (int row = 0; row < labels.rows; ++row) {
    if (labels.at<int>(row, 0) == predicted.at<int>(row, 0)) {
      ++correct;
    }
  }
  stats->classCount = classCount;
  stats->sampleCount = labels.rows;
  stats->trainingAccuracy = labels.rows > 0 ? static_cast<double>(correct) / labels.rows : 0.0;
  return true;
}

bool SegmentationClassifier::train(const cv::Mat& features,
                                   const cv::Mat& labels,
                                   int classCount,
                                   const SegmentationClassifierSettings& settings,
                                   SegmentationTrainingStats* stats) {
  clear();
  kind_ = settings.kind;
  if (stats) {
    *stats = {};
  }

  if (kind_ == SegmentationClassifierSettings::GaussianNaiveBayes) {
    return gnbModel_.train(features, labels, classCount, stats);
  }

  if (features.empty() || labels.empty()) {
    return false;
  }

  cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);
  if (!trainData) {
    return false;
  }

  if (kind_ == SegmentationClassifierSettings::RandomForest) {
    randomForest_ = cv::ml::RTrees::create();
    randomForest_->setMaxDepth(settings.randomForestMaxDepth);
    randomForest_->setMinSampleCount(2);
    randomForest_->setRegressionAccuracy(0.0f);
    randomForest_->setUseSurrogates(false);
    randomForest_->setMaxCategories(std::max(2, classCount));
    randomForest_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, settings.randomForestTrees, 0));
    if (!randomForest_->train(trainData)) {
      randomForest_.release();
      return false;
    }
    const cv::Mat predicted = predictLabels(features);
    return computeAccuracy(labels, predicted, stats, classCount);
  }

  if (kind_ == SegmentationClassifierSettings::SupportVectorMachine) {
    svm_ = cv::ml::SVM::create();
    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setKernel(cv::ml::SVM::RBF);
    svm_->setC(settings.svmC);
    svm_->setGamma(settings.svmGamma);
    svm_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 500, 1e-4));
    if (!svm_->train(trainData)) {
      svm_.release();
      return false;
    }
    const cv::Mat predicted = predictLabels(features);
    return computeAccuracy(labels, predicted, stats, classCount);
  }

  return false;
}

cv::Mat SegmentationClassifier::predictLabels(const cv::Mat& features) const {
  cv::Mat labels;
  if (!isValid() || features.empty()) {
    return labels;
  }

  if (kind_ == SegmentationClassifierSettings::GaussianNaiveBayes) {
    return gnbModel_.predictLabels(features);
  }

  labels = cv::Mat::zeros(features.rows, 1, CV_32S);
  for (int row = 0; row < features.rows; ++row) {
    float predicted = 0.0f;
    if (kind_ == SegmentationClassifierSettings::RandomForest) {
      predicted = randomForest_->predict(features.row(row));
    } else if (kind_ == SegmentationClassifierSettings::SupportVectorMachine) {
      predicted = svm_->predict(features.row(row));
    }
    labels.at<int>(row, 0) = static_cast<int>(std::lround(predicted));
  }
  return labels;
}

cv::Mat SegmentationClassifier::predictProbabilities(const cv::Mat& features) const {
  if (!isValid() || features.empty()) {
    return cv::Mat();
  }
  if (kind_ == SegmentationClassifierSettings::GaussianNaiveBayes) {
    return gnbModel_.predictProbabilities(features);
  }
  if (kind_ != SegmentationClassifierSettings::RandomForest || !randomForest_) {
    return cv::Mat();
  }

  cv::Mat votes;
  randomForest_->getVotes(features, votes, cv::ml::DTrees::PREDICT_AUTO);
  if (votes.empty() || votes.rows != features.rows + 1 || votes.cols <= 0) {
    return cv::Mat();
  }

  cv::Mat votesFloat;
  votes.convertTo(votesFloat, CV_32F);
  cv::Mat probs = cv::Mat::zeros(features.rows, votes.cols, CV_32F);
  for (int col = 0; col < votesFloat.cols; ++col) {
    const int classLabel = static_cast<int>(std::lround(votesFloat.at<float>(0, col)));
    if (classLabel < 0 || classLabel >= probs.cols) {
      continue;
    }
    for (int row = 1; row < votesFloat.rows; ++row) {
      probs.at<float>(row - 1, classLabel) = votesFloat.at<float>(row, col);
    }
  }

  for (int row = 0; row < probs.rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < probs.cols; ++col) {
      sum += probs.at<float>(row, col);
    }
    if (sum <= 0.0f) {
      continue;
    }
    const float inv = 1.0f / sum;
    for (int col = 0; col < probs.cols; ++col) {
      probs.at<float>(row, col) *= inv;
    }
  }
  return probs;
}

bool SegmentationClassifier::supportsProbability() const {
  return (kind_ == SegmentationClassifierSettings::GaussianNaiveBayes && gnbModel_.isValid())
      || (kind_ == SegmentationClassifierSettings::RandomForest && randomForest_ && !randomForest_->empty());
}

QString SegmentationClassifier::sidecarModelPath(const QString& metadataPath) {
  return metadataPath + QStringLiteral(".model.yml");
}

bool SegmentationClassifier::save(const QString& path,
                                  const std::vector<SegmentationClassInfo>& classes,
                                  const SegmentationFeatureSettings& featureSettings,
                                  const SegmentationClassifierSettings& classifierSettings,
                                  const SegmentationTrainingStats& stats) const {
  if (!isValid()) {
    return false;
  }

  cv::FileStorage fs(path.toStdString(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    return false;
  }

  fs << "classifier_kind" << classifierKindToString(kind_).toStdString();
  fs << "training_accuracy" << stats.trainingAccuracy;
  fs << "sample_count" << stats.sampleCount;
  fs << "class_count" << stats.classCount;

  fs << "feature_settings" << "{";
  fs << "intensity" << static_cast<int>(featureSettings.intensity);
  fs << "gaussian3" << static_cast<int>(featureSettings.gaussian3);
  fs << "gaussian7" << static_cast<int>(featureSettings.gaussian7);
  fs << "differenceOfGaussians" << static_cast<int>(featureSettings.differenceOfGaussians);
  fs << "gradient" << static_cast<int>(featureSettings.gradient);
  fs << "laplacian" << static_cast<int>(featureSettings.laplacian);
  fs << "hessian" << static_cast<int>(featureSettings.hessian);
  fs << "localMean" << static_cast<int>(featureSettings.localMean);
  fs << "localStd" << static_cast<int>(featureSettings.localStd);
  fs << "entropy" << static_cast<int>(featureSettings.entropy);
  fs << "texture" << static_cast<int>(featureSettings.texture);
  fs << "gabor" << static_cast<int>(featureSettings.gabor);
  fs << "membrane" << static_cast<int>(featureSettings.membrane);
  fs << "xPosition" << static_cast<int>(featureSettings.xPosition);
  fs << "yPosition" << static_cast<int>(featureSettings.yPosition);
  fs << "}";

  fs << "classifier_settings" << "{";
  fs << "kind" << classifierKindToString(classifierSettings.kind).toStdString();
  fs << "randomForestTrees" << classifierSettings.randomForestTrees;
  fs << "randomForestMaxDepth" << classifierSettings.randomForestMaxDepth;
  fs << "svmC" << classifierSettings.svmC;
  fs << "svmGamma" << classifierSettings.svmGamma;
  fs << "balanceClasses" << static_cast<int>(classifierSettings.balanceClasses);
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

  if (kind_ == SegmentationClassifierSettings::GaussianNaiveBayes) {
    fs << "gaussian_nb" << "{";
    gnbModel_.write(fs);
    fs << "}";
  } else {
    const QString sidecar = sidecarModelPath(path);
    fs << "opencv_sidecar_model" << sidecar.toStdString();
    if (kind_ == SegmentationClassifierSettings::RandomForest) {
      randomForest_->save(sidecar.toStdString());
    } else if (kind_ == SegmentationClassifierSettings::SupportVectorMachine) {
      svm_->save(sidecar.toStdString());
    }
  }
  return true;
}

bool SegmentationClassifier::load(const QString& path,
                                  std::vector<SegmentationClassInfo>* classes,
                                  SegmentationFeatureSettings* featureSettings,
                                  SegmentationClassifierSettings* classifierSettings,
                                  SegmentationTrainingStats* stats) {
  clear();
  cv::FileStorage fs(path.toStdString(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  const cv::FileNode kindNode = fs["classifier_kind"];
  const bool hasExplicitKind = !kindNode.empty();
  const QString kindName = hasExplicitKind
                               ? QString::fromStdString(static_cast<std::string>(kindNode))
                               : QString();
  kind_ = hasExplicitKind ? classifierKindFromString(kindName)
                          : SegmentationClassifierSettings::GaussianNaiveBayes;

  if (stats) {
    fs["training_accuracy"] >> stats->trainingAccuracy;
    fs["sample_count"] >> stats->sampleCount;
    fs["class_count"] >> stats->classCount;
  }

  if (featureSettings) {
    *featureSettings = {};
    cv::FileNode node = fs["feature_settings"];
    if (node.empty()) {
      node = fs["settings"];
    }
    if (!node.empty()) {
      featureSettings->intensity = static_cast<int>(node["intensity"]) != 0;
      featureSettings->gaussian3 = static_cast<int>(node["gaussian3"]) != 0;
      featureSettings->gaussian7 = static_cast<int>(node["gaussian7"]) != 0;
      featureSettings->differenceOfGaussians = static_cast<int>(node["differenceOfGaussians"]) != 0;
      featureSettings->gradient = static_cast<int>(node["gradient"]) != 0;
      featureSettings->laplacian = static_cast<int>(node["laplacian"]) != 0;
      featureSettings->hessian = static_cast<int>(node["hessian"]) != 0;
      featureSettings->localMean = static_cast<int>(node["localMean"]) != 0;
      featureSettings->localStd = static_cast<int>(node["localStd"]) != 0;
      featureSettings->entropy = static_cast<int>(node["entropy"]) != 0;
      featureSettings->texture = static_cast<int>(node["texture"]) != 0;
      featureSettings->gabor = static_cast<int>(node["gabor"]) != 0;
      featureSettings->membrane = static_cast<int>(node["membrane"]) != 0;
      featureSettings->xPosition = static_cast<int>(node["xPosition"]) != 0;
      featureSettings->yPosition = static_cast<int>(node["yPosition"]) != 0;
    }
  }

  if (classifierSettings) {
    *classifierSettings = {};
    const cv::FileNode node = fs["classifier_settings"];
    if (!node.empty()) {
      classifierSettings->kind = classifierKindFromString(QString::fromStdString(static_cast<std::string>(node["kind"])));
      classifierSettings->randomForestTrees = static_cast<int>(node["randomForestTrees"]);
      classifierSettings->randomForestMaxDepth = static_cast<int>(node["randomForestMaxDepth"]);
      classifierSettings->svmC = static_cast<double>(node["svmC"]);
      classifierSettings->svmGamma = static_cast<double>(node["svmGamma"]);
      classifierSettings->balanceClasses = static_cast<int>(node["balanceClasses"]) != 0;
    } else {
      classifierSettings->kind = kind_;
    }
  }

  if (classes) {
    classes->clear();
    for (const auto& item : fs["classes"]) {
      SegmentationClassInfo info;
      info.name = QString::fromStdString(static_cast<std::string>(item["name"]));
      info.color = QColor(static_cast<int>(item["color_r"]), static_cast<int>(item["color_g"]), static_cast<int>(item["color_b"]));
      info.enabled = static_cast<int>(item["enabled"]) != 0;
      classes->push_back(info);
    }
  }

  if (kind_ == SegmentationClassifierSettings::GaussianNaiveBayes) {
    const cv::FileNode legacyOrNestedNode = hasExplicitKind ? fs["gaussian_nb"] : fs.root();
    return gnbModel_.read(legacyOrNestedNode);
  }

  const QString sidecar = QString::fromStdString(static_cast<std::string>(fs["opencv_sidecar_model"]));
  if (sidecar.isEmpty()) {
    return false;
  }
  if (kind_ == SegmentationClassifierSettings::RandomForest) {
    randomForest_ = cv::Algorithm::load<cv::ml::RTrees>(sidecar.toStdString());
    return randomForest_ && !randomForest_->empty();
  }
  if (kind_ == SegmentationClassifierSettings::SupportVectorMachine) {
    svm_ = cv::Algorithm::load<cv::ml::SVM>(sidecar.toStdString());
    return svm_ && !svm_->empty();
  }
  return false;
}

QString SegmentationClassifier::classifierName() const {
  switch (kind_) {
    case SegmentationClassifierSettings::GaussianNaiveBayes: return QStringLiteral("Gaussian Naive Bayes");
    case SegmentationClassifierSettings::RandomForest: return QStringLiteral("Random Forest");
    case SegmentationClassifierSettings::SupportVectorMachine: return QStringLiteral("SVM (RBF)");
  }
  return QStringLiteral("Unknown");
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
  if (settings.differenceOfGaussians) {
    cv::Mat blurSmall, blurLarge, dog;
    cv::GaussianBlur(gray, blurSmall, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(gray, blurLarge, cv::Size(11, 11), 2.5);
    cv::absdiff(blurSmall, blurLarge, dog);
    appendFeature(dog, channels);
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
  if (settings.hessian) {
    cv::Mat dxx, dxy, dyy, hessianNorm;
    cv::Sobel(gray, dxx, CV_32F, 2, 0, 3);
    cv::Sobel(gray, dxy, CV_32F, 1, 1, 3);
    cv::Sobel(gray, dyy, CV_32F, 0, 2, 3);
    hessianNorm = dxx.mul(dxx) + dyy.mul(dyy) + dxy.mul(dxy) * 2.0f;
    cv::sqrt(hessianNorm, hessianNorm);
    appendFeature(hessianNorm, channels);
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
  if (settings.entropy) {
    appendFeature(computeLocalEntropy(gray), channels);
  }
  if (settings.texture) {
    cv::Mat gx, gy, mag, textureEnergy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    cv::blur(mag.mul(mag), textureEnergy, cv::Size(9, 9));
    appendFeature(textureEnergy, channels);
  }
  if (settings.gabor) {
    appendFeature(computeGaborResponse(gray), channels);
  }
  if (settings.membrane) {
    appendFeature(computeMembraneResponse(gray), channels);
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
                                          int maxSamplesPerClass,
                                          bool balanceClasses) {
  if (featureStack.empty() || classMasks.empty() || !labels) {
    return cv::Mat();
  }

  std::vector<cv::Mat> rows;
  std::vector<int> outputLabels;
  std::mt19937 rng(12345);
  std::vector<std::vector<int>> sampledIndices(classMasks.size());
  int balancedTarget = std::numeric_limits<int>::max();

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
    balancedTarget = std::min(balancedTarget, static_cast<int>(indices.size()));
    sampledIndices[cls] = std::move(indices);
  }

  if (balanceClasses && balancedTarget > 0 && balancedTarget != std::numeric_limits<int>::max()) {
    for (auto& indices : sampledIndices) {
      if (!indices.empty() && static_cast<int>(indices.size()) > balancedTarget) {
        indices.resize(balancedTarget);
      }
    }
  }

  for (int cls = 0; cls < static_cast<int>(sampledIndices.size()); ++cls) {
    for (int idx : sampledIndices[cls]) {
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

cv::Mat SegmentationEngine::applyModelLabels(const SegmentationClassifier& model,
                                             const cv::Mat& featureStack,
                                             int rows,
                                             int cols) {
  (void)cols;
  cv::Mat labels = model.predictLabels(featureStack);
  if (labels.empty()) {
    return cv::Mat();
  }
  return labels.reshape(1, rows).clone();
}

cv::Mat SegmentationEngine::applyModelProbabilities(const SegmentationClassifier& model,
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
