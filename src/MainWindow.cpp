#include "MainWindow.h"

#include "ImageIOUtils.h"
#include "QCustomPlot.h"
#include "SegmentationView.h"

#include <QApplication>
#include <QBoxLayout>
#include <QColorDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QEventLoop>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPainter>
#include <QProgressDialog>
#include <QScreen>
#include <QSplitter>
#include <QTemporaryDir>
#include <QTextEdit>
#include <QTextStream>
#include <QToolBar>
#include <QDateTime>
#include <QGuiApplication>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QDirIterator>
#include <QtConcurrent>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>

namespace {
std::string cvPath(const QString& path) {
  return QFile::encodeName(path).toStdString();
}

QString nowString() {
  return QDateTime::currentDateTime().toString("hh:mm:ss");
}

QColor defaultColorForIndex(int index) {
  static const QVector<QColor> palette = {
      QColor(224, 72, 72), QColor(73, 201, 118), QColor(72, 130, 235),
      QColor(235, 183, 72), QColor(188, 72, 235), QColor(54, 200, 211)};
  return palette[index % palette.size()];
}

std::vector<cv::Mat> cloneMaskVector(const std::vector<cv::Mat>& source) {
  std::vector<cv::Mat> cloned;
  cloned.reserve(source.size());
  for (const cv::Mat& mask : source) cloned.push_back(mask.clone());
  return cloned;
}

std::vector<std::vector<TraceRegion>> cloneTraceRegions(const std::vector<std::vector<TraceRegion>>& source) {
  return source;
}

QString sessionDataRoot() {
  const QString root = QDir(QDir::tempPath()).filePath("qt_trainable_segmentation_project");
  QDir().mkpath(root);
  return root;
}

QPainterPath pointsToPath(const QVector<QPointF>& points) {
  QPainterPath path;
  if (points.isEmpty()) return path;
  path.moveTo(points.first());
  for (int i = 1; i < points.size(); ++i) {
    path.lineTo(points[i]);
  }
  return path;
}

struct BinaryMetrics {
  QVector<QPointF> roc;
  QVector<QPointF> pr;
  double auc = 0.0;
  double ap = 0.0;
};

BinaryMetrics computeBinaryCurves(const std::vector<float>& scores, const std::vector<int>& truth) {
  BinaryMetrics metrics;
  if (scores.empty() || scores.size() != truth.size()) {
    return metrics;
  }
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

  const double positives = std::count(truth.begin(), truth.end(), 1);
  const double negatives = static_cast<double>(truth.size()) - positives;
  double tp = 0.0;
  double fp = 0.0;
  metrics.roc.push_back(QPointF(0.0, 0.0));
  metrics.pr.push_back(QPointF(0.0, positives > 0.0 ? 1.0 : 0.0));
  for (int idx : order) {
    if (truth[idx] == 1) {
      tp += 1.0;
    } else {
      fp += 1.0;
    }
    const double tpr = positives > 0.0 ? tp / positives : 0.0;
    const double fpr = negatives > 0.0 ? fp / negatives : 0.0;
    const double precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 1.0;
    const double recall = positives > 0.0 ? tp / positives : 0.0;
    metrics.roc.push_back(QPointF(fpr, tpr));
    metrics.pr.push_back(QPointF(recall, precision));
  }
  metrics.roc.push_back(QPointF(1.0, 1.0));

  for (int i = 1; i < metrics.roc.size(); ++i) {
    const QPointF a = metrics.roc[i - 1];
    const QPointF b = metrics.roc[i];
    metrics.auc += (b.x() - a.x()) * (a.y() + b.y()) * 0.5;
  }
  for (int i = 1; i < metrics.pr.size(); ++i) {
    const QPointF a = metrics.pr[i - 1];
    const QPointF b = metrics.pr[i];
    metrics.ap += std::abs(b.x() - a.x()) * (a.y() + b.y()) * 0.5;
  }
  return metrics;
}

QJsonObject featureSettingsToJson(const SegmentationFeatureSettings& settings) {
  return QJsonObject{{"intensity", settings.intensity},
                     {"gaussian3", settings.gaussian3},
                     {"gaussian7", settings.gaussian7},
                     {"gaussian15", settings.gaussian15},
                     {"differenceOfGaussians", settings.differenceOfGaussians},
                     {"minimum", settings.minimum},
                     {"maximum", settings.maximum},
                     {"median", settings.median},
                     {"bilateral", settings.bilateral},
                     {"gradient", settings.gradient},
                     {"laplacian", settings.laplacian},
                     {"laplacianOfGaussian", settings.laplacianOfGaussian},
                     {"hessian", settings.hessian},
                     {"localMean", settings.localMean},
                     {"localStd", settings.localStd},
                     {"localVariance", settings.localVariance},
                     {"entropy", settings.entropy},
                     {"texture", settings.texture},
                     {"clahe", settings.clahe},
                     {"canny", settings.canny},
                     {"structureTensor", settings.structureTensor},
                     {"gabor", settings.gabor},
                     {"membrane", settings.membrane},
                     {"channelRatios", settings.channelRatios},
                     {"xPosition", settings.xPosition},
                     {"yPosition", settings.yPosition}};
}

void loadFeatureSettingsFromJson(const QJsonObject& settingsObject, SegmentationFeatureSettings* settings) {
  if (!settings) return;
  settings->intensity = settingsObject["intensity"].toBool(true);
  settings->gaussian3 = settingsObject["gaussian3"].toBool(true);
  settings->gaussian7 = settingsObject["gaussian7"].toBool(true);
  settings->gaussian15 = settingsObject["gaussian15"].toBool(false);
  settings->differenceOfGaussians = settingsObject["differenceOfGaussians"].toBool(false);
  settings->minimum = settingsObject["minimum"].toBool(false);
  settings->maximum = settingsObject["maximum"].toBool(false);
  settings->median = settingsObject["median"].toBool(false);
  settings->bilateral = settingsObject["bilateral"].toBool(false);
  settings->gradient = settingsObject["gradient"].toBool(true);
  settings->laplacian = settingsObject["laplacian"].toBool(true);
  settings->laplacianOfGaussian = settingsObject["laplacianOfGaussian"].toBool(false);
  settings->hessian = settingsObject["hessian"].toBool(false);
  settings->localMean = settingsObject["localMean"].toBool(true);
  settings->localStd = settingsObject["localStd"].toBool(true);
  settings->localVariance = settingsObject["localVariance"].toBool(false);
  settings->entropy = settingsObject["entropy"].toBool(false);
  settings->texture = settingsObject["texture"].toBool(false);
  settings->clahe = settingsObject["clahe"].toBool(false);
  settings->canny = settingsObject["canny"].toBool(false);
  settings->structureTensor = settingsObject["structureTensor"].toBool(false);
  settings->gabor = settingsObject["gabor"].toBool(false);
  settings->membrane = settingsObject["membrane"].toBool(false);
  settings->channelRatios = settingsObject["channelRatios"].toBool(false);
  settings->xPosition = settingsObject["xPosition"].toBool(true);
  settings->yPosition = settingsObject["yPosition"].toBool(true);
}

QString arffEscape(const QString& value) {
  QString escaped = value;
  escaped.replace("\\", "\\\\");
  escaped.replace("'", "\\'");
  return QString("'%1'").arg(escaped);
}
}

struct ProjectTrainingBundle {
  QString imagePath;
  std::vector<cv::Mat> slices;
  std::vector<AnnotationSnapshot> annotations;
};

cv::Mat buildExclusionMask(const cv::Size& size, const std::vector<QPolygon>& regions);
cv::Mat buildInferenceMask(const cv::Size& size, const std::vector<QPolygon>& regions);

QString classifierRoiPath(const QString& classifierPath) {
  return classifierPath + QStringLiteral(".roi.json");
}

QJsonArray polygonToJson(const QPolygon& polygon) {
  QJsonArray points;
  for (const QPoint& point : polygon) {
    points.append(QJsonObject{{"x", point.x()}, {"y", point.y()}});
  }
  return points;
}

QPolygon polygonFromJson(const QJsonArray& points) {
  QPolygon polygon;
  for (const auto& value : points) {
    const QJsonObject pointObject = value.toObject();
    polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
  }
  return polygon;
}

bool saveClassifierRoiConfig(const QString& classifierPath,
                             const std::vector<AnnotationSnapshot>& sliceStates,
                             int currentSliceIndex) {
  QJsonObject root;
  root["version"] = 1;
  root["currentSliceIndex"] = currentSliceIndex;
  QJsonArray slices;
  for (int sliceIndex = 0; sliceIndex < static_cast<int>(sliceStates.size()); ++sliceIndex) {
    const AnnotationSnapshot& state = sliceStates[sliceIndex];
    QJsonObject sliceObject;
    sliceObject["index"] = sliceIndex;
    QJsonArray classTraces;
    for (const auto& traceList : state.classTraceRegions) {
      QJsonArray traces;
      for (const TraceRegion& region : traceList) {
        traces.append(QJsonObject{{"name", region.name}, {"points", polygonToJson(region.polygon)}});
      }
      classTraces.append(traces);
    }
    QJsonArray inferenceRegions;
    for (const QPolygon& polygon : state.inferenceRegions) inferenceRegions.append(polygonToJson(polygon));
    QJsonArray exclusionRegions;
    for (const QPolygon& polygon : state.exclusionRegions) exclusionRegions.append(polygonToJson(polygon));
    sliceObject["classTraces"] = classTraces;
    sliceObject["inferenceRegions"] = inferenceRegions;
    sliceObject["exclusionRegions"] = exclusionRegions;
    slices.append(sliceObject);
  }
  root["slices"] = slices;
  QFile file(classifierRoiPath(classifierPath));
  if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
    return false;
  }
  file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  return true;
}

bool loadClassifierRoiConfig(const QString& classifierPath,
                             int classCount,
                             std::vector<AnnotationSnapshot>* sliceStates,
                             int* currentSliceIndex) {
  if (!sliceStates) return false;
  QFile file(classifierRoiPath(classifierPath));
  if (!file.exists()) return false;
  if (!file.open(QIODevice::ReadOnly)) return false;
  const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (!doc.isObject()) return false;
  const QJsonObject root = doc.object();
  const QJsonArray slices = root["slices"].toArray();
  for (const auto& sliceValue : slices) {
    const QJsonObject sliceObject = sliceValue.toObject();
    const int sliceIndex = sliceObject["index"].toInt(-1);
    if (sliceIndex < 0 || sliceIndex >= static_cast<int>(sliceStates->size())) continue;
    AnnotationSnapshot& state = (*sliceStates)[sliceIndex];
    state.classTraceRegions.assign(classCount, {});
    const QJsonArray classTraces = sliceObject["classTraces"].toArray();
    for (int classIndex = 0; classIndex < classTraces.size() && classIndex < classCount; ++classIndex) {
      for (const auto& traceValue : classTraces[classIndex].toArray()) {
        const QJsonObject traceObject = traceValue.toObject();
        const QPolygon polygon = polygonFromJson(traceObject["points"].toArray());
        if (polygon.size() < 3) continue;
        state.classTraceRegions[classIndex].push_back({traceObject["name"].toString(), polygon});
      }
    }
    state.inferenceRegions.clear();
    for (const auto& value : sliceObject["inferenceRegions"].toArray()) {
      const QPolygon polygon = polygonFromJson(value.toArray());
      if (polygon.size() >= 3) state.inferenceRegions.push_back(polygon);
    }
    state.exclusionRegions.clear();
    for (const auto& value : sliceObject["exclusionRegions"].toArray()) {
      const QPolygon polygon = polygonFromJson(value.toArray());
      if (polygon.size() >= 3) state.exclusionRegions.push_back(polygon);
    }
  }
  if (currentSliceIndex) {
    *currentSliceIndex = root["currentSliceIndex"].toInt(*currentSliceIndex);
  }
  return true;
}

bool loadProjectTrainingBundle(const QString& path,
                               const std::vector<SegmentationClassInfo>& expectedClasses,
                               ProjectTrainingBundle* bundle) {
  if (!bundle) return false;
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) return false;
  const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (!doc.isObject()) return false;
  const QJsonObject root = doc.object();
  const QString imagePath = root["imagePath"].toString();
  if (imagePath.isEmpty()) return false;
  std::vector<cv::Mat> slices;
  QString error;
  if (!ImageIOUtils::loadImageVolume(imagePath, &slices, &error) || slices.empty()) return false;

  const QJsonArray classes = root["classes"].toArray();
  if (!expectedClasses.empty() && classes.size() != static_cast<int>(expectedClasses.size())) return false;

  bundle->imagePath = imagePath;
  bundle->slices = std::move(slices);
  bundle->annotations.assign(bundle->slices.size(), {});
  const QJsonArray slicesArray = root["slices"].toArray();
  for (const auto& sliceValue : slicesArray) {
    const QJsonObject sliceObject = sliceValue.toObject();
    const int sliceIndex = sliceObject["index"].toInt();
    if (sliceIndex < 0 || sliceIndex >= static_cast<int>(bundle->annotations.size())) continue;
    AnnotationSnapshot snapshot;
    snapshot.classBrushMasks.assign(classes.size(), cv::Mat());
    snapshot.classTraceRegions.assign(classes.size(), {});
    const QJsonArray classEntries = sliceObject["classes"].toArray();
    for (int classIndex = 0; classIndex < classEntries.size(); ++classIndex) {
      const QJsonObject clsObj = classEntries[classIndex].toObject();
      cv::Mat mask = cv::imread(cvPath(QFileInfo(path).dir().filePath(clsObj["mask"].toString())), cv::IMREAD_GRAYSCALE);
      if (mask.empty()) {
        mask = cv::Mat(bundle->slices[sliceIndex].rows, bundle->slices[sliceIndex].cols, CV_8U, cv::Scalar(0));
      }
      snapshot.classBrushMasks[classIndex] = mask;
      for (const auto& traceValue : clsObj["traces"].toArray()) {
        const QJsonObject traceObject = traceValue.toObject();
        QPolygon polygon;
        for (const auto& pointValue : traceObject["points"].toArray()) {
          const QJsonObject pointObject = pointValue.toObject();
          polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
        }
        snapshot.classTraceRegions[classIndex].push_back({traceObject["name"].toString(), polygon});
      }
    }
    for (const auto& regionValue : sliceObject["inferenceRegions"].toArray()) {
      QPolygon polygon;
      for (const auto& pointValue : regionValue.toArray()) {
        const QJsonObject pointObject = pointValue.toObject();
        polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
      }
      if (polygon.size() >= 3) snapshot.inferenceRegions.push_back(polygon);
    }
    for (const auto& regionValue : sliceObject["exclusionRegions"].toArray()) {
      QPolygon polygon;
      for (const auto& pointValue : regionValue.toArray()) {
        const QJsonObject pointObject = pointValue.toObject();
        polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
      }
      if (polygon.size() >= 3) snapshot.exclusionRegions.push_back(polygon);
    }
    bundle->annotations[sliceIndex] = snapshot;
  }
  return true;
}


std::vector<cv::Mat> masksFromAnnotationSnapshot(const AnnotationSnapshot& snapshot, const cv::Size& size, int classCount) {
  std::vector<cv::Mat> masks(classCount);
  for (int cls = 0; cls < classCount; ++cls) {
    if (cls < static_cast<int>(snapshot.classBrushMasks.size()) && !snapshot.classBrushMasks[cls].empty()) {
      masks[cls] = snapshot.classBrushMasks[cls].clone();
    } else {
      masks[cls] = cv::Mat(size, CV_8U, cv::Scalar(0));
    }
  }
  for (int cls = 0; cls < classCount && cls < static_cast<int>(snapshot.classTraceRegions.size()); ++cls) {
    for (const TraceRegion& region : snapshot.classTraceRegions[cls]) {
      if (region.polygon.size() < 3) continue;
      std::vector<cv::Point> points;
      points.reserve(region.polygon.size());
      for (const QPoint& pt : region.polygon) points.emplace_back(pt.x(), pt.y());
      std::vector<std::vector<cv::Point>> polygons{points};
      cv::Mat traceMask(size, CV_8U, cv::Scalar(0));
      cv::fillPoly(traceMask, polygons, cv::Scalar(255), cv::LINE_AA);
      for (int other = 0; other < classCount; ++other) {
        if (other == cls) continue;
        masks[other].setTo(cv::Scalar(0), traceMask);
      }
      masks[cls].setTo(cv::Scalar(255), traceMask);
    }
  }
  const cv::Mat inferenceMask = buildInferenceMask(size, snapshot.inferenceRegions);
  if (!inferenceMask.empty()) {
    cv::Mat outsideMask;
    cv::bitwise_not(inferenceMask, outsideMask);
    for (auto& mask : masks) {
      if (!mask.empty()) mask.setTo(cv::Scalar(0), outsideMask);
    }
  }
  const cv::Mat exclusionMask = buildExclusionMask(size, snapshot.exclusionRegions);
  if (!exclusionMask.empty()) {
    for (auto& mask : masks) {
      if (!mask.empty()) mask.setTo(cv::Scalar(0), exclusionMask);
    }
  }
  return masks;
}

cv::Mat buildExclusionMask(const cv::Size& size, const std::vector<QPolygon>& regions) {
  cv::Mat mask(size, CV_8U, cv::Scalar(0));
  for (const QPolygon& polygonQt : regions) {
    if (polygonQt.size() < 3) continue;
    std::vector<cv::Point> points;
    points.reserve(polygonQt.size());
    for (const QPoint& pt : polygonQt) points.emplace_back(pt.x(), pt.y());
    std::vector<std::vector<cv::Point>> polygons{points};
    cv::fillPoly(mask, polygons, cv::Scalar(255), cv::LINE_AA);
  }
  return mask;
}

cv::Mat buildInferenceMask(const cv::Size& size, const std::vector<QPolygon>& regions) {
  if (regions.empty()) return cv::Mat();
  cv::Mat mask(size, CV_8U, cv::Scalar(0));
  for (const QPolygon& polygonQt : regions) {
    if (polygonQt.size() < 3) continue;
    std::vector<cv::Point> points;
    points.reserve(polygonQt.size());
    for (const QPoint& pt : polygonQt) points.emplace_back(pt.x(), pt.y());
    std::vector<std::vector<cv::Point>> polygons{points};
    cv::fillPoly(mask, polygons, cv::Scalar(255), cv::LINE_AA);
  }
  return mask;
}

cv::Mat applyModelLabelsWithExclusion(const SegmentationClassifier& model,
                                      const cv::Mat& featureStack,
                                      int rows,
                                      int cols,
                                      const cv::Mat& inferenceMask,
                                      const cv::Mat& exclusionMask) {
  if (inferenceMask.empty() && exclusionMask.empty()) {
    return SegmentationEngine::applyModelLabels(model, featureStack, rows, cols);
  }
  std::vector<int> active;
  active.reserve(rows * cols);
  for (int row = 0; row < rows; ++row) {
    const uchar* includePtr = inferenceMask.empty() ? nullptr : inferenceMask.ptr<uchar>(row);
    const uchar* excludePtr = exclusionMask.empty() ? nullptr : exclusionMask.ptr<uchar>(row);
    for (int col = 0; col < cols; ++col) {
      const bool included = !includePtr || includePtr[col] > 0;
      const bool excluded = excludePtr && excludePtr[col] > 0;
      if (included && !excluded) {
        active.push_back(row * cols + col);
      }
    }
  }
  if (active.empty()) {
    return cv::Mat(rows, cols, CV_32S, cv::Scalar(0));
  }
  cv::Mat sampled(static_cast<int>(active.size()), featureStack.cols, featureStack.type());
  for (int i = 0; i < static_cast<int>(active.size()); ++i) {
    featureStack.row(active[i]).copyTo(sampled.row(i));
  }
  const cv::Mat sampledLabels = model.predictLabels(sampled);
  if (sampledLabels.empty()) return cv::Mat();
  cv::Mat full(rows * cols, 1, CV_32S, cv::Scalar(0));
  for (int i = 0; i < static_cast<int>(active.size()); ++i) {
    full.at<int>(active[i], 0) = sampledLabels.at<int>(i, 0);
  }
  return full.reshape(1, rows).clone();
}

cv::Mat applyModelProbabilitiesWithExclusion(const SegmentationClassifier& model,
                                             const cv::Mat& featureStack,
                                             int rows,
                                             int cols,
                                             int classIndex,
                                             const cv::Mat& inferenceMask,
                                             const cv::Mat& exclusionMask) {
  if (inferenceMask.empty() && exclusionMask.empty()) {
    return SegmentationEngine::applyModelProbabilities(model, featureStack, rows, cols, classIndex);
  }
  std::vector<int> active;
  active.reserve(rows * cols);
  for (int row = 0; row < rows; ++row) {
    const uchar* includePtr = inferenceMask.empty() ? nullptr : inferenceMask.ptr<uchar>(row);
    const uchar* excludePtr = exclusionMask.empty() ? nullptr : exclusionMask.ptr<uchar>(row);
    for (int col = 0; col < cols; ++col) {
      const bool included = !includePtr || includePtr[col] > 0;
      const bool excluded = excludePtr && excludePtr[col] > 0;
      if (included && !excluded) active.push_back(row * cols + col);
    }
  }
  if (active.empty()) {
    return cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
  }
  cv::Mat sampled(static_cast<int>(active.size()), featureStack.cols, featureStack.type());
  for (int i = 0; i < static_cast<int>(active.size()); ++i) {
    featureStack.row(active[i]).copyTo(sampled.row(i));
  }
  cv::Mat probs = model.predictProbabilities(sampled);
  if (probs.empty() || classIndex < 0 || classIndex >= probs.cols) return cv::Mat();
  cv::Mat full(rows * cols, 1, CV_32F, cv::Scalar(0));
  for (int i = 0; i < static_cast<int>(active.size()); ++i) {
    full.at<float>(active[i], 0) = probs.at<float>(i, classIndex);
  }
  return full.reshape(1, rows).clone();
}

QJsonArray readVersionHistory(const QString& path) {
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) return {};
  const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (!doc.isObject()) return {};
  return doc.object().value("versionHistory").toArray();
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
  initializeDefaultClasses();
  buildUi();
  connectSignals();
  updateUiState();
  resize(1640, 980);
  if (QScreen* screen = QGuiApplication::primaryScreen()) {
    const QRect available = screen->availableGeometry();
    move(available.center() - rect().center());
  }
  setWindowTitle("Qt Trainable Segmentation");
}

void MainWindow::buildUi() {
  auto* toolBar = addToolBar(tr("Main"));
  toolBar->setMovable(false);
  QAction* openImageAction = toolBar->addAction(tr("Open Image"));
  QAction* saveMaskAction = toolBar->addAction(tr("Export Mask"));
  toolBar->addSeparator();
  QAction* undoAction = toolBar->addAction(tr("Undo"));
  QAction* redoAction = toolBar->addAction(tr("Redo"));
  toolBar->addSeparator();
  QAction* paintAction = toolBar->addAction(tr("Paint"));
  QAction* eraseAction = toolBar->addAction(tr("Erase"));
  QAction* panAction = toolBar->addAction(tr("Pan"));
  QAction* traceAction = toolBar->addAction(tr("Trace ROI"));
  QAction* resetZoomAction = toolBar->addAction(tr("Reset Zoom"));

  connect(openImageAction, &QAction::triggered, this, &MainWindow::onOpenImage);
  connect(saveMaskAction, &QAction::triggered, this, &MainWindow::onExportMask);
  connect(undoAction, &QAction::triggered, this, &MainWindow::onUndo);
  connect(redoAction, &QAction::triggered, this, &MainWindow::onRedo);
  connect(paintAction, &QAction::triggered, this, &MainWindow::onPaintTool);
  connect(eraseAction, &QAction::triggered, this, &MainWindow::onEraseTool);
  connect(panAction, &QAction::triggered, this, &MainWindow::onPanTool);
  connect(traceAction, &QAction::triggered, this, &MainWindow::onTraceTool);
  connect(resetZoomAction, &QAction::triggered, this, &MainWindow::onResetZoom);

  auto* splitter = new QSplitter(this);
  splitter->setOrientation(Qt::Horizontal);
  setCentralWidget(splitter);

  auto* leftPanel = new QWidget(splitter);
  auto* leftLayout = new QVBoxLayout(leftPanel);
  leftLayout->setContentsMargins(12, 12, 12, 12);
  leftLayout->setSpacing(12);

  auto* trainingGroup = new QGroupBox(tr("Training"), leftPanel);
  auto* trainingLayout = new QVBoxLayout(trainingGroup);
  trainButton_ = new QPushButton(tr("Train classifier"), trainingGroup);
  stopTrainingButton_ = new QPushButton(tr("Stop training"), trainingGroup);
  QPushButton* overlayButton = new QPushButton(tr("Toggle overlay"), trainingGroup);
  createResultButton_ = new QPushButton(tr("Create result"), trainingGroup);
  binaryResultButton_ = new QPushButton(tr("Show binary result"), trainingGroup);
  probabilityButton_ = new QPushButton(tr("Get probability"), trainingGroup);
  QPushButton* plotButton = new QPushButton(tr("Plot result"), trainingGroup);
  evaluateButton_ = new QPushButton(tr("Evaluate"), trainingGroup);
  suggestButton_ = new QPushButton(tr("Suggest labels"), trainingGroup);
  trainingLayout->addWidget(trainButton_);
  trainingLayout->addWidget(stopTrainingButton_);
  trainingLayout->addWidget(overlayButton);
  trainingLayout->addWidget(createResultButton_);
  trainingLayout->addWidget(binaryResultButton_);
  trainingLayout->addWidget(probabilityButton_);
  trainingLayout->addWidget(plotButton);
  trainingLayout->addWidget(evaluateButton_);
  trainingLayout->addWidget(suggestButton_);
  leftLayout->addWidget(trainingGroup);

  auto* optionsGroup = new QGroupBox(tr("Options"), leftPanel);
  auto* optionsLayout = new QVBoxLayout(optionsGroup);
  applyButton_ = new QPushButton(tr("Apply classifier"), optionsGroup);
  QPushButton* loadClassifierButton = new QPushButton(tr("Load classifier"), optionsGroup);
  QPushButton* saveClassifierButton = new QPushButton(tr("Save classifier"), optionsGroup);
  QPushButton* loadDataButton = new QPushButton(tr("Load data"), optionsGroup);
  QPushButton* saveDataButton = new QPushButton(tr("Save data"), optionsGroup);
  QPushButton* createClassButton = new QPushButton(tr("Create new class"), optionsGroup);
  QPushButton* settingsButton = new QPushButton(tr("Settings"), optionsGroup);
  exportRoiMacroButton_ = new QPushButton(tr("Export ROI macro"), optionsGroup);
  optionsLayout->addWidget(applyButton_);
  optionsLayout->addWidget(loadClassifierButton);
  optionsLayout->addWidget(saveClassifierButton);
  optionsLayout->addWidget(loadDataButton);
  optionsLayout->addWidget(saveDataButton);
  optionsLayout->addWidget(createClassButton);
  optionsLayout->addWidget(settingsButton);
  optionsLayout->addWidget(exportRoiMacroButton_);
  leftLayout->addWidget(optionsGroup);

  auto* brushGroup = new QGroupBox(tr("Brush / Slice"), leftPanel);
  auto* brushLayout = new QFormLayout(brushGroup);
  brushSizeSpin_ = new QSpinBox(brushGroup);
  brushSizeSpin_->setRange(1, 256);
  brushSizeSpin_->setValue(12);
  opacitySlider_ = new QSlider(Qt::Horizontal, brushGroup);
  opacitySlider_->setRange(0, 100);
  opacitySlider_->setValue(static_cast<int>(overlayOpacity_ * 100.0));
  probabilityCombo_ = new QComboBox(brushGroup);
  classifierCombo_ = new QComboBox(brushGroup);
  for (const auto& descriptor : SegmentationClassifier::availableClassifiers()) {
    classifierCombo_->addItem(descriptor.displayName, descriptor.kind);
  }
  const int defaultClassifierIndex = classifierCombo_->findData(classifierSettings_.kind);
  if (defaultClassifierIndex >= 0) {
    classifierCombo_->setCurrentIndex(defaultClassifierIndex);
  }
  overlayCheck_ = new QCheckBox(tr("Show classifier overlay"), brushGroup);
  overlayCheck_->setChecked(true);
  contourCheck_ = new QCheckBox(tr("Draw contours"), brushGroup);
  contourCheck_->setChecked(true);
  sliceSlider_ = new QSlider(Qt::Horizontal, brushGroup);
  sliceSlider_->setRange(0, 0);
  sliceLabel_ = new QLabel(tr("Slice 1 / 1"), brushGroup);
  brushLayout->addRow(tr("Brush radius"), brushSizeSpin_);
  brushLayout->addRow(tr("Overlay alpha"), opacitySlider_);
  brushLayout->addRow(tr("Probability class"), probabilityCombo_);
  brushLayout->addRow(tr("Classifier"), classifierCombo_);
  brushLayout->addRow(tr("Slice"), sliceSlider_);
  brushLayout->addRow(QString(), sliceLabel_);
  brushLayout->addRow(overlayCheck_);
  brushLayout->addRow(contourCheck_);
  leftLayout->addWidget(brushGroup);

  undoButton_ = new QPushButton(tr("Undo"), leftPanel);
  redoButton_ = new QPushButton(tr("Redo"), leftPanel);
  leftLayout->addWidget(undoButton_);
  leftLayout->addWidget(redoButton_);

  auto* spacer = new QWidget(leftPanel);
  spacer->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  leftLayout->addWidget(spacer);

  connect(overlayButton, &QPushButton::clicked, this, &MainWindow::onToggleOverlay);
  connect(plotButton, &QPushButton::clicked, this, &MainWindow::onPlotResult);
  connect(loadClassifierButton, &QPushButton::clicked, this, &MainWindow::onLoadClassifier);
  connect(saveClassifierButton, &QPushButton::clicked, this, &MainWindow::onSaveClassifier);
  connect(loadDataButton, &QPushButton::clicked, this, &MainWindow::onLoadData);
  connect(saveDataButton, &QPushButton::clicked, this, &MainWindow::onSaveData);
  connect(createClassButton, &QPushButton::clicked, this, &MainWindow::onCreateNewClass);
  connect(settingsButton, &QPushButton::clicked, this, &MainWindow::onSettings);
  connect(exportRoiMacroButton_, &QPushButton::clicked, this, &MainWindow::onExportRoiMacro);

  auto* centerPanel = new QWidget(splitter);
  auto* centerLayout = new QVBoxLayout(centerPanel);
  centerLayout->setContentsMargins(8, 8, 8, 8);
  centerLayout->setSpacing(8);
  imagePathLabel_ = new QLabel(tr("No image loaded"), centerPanel);
  imagePathLabel_->setWordWrap(true);
  view_ = new SegmentationView(centerPanel);
  infoLabel_ = new QLabel(tr("Open a grayscale, color image, or TIFF stack to begin labeling."), centerPanel);
  probabilityLabel_ = new QLabel(tr("Probability: n/a"), centerPanel);
  modelLabel_ = new QLabel(tr("Model: not trained"), centerPanel);
  centerLayout->addWidget(imagePathLabel_);
  centerLayout->addWidget(view_, 1);
  centerLayout->addWidget(infoLabel_);
  centerLayout->addWidget(probabilityLabel_);
  centerLayout->addWidget(modelLabel_);

  auto* rightPanel = new QWidget(splitter);
  auto* rightLayout = new QVBoxLayout(rightPanel);
  rightLayout->setContentsMargins(12, 12, 12, 12);
  rightLayout->setSpacing(12);

  auto* labelsGroup = new QGroupBox(tr("Labels"), rightPanel);
  auto* labelsLayout = new QVBoxLayout(labelsGroup);
  addTraceButton_ = new QPushButton(tr("Add ROI to selected class"), labelsGroup);
  addInferenceRoiButton_ = new QPushButton(tr("Add recognition ROI"), labelsGroup);
  clearInferenceRoiButton_ = new QPushButton(tr("Clear recognition ROIs"), labelsGroup);
  addExclusionRoiButton_ = new QPushButton(tr("Add exclusion ROI"), labelsGroup);
  clearExclusionRoiButton_ = new QPushButton(tr("Clear exclusion ROIs"), labelsGroup);
  classList_ = new QListWidget(labelsGroup);
  traceList_ = new QListWidget(labelsGroup);
  removeTraceButton_ = new QPushButton(tr("Remove selected trace"), labelsGroup);
  renameTraceButton_ = new QPushButton(tr("Rename selected trace"), labelsGroup);
  clearTraceButton_ = new QPushButton(tr("Clear traces in class"), labelsGroup);
  traceGroup_ = new QGroupBox(tr("Traces"), labelsGroup);
  auto* traceLayout = new QVBoxLayout(traceGroup_);
  traceLayout->addWidget(traceList_);
  traceLayout->addWidget(removeTraceButton_);
  traceLayout->addWidget(renameTraceButton_);
  traceLayout->addWidget(clearTraceButton_);
  labelsLayout->addWidget(addTraceButton_);
  labelsLayout->addWidget(addInferenceRoiButton_);
  labelsLayout->addWidget(clearInferenceRoiButton_);
  labelsLayout->addWidget(addExclusionRoiButton_);
  labelsLayout->addWidget(clearExclusionRoiButton_);
  labelsLayout->addWidget(classList_);
  labelsLayout->addWidget(traceGroup_);
  rightLayout->addWidget(labelsGroup, 2);

  auto* logGroup = new QGroupBox(tr("Log"), rightPanel);
  auto* logLayout = new QVBoxLayout(logGroup);
  logEdit_ = new QTextEdit(logGroup);
  logEdit_->setReadOnly(true);
  logLayout->addWidget(logEdit_);
  rightLayout->addWidget(logGroup, 1);

  splitter->addWidget(leftPanel);
  splitter->addWidget(centerPanel);
  splitter->addWidget(rightPanel);
  splitter->setStretchFactor(0, 0);
  splitter->setStretchFactor(1, 1);
  splitter->setStretchFactor(2, 0);
  splitter->setSizes({290, 980, 320});

  refreshClassList();
}

void MainWindow::connectSignals() {
  connect(trainButton_, &QPushButton::clicked, this, &MainWindow::onTrainClassifier);
  connect(stopTrainingButton_, &QPushButton::clicked, this, &MainWindow::onStopTraining);
  connect(applyButton_, &QPushButton::clicked, this, &MainWindow::onApplyClassifier);
  connect(createResultButton_, &QPushButton::clicked, this, &MainWindow::onCreateResult);
  connect(binaryResultButton_, &QPushButton::clicked, this, &MainWindow::onShowBinaryResult);
  connect(probabilityButton_, &QPushButton::clicked, this, &MainWindow::onGetProbability);
  connect(evaluateButton_, &QPushButton::clicked, this, &MainWindow::onEvaluateModel);
  connect(suggestButton_, &QPushButton::clicked, this, &MainWindow::onSuggestLabels);
  connect(brushSizeSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::onBrushRadiusChanged);
  connect(classList_, &QListWidget::itemSelectionChanged, this, &MainWindow::onClassSelectionChanged);
  connect(probabilityCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::onProbabilityClassChanged);
  connect(classifierCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
    classifierSettings_.kind = static_cast<SegmentationClassifierSettings::Kind>(classifierCombo_->itemData(index).toInt());
    updateUiState();
  });
  connect(opacitySlider_, &QSlider::valueChanged, this, &MainWindow::onOpacityChanged);
  connect(sliceSlider_, &QSlider::valueChanged, this, &MainWindow::onSliceChanged);
  connect(overlayCheck_, &QCheckBox::toggled, this, [this](bool checked) {
    showOverlay_ = checked;
    updateViewer();
  });
  connect(contourCheck_, &QCheckBox::toggled, this, [this](bool) { updateViewer(); });
  connect(addTraceButton_, &QPushButton::clicked, this, &MainWindow::onAddTraceToSelectedClass);
  connect(addInferenceRoiButton_, &QPushButton::clicked, this, &MainWindow::onAddInferenceRoi);
  connect(clearInferenceRoiButton_, &QPushButton::clicked, this, &MainWindow::onClearInferenceRois);
  connect(addExclusionRoiButton_, &QPushButton::clicked, this, &MainWindow::onAddExclusionRoi);
  connect(clearExclusionRoiButton_, &QPushButton::clicked, this, &MainWindow::onClearExclusionRois);
  connect(removeTraceButton_, &QPushButton::clicked, this, &MainWindow::onRemoveSelectedTrace);
  connect(renameTraceButton_, &QPushButton::clicked, this, &MainWindow::onRenameSelectedTrace);
  connect(clearTraceButton_, &QPushButton::clicked, this, &MainWindow::onClearTracesForSelectedClass);
  connect(undoButton_, &QPushButton::clicked, this, &MainWindow::onUndo);
  connect(redoButton_, &QPushButton::clicked, this, &MainWindow::onRedo);
  view_->onBrushStroke = [this](const QPoint& imagePos, int radius, bool erase) {
    applyBrushStroke(imagePos, radius, erase);
  };
  view_->onTraceFinished = [this](const QPolygon& trace) {
    setPendingTrace(trace);
  };
  view_->onMouseHover = [this](const QPoint& imagePos) {
    if (originalImage_.empty() || imagePos.x() < 0 || imagePos.y() < 0) {
      probabilityLabel_->setText(tr("Probability: n/a"));
      return;
    }
    if (!probabilityImage_.empty()) {
      const float value = probabilityImage_.at<float>(imagePos.y(), imagePos.x());
      probabilityLabel_->setText(tr("Probability: %1%  @ (%2, %3)")
                                     .arg(QString::number(value * 100.0f, 'f', 2))
                                     .arg(imagePos.x())
                                     .arg(imagePos.y()));
    } else {
      probabilityLabel_->setText(tr("Cursor: (%1, %2)").arg(imagePos.x()).arg(imagePos.y()));
    }
  };
}

void MainWindow::initializeDefaultClasses() {
  classes_.clear();
  classes_.push_back({tr("Foreground target"), defaultColorForIndex(0), true});
  classes_.push_back({tr("Background object"), defaultColorForIndex(1), true});
}

void MainWindow::refreshClassList() {
  classList_->clear();
  probabilityCombo_->clear();
  for (int i = 0; i < static_cast<int>(classes_.size()); ++i) {
    const auto& cls = classes_[i];
    QListWidgetItem* item = new QListWidgetItem(QString("%1  (Class %2)").arg(cls.name).arg(i + 1), classList_);
    item->setForeground(cls.color);
    item->setData(Qt::UserRole, i);
    probabilityCombo_->addItem(cls.name, i);
  }
  if (classList_->count() > 0 && !classList_->currentItem()) {
    classList_->setCurrentRow(0);
  }
}

void MainWindow::updateUiState() {
  const bool hasImage = !originalImage_.empty();
  const bool hasModel = model_.isValid();
  const bool hasProbability = hasModel && model_.supportsProbability();
  const int currentClass = currentSelectedClassIndex();
  const bool hasClassSelection = currentClass >= 0 && currentClass < static_cast<int>(classes_.size());
  trainButton_->setEnabled(hasImage && !trainingInProgress_);
  stopTrainingButton_->setEnabled(trainingInProgress_);
  removeTraceButton_->setEnabled(traceList_ && traceList_->currentRow() >= 0);
  renameTraceButton_->setEnabled(traceList_ && traceList_->currentRow() >= 0);
  clearTraceButton_->setEnabled(hasImage && hasClassSelection);
  undoButton_->setEnabled(!undoStack_.empty());
  redoButton_->setEnabled(!redoStack_.empty());
  applyButton_->setEnabled(hasModel && !trainingInProgress_);
  createResultButton_->setEnabled(hasImage && hasModel && !trainingInProgress_);
  binaryResultButton_->setEnabled(hasImage && hasModel && !trainingInProgress_);
  probabilityButton_->setEnabled(hasImage && hasProbability && !trainingInProgress_);
  evaluateButton_->setEnabled(hasImage && hasModel && !trainingInProgress_);
  suggestButton_->setEnabled(hasImage && hasProbability && !trainingInProgress_);
  brushSizeSpin_->setEnabled(hasImage);
  probabilityCombo_->setEnabled(hasProbability && !trainingInProgress_);
  classifierCombo_->setEnabled(!trainingInProgress_);
  addTraceButton_->setEnabled(hasImage && hasClassSelection && !trainingInProgress_);
  addInferenceRoiButton_->setEnabled(hasImage && !trainingInProgress_);
  clearInferenceRoiButton_->setEnabled(hasImage && !trainingInProgress_ && !inferenceRegions_.empty());
  addExclusionRoiButton_->setEnabled(hasImage && !trainingInProgress_);
  clearExclusionRoiButton_->setEnabled(hasImage && !trainingInProgress_ && !exclusionRegions_.empty());
  traceList_->setEnabled(hasImage && hasClassSelection && !trainingInProgress_);
  sliceSlider_->setEnabled(imageVolume_.size() > 1 && !trainingInProgress_);
  if (traceGroup_) {
    traceGroup_->setEnabled(hasImage && hasClassSelection);
  }
  overlayCheck_->setEnabled(hasModel || !labelResult_.empty());
  contourCheck_->setEnabled(hasModel || !labelResult_.empty());
  view_->setPaintEnabled(hasImage);
  modelLabel_->setText(hasModel
                           ? tr("Model: %1 | %2 classes, %3 samples, acc=%4%")
                                 .arg(model_.classifierName())
                                 .arg(trainingStats_.classCount)
                                 .arg(trainingStats_.sampleCount)
                                 .arg(QString::number(trainingStats_.trainingAccuracy * 100.0, 'f', 2))
                           : (!importedTrainingSamples_.empty()
                                  ? tr("Model: not trained | imported vectors=%1").arg(importedTrainingSamples_.rows)
                                  : tr("Model: not trained")));
  sliceLabel_->setText(tr("Slice %1 / %2").arg(currentSliceIndex_ + 1).arg(std::max(1, static_cast<int>(imageVolume_.size()))));
}

void MainWindow::ensureSliceStorage() {
  const int count = std::max(1, static_cast<int>(imageVolume_.size()));
  if (static_cast<int>(sliceAnnotationStates_.size()) != count) {
    sliceAnnotationStates_.resize(count);
  }
  if (static_cast<int>(sliceLabelResults_.size()) != count) {
    sliceLabelResults_.resize(count);
  }
  if (static_cast<int>(sliceProbabilityResults_.size()) != count) {
    sliceProbabilityResults_.resize(count);
  }
  if (static_cast<int>(featureVolume_.size()) != count) {
    featureVolume_.resize(count);
  }
}

void MainWindow::saveCurrentSliceState() {
  if (originalImage_.empty()) {
    return;
  }
  ensureSliceStorage();
  if (currentSliceIndex_ < 0 || currentSliceIndex_ >= static_cast<int>(sliceAnnotationStates_.size())) {
    return;
  }
  sliceAnnotationStates_[currentSliceIndex_].classBrushMasks = cloneMaskVector(classBrushMasks_);
  sliceAnnotationStates_[currentSliceIndex_].classTraceRegions = cloneTraceRegions(classTraceRegions_);
  sliceAnnotationStates_[currentSliceIndex_].inferenceRegions = inferenceRegions_;
  sliceAnnotationStates_[currentSliceIndex_].exclusionRegions = exclusionRegions_;
  sliceLabelResults_[currentSliceIndex_] = labelResult_.clone();
  sliceProbabilityResults_[currentSliceIndex_] = probabilityImage_.clone();
}

void MainWindow::restoreSliceState(int sliceIndex) {
  ensureSliceStorage();
  if (sliceIndex < 0 || sliceIndex >= static_cast<int>(sliceAnnotationStates_.size())) {
    return;
  }
  classBrushMasks_ = cloneMaskVector(sliceAnnotationStates_[sliceIndex].classBrushMasks);
  classTraceRegions_ = cloneTraceRegions(sliceAnnotationStates_[sliceIndex].classTraceRegions);
  inferenceRegions_ = sliceAnnotationStates_[sliceIndex].inferenceRegions;
  exclusionRegions_ = sliceAnnotationStates_[sliceIndex].exclusionRegions;
  labelResult_ = sliceLabelResults_[sliceIndex].clone();
  probabilityImage_ = sliceProbabilityResults_[sliceIndex].clone();
  pendingTrace_.clear();
  view_->clearPendingTrace();
  undoStack_.clear();
  redoStack_.clear();
  rebuildMasksFromAnnotations();
}

QString MainWindow::defaultTraceNameForClass(int classIndex) const {
  const QString className = (classIndex >= 0 && classIndex < static_cast<int>(classes_.size())) ? classes_[classIndex].name : tr("Class");
  int count = 1;
  if (classIndex >= 0 && classIndex < static_cast<int>(classTraceRegions_.size())) {
    count = static_cast<int>(classTraceRegions_[classIndex].size()) + 1;
  }
  return tr("%1 ROI %2").arg(className).arg(count);
}

void MainWindow::pushUndoSnapshot() {
  AnnotationSnapshot snapshot;
  snapshot.classBrushMasks = cloneMaskVector(classBrushMasks_);
  snapshot.classTraceRegions = cloneTraceRegions(classTraceRegions_);
  snapshot.inferenceRegions = inferenceRegions_;
  snapshot.exclusionRegions = exclusionRegions_;
  undoStack_.push_back(std::move(snapshot));
  trimUndoHistory();
  redoStack_.clear();
}

void MainWindow::trimUndoHistory() {
  constexpr int kMaxHistory = 30;
  if (static_cast<int>(undoStack_.size()) > kMaxHistory) {
    undoStack_.erase(undoStack_.begin(), undoStack_.begin() + (undoStack_.size() - kMaxHistory));
  }
}

void MainWindow::restoreSnapshot(const AnnotationSnapshot& snapshot) {
  classBrushMasks_ = cloneMaskVector(snapshot.classBrushMasks);
  classTraceRegions_ = cloneTraceRegions(snapshot.classTraceRegions);
  inferenceRegions_ = snapshot.inferenceRegions;
  exclusionRegions_ = snapshot.exclusionRegions;
  rebuildMasksFromAnnotations();
}

void MainWindow::rebuildFeatureStack() {
  if (originalImage_.empty()) {
    featureStack_.release();
    return;
  }
  featureStack_ = SegmentationEngine::computeFeatureStack(originalImage_, featureSettings_);
  ensureSliceStorage();
  featureVolume_[currentSliceIndex_] = featureStack_.clone();
}

void MainWindow::updateViewer() {
  if (originalImage_.empty()) {
    view_->clearAllLayers();
    return;
  }
  view_->setBaseImage(cvMatToQImage(originalImage_));
  if (showOverlay_ && !labelResult_.empty()) {
    overlayImage_ = SegmentationEngine::makeOverlay(originalImage_, labelResult_, classes_, overlayOpacity_, contourCheck_->isChecked());
    view_->setOverlayImage(cvMatToQImage(overlayImage_));
  } else {
    view_->setOverlayImage(QImage());
  }
  repaintAnnotationPreview();
}

bool MainWindow::computeProbabilityOutputs(const QString& title,
                                           int classIndex,
                                           cv::Mat* probabilityImage,
                                           cv::Mat* fullProbabilities) {
  if (!probabilityImage && !fullProbabilities) {
    return false;
  }
  if (originalImage_.empty() || !model_.isValid() || featureStack_.empty() || !model_.supportsProbability()) {
    if (probabilityImage) probabilityImage->release();
    if (fullProbabilities) fullProbabilities->release();
    return false;
  }

  QProgressDialog progress(title.isEmpty() ? tr("Computing probability map...") : title,
                           tr("Cancel"),
                           0,
                           featureStack_.rows,
                           this);
  progress.setWindowModality(Qt::ApplicationModal);
  progress.setMinimumDuration(0);
  progress.setAutoClose(false);
  progress.setValue(0);
  progress.show();
  QApplication::processEvents();

  cv::Mat probabilities;
  constexpr int kBatchSize = 4096;
  for (int start = 0; start < featureStack_.rows; start += kBatchSize) {
    if (progress.wasCanceled()) {
      if (probabilityImage) probabilityImage->release();
      if (fullProbabilities) fullProbabilities->release();
      return false;
    }
    const int end = std::min(featureStack_.rows, start + kBatchSize);
    const cv::Mat batch = model_.predictProbabilities(featureStack_.rowRange(start, end));
    if (batch.empty()) {
      if (probabilityImage) probabilityImage->release();
      if (fullProbabilities) fullProbabilities->release();
      return false;
    }
    if (probabilities.empty()) {
      probabilities = cv::Mat::zeros(featureStack_.rows, batch.cols, CV_32F);
    } else if (batch.cols != probabilities.cols) {
      if (probabilityImage) probabilityImage->release();
      if (fullProbabilities) fullProbabilities->release();
      return false;
    }
    batch.copyTo(probabilities.rowRange(start, end));
    progress.setValue(end);
    QApplication::processEvents();
  }

  if (probabilities.empty() || classIndex < 0 || classIndex >= probabilities.cols) {
    if (probabilityImage) probabilityImage->release();
    if (fullProbabilities) fullProbabilities->release();
    return false;
  }

  if (fullProbabilities) {
    *fullProbabilities = probabilities;
  }
  if (probabilityImage) {
    cv::Mat channel(originalImage_.rows, originalImage_.cols, CV_32F);
    for (int row = 0; row < originalImage_.rows; ++row) {
      float* out = channel.ptr<float>(row);
      for (int col = 0; col < originalImage_.cols; ++col) {
        out[col] = probabilities.at<float>(row * originalImage_.cols + col, classIndex);
      }
    }
    if (!inferenceMask_.empty()) {
      cv::Mat outsideMask;
      cv::bitwise_not(inferenceMask_, outsideMask);
      channel.setTo(cv::Scalar(0), outsideMask);
    }
    if (!exclusionMask_.empty()) {
      channel.setTo(cv::Scalar(0), exclusionMask_);
    }
    *probabilityImage = channel;
  }
  progress.setValue(featureStack_.rows);
  return true;
}

bool MainWindow::updateProbabilityView(bool showProgress, const QString& title) {
  if (originalImage_.empty() || !model_.isValid() || featureStack_.empty()) {
    probabilityImage_.release();
    return false;
  }
  const int cls = probabilityCombo_->currentData().toInt();
  if (showProgress) {
    cv::Mat updatedProbability;
    probabilityImage_ = computeProbabilityOutputs(title, cls, &updatedProbability, nullptr)
                            ? updatedProbability
                            : cv::Mat();
  } else {
    probabilityImage_ = model_.supportsProbability()
                            ? applyModelProbabilitiesWithExclusion(model_, featureStack_, originalImage_.rows, originalImage_.cols, cls, inferenceMask_, exclusionMask_)
                            : cv::Mat();
  }
  if (!sliceProbabilityResults_.empty() && currentSliceIndex_ < static_cast<int>(sliceProbabilityResults_.size())) {
    sliceProbabilityResults_[currentSliceIndex_] = probabilityImage_.clone();
  }
  return !probabilityImage_.empty();
}

int MainWindow::currentSelectedClassIndex() const {
  return classList_ && classList_->currentItem() ? classList_->currentItem()->data(Qt::UserRole).toInt() : -1;
}

void MainWindow::repaintAnnotationPreview() {
  if (originalImage_.empty()) {
    view_->setAnnotationPreview(QImage());
    return;
  }
  QImage annotation(originalImage_.cols, originalImage_.rows, QImage::Format_ARGB32_Premultiplied);
  annotation.fill(Qt::transparent);
  QPainter painter(&annotation);
  painter.setRenderHint(QPainter::Antialiasing, false);
  for (int cls = 0; cls < static_cast<int>(classMasks_.size()) && cls < static_cast<int>(classes_.size()); ++cls) {
    if (classMasks_[cls].empty()) continue;
    const bool outlineOnly = (classes_[cls].name == tr("Background object"));
    QImage maskImg(classMasks_[cls].data, classMasks_[cls].cols, classMasks_[cls].rows, static_cast<int>(classMasks_[cls].step), QImage::Format_Grayscale8);
    QImage colored(maskImg.size(), QImage::Format_ARGB32_Premultiplied);
    colored.fill(Qt::transparent);
    if (!outlineOnly) {
      for (int y = 0; y < maskImg.height(); ++y) {
        const uchar* src = maskImg.constScanLine(y);
        QRgb* dst = reinterpret_cast<QRgb*>(colored.scanLine(y));
        for (int x = 0; x < maskImg.width(); ++x) {
          if (src[x] > 0) {
            const QColor c = classes_[cls].color;
            dst[x] = qRgba(c.red(), c.green(), c.blue(), 140);
          }
        }
      }
    }
    painter.drawImage(0, 0, colored);
    if (outlineOnly) {
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(classMasks_[cls].clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      QPen pen(classes_[cls].color);
      pen.setWidth(2);
      painter.setPen(pen);
      for (const auto& contour : contours) {
        if (contour.empty()) continue;
        QPolygon polygon;
        polygon.reserve(static_cast<int>(contour.size()));
        for (const auto& pt : contour) {
          polygon << QPoint(pt.x, pt.y);
        }
        painter.drawPolygon(polygon);
      }
    }
  }
  if (!inferenceMask_.empty()) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(inferenceMask_.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    QPen pen(QColor(60, 200, 90, 220));
    pen.setWidth(2);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    for (const auto& contour : contours) {
      if (contour.empty()) continue;
      QPolygon polygon;
      polygon.reserve(static_cast<int>(contour.size()));
      for (const auto& pt : contour) {
        polygon << QPoint(pt.x, pt.y);
      }
      if (polygon.size() >= 3) {
        painter.drawPolygon(polygon);
      }
    }
  }
  if (!exclusionMask_.empty()) {
    QImage exMask(exclusionMask_.data, exclusionMask_.cols, exclusionMask_.rows, static_cast<int>(exclusionMask_.step), QImage::Format_Grayscale8);
    QImage colored(exMask.size(), QImage::Format_ARGB32_Premultiplied);
    colored.fill(Qt::transparent);
    for (int y = 0; y < exMask.height(); ++y) {
      const uchar* src = exMask.constScanLine(y);
      QRgb* dst = reinterpret_cast<QRgb*>(colored.scanLine(y));
      for (int x = 0; x < exMask.width(); ++x) {
        if (src[x] > 0) dst[x] = qRgba(255, 60, 60, 70);
      }
    }
    painter.drawImage(0, 0, colored);
  }
  painter.end();
  view_->setAnnotationPreview(annotation);
}

void MainWindow::applyBrushStroke(const QPoint& imagePos, int radius, bool erase) {
  if (originalImage_.empty()) {
    return;
  }
  pushUndoSnapshot();
  ensureAnnotationStorage();
  const int cls = currentSelectedClassIndex() >= 0 ? currentSelectedClassIndex() : 0;
  if (cls < 0 || cls >= static_cast<int>(classMasks_.size())) {
    return;
  }
  cv::Scalar value = erase ? cv::Scalar(0) : cv::Scalar(255);
  cv::circle(classBrushMasks_[cls], cv::Point(imagePos.x(), imagePos.y()), radius, value, -1, cv::LINE_AA);
  if (!erase) {
    for (int other = 0; other < static_cast<int>(classMasks_.size()); ++other) {
      if (other == cls) continue;
      cv::circle(classBrushMasks_[other], cv::Point(imagePos.x(), imagePos.y()), radius, cv::Scalar(0), -1, cv::LINE_AA);
    }
  }
  rebuildMasksFromAnnotations();
}

bool MainWindow::ensureImageLoaded(const QString& actionName) {
  if (!originalImage_.empty()) {
    return true;
  }
  QMessageBox::information(this, tr("No image"), tr("Please open an image before %1.").arg(actionName));
  return false;
}

bool MainWindow::ensureModelReady(const QString& actionName) {
  if (model_.isValid()) {
    return true;
  }
  QMessageBox::information(this, tr("No classifier"), tr("Please train or load a classifier before %1.").arg(actionName));
  return false;
}

void MainWindow::logMessage(const QString& message) {
  logEdit_->append(QString("[%1] %2").arg(nowString(), message));
}

void MainWindow::activateLabelShortcut(int classIndex, const QString& semanticDescription) {
  if (classIndex < 0 || classIndex >= classList_->count()) {
    QMessageBox::information(this, tr("Class unavailable"), tr("The requested class shortcut is not available."));
    return;
  }
  classList_->setCurrentRow(classIndex);
  view_->setToolMode(SegmentationView::PaintTool);
  const QString className = classes_[classIndex].name;
  infoLabel_->setText(tr("Now painting %1 samples into %2.").arg(semanticDescription, className));
  logMessage(tr("Activated %1 shortcut for %2.").arg(semanticDescription, className));
}

void MainWindow::ensureAnnotationStorage() {
  if (originalImage_.empty()) {
    return;
  }
  const int rows = originalImage_.rows;
  const int cols = originalImage_.cols;
  if (static_cast<int>(classMasks_.size()) != static_cast<int>(classes_.size())) {
    classMasks_.assign(classes_.size(), cv::Mat(rows, cols, CV_8U, cv::Scalar(0)));
  }
  if (static_cast<int>(classBrushMasks_.size()) != static_cast<int>(classes_.size())) {
    classBrushMasks_.assign(classes_.size(), cv::Mat(rows, cols, CV_8U, cv::Scalar(0)));
  }
  if (static_cast<int>(classTraceRegions_.size()) != static_cast<int>(classes_.size())) {
    classTraceRegions_.resize(classes_.size());
  }
  for (auto& mask : classMasks_) {
    if (mask.empty() || mask.rows != rows || mask.cols != cols) {
      mask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    }
  }
  for (auto& mask : classBrushMasks_) {
    if (mask.empty() || mask.rows != rows || mask.cols != cols) {
      mask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    }
  }
}

void MainWindow::clearAnnotationsForCurrentImage() {
  if (originalImage_.empty()) {
    classMasks_.clear();
    classBrushMasks_.clear();
    classTraceRegions_.clear();
    pendingTrace_.clear();
    view_->clearPendingTrace();
    return;
  }
  ensureSliceStorage();
  for (auto& state : sliceAnnotationStates_) {
    state.classBrushMasks.clear();
    state.classTraceRegions.clear();
    state.inferenceRegions.clear();
    state.exclusionRegions.clear();
  }
  for (auto& result : sliceLabelResults_) result.release();
  for (auto& result : sliceProbabilityResults_) result.release();
  classMasks_.assign(classes_.size(), cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0)));
  classBrushMasks_.assign(classes_.size(), cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0)));
  classTraceRegions_.assign(classes_.size(), {});
  inferenceRegions_.clear();
  inferenceMask_.release();
  exclusionRegions_.clear();
  exclusionMask_.release();
  undoStack_.clear();
  redoStack_.clear();
  pendingTrace_.clear();
  view_->clearPendingTrace();
  rebuildMasksFromAnnotations();
}

void MainWindow::clearInferenceOutputs() {
  labelResult_.release();
  overlayImage_.release();
  probabilityImage_.release();
  for (auto& result : sliceLabelResults_) result.release();
  for (auto& result : sliceProbabilityResults_) result.release();
}

void MainWindow::clearImportedTrainingData() {
  importedTrainingSamples_.release();
  importedTrainingLabels_.release();
  importedTrainingSource_.clear();
}

void MainWindow::rebuildMasksFromAnnotations() {
  if (originalImage_.empty()) {
    return;
  }
  ensureAnnotationStorage();
  for (int i = 0; i < static_cast<int>(classes_.size()); ++i) {
    classMasks_[i] = classBrushMasks_[i].clone();
  }

  for (int cls = 0; cls < static_cast<int>(classTraceRegions_.size()); ++cls) {
    for (const TraceRegion& region : classTraceRegions_[cls]) {
      if (region.polygon.size() < 3) continue;
      std::vector<cv::Point> cvPoints;
      cvPoints.reserve(region.polygon.size());
      for (const QPoint& pt : region.polygon) {
        cvPoints.emplace_back(pt.x(), pt.y());
      }
      std::vector<std::vector<cv::Point>> fillPoints{cvPoints};
      cv::Mat traceMask(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0));
      cv::fillPoly(traceMask, fillPoints, cv::Scalar(255), cv::LINE_AA);
      for (int other = 0; other < static_cast<int>(classes_.size()); ++other) {
        if (other == cls) continue;
        classMasks_[other].setTo(cv::Scalar(0), traceMask);
      }
      classMasks_[cls].setTo(cv::Scalar(255), traceMask);
    }
  }

  inferenceMask_ = buildInferenceMask(originalImage_.size(), inferenceRegions_);
  exclusionMask_ = buildExclusionMask(originalImage_.size(), exclusionRegions_);
  if (!inferenceMask_.empty()) {
    cv::Mat outsideMask;
    cv::bitwise_not(inferenceMask_, outsideMask);
    for (auto& mask : classMasks_) {
      if (!mask.empty()) mask.setTo(cv::Scalar(0), outsideMask);
    }
    for (auto& mask : classBrushMasks_) {
      if (!mask.empty()) mask.setTo(cv::Scalar(0), outsideMask);
    }
  }
  if (!exclusionMask_.empty()) {
    for (auto& mask : classMasks_) {
      if (!mask.empty()) mask.setTo(cv::Scalar(0), exclusionMask_);
    }
    for (auto& mask : classBrushMasks_) {
      if (!mask.empty()) mask.setTo(cv::Scalar(0), exclusionMask_);
    }
  }

  repaintAnnotationPreview();
  updateTraceLists();
  updateUiState();
  saveCurrentSliceState();
}

void MainWindow::updateTraceLists() {
  if (!traceList_ || !traceGroup_) {
    return;
  }
  traceList_->clear();
  const int cls = currentSelectedClassIndex();
  if (cls < 0 || cls >= static_cast<int>(classes_.size())) {
    traceGroup_->setTitle(tr("Traces"));
    return;
  }
  traceGroup_->setTitle(tr("Traces for %1 (slice %2)").arg(classes_[cls].name).arg(currentSliceIndex_ + 1));
  if (cls < static_cast<int>(classTraceRegions_.size())) {
    for (int i = 0; i < static_cast<int>(classTraceRegions_[cls].size()); ++i) {
      const TraceRegion& region = classTraceRegions_[cls][i];
      QListWidgetItem* item = new QListWidgetItem(region.name.isEmpty() ? tr("Trace %1").arg(i + 1) : region.name, traceList_);
      item->setData(Qt::UserRole, i);
      item->setForeground(classes_[cls].color);
    }
  }
}

void MainWindow::setPendingTrace(const QPolygon& trace) {
  pendingTrace_ = trace;
  const int cls = currentSelectedClassIndex();
  const QColor color = (cls >= 0 && cls < static_cast<int>(classes_.size())) ? classes_[cls].color : QColor(255, 210, 90);
  view_->setPendingTrace(pendingTrace_, color);
  const QString className = (cls >= 0 && cls < static_cast<int>(classes_.size())) ? classes_[cls].name : tr("the selected class");
  infoLabel_->setText(tr("ROI trace captured. Click Add ROI/Add recognition ROI/Add exclusion ROI to use it."));
  logMessage(tr("Captured a pending ROI trace with %1 points for %2.").arg(pendingTrace_.size()).arg(className));
}

void MainWindow::addPendingTraceToSelectedClass() {
  const int classIndex = currentSelectedClassIndex();
  if (classIndex < 0 || classIndex >= static_cast<int>(classes_.size())) {
    QMessageBox::information(this, tr("Class unavailable"), tr("Please select a class before adding a ROI trace."));
    return;
  }
  if (pendingTrace_.size() < 3) {
    view_->setToolMode(SegmentationView::TraceTool);
    infoLabel_->setText(tr("Draw a freehand ROI for %1, then click Add ROI to selected class.").arg(classes_[classIndex].name));
    return;
  }
  pushUndoSnapshot();
  ensureAnnotationStorage();
  classTraceRegions_[classIndex].push_back({defaultTraceNameForClass(classIndex), pendingTrace_});
  pendingTrace_.clear();
  view_->clearPendingTrace();
  view_->setToolMode(SegmentationView::TraceTool);
  rebuildMasksFromAnnotations();
  if (traceList_) {
    traceList_->setCurrentRow(traceList_->count() - 1);
  }
  infoLabel_->setText(tr("Added ROI trace to %1.").arg(classes_[classIndex].name));
  logMessage(tr("Added ROI trace to %1.").arg(classes_[classIndex].name));
}

void MainWindow::removeSelectedTraceFromClass() {
  const int classIndex = currentSelectedClassIndex();
  if (!traceList_ || classIndex < 0 || classIndex >= static_cast<int>(classTraceRegions_.size())) {
    return;
  }
  const int row = traceList_->currentRow();
  if (row < 0 || row >= static_cast<int>(classTraceRegions_[classIndex].size())) {
    return;
  }
  pushUndoSnapshot();
  classTraceRegions_[classIndex].erase(classTraceRegions_[classIndex].begin() + row);
  rebuildMasksFromAnnotations();
  updateTraceLists();
  logMessage(tr("Removed trace %1 from %2.").arg(row + 1).arg(classes_[classIndex].name));
}

bool MainWindow::loadImageFile(const QString& path) {
  std::vector<cv::Mat> slices;
  QString error;
  if (!ImageIOUtils::loadImageVolume(path, &slices, &error) || slices.empty()) {
    QMessageBox::warning(this, tr("Open image failed"), error.isEmpty() ? tr("Unable to load image: %1").arg(path) : error);
    return false;
  }

  imageVolume_ = std::move(slices);
  currentSliceIndex_ = 0;
  originalImage_ = imageVolume_.front().clone();
  featureVolume_.assign(imageVolume_.size(), cv::Mat());
  sliceAnnotationStates_.assign(imageVolume_.size(), {});
  sliceLabelResults_.assign(imageVolume_.size(), cv::Mat());
  sliceProbabilityResults_.assign(imageVolume_.size(), cv::Mat());
  imagePath_ = path;
  imagePathLabel_->setText(path);
  model_.clear();
  trainingStats_ = {};
  clearAnnotationsForCurrentImage();
  clearInferenceOutputs();
  rebuildFeatureStack();
  updateViewer();
  sliceSlider_->blockSignals(true);
  sliceSlider_->setRange(0, std::max(0, static_cast<int>(imageVolume_.size()) - 1));
  sliceSlider_->setValue(0);
  sliceSlider_->blockSignals(false);
  updateUiState();
  logMessage(tr("Loaded image %1 (%2 x %3, %4 slice(s)).")
                 .arg(QFileInfo(path).fileName())
                 .arg(originalImage_.cols)
                 .arg(originalImage_.rows)
                 .arg(imageVolume_.size()));
  return true;
}

bool MainWindow::applyClassifierToImage(const cv::Mat& image, const QString& path, bool resetAnnotations) {
  if (image.empty() || !model_.isValid()) {
    return false;
  }
  originalImage_ = image.clone();
  imageVolume_.assign(1, originalImage_);
  currentSliceIndex_ = 0;
  imagePath_ = path;
  imagePathLabel_->setText(path.isEmpty() ? tr("Unsaved image") : path);
  featureVolume_.assign(1, cv::Mat());
  sliceAnnotationStates_.assign(1, {});
  sliceLabelResults_.assign(1, cv::Mat());
  sliceProbabilityResults_.assign(1, cv::Mat());
  clearInferenceOutputs();
  if (resetAnnotations) {
    clearAnnotationsForCurrentImage();
  } else {
    ensureAnnotationStorage();
    rebuildMasksFromAnnotations();
  }
  rebuildFeatureStack();
  labelResult_ = applyModelLabelsWithExclusion(model_, featureStack_, originalImage_.rows, originalImage_.cols, inferenceMask_, exclusionMask_);
  sliceLabelResults_[0] = labelResult_.clone();
  updateProbabilityView();
  updateViewer();
  updateUiState();
  return !labelResult_.empty();
}

bool MainWindow::saveTrainingData(const QString& path) {
  if (QFileInfo(path).suffix().compare("arff", Qt::CaseInsensitive) == 0) {
    return saveTrainingDataArff(path);
  }
  return saveTrainingDataJson(path);
}

bool MainWindow::saveTrainingDataJson(const QString& path) {
  if (!ensureImageLoaded(tr("saving training data"))) {
    return false;
  }
  saveCurrentSliceState();
  ensureSliceStorage();
  QFileInfo info(path);
  QDir dir = info.dir();
  const QString base = info.completeBaseName();
  if (!dir.exists() && !dir.mkpath(".")) {
    return false;
  }

  QJsonObject root;
  root["imagePath"] = imagePath_;
  root["currentSliceIndex"] = currentSliceIndex_;
  root["sliceCount"] = static_cast<int>(imageVolume_.size());
  QJsonArray classArray;
  for (const auto& cls : classes_) {
    QJsonObject clsObj;
    clsObj["name"] = cls.name;
    clsObj["color"] = cls.color.name(QColor::HexRgb);
    classArray.append(clsObj);
  }
  root["classes"] = classArray;
  root["featureSettings"] = featureSettingsToJson(featureSettings_);
  root["classifierSettings"] = QJsonObject{{"kind", static_cast<int>(classifierSettings_.kind)},
                                           {"randomForestTrees", classifierSettings_.randomForestTrees},
                                           {"randomForestMaxDepth", classifierSettings_.randomForestMaxDepth},
                                           {"svmC", classifierSettings_.svmC},
                                           {"svmGamma", classifierSettings_.svmGamma},
                                           {"knnNeighbors", classifierSettings_.knnNeighbors},
                                           {"logisticLearningRate", classifierSettings_.logisticLearningRate},
                                           {"logisticIterations", classifierSettings_.logisticIterations},
                                           {"balanceClasses", classifierSettings_.balanceClasses}};

  QJsonArray slicesArray;
  for (int sliceIndex = 0; sliceIndex < static_cast<int>(sliceAnnotationStates_.size()); ++sliceIndex) {
    const AnnotationSnapshot& state = sliceAnnotationStates_[sliceIndex];
    QJsonObject sliceObject;
    sliceObject["index"] = sliceIndex;
    QJsonArray sliceClasses;
    for (int classIndex = 0; classIndex < static_cast<int>(classes_.size()); ++classIndex) {
      const QString maskName = QString("%1_slice_%2_class_%3.png").arg(base).arg(sliceIndex).arg(classIndex);
      cv::Mat mask = classIndex < static_cast<int>(state.classBrushMasks.size())
                         ? state.classBrushMasks[classIndex]
                         : cv::Mat(imageVolume_[sliceIndex].rows, imageVolume_[sliceIndex].cols, CV_8U, cv::Scalar(0));
      if (mask.empty()) {
        mask = cv::Mat(imageVolume_[sliceIndex].rows, imageVolume_[sliceIndex].cols, CV_8U, cv::Scalar(0));
      }
      cv::imwrite(cvPath(dir.filePath(maskName)), mask);
      QJsonObject cls;
      cls["mask"] = maskName;
      QJsonArray traces;
      if (classIndex < static_cast<int>(state.classTraceRegions.size())) {
        for (const TraceRegion& region : state.classTraceRegions[classIndex]) {
          QJsonObject traceObject;
          traceObject["name"] = region.name;
          QJsonArray tracePoints;
          for (const QPoint& point : region.polygon) {
            tracePoints.append(QJsonObject{{"x", point.x()}, {"y", point.y()}});
          }
          traceObject["points"] = tracePoints;
          traces.append(traceObject);
        }
      }
      cls["traces"] = traces;
      sliceClasses.append(cls);
    }
    QJsonArray inferenceRegions;
    for (const QPolygon& polygon : state.inferenceRegions) {
      QJsonArray points;
      for (const QPoint& point : polygon) {
        points.append(QJsonObject{{"x", point.x()}, {"y", point.y()}});
      }
      inferenceRegions.append(points);
    }
    QJsonArray exclusionRegions;
    for (const QPolygon& polygon : state.exclusionRegions) {
      QJsonArray points;
      for (const QPoint& point : polygon) {
        points.append(QJsonObject{{"x", point.x()}, {"y", point.y()}});
      }
      exclusionRegions.append(points);
    }
    sliceObject["classes"] = sliceClasses;
    sliceObject["inferenceRegions"] = inferenceRegions;
    sliceObject["exclusionRegions"] = exclusionRegions;
    slicesArray.append(sliceObject);
  }
  root["slices"] = slicesArray;

  QFile file(path);
  if (!file.open(QIODevice::WriteOnly)) {
    return false;
  }
  file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  logMessage(tr("Saved training data to %1.").arg(path));
  return true;
}

bool MainWindow::loadTrainingData(const QString& path) {
  if (QFileInfo(path).suffix().compare("arff", Qt::CaseInsensitive) == 0) {
    return loadTrainingDataArff(path);
  }
  return loadTrainingDataJson(path);
}

bool MainWindow::loadTrainingDataJson(const QString& path) {
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) {
    return false;
  }
  const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (!doc.isObject()) {
    return false;
  }
  const QJsonObject root = doc.object();
  const QString imagePath = root["imagePath"].toString();
  if (!loadImageFile(imagePath)) {
    return false;
  }

  const QJsonArray classArray = root["classes"].toArray();
  classes_.clear();
  for (const auto& value : classArray) {
    const QJsonObject clsObj = value.toObject();
    classes_.push_back({clsObj["name"].toString(), QColor(clsObj["color"].toString()), true});
  }
  if (classes_.empty()) {
    initializeDefaultClasses();
  }
  refreshClassList();

  loadFeatureSettingsFromJson(root["featureSettings"].toObject(), &featureSettings_);

  const QJsonObject classifierSettings = root["classifierSettings"].toObject();
  if (!classifierSettings.isEmpty()) {
    classifierSettings_.kind = static_cast<SegmentationClassifierSettings::Kind>(classifierSettings["kind"].toInt(static_cast<int>(classifierSettings_.kind)));
    classifierSettings_.randomForestTrees = classifierSettings["randomForestTrees"].toInt(classifierSettings_.randomForestTrees);
    classifierSettings_.randomForestMaxDepth = classifierSettings["randomForestMaxDepth"].toInt(classifierSettings_.randomForestMaxDepth);
    classifierSettings_.svmC = classifierSettings["svmC"].toDouble(classifierSettings_.svmC);
    classifierSettings_.svmGamma = classifierSettings["svmGamma"].toDouble(classifierSettings_.svmGamma);
    classifierSettings_.knnNeighbors = classifierSettings["knnNeighbors"].toInt(classifierSettings_.knnNeighbors);
    classifierSettings_.logisticLearningRate = classifierSettings["logisticLearningRate"].toDouble(classifierSettings_.logisticLearningRate);
    classifierSettings_.logisticIterations = classifierSettings["logisticIterations"].toInt(classifierSettings_.logisticIterations);
    classifierSettings_.balanceClasses = classifierSettings["balanceClasses"].toBool(classifierSettings_.balanceClasses);
  }

  ensureSliceStorage();
  for (auto& state : sliceAnnotationStates_) {
    state.classBrushMasks.clear();
    state.classTraceRegions.clear();
    state.inferenceRegions.clear();
    state.exclusionRegions.clear();
  }
  const QJsonArray slicesArray = root["slices"].toArray();
  for (const auto& sliceValue : slicesArray) {
    const QJsonObject sliceObject = sliceValue.toObject();
    const int sliceIndex = sliceObject["index"].toInt();
    if (sliceIndex < 0 || sliceIndex >= static_cast<int>(sliceAnnotationStates_.size())) continue;
    AnnotationSnapshot snapshot;
    snapshot.classBrushMasks.assign(classes_.size(), cv::Mat());
    snapshot.classTraceRegions.assign(classes_.size(), {});
    const QJsonArray sliceClasses = sliceObject["classes"].toArray();
    for (int classIndex = 0; classIndex < sliceClasses.size() && classIndex < static_cast<int>(classes_.size()); ++classIndex) {
      const QJsonObject clsObj = sliceClasses[classIndex].toObject();
      cv::Mat mask = cv::imread(cvPath(QFileInfo(path).dir().filePath(clsObj["mask"].toString())), cv::IMREAD_GRAYSCALE);
      if (mask.empty()) {
        mask = cv::Mat(imageVolume_[sliceIndex].rows, imageVolume_[sliceIndex].cols, CV_8U, cv::Scalar(0));
      }
      snapshot.classBrushMasks[classIndex] = mask;
      for (const auto& traceValue : clsObj["traces"].toArray()) {
        const QJsonObject traceObject = traceValue.toObject();
        QPolygon polygon;
        for (const auto& pointValue : traceObject["points"].toArray()) {
          const QJsonObject pointObject = pointValue.toObject();
          polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
        }
        snapshot.classTraceRegions[classIndex].push_back({traceObject["name"].toString(), polygon});
      }
    }
    for (const auto& regionValue : sliceObject["inferenceRegions"].toArray()) {
      QPolygon polygon;
      for (const auto& pointValue : regionValue.toArray()) {
        const QJsonObject pointObject = pointValue.toObject();
        polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
      }
      if (polygon.size() >= 3) snapshot.inferenceRegions.push_back(polygon);
    }
    for (const auto& regionValue : sliceObject["exclusionRegions"].toArray()) {
      QPolygon polygon;
      for (const auto& pointValue : regionValue.toArray()) {
        const QJsonObject pointObject = pointValue.toObject();
        polygon << QPoint(pointObject["x"].toInt(), pointObject["y"].toInt());
      }
      if (polygon.size() >= 3) snapshot.exclusionRegions.push_back(polygon);
    }
    sliceAnnotationStates_[sliceIndex] = snapshot;
  }

  currentSliceIndex_ = std::clamp(root["currentSliceIndex"].toInt(0), 0, std::max(0, static_cast<int>(imageVolume_.size()) - 1));
  originalImage_ = imageVolume_[currentSliceIndex_].clone();
  restoreSliceState(currentSliceIndex_);
  rebuildFeatureStack();
  const int comboIndex = classifierCombo_->findData(classifierSettings_.kind);
  if (comboIndex >= 0) classifierCombo_->setCurrentIndex(comboIndex);
  sliceSlider_->blockSignals(true);
  sliceSlider_->setRange(0, std::max(0, static_cast<int>(imageVolume_.size()) - 1));
  sliceSlider_->setValue(currentSliceIndex_);
  sliceSlider_->blockSignals(false);
  updateViewer();
  updateUiState();
  clearImportedTrainingData();
  logMessage(tr("Loaded training data from %1.").arg(path));
  return true;
}

bool MainWindow::saveTrainingDataArff(const QString& path) {
  if (!ensureImageLoaded(tr("saving training data"))) {
    return false;
  }
  saveCurrentSliceState();

  cv::Mat allSamples;
  cv::Mat allLabels;
  for (int sliceIndex = 0; sliceIndex < static_cast<int>(imageVolume_.size()); ++sliceIndex) {
    cv::Mat features = sliceIndex < static_cast<int>(featureVolume_.size()) && !featureVolume_[sliceIndex].empty()
                           ? featureVolume_[sliceIndex]
                           : SegmentationEngine::computeFeatureStack(imageVolume_[sliceIndex], featureSettings_);
    if (features.empty()) continue;
    cv::Mat labels;
    const auto masks = masksFromAnnotationSnapshot(sliceAnnotationStates_[sliceIndex], imageVolume_[sliceIndex].size(), static_cast<int>(classes_.size()));
    const cv::Mat samples = SegmentationEngine::gatherSamples(features, masks, &labels, 120000, false);
    if (samples.empty() || labels.empty()) continue;
    if (allSamples.empty()) allSamples = samples.clone();
    else cv::vconcat(allSamples, samples, allSamples);
    if (allLabels.empty()) allLabels = labels.clone();
    else cv::vconcat(allLabels, labels, allLabels);
  }

  if (allSamples.empty() || allLabels.empty()) {
    return false;
  }

  QFile file(path);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
    return false;
  }

  QTextStream stream(&file);
  stream.setRealNumberNotation(QTextStream::FixedNotation);
  stream.setRealNumberPrecision(7);
  stream << "% QtTrainableSegmentation ARFF export\n";
  stream << "% imagePath=" << imagePath_ << "\n";
  stream << "% classCount=" << classes_.size() << "\n";
  stream << "% featureSettings=" << QString::fromUtf8(QJsonDocument(featureSettingsToJson(featureSettings_)).toJson(QJsonDocument::Compact)) << "\n";
  stream << "@RELATION qt_trainable_segmentation\n\n";
  for (int col = 0; col < allSamples.cols; ++col) {
    stream << "@ATTRIBUTE feature_" << QString("%1").arg(col, 4, 10, QLatin1Char('0')) << " NUMERIC\n";
  }
  QStringList classNames;
  for (const auto& cls : classes_) {
    classNames << arffEscape(cls.name);
  }
  stream << "@ATTRIBUTE class {" << classNames.join(",") << "}\n\n";
  stream << "@DATA\n";
  for (int row = 0; row < allSamples.rows; ++row) {
    QStringList values;
    values.reserve(allSamples.cols + 1);
    for (int col = 0; col < allSamples.cols; ++col) {
      values << QString::number(allSamples.at<float>(row, col), 'f', 7);
    }
    const int cls = allLabels.at<int>(row, 0);
    if (cls < 0 || cls >= static_cast<int>(classes_.size())) {
      continue;
    }
    values << arffEscape(classes_[cls].name);
    stream << values.join(",") << "\n";
  }
  logMessage(tr("Saved ARFF training data to %1 (%2 samples).").arg(path).arg(allSamples.rows));
  return true;
}

bool MainWindow::loadTrainingDataArff(const QString& path) {
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return false;
  }

  QStringList classNames;
  std::vector<std::vector<float>> rows;
  std::vector<int> labels;
  bool inData = false;
  int featureCount = -1;
  QTextStream stream(&file);
  while (!stream.atEnd()) {
    const QString rawLine = stream.readLine();
    const QString line = rawLine.trimmed();
    if (line.isEmpty() || line.startsWith('%')) continue;
    const QString lower = line.toLower();
    if (!inData) {
      if (lower.startsWith("@attribute")) {
        if (line.contains('{') && line.contains('}')) {
          const int start = line.indexOf('{');
          const int end = line.lastIndexOf('}');
          const QStringList tokens = line.mid(start + 1, end - start - 1).split(',', Qt::SkipEmptyParts);
          classNames.clear();
          for (QString token : tokens) {
            token = token.trimmed();
            if (token.startsWith('\'')) token.remove(0, 1);
            if (token.endsWith('\'')) token.chop(1);
            token.replace("\\'", "'");
            token.replace("\\\\", "\\");
            classNames << token;
          }
        } else {
          featureCount += 1;
        }
      } else if (lower == "@data") {
        inData = true;
      }
      continue;
    }

    const QStringList tokens = line.split(',', Qt::KeepEmptyParts);
    if (tokens.size() != featureCount + 2 || classNames.isEmpty()) {
      return false;
    }
    std::vector<float> featureRow;
    featureRow.reserve(featureCount + 1);
    for (int index = 0; index <= featureCount; ++index) {
      bool ok = false;
      const float value = tokens[index].trimmed().toFloat(&ok);
      if (!ok) return false;
      featureRow.push_back(value);
    }
    QString classToken = tokens.back().trimmed();
    if (classToken.startsWith('\'')) classToken.remove(0, 1);
    if (classToken.endsWith('\'')) classToken.chop(1);
    classToken.replace("\\'", "'");
    classToken.replace("\\\\", "\\");
    const int classIndex = classNames.indexOf(classToken);
    if (classIndex < 0) {
      return false;
    }
    rows.push_back(std::move(featureRow));
    labels.push_back(classIndex);
  }

  if (rows.empty() || classNames.isEmpty()) {
    return false;
  }
  const int importedFeatureCount = featureCount + 1;
  if (importedFeatureCount <= 0) {
    return false;
  }
  if (!featureStack_.empty() && featureStack_.cols != importedFeatureCount) {
    return false;
  }

  if (static_cast<int>(classes_.size()) != classNames.size()) {
    return false;
  }
  for (int i = 0; i < classNames.size(); ++i) {
    if (classes_[i].name != classNames[i]) {
      return false;
    }
  }

  importedTrainingSamples_ = cv::Mat(static_cast<int>(rows.size()), importedFeatureCount, CV_32F);
  importedTrainingLabels_ = cv::Mat(static_cast<int>(rows.size()), 1, CV_32S);
  for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
    for (int col = 0; col < importedFeatureCount; ++col) {
      importedTrainingSamples_.at<float>(row, col) = rows[row][col];
    }
    importedTrainingLabels_.at<int>(row, 0) = labels[row];
  }

  importedTrainingSource_ = path;
  model_.clear();
  trainingStats_ = {};
  clearInferenceOutputs();
  updateUiState();
  logMessage(tr("Loaded ARFF training vectors from %1 (%2 samples).").arg(path).arg(importedTrainingSamples_.rows));
  return true;
}

bool MainWindow::saveProject(const QString& path) {
  persistCurrentProjectImageState();
  QFileInfo info(path);
  QDir dir = info.dir();
  if (!dir.exists() && !dir.mkpath(".")) {
    return false;
  }
  const QString annotationDirPath = dir.filePath(info.completeBaseName() + "_annotations");
  QDir().mkpath(annotationDirPath);

  QJsonObject root;
  QJsonArray versionHistory = readVersionHistory(path);
  versionHistory.append(QJsonObject{{"timestamp", QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
                                    {"imageCount", static_cast<int>(projectImages_.size())},
                                    {"classifier", model_.classifierName()}});
  root["versionHistory"] = versionHistory;
  QJsonArray images;
  for (int index = 0; index < static_cast<int>(projectImages_.size()); ++index) {
    ProjectImageEntry& entry = projectImages_[index];
    if (entry.workingDataPath.isEmpty()) {
      entry.workingDataPath = ensureEntryWorkingPath(index);
    }
    const QString targetAnnotation = QDir(annotationDirPath).filePath(QString("image_%1.json").arg(index, 3, 10, QLatin1Char('0')));
    QFile::remove(targetAnnotation);
    if (!entry.workingDataPath.isEmpty() && QFile::exists(entry.workingDataPath)) {
      QFile::copy(entry.workingDataPath, targetAnnotation);
    }
    QJsonObject imageObject;
    imageObject["imagePath"] = entry.imagePath;
    imageObject["annotationPath"] = QDir(info.dir()).relativeFilePath(targetAnnotation);
    images.append(imageObject);
  }
  root["images"] = images;
  root["currentIndex"] = currentProjectImageIndex_;

  QFile file(path);
  if (!file.open(QIODevice::WriteOnly)) {
    return false;
  }
  const QByteArray manifestBytes = QJsonDocument(root).toJson(QJsonDocument::Indented);
  file.write(manifestBytes);
  const QString versionDir = dir.filePath(info.completeBaseName() + "_versions");
  QDir().mkpath(versionDir);
  const QString versionName = QString("project_%1.json").arg(QDateTime::currentDateTimeUtc().toString("yyyyMMdd_hhmmss"));
  QFile versionFile(QDir(versionDir).filePath(versionName));
  if (versionFile.open(QIODevice::WriteOnly)) {
    versionFile.write(manifestBytes);
  }
  projectPath_ = path;
  logMessage(tr("Saved project to %1.").arg(path));
  return true;
}

bool MainWindow::loadProject(const QString& path) {
  QFile file(path);
  if (!file.open(QIODevice::ReadOnly)) {
    return false;
  }
  const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
  if (!doc.isObject()) {
    return false;
  }
  const QJsonObject root = doc.object();
  projectImages_.clear();
  const QJsonArray images = root["images"].toArray();
  for (const auto& value : images) {
    const QJsonObject imageObject = value.toObject();
    ProjectImageEntry entry;
    entry.imagePath = imageObject["imagePath"].toString();
    entry.workingDataPath = QFileInfo(path).dir().filePath(imageObject["annotationPath"].toString());
    projectImages_.push_back(entry);
  }
  projectPath_ = path;
  updateProjectList();
  if (!projectImages_.empty()) {
    const int index = std::clamp(root["currentIndex"].toInt(0), 0, static_cast<int>(projectImages_.size()) - 1);
    switchToProjectImage(index);
  }
  logMessage(tr("Loaded project %1.").arg(path));
  return true;
}

QString MainWindow::ensureEntryWorkingPath(int index) {
  if (index < 0 || index >= static_cast<int>(projectImages_.size())) {
    return QString();
  }
  if (projectImages_[index].workingDataPath.isEmpty()) {
    const QString root = sessionDataRoot();
    projectImages_[index].workingDataPath = QDir(root).filePath(QString("entry_%1.json").arg(index, 3, 10, QLatin1Char('0')));
  }
  return projectImages_[index].workingDataPath;
}

void MainWindow::persistCurrentProjectImageState() {
  if (currentProjectImageIndex_ < 0 || currentProjectImageIndex_ >= static_cast<int>(projectImages_.size())) {
    return;
  }
  const QString dataPath = ensureEntryWorkingPath(currentProjectImageIndex_);
  if (!dataPath.isEmpty()) {
    saveTrainingData(dataPath);
  }
}

void MainWindow::updateProjectList() {
  if (!projectList_) return;
  switchingProjectSelection_ = true;
  projectList_->clear();
  for (int i = 0; i < static_cast<int>(projectImages_.size()); ++i) {
    QListWidgetItem* item = new QListWidgetItem(QFileInfo(projectImages_[i].imagePath).fileName(), projectList_);
    item->setData(Qt::UserRole, i);
  }
  if (currentProjectImageIndex_ >= 0 && currentProjectImageIndex_ < projectList_->count()) {
    projectList_->setCurrentRow(currentProjectImageIndex_);
  }
  switchingProjectSelection_ = false;
}

void MainWindow::switchToProjectImage(int index) {
  if (index < 0 || index >= static_cast<int>(projectImages_.size())) {
    return;
  }
  persistCurrentProjectImageState();
  currentProjectImageIndex_ = index;
  const ProjectImageEntry& entry = projectImages_[index];
  if (!entry.workingDataPath.isEmpty() && QFile::exists(entry.workingDataPath)) {
    loadTrainingData(entry.workingDataPath);
  } else {
    loadImageFile(entry.imagePath);
  }
  updateProjectList();
}

QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
  if (mat.empty()) {
    return QImage();
  }
  if (mat.type() == CV_8UC3) {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
  }
  if (mat.type() == CV_8UC1) {
    return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8).copy();
  }
  if (mat.type() == CV_32F) {
    cv::Mat normalized;
    cv::normalize(mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_TURBO);
    return cvMatToQImage(colored);
  }
  if (mat.type() == CV_32S) {
    cv::Mat converted;
    mat.convertTo(converted, CV_8U);
    return cvMatToQImage(converted);
  }
  if (mat.type() == CV_8UC4) {
    return QImage(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32).copy();
  }
  return QImage();
}

cv::Mat MainWindow::qImageToCvMat(const QImage& image) {
  QImage converted = image.convertToFormat(QImage::Format_RGB888);
  cv::Mat mat(converted.height(), converted.width(), CV_8UC3, const_cast<uchar*>(converted.bits()), converted.bytesPerLine());
  cv::Mat bgr;
  cv::cvtColor(mat, bgr, cv::COLOR_RGB2BGR);
  return bgr.clone();
}

void MainWindow::onOpenImage() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Open image"), QString(), tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
  if (!path.isEmpty()) {
    loadImageFile(path);
  }
}

void MainWindow::onTrainClassifier() {
  if (!ensureImageLoaded(tr("training the classifier"))) {
    return;
  }
  if (trainingTaskRunning_) {
    return;
  }
  trainingInProgress_ = true;
  trainingTaskRunning_ = true;
  stopTrainingRequested_ = false;
  updateUiState();
  QProgressDialog progress(tr("Training classifier in background..."), tr("Cancel"), 0, 0, this);
  progress.setWindowModality(Qt::ApplicationModal);
  progress.setMinimumDuration(0);
  progress.setAutoClose(false);
  progress.setValue(0);
  progress.show();
  QApplication::processEvents();

  rebuildFeatureStack();
  const cv::Mat features = featureStack_.clone();
  const auto masks = cloneMaskVector(classMasks_);
  const cv::Mat importedSamples = importedTrainingSamples_.clone();
  const cv::Mat importedLabels = importedTrainingLabels_.clone();
  const auto classesCount = static_cast<int>(classes_.size());
  const SegmentationClassifierSettings settings = classifierSettings_;
  std::atomic_bool cancelRequested{false};

  struct TrainingTaskResult {
    bool canceled = false;
    bool success = false;
    SegmentationClassifier model;
    SegmentationTrainingStats stats;
    QString error;
  };

  QFuture<TrainingTaskResult> future = QtConcurrent::run([features, masks, importedSamples, importedLabels, classesCount, settings, &cancelRequested]() {
    TrainingTaskResult result;
    if (cancelRequested.load()) {
      result.canceled = true;
      return result;
    }
    cv::Mat labels;
    cv::Mat samples = SegmentationEngine::gatherSamples(features, masks, &labels, 12000, settings.balanceClasses);
    if (!importedSamples.empty() && !importedLabels.empty()) {
      if (samples.empty()) samples = importedSamples.clone();
      else cv::vconcat(samples, importedSamples, samples);
      if (labels.empty()) labels = importedLabels.clone();
      else cv::vconcat(labels, importedLabels, labels);
    }
    if (samples.empty() || labels.empty()) {
      result.error = QStringLiteral("no_samples");
      return result;
    }
    if (cancelRequested.load()) {
      result.canceled = true;
      return result;
    }
    if (!result.model.train(samples, labels, classesCount, settings, &result.stats)) {
      result.error = QStringLiteral("train_failed");
      return result;
    }
    result.success = true;
    return result;
  });

  while (!future.isFinished()) {
    QApplication::processEvents(QEventLoop::AllEvents, 30);
    if (progress.wasCanceled()) {
      cancelRequested.store(true);
      stopTrainingRequested_ = true;
    }
  }
  const TrainingTaskResult result = future.result();
  progress.setValue(progress.maximum());
  trainingInProgress_ = false;
  trainingTaskRunning_ = false;
  updateUiState();

  if (result.canceled || stopTrainingRequested_) {
    logMessage(tr("Training cancelled."));
    return;
  }
  if (!result.success) {
    if (result.error == QStringLiteral("no_samples")) {
      QMessageBox::information(this, tr("No annotations"), tr("Please paint at least one pixel for each class before training."));
    } else {
      QMessageBox::warning(this, tr("Training failed"), tr("Training failed. Ensure every class has annotations."));
    }
    return;
  }
  model_ = result.model;
  trainingStats_ = result.stats;
  labelResult_ = applyModelLabelsWithExclusion(model_, featureStack_, originalImage_.rows, originalImage_.cols, inferenceMask_, exclusionMask_);
  if (!sliceLabelResults_.empty()) {
    sliceLabelResults_[currentSliceIndex_] = labelResult_.clone();
  }
  updateProbabilityView(false);
  updateViewer();
  logMessage(tr("Trained classifier with %1 samples across %2 classes.").arg(trainingStats_.sampleCount).arg(trainingStats_.classCount));
}

void MainWindow::onStopTraining() {
  if (!trainingInProgress_) return;
  stopTrainingRequested_ = true;
  logMessage(tr("Stop requested. The current training stage will stop as soon as it can."));
}

void MainWindow::onApplyClassifier() {
  if (!ensureModelReady(tr("applying the classifier"))) {
    return;
  }
  QString targetPath;
  bool useCurrentImage = !imageVolume_.empty();
  if (!imageVolume_.empty()) {
    QMessageBox choiceBox(this);
    choiceBox.setWindowTitle(tr("Apply classifier"));
    choiceBox.setText(tr("Apply the classifier to the current image/stack, or choose another image file?"));
    QPushButton* currentButton = choiceBox.addButton(tr("Current image"), QMessageBox::AcceptRole);
    QPushButton* chooseButton = choiceBox.addButton(tr("Choose image..."), QMessageBox::ActionRole);
    choiceBox.addButton(QMessageBox::Cancel);
    choiceBox.exec();
    if (choiceBox.clickedButton() == chooseButton) {
      useCurrentImage = false;
    } else if (choiceBox.clickedButton() != currentButton) {
      return;
    }
  }

  std::vector<cv::Mat> targetVolume = imageVolume_;
  if (!useCurrentImage) {
    targetPath = QFileDialog::getOpenFileName(this, tr("Apply classifier to image"), imagePath_, tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (targetPath.isEmpty()) return;
    QString error;
    if (!ImageIOUtils::loadImageVolume(targetPath, &targetVolume, &error) || targetVolume.empty()) {
      QMessageBox::warning(this, tr("Open image failed"), error);
      return;
    }
  }

  if (targetVolume.empty()) return;
  imageVolume_ = targetVolume;
  imagePath_ = useCurrentImage ? imagePath_ : targetPath;
  imagePathLabel_->setText(imagePath_);
  featureVolume_.assign(imageVolume_.size(), cv::Mat());
  if (!useCurrentImage) {
    sliceAnnotationStates_.assign(imageVolume_.size(), {});
  }
  sliceLabelResults_.assign(imageVolume_.size(), cv::Mat());
  sliceProbabilityResults_.assign(imageVolume_.size(), cv::Mat());
  QProgressDialog progress(tr("Applying classifier..."), QString(), 0, static_cast<int>(imageVolume_.size()), this);
  progress.setWindowModality(Qt::ApplicationModal);
  progress.setCancelButton(nullptr);
  progress.setMinimumDuration(0);
  progress.setValue(0);
  progress.show();
  QApplication::processEvents();
  for (int sliceIndex = 0; sliceIndex < static_cast<int>(imageVolume_.size()); ++sliceIndex) {
    const cv::Mat features = SegmentationEngine::computeFeatureStack(imageVolume_[sliceIndex], featureSettings_);
    featureVolume_[sliceIndex] = features;
    sliceLabelResults_[sliceIndex] = applyModelLabelsWithExclusion(model_, features, imageVolume_[sliceIndex].rows, imageVolume_[sliceIndex].cols, inferenceMask_, exclusionMask_);
    if (model_.supportsProbability()) {
      sliceProbabilityResults_[sliceIndex] = applyModelProbabilitiesWithExclusion(model_, features, imageVolume_[sliceIndex].rows, imageVolume_[sliceIndex].cols,
                                                                                  probabilityCombo_->currentData().toInt(), inferenceMask_, exclusionMask_);
    }
    progress.setValue(sliceIndex + 1);
    QApplication::processEvents();
  }
  currentSliceIndex_ = std::clamp(currentSliceIndex_, 0, std::max(0, static_cast<int>(imageVolume_.size()) - 1));
  originalImage_ = imageVolume_[currentSliceIndex_].clone();
  featureStack_ = featureVolume_[currentSliceIndex_].clone();
  labelResult_ = sliceLabelResults_[currentSliceIndex_].clone();
  probabilityImage_ = sliceProbabilityResults_[currentSliceIndex_].clone();
  sliceSlider_->blockSignals(true);
  sliceSlider_->setRange(0, std::max(0, static_cast<int>(imageVolume_.size()) - 1));
  sliceSlider_->setValue(currentSliceIndex_);
  sliceSlider_->blockSignals(false);
  updateViewer();
  updateUiState();
  logMessage(tr("Applied classifier to %1 slice(s).").arg(imageVolume_.size()));
}

void MainWindow::onToggleOverlay() {
  showOverlay_ = !showOverlay_;
  overlayCheck_->setChecked(showOverlay_);
  updateViewer();
}

void MainWindow::onCreateResult() {
  if (!ensureImageLoaded(tr("creating the result")) || !ensureModelReady(tr("creating the result"))) {
    return;
  }
  if (labelResult_.empty()) {
    onApplyClassifier();
    if (labelResult_.empty()) return;
  }
  QDialog dialog(this);
  dialog.setWindowTitle(tr("Segmentation result"));
  dialog.resize(1200, 760);
  auto* layout = new QHBoxLayout(&dialog);
  auto* resultView = new SegmentationView(&dialog);
  resultView->setPaintEnabled(false);
  resultView->setToolMode(SegmentationView::PanTool);
  resultView->setBaseImage(cvMatToQImage(originalImage_));
  resultView->setOverlayImage(cvMatToQImage(SegmentationEngine::makeOverlay(originalImage_, labelResult_, classes_, overlayOpacity_, contourCheck_->isChecked())));
  layout->addWidget(resultView);
  dialog.exec();
}

void MainWindow::onShowBinaryResult() {
  if (!ensureImageLoaded(tr("showing binary result")) || !ensureModelReady(tr("showing binary result"))) {
    return;
  }
  if (labelResult_.empty()) {
    onApplyClassifier();
    if (labelResult_.empty()) return;
  }
  if (classes_.empty()) {
    QMessageBox::information(this, tr("No classes"), tr("No classes are available for binary export."));
    return;
  }

  QStringList classNames;
  classNames.reserve(static_cast<int>(classes_.size()));
  for (int i = 0; i < static_cast<int>(classes_.size()); ++i) {
    classNames << QString("%1 (Class %2)").arg(classes_[i].name).arg(i + 1);
  }
  const int defaultClass = std::clamp(probabilityCombo_->currentData().toInt(), 0, std::max(0, static_cast<int>(classes_.size()) - 1));
  bool ok = false;
  const QString selected = QInputDialog::getItem(this,
                                                 tr("Binary result class"),
                                                 tr("Choose target class"),
                                                 classNames,
                                                 defaultClass,
                                                 false,
                                                 &ok);
  if (!ok || selected.isEmpty()) return;
  const int targetClass = classNames.indexOf(selected);
  if (targetClass < 0) return;

  cv::Mat binaryMask = (labelResult_ == targetClass);
  cv::Mat binaryU8;
  binaryMask.convertTo(binaryU8, CV_8U, 255.0);
  QDialog dialog(this);
  dialog.setWindowTitle(tr("Binary result - %1").arg(classes_[targetClass].name));
  dialog.resize(1200, 760);
  auto* layout = new QHBoxLayout(&dialog);
  auto* resultView = new SegmentationView(&dialog);
  resultView->setPaintEnabled(false);
  resultView->setToolMode(SegmentationView::PanTool);
  resultView->setBaseImage(cvMatToQImage(binaryU8));
  layout->addWidget(resultView);
  dialog.exec();
}

void MainWindow::onGetProbability() {
  if (!ensureImageLoaded(tr("getting probability")) || !ensureModelReady(tr("getting probability"))) {
    return;
  }
  if (probabilityTaskRunning_) {
    return;
  }
  if (!model_.supportsProbability()) {
    QMessageBox::information(this, tr("Probability unavailable"), tr("The current classifier does not expose probabilities for this view."));
    return;
  }
  probabilityTaskRunning_ = true;
  const int cls = probabilityCombo_->currentData().toInt();
  const cv::Mat features = featureStack_.clone();
  const cv::Mat inferenceMask = inferenceMask_.clone();
  const cv::Mat exclusionMask = exclusionMask_.clone();
  const int rows = originalImage_.rows;
  const int cols = originalImage_.cols;
  const SegmentationClassifier modelCopy = model_;
  std::atomic_bool cancelRequested{false};

  QProgressDialog progress(tr("Computing probability map in background..."), tr("Cancel"), 0, 0, this);
  progress.setWindowModality(Qt::ApplicationModal);
  progress.setMinimumDuration(0);
  progress.setAutoClose(false);
  progress.show();
  QApplication::processEvents();

  QFuture<cv::Mat> future = QtConcurrent::run([modelCopy, features, inferenceMask, exclusionMask, rows, cols, cls, &cancelRequested]() {
    if (cancelRequested.load()) return cv::Mat();
    std::vector<int> active;
    if (!inferenceMask.empty() || !exclusionMask.empty()) {
      active.reserve(rows * cols);
      for (int row = 0; row < rows; ++row) {
        const uchar* includePtr = inferenceMask.empty() ? nullptr : inferenceMask.ptr<uchar>(row);
        const uchar* excludePtr = exclusionMask.empty() ? nullptr : exclusionMask.ptr<uchar>(row);
        for (int col = 0; col < cols; ++col) {
          const bool included = !includePtr || includePtr[col] > 0;
          const bool excluded = excludePtr && excludePtr[col] > 0;
          if (included && !excluded) active.push_back(row * cols + col);
        }
      }
      if (active.empty()) return cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
    }
    const bool useSubset = !active.empty();
    const cv::Mat inferFeatures = [&]() {
      if (!useSubset) return features;
      cv::Mat sampled(static_cast<int>(active.size()), features.cols, features.type());
      for (int i = 0; i < static_cast<int>(active.size()); ++i) {
        features.row(active[i]).copyTo(sampled.row(i));
      }
      return sampled;
    }();
    constexpr int kBatchSize = 4096;
    cv::Mat probabilities;
    for (int start = 0; start < inferFeatures.rows; start += kBatchSize) {
      if (cancelRequested.load()) return cv::Mat();
      const int end = std::min(inferFeatures.rows, start + kBatchSize);
      const cv::Mat batch = modelCopy.predictProbabilities(inferFeatures.rowRange(start, end));
      if (batch.empty()) return cv::Mat();
      if (probabilities.empty()) probabilities = cv::Mat::zeros(inferFeatures.rows, batch.cols, CV_32F);
      batch.copyTo(probabilities.rowRange(start, end));
    }
    if (probabilities.empty() || cls < 0 || cls >= probabilities.cols) return cv::Mat();
    cv::Mat channel(rows, cols, CV_32F);
    channel.setTo(0);
    if (useSubset) {
      for (int i = 0; i < static_cast<int>(active.size()); ++i) {
        channel.at<float>(active[i] / cols, active[i] % cols) = probabilities.at<float>(i, cls);
      }
    } else {
      for (int row = 0; row < rows; ++row) {
        float* out = channel.ptr<float>(row);
        for (int col = 0; col < cols; ++col) {
          out[col] = probabilities.at<float>(row * cols + col, cls);
        }
      }
    }
    return channel;
  });

  while (!future.isFinished()) {
    QApplication::processEvents(QEventLoop::AllEvents, 30);
    if (progress.wasCanceled()) {
      cancelRequested.store(true);
    }
  }
  probabilityTaskRunning_ = false;
  progress.setValue(progress.maximum());
  probabilityImage_ = future.result();
  if (probabilityImage_.empty()) {
    QMessageBox::information(this, tr("Probability unavailable"), tr("Probability output is not available for the current image/classifier combination."));
    return;
  }
  if (!sliceProbabilityResults_.empty() && currentSliceIndex_ < static_cast<int>(sliceProbabilityResults_.size())) {
    sliceProbabilityResults_[currentSliceIndex_] = probabilityImage_.clone();
  }
  QDialog dialog(this);
  dialog.setWindowTitle(tr("Probability map"));
  dialog.resize(1180, 760);
  auto* layout = new QHBoxLayout(&dialog);
  auto* resultView = new SegmentationView(&dialog);
  resultView->setPaintEnabled(false);
  resultView->setToolMode(SegmentationView::PanTool);
  resultView->setBaseImage(cvMatToQImage(probabilityImage_));
  layout->addWidget(resultView);
  dialog.exec();
}

void MainWindow::onPlotResult() {
  if (!ensureImageLoaded(tr("plotting the result"))) return;
  if (labelResult_.empty()) {
    if (!ensureModelReady(tr("plotting the result"))) return;
    onApplyClassifier();
    if (labelResult_.empty()) return;
  }
  QVector<double> ticks(classes_.size()), values(classes_.size());
  QVector<QString> labels(classes_.size());
  for (int cls = 0; cls < static_cast<int>(classes_.size()); ++cls) {
    cv::Mat mask = (labelResult_ == cls);
    values[cls] = static_cast<double>(cv::countNonZero(mask));
    ticks[cls] = cls + 1;
    labels[cls] = classes_[cls].name;
  }

  QDialog dialog(this);
  dialog.setWindowTitle(tr("Result statistics"));
  dialog.resize(960, 600);
  auto* layout = new QVBoxLayout(&dialog);
  auto* plot = new QCustomPlot(&dialog);
  plot->addGraph();
  plot->graph(0)->setLineStyle(QCPGraph::lsNone);
  plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 8));
  plot->graph(0)->setData(ticks, values);
  plot->xAxis->setAutoTicks(false);
  plot->xAxis->setAutoTickLabels(false);
  plot->xAxis->setTickVector(ticks);
  plot->xAxis->setTickVectorLabels(labels);
  plot->xAxis->setLabel(tr("Class"));
  plot->yAxis->setLabel(tr("Pixel count"));
  plot->rescaleAxes();
  plot->replot();
  layout->addWidget(plot);
  dialog.exec();
}

void MainWindow::onEvaluateModel() {
  if (!ensureImageLoaded(tr("evaluating the model")) || !ensureModelReady(tr("evaluating the model"))) {
    return;
  }
  cv::Mat labels;
  cv::Mat samples = SegmentationEngine::gatherSamples(featureStack_, classMasks_, &labels, 120000, false);
  if (samples.empty() || labels.empty()) {
    QMessageBox::information(this, tr("No annotations"), tr("Please annotate the current slice before evaluating."));
    return;
  }
  const cv::Mat predicted = model_.predictLabels(samples);
  if (predicted.empty()) {
    QMessageBox::warning(this, tr("Evaluation failed"), tr("The classifier could not produce predictions for the current slice annotations."));
    return;
  }

  std::vector<std::vector<int>> confusion(classes_.size(), std::vector<int>(classes_.size(), 0));
  for (int row = 0; row < labels.rows; ++row) {
    const int truth = labels.at<int>(row, 0);
    const int pred = predicted.at<int>(row, 0);
    if (truth >= 0 && truth < static_cast<int>(classes_.size()) && pred >= 0 && pred < static_cast<int>(classes_.size())) {
      confusion[truth][pred] += 1;
    }
  }

  QString summary;
  summary += tr("Samples: %1\n\nConfusion Matrix (rows=true, cols=pred):\n").arg(labels.rows);
  for (int r = 0; r < static_cast<int>(classes_.size()); ++r) {
    summary += classes_[r].name + ": ";
    for (int c = 0; c < static_cast<int>(classes_.size()); ++c) {
      summary += QString::number(confusion[r][c]) + (c + 1 == static_cast<int>(classes_.size()) ? "\n" : "  ");
    }
  }
  summary += "\nMetrics:\n";
  QVector<double> classTicks;
  QVector<double> classF1;
  QVector<QString> classNames;
  for (int cls = 0; cls < static_cast<int>(classes_.size()); ++cls) {
    double tp = confusion[cls][cls];
    double fp = 0.0;
    double fn = 0.0;
    for (int i = 0; i < static_cast<int>(classes_.size()); ++i) {
      if (i != cls) {
        fp += confusion[i][cls];
        fn += confusion[cls][i];
      }
    }
    const double precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
    const double recall = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
    const double f1 = (precision + recall) > 0.0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;
    classTicks.push_back(cls + 1);
    classF1.push_back(f1);
    classNames.push_back(classes_[cls].name);
    summary += tr("- %1: Precision=%2  Recall=%3  F1=%4\n")
                   .arg(classes_[cls].name)
                   .arg(QString::number(precision, 'f', 3))
                   .arg(QString::number(recall, 'f', 3))
                   .arg(QString::number(f1, 'f', 3));
  }

  cv::Mat probs;
  BinaryMetrics metrics;
  if (model_.supportsProbability()) {
    probs = model_.predictProbabilities(samples);
    const int classIndex = std::clamp(probabilityCombo_->currentData().toInt(), 0, std::max(0, static_cast<int>(classes_.size()) - 1));
    std::vector<float> scores;
    std::vector<int> truth;
    scores.reserve(probs.rows);
    truth.reserve(probs.rows);
    for (int row = 0; row < probs.rows; ++row) {
      scores.push_back(probs.at<float>(row, classIndex));
      truth.push_back(labels.at<int>(row, 0) == classIndex ? 1 : 0);
    }
    metrics = computeBinaryCurves(scores, truth);
    summary += tr("\nOne-vs-rest for %1: AUC=%2  AP=%3")
                   .arg(classes_[classIndex].name)
                   .arg(QString::number(metrics.auc, 'f', 3))
                   .arg(QString::number(metrics.ap, 'f', 3));
  }

  QDialog dialog(this);
  dialog.setWindowTitle(tr("Evaluation"));
  dialog.resize(1100, 720);
  auto* layout = new QVBoxLayout(&dialog);
  auto* text = new QTextEdit(&dialog);
  text->setReadOnly(true);
  text->setPlainText(summary);
  layout->addWidget(text, 1);
  if (!classF1.isEmpty()) {
    auto* f1Plot = new QCustomPlot(&dialog);
    f1Plot->addGraph();
    f1Plot->graph(0)->setLineStyle(QCPGraph::lsNone);
    f1Plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 7));
    f1Plot->graph(0)->setData(classTicks, classF1);
    f1Plot->xAxis->setAutoTicks(false);
    f1Plot->xAxis->setAutoTickLabels(false);
    f1Plot->xAxis->setTickVector(classTicks);
    f1Plot->xAxis->setTickVectorLabels(classNames);
    f1Plot->xAxis->setLabel(tr("Class"));
    f1Plot->yAxis->setLabel(tr("F1 score"));
    f1Plot->yAxis->setRange(0, 1);
    f1Plot->replot();
    layout->addWidget(f1Plot, 1);
  }
  if (!metrics.roc.isEmpty()) {
    auto* plot = new QCustomPlot(&dialog);
    plot->legend->setVisible(true);
    plot->addGraph();
    plot->graph(0)->setName(tr("ROC"));
    QVector<double> rocX, rocY, prX, prY;
    for (const QPointF& point : metrics.roc) { rocX.push_back(point.x()); rocY.push_back(point.y()); }
    for (const QPointF& point : metrics.pr) { prX.push_back(point.x()); prY.push_back(point.y()); }
    plot->graph(0)->setData(rocX, rocY);
    plot->graph(0)->setPen(QPen(QColor(80, 180, 255), 2));
    plot->addGraph();
    plot->graph(1)->setName(tr("PR"));
    plot->graph(1)->setData(prX, prY);
    plot->graph(1)->setPen(QPen(QColor(255, 170, 80), 2));
    plot->xAxis->setLabel(tr("Rate / Recall"));
    plot->yAxis->setLabel(tr("Value"));
    plot->xAxis->setRange(0, 1);
    plot->yAxis->setRange(0, 1);
    plot->replot();
    layout->addWidget(plot, 1);
  }
  dialog.exec();
}


void MainWindow::onSuggestLabels() {
  if (!ensureImageLoaded(tr("suggesting labels")) || !ensureModelReady(tr("suggesting labels"))) {
    return;
  }
  if (!model_.supportsProbability()) {
    QMessageBox::information(this, tr("Suggestions unavailable"), tr("Automatic suggestions currently require a classifier with probability output."));
    return;
  }
  cv::Mat probs;
  if (!computeProbabilityOutputs(tr("Scoring uncertainty for label suggestions..."),
                                 probabilityCombo_->currentData().toInt(),
                                 nullptr,
                                 &probs)) {
    QMessageBox::information(this, tr("Suggestions unavailable"), tr("The current model did not return probability estimates."));
    return;
  }
  if (probs.empty()) {
    QMessageBox::information(this, tr("Suggestions unavailable"), tr("The current model did not return probability estimates."));
    return;
  }

  struct Candidate { double uncertainty; QPoint point; };
  std::vector<Candidate> candidates;
  candidates.reserve(probs.rows);
  for (int row = 0; row < originalImage_.rows; ++row) {
    for (int col = 0; col < originalImage_.cols; ++col) {
      if (!inferenceMask_.empty() && inferenceMask_.at<uchar>(row, col) == 0) {
        continue;
      }
      if (!exclusionMask_.empty() && exclusionMask_.at<uchar>(row, col) > 0) {
        continue;
      }
      const cv::Mat probRow = probs.row(row * originalImage_.cols + col);
      double maxValue = 0.0;
      cv::minMaxLoc(probRow, nullptr, &maxValue, nullptr, nullptr);
      candidates.push_back({1.0 - maxValue, QPoint(col, row)});
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.uncertainty > b.uncertainty; });

  QVector<QPoint> selected;
  QString summary = tr("Top active-learning suggestions (highest uncertainty):\n");
  for (const Candidate& candidate : candidates) {
    bool farEnough = true;
    for (const QPoint& existing : selected) {
      if (QLineF(existing, candidate.point).length() < 24.0) {
        farEnough = false;
        break;
      }
    }
    if (!farEnough) continue;
    selected.push_back(candidate.point);
    summary += tr("- (%1, %2) uncertainty=%3\n")
                   .arg(candidate.point.x())
                   .arg(candidate.point.y())
                   .arg(QString::number(candidate.uncertainty, 'f', 3));
    if (selected.size() >= 10) break;
  }
  QMessageBox::information(this, tr("Suggested labels"), summary);
  logMessage(tr("Generated %1 active-learning suggestions for the current slice.").arg(selected.size()));
}

void MainWindow::onSaveClassifier() {
  if (!ensureModelReady(tr("saving the classifier"))) return;
  const QString path = QFileDialog::getSaveFileName(this, tr("Save classifier"), QString(), tr("Classifier (*.yml *.yaml)"));
  if (path.isEmpty()) return;
  saveCurrentSliceState();
  if (!model_.save(path, classes_, featureSettings_, classifierSettings_, trainingStats_)) {
    QMessageBox::warning(this, tr("Save failed"),
                         tr("Failed to save classifier to:\n%1\n\nPlease check path permissions and whether the path contains unsupported characters.")
                             .arg(path));
    return;
  }
  ensureSliceStorage();
  if (saveClassifierRoiConfig(path, sliceAnnotationStates_, currentSliceIndex_)) {
    logMessage(tr("Saved ROI settings to %1.").arg(classifierRoiPath(path)));
  } else {
    logMessage(tr("Warning: classifier saved, but ROI settings could not be written."));
  }
  logMessage(tr("Saved classifier to %1.").arg(path));
}

void MainWindow::onLoadClassifier() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Load classifier"), QString(), tr("Classifier (*.yml *.yaml)"));
  if (path.isEmpty()) return;
  if (!QFile::exists(path)) {
    QMessageBox::warning(this, tr("Load failed"), tr("Classifier file does not exist:\n%1").arg(path));
    return;
  }
  std::vector<SegmentationClassInfo> loadedClasses;
  SegmentationFeatureSettings settings;
  SegmentationClassifierSettings classifierSettings;
  SegmentationTrainingStats stats;
  SegmentationClassifier loadedModel;
  QProgressDialog progress(tr("Loading classifier..."), QString(), 0, 0, this);
  progress.setWindowModality(Qt::ApplicationModal);
  progress.setCancelButton(nullptr);
  progress.setMinimumDuration(0);
  progress.show();
  QApplication::processEvents();
  if (!loadedModel.load(path, &loadedClasses, &settings, &classifierSettings, &stats)) {
    progress.close();
    QMessageBox::warning(this, tr("Load failed"),
                         tr("Failed to load classifier:\n%1\n\nIf this model uses a sidecar file, ensure \"%2\" exists beside the metadata file.")
                             .arg(path, path + ".model.yml"));
    return;
  }
  model_ = loadedModel;
  classes_ = loadedClasses;
  featureSettings_ = settings;
  classifierSettings_ = classifierSettings;
  trainingStats_ = stats;
  clearImportedTrainingData();
  clearInferenceOutputs();
  refreshClassList();
  const int comboIndex = classifierCombo_->findData(classifierSettings_.kind);
  if (comboIndex >= 0) classifierCombo_->setCurrentIndex(comboIndex);
  if (!originalImage_.empty()) {
    ensureSliceStorage();
    int loadedSliceIndex = currentSliceIndex_;
    if (loadClassifierRoiConfig(path, static_cast<int>(classes_.size()), &sliceAnnotationStates_, &loadedSliceIndex)) {
      currentSliceIndex_ = std::clamp(loadedSliceIndex, 0, std::max(0, static_cast<int>(imageVolume_.size()) - 1));
      originalImage_ = imageVolume_[currentSliceIndex_].clone();
      restoreSliceState(currentSliceIndex_);
      logMessage(tr("Loaded ROI settings from %1.").arg(classifierRoiPath(path)));
    } else {
      logMessage(tr("No ROI settings sidecar found for this classifier."));
    }
  }
  progress.close();
  updateUiState();
  updateViewer();
  logMessage(tr("Loaded classifier from %1.").arg(path));
}

void MainWindow::onSaveData() {
  const QString path = QFileDialog::getSaveFileName(this, tr("Save training data"), QString(), tr("Training data (*.json *.arff)"));
  if (!path.isEmpty() && !saveTrainingData(path)) {
    QMessageBox::warning(this, tr("Save failed"), tr("Failed to save training data."));
  }
}

void MainWindow::onLoadData() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Load training data"), QString(), tr("Training data (*.json *.arff)"));
  if (!path.isEmpty() && !loadTrainingData(path)) {
    QMessageBox::warning(this, tr("Load failed"), tr("Failed to load training data."));
  }
}

void MainWindow::onCreateNewClass() {
  bool ok = false;
  const QString name = QInputDialog::getText(this, tr("Create class"), tr("Class name"), QLineEdit::Normal, tr("Class %1").arg(classes_.size() + 1), &ok);
  if (!ok || name.trimmed().isEmpty()) return;
  const QColor color = QColorDialog::getColor(defaultColorForIndex(classes_.size()), this, tr("Choose class color"));
  if (!color.isValid()) return;
  classes_.push_back({name.trimmed(), color, true});
  if (!originalImage_.empty()) {
    classMasks_.push_back(cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0)));
    classBrushMasks_.push_back(cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0)));
    classTraceRegions_.push_back({});
    saveCurrentSliceState();
  }
  model_.clear();
  trainingStats_ = {};
  clearImportedTrainingData();
  clearInferenceOutputs();
  refreshClassList();
  rebuildMasksFromAnnotations();
  updateViewer();
  logMessage(tr("Added class %1.").arg(name.trimmed()));
}

void MainWindow::onSettings() {
  QDialog dialog(this);
  dialog.setWindowTitle(tr("Feature and classifier settings"));
  auto* layout = new QVBoxLayout(&dialog);
  auto* presetRow = new QHBoxLayout();
  auto* presetLabel = new QLabel(tr("Feature preset"), &dialog);
  auto* presetCombo = new QComboBox(&dialog);
  presetCombo->addItem(tr("Custom"));
  presetCombo->addItem(tr("TWS-like 2D basic"));
  presetCombo->addItem(tr("TWS-like edges"));
  presetCombo->addItem(tr("TWS-like texture"));
  presetCombo->addItem(tr("TWS-like dense"));
  auto* presetApply = new QPushButton(tr("Apply preset"), &dialog);
  presetRow->addWidget(presetLabel);
  presetRow->addWidget(presetCombo, 1);
  presetRow->addWidget(presetApply);
  layout->addLayout(presetRow);
  auto* featureGroup = new QGroupBox(tr("Feature extraction"), &dialog);
  auto* featureLayout = new QVBoxLayout(featureGroup);
  auto* intensity = new QCheckBox(tr("Intensity"), &dialog);
  auto* gaussian3 = new QCheckBox(tr("Gaussian sigma small"), &dialog);
  auto* gaussian7 = new QCheckBox(tr("Gaussian sigma large"), &dialog);
  auto* gaussian15 = new QCheckBox(tr("Gaussian sigma extra large"), &dialog);
  auto* differenceOfGaussians = new QCheckBox(tr("Difference of Gaussians"), &dialog);
  auto* minimum = new QCheckBox(tr("Minimum filter"), &dialog);
  auto* maximum = new QCheckBox(tr("Maximum filter"), &dialog);
  auto* median = new QCheckBox(tr("Median filter"), &dialog);
  auto* bilateral = new QCheckBox(tr("Bilateral filter"), &dialog);
  auto* gradient = new QCheckBox(tr("Gradient magnitude"), &dialog);
  auto* laplacian = new QCheckBox(tr("Laplacian"), &dialog);
  auto* laplacianOfGaussian = new QCheckBox(tr("Laplacian of Gaussian"), &dialog);
  auto* hessian = new QCheckBox(tr("Hessian norm"), &dialog);
  auto* localMean = new QCheckBox(tr("Local mean"), &dialog);
  auto* localStd = new QCheckBox(tr("Local std-dev"), &dialog);
  auto* localVariance = new QCheckBox(tr("Local variance"), &dialog);
  auto* entropy = new QCheckBox(tr("Local entropy"), &dialog);
  auto* texture = new QCheckBox(tr("Texture energy"), &dialog);
  auto* clahe = new QCheckBox(tr("CLAHE"), &dialog);
  auto* canny = new QCheckBox(tr("Canny edges"), &dialog);
  auto* structureTensor = new QCheckBox(tr("Structure tensor"), &dialog);
  auto* gabor = new QCheckBox(tr("Gabor response"), &dialog);
  auto* membrane = new QCheckBox(tr("Membrane response"), &dialog);
  auto* channelRatios = new QCheckBox(tr("Channel ratios"), &dialog);
  auto* xpos = new QCheckBox(tr("X position"), &dialog);
  auto* ypos = new QCheckBox(tr("Y position"), &dialog);
  intensity->setChecked(featureSettings_.intensity);
  gaussian3->setChecked(featureSettings_.gaussian3);
  gaussian7->setChecked(featureSettings_.gaussian7);
  gaussian15->setChecked(featureSettings_.gaussian15);
  differenceOfGaussians->setChecked(featureSettings_.differenceOfGaussians);
  minimum->setChecked(featureSettings_.minimum);
  maximum->setChecked(featureSettings_.maximum);
  median->setChecked(featureSettings_.median);
  bilateral->setChecked(featureSettings_.bilateral);
  gradient->setChecked(featureSettings_.gradient);
  laplacian->setChecked(featureSettings_.laplacian);
  laplacianOfGaussian->setChecked(featureSettings_.laplacianOfGaussian);
  hessian->setChecked(featureSettings_.hessian);
  localMean->setChecked(featureSettings_.localMean);
  localStd->setChecked(featureSettings_.localStd);
  localVariance->setChecked(featureSettings_.localVariance);
  entropy->setChecked(featureSettings_.entropy);
  texture->setChecked(featureSettings_.texture);
  clahe->setChecked(featureSettings_.clahe);
  canny->setChecked(featureSettings_.canny);
  structureTensor->setChecked(featureSettings_.structureTensor);
  gabor->setChecked(featureSettings_.gabor);
  membrane->setChecked(featureSettings_.membrane);
  channelRatios->setChecked(featureSettings_.channelRatios);
  xpos->setChecked(featureSettings_.xPosition);
  ypos->setChecked(featureSettings_.yPosition);
  for (QCheckBox* checkbox : {intensity, gaussian3, gaussian7, gaussian15, differenceOfGaussians, minimum, maximum, median, bilateral, gradient, laplacian,
                              laplacianOfGaussian, hessian, localMean, localStd, localVariance, entropy, texture, clahe, canny, structureTensor, gabor,
                              membrane, channelRatios, xpos, ypos}) {
    featureLayout->addWidget(checkbox);
  }
  connect(presetApply, &QPushButton::clicked, &dialog, [=]() {
    const int preset = presetCombo->currentIndex();
    if (preset <= 0) return;
    for (QCheckBox* checkbox : {intensity, gaussian3, gaussian7, gaussian15, differenceOfGaussians, minimum, maximum, median, bilateral, gradient, laplacian,
                                laplacianOfGaussian, hessian, localMean, localStd, localVariance, entropy, texture, clahe, canny, structureTensor, gabor,
                                membrane, channelRatios, xpos, ypos}) {
      checkbox->setChecked(false);
    }
    if (preset == 1) {
      intensity->setChecked(true);
      gaussian3->setChecked(true);
      gaussian7->setChecked(true);
      localMean->setChecked(true);
      localStd->setChecked(true);
      xpos->setChecked(true);
      ypos->setChecked(true);
    } else if (preset == 2) {
      intensity->setChecked(true);
      gaussian3->setChecked(true);
      gradient->setChecked(true);
      laplacian->setChecked(true);
      laplacianOfGaussian->setChecked(true);
      canny->setChecked(true);
      structureTensor->setChecked(true);
    } else if (preset == 3) {
      intensity->setChecked(true);
      median->setChecked(true);
      bilateral->setChecked(true);
      entropy->setChecked(true);
      texture->setChecked(true);
      gabor->setChecked(true);
      membrane->setChecked(true);
    } else if (preset == 4) {
      for (QCheckBox* checkbox : {intensity, gaussian3, gaussian7, gaussian15, differenceOfGaussians, minimum, maximum, median, bilateral, gradient, laplacian,
                                  laplacianOfGaussian, hessian, localMean, localStd, localVariance, entropy, texture, clahe, canny, structureTensor, gabor,
                                  membrane, channelRatios, xpos, ypos}) {
        checkbox->setChecked(true);
      }
    }
  });

  auto* classifierGroup = new QGroupBox(tr("Classifier parameters"), &dialog);
  auto* classifierLayout = new QFormLayout(classifierGroup);
  auto* balanceClasses = new QCheckBox(tr("Balance class sampling"), classifierGroup);
  balanceClasses->setChecked(classifierSettings_.balanceClasses);
  auto* rfTrees = new QSpinBox(classifierGroup);
  rfTrees->setRange(10, 5000);
  rfTrees->setValue(classifierSettings_.randomForestTrees);
  auto* rfDepth = new QSpinBox(classifierGroup);
  rfDepth->setRange(1, 128);
  rfDepth->setValue(classifierSettings_.randomForestMaxDepth);
  auto* svmC = new QDoubleSpinBox(classifierGroup);
  svmC->setDecimals(3);
  svmC->setRange(0.001, 10000.0);
  svmC->setSingleStep(0.25);
  svmC->setValue(classifierSettings_.svmC);
  auto* svmGamma = new QDoubleSpinBox(classifierGroup);
  svmGamma->setDecimals(4);
  svmGamma->setRange(0.0001, 1000.0);
  svmGamma->setSingleStep(0.05);
  svmGamma->setValue(classifierSettings_.svmGamma);
  auto* knnNeighbors = new QSpinBox(classifierGroup);
  knnNeighbors->setRange(1, 64);
  knnNeighbors->setValue(classifierSettings_.knnNeighbors);
  auto* logisticLearningRate = new QDoubleSpinBox(classifierGroup);
  logisticLearningRate->setDecimals(4);
  logisticLearningRate->setRange(0.0001, 10.0);
  logisticLearningRate->setValue(classifierSettings_.logisticLearningRate);
  auto* logisticIterations = new QSpinBox(classifierGroup);
  logisticIterations->setRange(10, 5000);
  logisticIterations->setValue(classifierSettings_.logisticIterations);
  classifierLayout->addRow(balanceClasses);
  classifierLayout->addRow(tr("RF trees"), rfTrees);
  classifierLayout->addRow(tr("RF max depth"), rfDepth);
  classifierLayout->addRow(tr("SVM C"), svmC);
  classifierLayout->addRow(tr("SVM gamma"), svmGamma);
  classifierLayout->addRow(tr("KNN neighbors"), knnNeighbors);
  classifierLayout->addRow(tr("Logistic lr"), logisticLearningRate);
  classifierLayout->addRow(tr("Logistic iters"), logisticIterations);

  layout->addWidget(featureGroup);
  layout->addWidget(classifierGroup);
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
  layout->addWidget(buttons);
  connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  if (dialog.exec() != QDialog::Accepted) return;
  featureSettings_.intensity = intensity->isChecked();
  featureSettings_.gaussian3 = gaussian3->isChecked();
  featureSettings_.gaussian7 = gaussian7->isChecked();
  featureSettings_.gaussian15 = gaussian15->isChecked();
  featureSettings_.differenceOfGaussians = differenceOfGaussians->isChecked();
  featureSettings_.minimum = minimum->isChecked();
  featureSettings_.maximum = maximum->isChecked();
  featureSettings_.median = median->isChecked();
  featureSettings_.bilateral = bilateral->isChecked();
  featureSettings_.gradient = gradient->isChecked();
  featureSettings_.laplacian = laplacian->isChecked();
  featureSettings_.laplacianOfGaussian = laplacianOfGaussian->isChecked();
  featureSettings_.hessian = hessian->isChecked();
  featureSettings_.localMean = localMean->isChecked();
  featureSettings_.localStd = localStd->isChecked();
  featureSettings_.localVariance = localVariance->isChecked();
  featureSettings_.entropy = entropy->isChecked();
  featureSettings_.texture = texture->isChecked();
  featureSettings_.clahe = clahe->isChecked();
  featureSettings_.canny = canny->isChecked();
  featureSettings_.structureTensor = structureTensor->isChecked();
  featureSettings_.gabor = gabor->isChecked();
  featureSettings_.membrane = membrane->isChecked();
  featureSettings_.channelRatios = channelRatios->isChecked();
  featureSettings_.xPosition = xpos->isChecked();
  featureSettings_.yPosition = ypos->isChecked();
  classifierSettings_.balanceClasses = balanceClasses->isChecked();
  classifierSettings_.randomForestTrees = rfTrees->value();
  classifierSettings_.randomForestMaxDepth = rfDepth->value();
  classifierSettings_.svmC = svmC->value();
  classifierSettings_.svmGamma = svmGamma->value();
  classifierSettings_.knnNeighbors = knnNeighbors->value();
  classifierSettings_.logisticLearningRate = logisticLearningRate->value();
  classifierSettings_.logisticIterations = logisticIterations->value();
  model_.clear();
  trainingStats_ = {};
  clearImportedTrainingData();
  rebuildFeatureStack();
  clearInferenceOutputs();
  updateViewer();
  updateUiState();
  logMessage(tr("Updated feature extraction settings."));
}

void MainWindow::onExportMask() {
  if (labelResult_.empty()) {
    QMessageBox::information(this, tr("No result"), tr("Please train/apply the classifier before exporting a mask."));
    return;
  }
  QString defaultName = imageVolume_.size() > 1 ? tr("slice_%1_mask.png").arg(currentSliceIndex_ + 1) : QString();
  const QString path = QFileDialog::getSaveFileName(this, tr("Export label mask"), defaultName, tr("PNG (*.png);;TIFF (*.tif *.tiff)"));
  if (path.isEmpty()) return;
  cv::Mat exportMask;
  labelResult_.convertTo(exportMask, CV_8U);
  if (!cv::imwrite(cvPath(path), exportMask)) {
    QMessageBox::warning(this, tr("Export failed"), tr("Failed to export the label mask."));
    return;
  }
  logMessage(tr("Exported label mask to %1.").arg(path));
}

void MainWindow::onExportRoiMacro() {
  if (classTraceRegions_.empty()) {
    QMessageBox::information(this, tr("No ROIs"), tr("No ROI traces are available to export."));
    return;
  }
  const QString path = QFileDialog::getSaveFileName(this, tr("Export ROI macro"), QString(), tr("ImageJ Macro (*.ijm)"));
  if (path.isEmpty()) return;
  QFile file(path);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) {
    QMessageBox::warning(this, tr("Export failed"), tr("Failed to write macro file."));
    return;
  }
  QTextStream out(&file);
  out << "// Auto-generated ROI macro from QtTrainableSegmentation\n";
  out << "roiManager(\"reset\");\n";
  for (int cls = 0; cls < static_cast<int>(classTraceRegions_.size()) && cls < static_cast<int>(classes_.size()); ++cls) {
    for (int roiIndex = 0; roiIndex < static_cast<int>(classTraceRegions_[cls].size()); ++roiIndex) {
      const TraceRegion& region = classTraceRegions_[cls][roiIndex];
      if (region.polygon.size() < 3) continue;
      QStringList xs;
      QStringList ys;
      xs.reserve(region.polygon.size());
      ys.reserve(region.polygon.size());
      for (const QPoint& point : region.polygon) {
        xs << QString::number(point.x());
        ys << QString::number(point.y());
      }
      out << "makeSelection(\"polygon\", newArray(" << xs.join(",") << "), newArray(" << ys.join(",") << "));\n";
      out << "roiManager(\"Add\");\n";
      out << "roiManager(\"Select\", roiManager(\"count\")-1);\n";
      out << "roiManager(\"Rename\", \"" << classes_[cls].name << "_ROI_" << (roiIndex + 1) << "\");\n";
    }
  }
  logMessage(tr("Exported ROI macro to %1.").arg(path));
}

void MainWindow::onBrushRadiusChanged(int value) {
  view_->setBrushRadius(value);
}

void MainWindow::onClassSelectionChanged() {
  const int cls = currentSelectedClassIndex();
  if (cls >= 0 && cls < static_cast<int>(classes_.size())) {
    view_->setActiveClass(cls, classes_[cls].color);
  }
  updateTraceLists();
}

void MainWindow::onProbabilityClassChanged(int) {
  updateProbabilityView(true, tr("Updating probability map..."));
  updateUiState();
}

void MainWindow::onOpacityChanged(int value) {
  overlayOpacity_ = static_cast<double>(value) / 100.0;
  updateViewer();
}

void MainWindow::onPaintTool() { view_->setToolMode(SegmentationView::PaintTool); }
void MainWindow::onEraseTool() { view_->setToolMode(SegmentationView::EraseTool); }
void MainWindow::onPanTool() { view_->setToolMode(SegmentationView::PanTool); }
void MainWindow::onTraceTool() { view_->setToolMode(SegmentationView::TraceTool); }
void MainWindow::onResetZoom() { view_->resetView(); }
void MainWindow::onAddTraceToSelectedClass() { addPendingTraceToSelectedClass(); }
void MainWindow::onAddInferenceRoi() {
  if (pendingTrace_.size() < 3) {
    QMessageBox::information(this, tr("No ROI"), tr("Draw a ROI trace first, then add it as a recognition region."));
    return;
  }
  pushUndoSnapshot();
  inferenceRegions_.push_back(pendingTrace_);
  pendingTrace_.clear();
  view_->clearPendingTrace();
  infoLabel_->setText(tr("Added recognition ROI (%1 total).").arg(inferenceRegions_.size()));
  logMessage(tr("Added recognition ROI: only this ROI area will be used for inference."));
  rebuildMasksFromAnnotations();
}
void MainWindow::onClearInferenceRois() {
  if (inferenceRegions_.empty()) return;
  pushUndoSnapshot();
  inferenceRegions_.clear();
  inferenceMask_.release();
  logMessage(tr("Cleared recognition ROIs. Inference now runs on full image (except exclusion ROIs)."));
  rebuildMasksFromAnnotations();
}
void MainWindow::onAddExclusionRoi() {
  if (pendingTrace_.size() < 3) {
    QMessageBox::information(this, tr("No ROI"), tr("Draw a ROI trace first, then add it as an exclusion region."));
    return;
  }
  pushUndoSnapshot();
  exclusionRegions_.push_back(pendingTrace_);
  pendingTrace_.clear();
  view_->clearPendingTrace();
  infoLabel_->setText(tr("Added exclusion ROI (%1 total).").arg(exclusionRegions_.size()));
  logMessage(tr("Added exclusion ROI to ignore area during training/inference."));
  rebuildMasksFromAnnotations();
}
void MainWindow::onClearExclusionRois() {
  if (exclusionRegions_.empty()) return;
  pushUndoSnapshot();
  exclusionRegions_.clear();
  exclusionMask_.release();
  logMessage(tr("Cleared exclusion ROIs."));
  rebuildMasksFromAnnotations();
}
void MainWindow::onRemoveSelectedTrace() { removeSelectedTraceFromClass(); }

void MainWindow::onRenameSelectedTrace() {
  const int classIndex = currentSelectedClassIndex();
  const int row = traceList_ ? traceList_->currentRow() : -1;
  if (classIndex < 0 || row < 0 || classIndex >= static_cast<int>(classTraceRegions_.size()) || row >= static_cast<int>(classTraceRegions_[classIndex].size())) {
    return;
  }
  bool ok = false;
  const QString currentName = classTraceRegions_[classIndex][row].name;
  const QString name = QInputDialog::getText(this, tr("Rename trace"), tr("Trace name"), QLineEdit::Normal, currentName, &ok);
  if (!ok || name.trimmed().isEmpty()) return;
  pushUndoSnapshot();
  classTraceRegions_[classIndex][row].name = name.trimmed();
  updateTraceLists();
  saveCurrentSliceState();
}

void MainWindow::onClearTracesForSelectedClass() {
  const int classIndex = currentSelectedClassIndex();
  if (classIndex < 0 || classIndex >= static_cast<int>(classTraceRegions_.size())) return;
  if (classTraceRegions_[classIndex].empty()) return;
  pushUndoSnapshot();
  classTraceRegions_[classIndex].clear();
  rebuildMasksFromAnnotations();
}

void MainWindow::onUndo() {
  if (undoStack_.empty()) return;
  AnnotationSnapshot current;
  current.classBrushMasks = cloneMaskVector(classBrushMasks_);
  current.classTraceRegions = cloneTraceRegions(classTraceRegions_);
  current.inferenceRegions = inferenceRegions_;
  current.exclusionRegions = exclusionRegions_;
  redoStack_.push_back(std::move(current));
  const AnnotationSnapshot snapshot = undoStack_.back();
  undoStack_.pop_back();
  restoreSnapshot(snapshot);
}

void MainWindow::onRedo() {
  if (redoStack_.empty()) return;
  AnnotationSnapshot current;
  current.classBrushMasks = cloneMaskVector(classBrushMasks_);
  current.classTraceRegions = cloneTraceRegions(classTraceRegions_);
  current.inferenceRegions = inferenceRegions_;
  current.exclusionRegions = exclusionRegions_;
  undoStack_.push_back(std::move(current));
  const AnnotationSnapshot snapshot = redoStack_.back();
  redoStack_.pop_back();
  restoreSnapshot(snapshot);
}

void MainWindow::onSliceChanged(int value) {
  if (value < 0 || value >= static_cast<int>(imageVolume_.size()) || value == currentSliceIndex_) {
    return;
  }
  saveCurrentSliceState();
  currentSliceIndex_ = value;
  originalImage_ = imageVolume_[currentSliceIndex_].clone();
  featureStack_ = featureVolume_[currentSliceIndex_].clone();
  if (featureStack_.empty()) rebuildFeatureStack();
  restoreSliceState(currentSliceIndex_);
  updateViewer();
  updateUiState();
}

void MainWindow::onNewProject() {
  persistCurrentProjectImageState();
  projectImages_.clear();
  currentProjectImageIndex_ = -1;
  projectPath_.clear();
  updateProjectList();
  logMessage(tr("Started a new empty project."));
}

void MainWindow::onOpenProject() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Open project"), QString(), tr("Project (*.json)"));
  if (!path.isEmpty() && !loadProject(path)) {
    QMessageBox::warning(this, tr("Open failed"), tr("Failed to load project."));
  }
}

void MainWindow::onSaveProject() {
  QString path = projectPath_;
  if (path.isEmpty()) {
    path = QFileDialog::getSaveFileName(this, tr("Save project"), QString(), tr("Project (*.json)"));
  }
  if (!path.isEmpty() && !saveProject(path)) {
    QMessageBox::warning(this, tr("Save failed"), tr("Failed to save project."));
  }
}

void MainWindow::onAddImageToProject() {
  const QStringList paths = QFileDialog::getOpenFileNames(this, tr("Add image(s) to project"), QString(), tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
  if (paths.isEmpty()) return;
  for (const QString& path : paths) {
    projectImages_.push_back({path, QString()});
  }
  if (currentProjectImageIndex_ < 0 && !projectImages_.empty()) {
    currentProjectImageIndex_ = 0;
    switchToProjectImage(0);
  }
  updateProjectList();
}

void MainWindow::onRemoveProjectImage() {
  if (currentProjectImageIndex_ < 0 || currentProjectImageIndex_ >= static_cast<int>(projectImages_.size())) return;
  projectImages_.erase(projectImages_.begin() + currentProjectImageIndex_);
  if (projectImages_.empty()) {
    currentProjectImageIndex_ = -1;
  } else {
    currentProjectImageIndex_ = std::clamp(currentProjectImageIndex_, 0, static_cast<int>(projectImages_.size()) - 1);
    switchToProjectImage(currentProjectImageIndex_);
  }
  updateProjectList();
}

void MainWindow::onProjectSelectionChanged() {
  if (switchingProjectSelection_ || !projectList_ || !projectList_->currentItem()) {
    return;
  }
  const int index = projectList_->currentItem()->data(Qt::UserRole).toInt();
  if (index != currentProjectImageIndex_) {
    switchToProjectImage(index);
  }
}
