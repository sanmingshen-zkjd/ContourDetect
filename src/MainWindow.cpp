#include "MainWindow.h"

#include "QCustomPlot.h"
#include "SegmentationView.h"

#include <QApplication>
#include <QBoxLayout>
#include <QColorDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QPainter>
#include <QScreen>
#include <QSplitter>
#include <QToolBar>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QLineEdit>
#include <QDoubleSpinBox>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
QString nowString() {
  return QDateTime::currentDateTime().toString("hh:mm:ss");
}

QColor defaultColorForIndex(int index) {
  static const QVector<QColor> palette = {
      QColor(224, 72, 72), QColor(73, 201, 118), QColor(72, 130, 235),
      QColor(235, 183, 72), QColor(188, 72, 235), QColor(54, 200, 211)};
  return palette[index % palette.size()];
}

}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
  initializeDefaultClasses();
  buildUi();
  connectSignals();
  updateUiState();
  resize(1540, 920);
  if (QScreen* screen = QGuiApplication::primaryScreen()) {
    const QRect available = screen->availableGeometry();
    move(available.center() - rect().center());
  }
  setWindowTitle("Qt Trainable Segmentation");
}

void MainWindow::buildUi() {
  auto* toolBar = addToolBar(tr("File"));
  toolBar->setMovable(false);
  QAction* openImageAction = toolBar->addAction(tr("Open Image"));
  QAction* saveMaskAction = toolBar->addAction(tr("Export Mask"));
  toolBar->addSeparator();
  QAction* paintAction = toolBar->addAction(tr("Paint"));
  QAction* eraseAction = toolBar->addAction(tr("Erase"));
  QAction* panAction = toolBar->addAction(tr("Pan"));
  QAction* traceAction = toolBar->addAction(tr("Trace ROI"));
  QAction* resetZoomAction = toolBar->addAction(tr("Reset Zoom"));

  connect(openImageAction, &QAction::triggered, this, &MainWindow::onOpenImage);
  connect(saveMaskAction, &QAction::triggered, this, &MainWindow::onExportMask);
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
  probabilityButton_ = new QPushButton(tr("Get probability"), trainingGroup);
  QPushButton* plotButton = new QPushButton(tr("Plot result"), trainingGroup);
  trainingLayout->addWidget(trainButton_);
  trainingLayout->addWidget(stopTrainingButton_);
  trainingLayout->addWidget(overlayButton);
  trainingLayout->addWidget(createResultButton_);
  trainingLayout->addWidget(probabilityButton_);
  trainingLayout->addWidget(plotButton);
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
  optionsLayout->addWidget(applyButton_);
  optionsLayout->addWidget(loadClassifierButton);
  optionsLayout->addWidget(saveClassifierButton);
  optionsLayout->addWidget(loadDataButton);
  optionsLayout->addWidget(saveDataButton);
  optionsLayout->addWidget(createClassButton);
  optionsLayout->addWidget(settingsButton);
  leftLayout->addWidget(optionsGroup);

  auto* brushGroup = new QGroupBox(tr("Brush"), leftPanel);
  auto* brushLayout = new QFormLayout(brushGroup);
  brushSizeSpin_ = new QSpinBox(brushGroup);
  brushSizeSpin_->setRange(1, 256);
  brushSizeSpin_->setValue(12);
  opacitySlider_ = new QSlider(Qt::Horizontal, brushGroup);
  opacitySlider_->setRange(0, 100);
  opacitySlider_->setValue(static_cast<int>(overlayOpacity_ * 100.0));
  probabilityCombo_ = new QComboBox(brushGroup);
  classifierCombo_ = new QComboBox(brushGroup);
  classifierCombo_->addItem(tr("Gaussian Naive Bayes"), SegmentationClassifierSettings::GaussianNaiveBayes);
  classifierCombo_->addItem(tr("Random Forest"), SegmentationClassifierSettings::RandomForest);
  classifierCombo_->addItem(tr("SVM (RBF)"), SegmentationClassifierSettings::SupportVectorMachine);
  overlayCheck_ = new QCheckBox(tr("Show classifier overlay"), brushGroup);
  overlayCheck_->setChecked(true);
  contourCheck_ = new QCheckBox(tr("Draw contours"), brushGroup);
  contourCheck_->setChecked(true);
  brushLayout->addRow(tr("Brush radius"), brushSizeSpin_);
  brushLayout->addRow(tr("Overlay alpha"), opacitySlider_);
  brushLayout->addRow(tr("Probability class"), probabilityCombo_);
  brushLayout->addRow(tr("Classifier"), classifierCombo_);
  brushLayout->addRow(overlayCheck_);
  brushLayout->addRow(contourCheck_);
  leftLayout->addWidget(brushGroup);

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

  auto* centerPanel = new QWidget(splitter);
  auto* centerLayout = new QVBoxLayout(centerPanel);
  centerLayout->setContentsMargins(8, 8, 8, 8);
  centerLayout->setSpacing(8);
  imagePathLabel_ = new QLabel(tr("No image loaded"), centerPanel);
  imagePathLabel_->setWordWrap(true);
  view_ = new SegmentationView(centerPanel);
  infoLabel_ = new QLabel(tr("Open a grayscale or color image to begin labeling."), centerPanel);
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
  addTraceButton_->setToolTip(tr("Commit the current ROI trace to the currently selected class."));
  classList_ = new QListWidget(labelsGroup);
  traceList_ = new QListWidget(labelsGroup);
  removeTraceButton_ = new QPushButton(tr("Remove selected trace"), labelsGroup);
  traceGroup_ = new QGroupBox(tr("Traces"), labelsGroup);
  auto* traceLayout = new QVBoxLayout(traceGroup_);
  traceLayout->addWidget(traceList_);
  traceLayout->addWidget(removeTraceButton_);
  labelsLayout->addWidget(addTraceButton_);
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
  splitter->setSizes({250, 980, 280});

  refreshClassList();
}

void MainWindow::connectSignals() {
  connect(trainButton_, &QPushButton::clicked, this, &MainWindow::onTrainClassifier);
  connect(stopTrainingButton_, &QPushButton::clicked, this, &MainWindow::onStopTraining);
  connect(applyButton_, &QPushButton::clicked, this, &MainWindow::onApplyClassifier);
  connect(createResultButton_, &QPushButton::clicked, this, &MainWindow::onCreateResult);
  connect(probabilityButton_, &QPushButton::clicked, this, &MainWindow::onGetProbability);
  connect(brushSizeSpin_, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::onBrushRadiusChanged);
  connect(classList_, &QListWidget::itemSelectionChanged, this, &MainWindow::onClassSelectionChanged);
  connect(probabilityCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::onProbabilityClassChanged);
  connect(classifierCombo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
    classifierSettings_.kind = static_cast<SegmentationClassifierSettings::Kind>(classifierCombo_->itemData(index).toInt());
    updateUiState();
  });
  connect(opacitySlider_, &QSlider::valueChanged, this, &MainWindow::onOpacityChanged);
  connect(overlayCheck_, &QCheckBox::toggled, this, [this](bool checked) {
    showOverlay_ = checked;
    updateViewer();
  });
  connect(contourCheck_, &QCheckBox::toggled, this, [this](bool) {
    updateViewer();
  });
  connect(addTraceButton_, &QPushButton::clicked, this, &MainWindow::onAddTraceToSelectedClass);
  connect(removeTraceButton_, &QPushButton::clicked, this, &MainWindow::onRemoveSelectedTrace);
  connect(traceList_, &QListWidget::itemSelectionChanged, this, [this]() { updateUiState(); });

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
    QString roleText;
    if (i == 0) roleText = tr(" - foreground target");
    else if (i == 1) roleText = tr(" - background object");
    QListWidgetItem* item = new QListWidgetItem(QString("%1  (Class %2)%3").arg(cls.name).arg(i + 1).arg(roleText), classList_);
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
  applyButton_->setEnabled(hasModel && !trainingInProgress_);
  createResultButton_->setEnabled(hasImage && hasModel && !trainingInProgress_);
  probabilityButton_->setEnabled(hasImage && hasProbability && !trainingInProgress_);
  brushSizeSpin_->setEnabled(hasImage);
  probabilityCombo_->setEnabled(hasProbability && !trainingInProgress_);
  classifierCombo_->setEnabled(!trainingInProgress_);
  addTraceButton_->setEnabled(hasImage && hasClassSelection && !trainingInProgress_);
  traceList_->setEnabled(hasImage && hasClassSelection && !trainingInProgress_);
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
                           : tr("Model: not trained"));
}

void MainWindow::rebuildFeatureStack() {
  if (originalImage_.empty()) {
    featureStack_.release();
    return;
  }
  featureStack_ = SegmentationEngine::computeFeatureStack(originalImage_, featureSettings_);
}

void MainWindow::updateViewer() {
  if (originalImage_.empty()) {
    view_->clearAllLayers();
    return;
  }
  if (view_->sceneRect().isEmpty()) {
    view_->setBaseImage(cvMatToQImage(originalImage_));
  } else {
    view_->setBaseImage(cvMatToQImage(originalImage_));
  }

  if (showOverlay_ && !labelResult_.empty()) {
    overlayImage_ = SegmentationEngine::makeOverlay(originalImage_, labelResult_, classes_, overlayOpacity_, contourCheck_->isChecked());
    view_->setOverlayImage(cvMatToQImage(overlayImage_));
  } else {
    view_->setOverlayImage(QImage());
  }
  rebuildMasksFromAnnotations();
}

void MainWindow::updateProbabilityView() {
  if (originalImage_.empty() || !model_.isValid() || featureStack_.empty()) {
    probabilityImage_.release();
    return;
  }
  const int cls = probabilityCombo_->currentData().toInt();
  probabilityImage_ = model_.supportsProbability()
                          ? SegmentationEngine::applyModelProbabilities(model_, featureStack_, originalImage_.rows, originalImage_.cols, cls)
                          : cv::Mat();
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
    QImage maskImg(classMasks_[cls].data, classMasks_[cls].cols, classMasks_[cls].rows, static_cast<int>(classMasks_[cls].step), QImage::Format_Grayscale8);
    QImage colored(maskImg.size(), QImage::Format_ARGB32_Premultiplied);
    colored.fill(Qt::transparent);
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
    painter.drawImage(0, 0, colored);
  }
  painter.end();
  view_->setAnnotationPreview(annotation);
}

void MainWindow::applyBrushStroke(const QPoint& imagePos, int radius, bool erase) {
  if (originalImage_.empty()) {
    return;
  }
  ensureAnnotationStorage();
  const int cls = classList_->currentItem() ? classList_->currentItem()->data(Qt::UserRole).toInt() : 0;
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
  if (static_cast<int>(classTracePolygons_.size()) != static_cast<int>(classes_.size())) {
    classTracePolygons_.resize(classes_.size());
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
    classTracePolygons_.clear();
    pendingTrace_.clear();
    view_->clearPendingTrace();
    return;
  }
  ensureAnnotationStorage();
  for (auto& mask : classBrushMasks_) {
    mask = cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0));
  }
  for (auto& traces : classTracePolygons_) {
    traces.clear();
  }
  pendingTrace_.clear();
  view_->clearPendingTrace();
  rebuildMasksFromAnnotations();
}

void MainWindow::clearInferenceOutputs() {
  labelResult_.release();
  overlayImage_.release();
  probabilityImage_.release();
}

void MainWindow::rebuildMasksFromAnnotations() {
  if (originalImage_.empty()) {
    return;
  }
  ensureAnnotationStorage();
  for (int i = 0; i < static_cast<int>(classes_.size()); ++i) {
    classMasks_[i] = classBrushMasks_[i].clone();
  }

  for (int cls = 0; cls < static_cast<int>(classTracePolygons_.size()); ++cls) {
    for (const QPolygon& polygon : classTracePolygons_[cls]) {
      if (polygon.size() < 3) continue;
      std::vector<cv::Point> cvPoints;
      cvPoints.reserve(polygon.size());
      for (const QPoint& pt : polygon) {
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

  repaintAnnotationPreview();
  updateTraceLists();
  updateUiState();
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
  traceGroup_->setTitle(tr("Traces for %1").arg(classes_[cls].name));
  if (cls < static_cast<int>(classTracePolygons_.size())) {
    for (int i = 0; i < static_cast<int>(classTracePolygons_[cls].size()); ++i) {
      QListWidgetItem* item = new QListWidgetItem(tr("Trace %1").arg(i + 1), traceList_);
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
  infoLabel_->setText(tr("ROI trace captured. Click Add ROI to selected class to commit it to %1.").arg(className));
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
    logMessage(tr("Waiting for a ROI trace before adding to %1.").arg(classes_[classIndex].name));
    return;
  }
  ensureAnnotationStorage();
  classTracePolygons_[classIndex].push_back(pendingTrace_);
  pendingTrace_.clear();
  view_->clearPendingTrace();
  view_->setToolMode(SegmentationView::TraceTool);
  rebuildMasksFromAnnotations();
  updateTraceLists();
  if (traceList_) {
    traceList_->setCurrentRow(traceList_->count() - 1);
  }
  infoLabel_->setText(tr("Added ROI trace to %1.").arg(classes_[classIndex].name));
  logMessage(tr("Added ROI trace to %1.").arg(classes_[classIndex].name));
}

void MainWindow::removeSelectedTraceFromClass() {
  const int classIndex = currentSelectedClassIndex();
  if (!traceList_ || classIndex < 0 || classIndex >= static_cast<int>(classTracePolygons_.size())) {
    return;
  }
  const int row = traceList_->currentRow();
  if (row < 0 || row >= static_cast<int>(classTracePolygons_[classIndex].size())) {
    return;
  }
  classTracePolygons_[classIndex].erase(classTracePolygons_[classIndex].begin() + row);
  rebuildMasksFromAnnotations();
  updateTraceLists();
  logMessage(tr("Removed trace %1 from %2.").arg(row).arg(classes_[classIndex].name));
}

bool MainWindow::loadImageFile(const QString& path) {
  cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
  if (image.empty()) {
    QMessageBox::warning(this, tr("Open image failed"), tr("Unable to load image: %1").arg(path));
    return false;
  }
  originalImage_ = image;
  imagePath_ = path;
  imagePathLabel_->setText(path);
  clearInferenceOutputs();
  clearAnnotationsForCurrentImage();
  model_.clear();
  trainingStats_ = {};
  rebuildFeatureStack();
  updateViewer();
  updateUiState();
  logMessage(tr("Loaded image %1 (%2 x %3).")
                 .arg(QFileInfo(path).fileName())
                 .arg(originalImage_.cols)
                 .arg(originalImage_.rows));
  return true;
}

bool MainWindow::applyClassifierToImage(const cv::Mat& image, const QString& path, bool resetAnnotations) {
  if (image.empty() || !model_.isValid()) {
    return false;
  }

  originalImage_ = image.clone();
  imagePath_ = path;
  imagePathLabel_->setText(path.isEmpty() ? tr("Unsaved image") : path);
  clearInferenceOutputs();
  if (resetAnnotations) {
    clearAnnotationsForCurrentImage();
  } else {
    ensureAnnotationStorage();
    rebuildMasksFromAnnotations();
  }
  rebuildFeatureStack();
  labelResult_ = SegmentationEngine::applyModelLabels(model_, featureStack_, originalImage_.rows, originalImage_.cols);
  updateProbabilityView();
  updateViewer();
  updateUiState();
  return !labelResult_.empty();
}

bool MainWindow::saveTrainingData(const QString& path) {
  if (!ensureImageLoaded(tr("saving training data"))) {
    return false;
  }
  QFileInfo info(path);
  QDir dir = info.dir();
  const QString base = info.completeBaseName();
  if (!dir.exists() && !dir.mkpath(".")) {
    return false;
  }

  QJsonObject root;
  root["imagePath"] = imagePath_;
  root["maskWidth"] = originalImage_.cols;
  root["maskHeight"] = originalImage_.rows;
  QJsonArray classArray;
  for (int i = 0; i < static_cast<int>(classes_.size()); ++i) {
    const QString maskName = QString("%1_class_%2.png").arg(base).arg(i);
    cv::imwrite(dir.filePath(maskName).toStdString(), classMasks_[i]);
    QJsonObject cls;
    cls["name"] = classes_[i].name;
    cls["color"] = classes_[i].color.name(QColor::HexRgb);
    cls["mask"] = maskName;
    QJsonArray traces;
    if (i < static_cast<int>(classTracePolygons_.size())) {
      for (const QPolygon& polygon : classTracePolygons_[i]) {
        QJsonArray tracePoints;
        for (const QPoint& point : polygon) {
          tracePoints.append(QJsonObject{{"x", point.x()}, {"y", point.y()}});
        }
        traces.append(tracePoints);
      }
    }
    cls["traces"] = traces;
    classArray.append(cls);
  }
  root["classes"] = classArray;
  root["featureSettings"] = QJsonObject{{"intensity", featureSettings_.intensity},
                                         {"gaussian3", featureSettings_.gaussian3},
                                         {"gaussian7", featureSettings_.gaussian7},
                                         {"differenceOfGaussians", featureSettings_.differenceOfGaussians},
                                         {"gradient", featureSettings_.gradient},
                                         {"laplacian", featureSettings_.laplacian},
                                         {"hessian", featureSettings_.hessian},
                                         {"localMean", featureSettings_.localMean},
                                         {"localStd", featureSettings_.localStd},
                                         {"entropy", featureSettings_.entropy},
                                         {"texture", featureSettings_.texture},
                                         {"gabor", featureSettings_.gabor},
                                         {"membrane", featureSettings_.membrane},
                                         {"xPosition", featureSettings_.xPosition},
                                         {"yPosition", featureSettings_.yPosition}};
  root["classifierSettings"] = QJsonObject{{"kind", static_cast<int>(classifierSettings_.kind)},
                                           {"randomForestTrees", classifierSettings_.randomForestTrees},
                                           {"randomForestMaxDepth", classifierSettings_.randomForestMaxDepth},
                                           {"svmC", classifierSettings_.svmC},
                                           {"svmGamma", classifierSettings_.svmGamma},
                                           {"balanceClasses", classifierSettings_.balanceClasses}};

  QFile file(path);
  if (!file.open(QIODevice::WriteOnly)) {
    return false;
  }
  file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
  logMessage(tr("Saved training data to %1.").arg(path));
  return true;
}

bool MainWindow::loadTrainingData(const QString& path) {
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
  classMasks_.clear();
  classBrushMasks_.clear();
  classTracePolygons_.clear();
  for (int i = 0; i < classArray.size(); ++i) {
    const QJsonObject clsObj = classArray[i].toObject();
    classes_.push_back({clsObj["name"].toString(), QColor(clsObj["color"].toString()), true});
    cv::Mat mask = cv::imread(QFileInfo(path).dir().filePath(clsObj["mask"].toString()).toStdString(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
      mask = cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0));
    }
    classBrushMasks_.push_back(mask.clone());
    classMasks_.push_back(mask);
    std::vector<QPolygon> traces;
    for (const auto& traceValue : clsObj["traces"].toArray()) {
      QPolygon poly;
      for (const auto& ptValue : traceValue.toArray()) {
        const QJsonObject pt = ptValue.toObject();
        poly << QPoint(pt["x"].toInt(), pt["y"].toInt());
      }
      if (!poly.isEmpty()) traces.push_back(poly);
    }
    classTracePolygons_.push_back(traces);
  }
  const QJsonObject settings = root["featureSettings"].toObject();
  featureSettings_.intensity = settings["intensity"].toBool(true);
  featureSettings_.gaussian3 = settings["gaussian3"].toBool(true);
  featureSettings_.gaussian7 = settings["gaussian7"].toBool(true);
  featureSettings_.differenceOfGaussians = settings["differenceOfGaussians"].toBool(false);
  featureSettings_.gradient = settings["gradient"].toBool(true);
  featureSettings_.laplacian = settings["laplacian"].toBool(true);
  featureSettings_.hessian = settings["hessian"].toBool(false);
  featureSettings_.localMean = settings["localMean"].toBool(true);
  featureSettings_.localStd = settings["localStd"].toBool(true);
  featureSettings_.entropy = settings["entropy"].toBool(false);
  featureSettings_.texture = settings["texture"].toBool(false);
  featureSettings_.gabor = settings["gabor"].toBool(false);
  featureSettings_.membrane = settings["membrane"].toBool(false);
  featureSettings_.xPosition = settings["xPosition"].toBool(true);
  featureSettings_.yPosition = settings["yPosition"].toBool(true);
  const QJsonObject classifierSettings = root["classifierSettings"].toObject();
  if (!classifierSettings.isEmpty()) {
    classifierSettings_.kind = static_cast<SegmentationClassifierSettings::Kind>(
        classifierSettings["kind"].toInt(static_cast<int>(classifierSettings_.kind)));
    classifierSettings_.randomForestTrees = classifierSettings["randomForestTrees"].toInt(classifierSettings_.randomForestTrees);
    classifierSettings_.randomForestMaxDepth = classifierSettings["randomForestMaxDepth"].toInt(classifierSettings_.randomForestMaxDepth);
    classifierSettings_.svmC = classifierSettings["svmC"].toDouble(classifierSettings_.svmC);
    classifierSettings_.svmGamma = classifierSettings["svmGamma"].toDouble(classifierSettings_.svmGamma);
    classifierSettings_.balanceClasses = classifierSettings["balanceClasses"].toBool(classifierSettings_.balanceClasses);
  }

  refreshClassList();
  pendingTrace_.clear();
  view_->clearPendingTrace();
  clearInferenceOutputs();
  rebuildMasksFromAnnotations();
  rebuildFeatureStack();
  const int comboIndex = classifierCombo_->findData(classifierSettings_.kind);
  if (comboIndex >= 0) {
    classifierCombo_->setCurrentIndex(comboIndex);
  }
  updateUiState();
  logMessage(tr("Loaded training data from %1.").arg(path));
  return true;
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
  trainingInProgress_ = true;
  stopTrainingRequested_ = false;
  updateUiState();
  QApplication::processEvents();
  rebuildFeatureStack();
  if (stopTrainingRequested_) {
    trainingInProgress_ = false;
    updateUiState();
    logMessage(tr("Training stopped before sample gathering."));
    return;
  }
  cv::Mat labels;
  cv::Mat samples = SegmentationEngine::gatherSamples(featureStack_, classMasks_, &labels, 12000, classifierSettings_.balanceClasses);
  QApplication::processEvents();
  if (stopTrainingRequested_) {
    trainingInProgress_ = false;
    updateUiState();
    logMessage(tr("Training stopped before classifier fitting."));
    return;
  }
  if (samples.empty()) {
    trainingInProgress_ = false;
    updateUiState();
    QMessageBox::information(this, tr("No annotations"), tr("Please paint at least one pixel for each class before training."));
    return;
  }
  if (!model_.train(samples, labels, static_cast<int>(classes_.size()), classifierSettings_, &trainingStats_)) {
    trainingInProgress_ = false;
    updateUiState();
    QMessageBox::warning(this, tr("Training failed"), tr("Training failed. Ensure every class has annotations."));
    return;
  }
  if (stopTrainingRequested_) {
    trainingInProgress_ = false;
    updateUiState();
    clearInferenceOutputs();
    logMessage(tr("Training finished fitting but stopped before result generation."));
    return;
  }
  labelResult_ = SegmentationEngine::applyModelLabels(model_, featureStack_, originalImage_.rows, originalImage_.cols);
  updateProbabilityView();
  updateViewer();
  trainingInProgress_ = false;
  updateUiState();
  logMessage(tr("Trained classifier with %1 samples across %2 classes.").arg(trainingStats_.sampleCount).arg(trainingStats_.classCount));
}

void MainWindow::onStopTraining() {
  if (!trainingInProgress_) {
    return;
  }
  stopTrainingRequested_ = true;
  logMessage(tr("Stop requested. The current training stage will stop as soon as it can."));
}

void MainWindow::onApplyClassifier() {
  if (!ensureModelReady(tr("applying the classifier"))) {
    return;
  }
  QString targetPath;
  bool useCurrentImage = !originalImage_.empty();
  if (!originalImage_.empty()) {
    QMessageBox choiceBox(this);
    choiceBox.setWindowTitle(tr("Apply classifier"));
    choiceBox.setText(tr("Apply the classifier to the current image, or choose another image file?"));
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

  if (!useCurrentImage) {
    targetPath = QFileDialog::getOpenFileName(this, tr("Apply classifier to image"), imagePath_,
                                              tr("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"));
    if (targetPath.isEmpty()) {
      return;
    }
    cv::Mat targetImage = cv::imread(targetPath.toStdString(), cv::IMREAD_COLOR);
    if (targetImage.empty()) {
      QMessageBox::warning(this, tr("Open image failed"), tr("Unable to load image: %1").arg(targetPath));
      return;
    }
    if (!applyClassifierToImage(targetImage, targetPath, true)) {
      QMessageBox::warning(this, tr("Apply failed"), tr("Failed to apply the classifier to %1.").arg(targetPath));
      return;
    }
    logMessage(tr("Applied classifier to %1.").arg(QFileInfo(targetPath).fileName()));
    return;
  }

  if (!ensureImageLoaded(tr("applying the classifier"))) {
    return;
  }
  if (!applyClassifierToImage(originalImage_, imagePath_, false)) {
    QMessageBox::warning(this, tr("Apply failed"), tr("Failed to apply the classifier to the current image."));
    return;
  }
  logMessage(tr("Applied classifier to the full current image."));
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
    if (labelResult_.empty()) {
      return;
    }
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

void MainWindow::onGetProbability() {
  if (!ensureImageLoaded(tr("getting probability")) || !ensureModelReady(tr("getting probability"))) {
    return;
  }
  if (!model_.supportsProbability()) {
    QMessageBox::information(this, tr("Probability unavailable"),
                             tr("The current classifier does not expose probabilities for this view."));
    return;
  }
  updateProbabilityView();
  if (probabilityImage_.empty()) {
    QMessageBox::information(this, tr("Probability unavailable"),
                             tr("Probability output is not available for the current image/classifier combination."));
    return;
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
  if (!ensureImageLoaded(tr("plotting the result"))) {
    return;
  }
  if (labelResult_.empty()) {
    if (!ensureModelReady(tr("plotting the result"))) {
      return;
    }
    onApplyClassifier();
    if (labelResult_.empty()) {
      return;
    }
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

void MainWindow::onSaveClassifier() {
  if (!ensureModelReady(tr("saving the classifier"))) {
    return;
  }
  const QString path = QFileDialog::getSaveFileName(this, tr("Save classifier"), QString(), tr("Classifier (*.yml *.yaml)"));
  if (path.isEmpty()) return;
  if (!model_.save(path, classes_, featureSettings_, classifierSettings_, trainingStats_)) {
    QMessageBox::warning(this, tr("Save failed"), tr("Failed to save classifier."));
    return;
  }
  logMessage(tr("Saved classifier to %1.").arg(path));
}

void MainWindow::onLoadClassifier() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Load classifier"), QString(), tr("Classifier (*.yml *.yaml)"));
  if (path.isEmpty()) return;
  std::vector<SegmentationClassInfo> loadedClasses;
  SegmentationFeatureSettings settings;
  SegmentationClassifierSettings classifierSettings;
  SegmentationTrainingStats stats;
  SegmentationClassifier loadedModel;
  if (!loadedModel.load(path, &loadedClasses, &settings, &classifierSettings, &stats)) {
    QMessageBox::warning(this, tr("Load failed"), tr("Failed to load classifier."));
    return;
  }
  model_ = loadedModel;
  classes_ = loadedClasses;
  featureSettings_ = settings;
  classifierSettings_ = classifierSettings;
  trainingStats_ = stats;
  if (!originalImage_.empty()) {
    clearAnnotationsForCurrentImage();
    rebuildFeatureStack();
    clearInferenceOutputs();
  }
  refreshClassList();
  const int comboIndex = classifierCombo_->findData(classifierSettings_.kind);
  if (comboIndex >= 0) classifierCombo_->setCurrentIndex(comboIndex);
  updateUiState();
  updateViewer();
  logMessage(tr("Loaded classifier from %1.").arg(path));
}

void MainWindow::onSaveData() {
  const QString path = QFileDialog::getSaveFileName(this, tr("Save training data"), QString(), tr("Training data (*.json)"));
  if (!path.isEmpty() && !saveTrainingData(path)) {
    QMessageBox::warning(this, tr("Save failed"), tr("Failed to save training data."));
  }
}

void MainWindow::onLoadData() {
  const QString path = QFileDialog::getOpenFileName(this, tr("Load training data"), QString(), tr("Training data (*.json)"));
  if (!path.isEmpty() && !loadTrainingData(path)) {
    QMessageBox::warning(this, tr("Load failed"), tr("Failed to load training data."));
  }
}

void MainWindow::onCreateNewClass() {
  bool ok = false;
  const QString name = QInputDialog::getText(this, tr("Create class"), tr("Class name"), QLineEdit::Normal,
                                             tr("Class %1").arg(classes_.size() + 1), &ok);
  if (!ok || name.trimmed().isEmpty()) {
    return;
  }
  const QColor color = QColorDialog::getColor(defaultColorForIndex(classes_.size()), this, tr("Choose class color"));
  if (!color.isValid()) {
    return;
  }
  classes_.push_back({name.trimmed(), color, true});
  if (!originalImage_.empty()) {
    classMasks_.push_back(cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0)));
    classBrushMasks_.push_back(cv::Mat(originalImage_.rows, originalImage_.cols, CV_8U, cv::Scalar(0)));
    classTracePolygons_.push_back({});
  }
  model_.clear();
  trainingStats_ = {};
  clearInferenceOutputs();
  refreshClassList();
  updateTraceLists();
  rebuildMasksFromAnnotations();
  updateViewer();
  logMessage(tr("Added class %1.").arg(name.trimmed()));
}

void MainWindow::onSettings() {
  QDialog dialog(this);
  dialog.setWindowTitle(tr("Feature and classifier settings"));
  auto* layout = new QVBoxLayout(&dialog);
  auto* featureGroup = new QGroupBox(tr("Feature extraction"), &dialog);
  auto* featureLayout = new QVBoxLayout(featureGroup);
  auto* intensity = new QCheckBox(tr("Intensity"), &dialog);
  auto* gaussian3 = new QCheckBox(tr("Gaussian sigma small"), &dialog);
  auto* gaussian7 = new QCheckBox(tr("Gaussian sigma large"), &dialog);
  auto* differenceOfGaussians = new QCheckBox(tr("Difference of Gaussians"), &dialog);
  auto* gradient = new QCheckBox(tr("Gradient magnitude"), &dialog);
  auto* laplacian = new QCheckBox(tr("Laplacian"), &dialog);
  auto* hessian = new QCheckBox(tr("Hessian norm"), &dialog);
  auto* localMean = new QCheckBox(tr("Local mean"), &dialog);
  auto* localStd = new QCheckBox(tr("Local std-dev"), &dialog);
  auto* entropy = new QCheckBox(tr("Local entropy"), &dialog);
  auto* texture = new QCheckBox(tr("Texture energy"), &dialog);
  auto* gabor = new QCheckBox(tr("Gabor response"), &dialog);
  auto* membrane = new QCheckBox(tr("Membrane response"), &dialog);
  auto* xpos = new QCheckBox(tr("X position"), &dialog);
  auto* ypos = new QCheckBox(tr("Y position"), &dialog);
  intensity->setChecked(featureSettings_.intensity);
  gaussian3->setChecked(featureSettings_.gaussian3);
  gaussian7->setChecked(featureSettings_.gaussian7);
  differenceOfGaussians->setChecked(featureSettings_.differenceOfGaussians);
  gradient->setChecked(featureSettings_.gradient);
  laplacian->setChecked(featureSettings_.laplacian);
  hessian->setChecked(featureSettings_.hessian);
  localMean->setChecked(featureSettings_.localMean);
  localStd->setChecked(featureSettings_.localStd);
  entropy->setChecked(featureSettings_.entropy);
  texture->setChecked(featureSettings_.texture);
  gabor->setChecked(featureSettings_.gabor);
  membrane->setChecked(featureSettings_.membrane);
  xpos->setChecked(featureSettings_.xPosition);
  ypos->setChecked(featureSettings_.yPosition);
  for (QCheckBox* checkbox : {intensity, gaussian3, gaussian7, differenceOfGaussians, gradient, laplacian, hessian,
                              localMean, localStd, entropy, texture, gabor, membrane, xpos, ypos}) {
    featureLayout->addWidget(checkbox);
  }

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
  classifierLayout->addRow(balanceClasses);
  classifierLayout->addRow(tr("RF trees"), rfTrees);
  classifierLayout->addRow(tr("RF max depth"), rfDepth);
  classifierLayout->addRow(tr("SVM C"), svmC);
  classifierLayout->addRow(tr("SVM gamma"), svmGamma);

  layout->addWidget(featureGroup);
  layout->addWidget(classifierGroup);
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
  layout->addWidget(buttons);
  connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  if (dialog.exec() != QDialog::Accepted) {
    return;
  }
  featureSettings_.intensity = intensity->isChecked();
  featureSettings_.gaussian3 = gaussian3->isChecked();
  featureSettings_.gaussian7 = gaussian7->isChecked();
  featureSettings_.differenceOfGaussians = differenceOfGaussians->isChecked();
  featureSettings_.gradient = gradient->isChecked();
  featureSettings_.laplacian = laplacian->isChecked();
  featureSettings_.hessian = hessian->isChecked();
  featureSettings_.localMean = localMean->isChecked();
  featureSettings_.localStd = localStd->isChecked();
  featureSettings_.entropy = entropy->isChecked();
  featureSettings_.texture = texture->isChecked();
  featureSettings_.gabor = gabor->isChecked();
  featureSettings_.membrane = membrane->isChecked();
  featureSettings_.xPosition = xpos->isChecked();
  featureSettings_.yPosition = ypos->isChecked();
  classifierSettings_.balanceClasses = balanceClasses->isChecked();
  classifierSettings_.randomForestTrees = rfTrees->value();
  classifierSettings_.randomForestMaxDepth = rfDepth->value();
  classifierSettings_.svmC = svmC->value();
  classifierSettings_.svmGamma = svmGamma->value();
  model_.clear();
  trainingStats_ = {};
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
  const QString path = QFileDialog::getSaveFileName(this, tr("Export label mask"), QString(), tr("PNG (*.png);;TIFF (*.tif *.tiff)"));
  if (path.isEmpty()) return;
  cv::Mat exportMask;
  labelResult_.convertTo(exportMask, CV_8U);
  if (!cv::imwrite(path.toStdString(), exportMask)) {
    QMessageBox::warning(this, tr("Export failed"), tr("Failed to export the label mask."));
    return;
  }
  logMessage(tr("Exported label mask to %1.").arg(path));
}

void MainWindow::onBrushRadiusChanged(int value) {
  view_->setBrushRadius(value);
}

void MainWindow::onClassSelectionChanged() {
  const int cls = currentSelectedClassIndex();
  if (cls >= 0 && cls < static_cast<int>(classes_.size())) {
    view_->setActiveClass(cls, classes_[cls].color);
    if (!pendingTrace_.isEmpty()) {
      view_->setPendingTrace(pendingTrace_, classes_[cls].color);
    }
    addTraceButton_->setText(tr("Add ROI to %1").arg(classes_[cls].name));
    addTraceButton_->setToolTip(tr("Commit the current ROI trace to %1.").arg(classes_[cls].name));
    infoLabel_->setText(tr("Active class: %1 — use left mouse to paint samples, or draw a ROI trace and commit it to the selected class.").arg(classes_[cls].name));
  } else if (addTraceButton_) {
    addTraceButton_->setText(tr("Add ROI to selected class"));
    addTraceButton_->setToolTip(tr("Commit the current ROI trace to the currently selected class."));
  }
  updateTraceLists();
  updateUiState();
}

void MainWindow::onProbabilityClassChanged(int index) {
  Q_UNUSED(index);
  updateProbabilityView();
}

void MainWindow::onOpacityChanged(int value) {
  overlayOpacity_ = static_cast<double>(value) / 100.0;
  updateViewer();
}

void MainWindow::onPaintTool() {
  view_->setToolMode(SegmentationView::PaintTool);
}

void MainWindow::onTraceTool() {
  view_->setToolMode(SegmentationView::TraceTool);
  infoLabel_->setText(tr("Draw a freehand ROI, then click Add ROI to selected class to commit it."));
}

void MainWindow::onEraseTool() {
  view_->setToolMode(SegmentationView::EraseTool);
}

void MainWindow::onPanTool() {
  view_->setToolMode(SegmentationView::PanTool);
}

void MainWindow::onResetZoom() {
  view_->resetView();
}

void MainWindow::onAddTraceToSelectedClass() {
  addPendingTraceToSelectedClass();
}

void MainWindow::onRemoveSelectedTrace() {
  removeSelectedTraceFromClass();
}
