#pragma once

#include <QMainWindow>
#include <QVector>
#include <QColor>
#include <QListWidget>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>
#include <QTextEdit>
#include <QComboBox>
#include <QPolygon>

#include <opencv2/core.hpp>

#include <vector>

#include "SegmentationEngine.h"

class QCustomPlot;
class SegmentationView;

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override = default;

private slots:
  void onOpenImage();
  void onTrainClassifier();
  void onApplyClassifier();
  void onToggleOverlay();
  void onCreateResult();
  void onGetProbability();
  void onPlotResult();
  void onSaveClassifier();
  void onLoadClassifier();
  void onSaveData();
  void onLoadData();
  void onCreateNewClass();
  void onSettings();
  void onExportMask();
  void onBrushRadiusChanged(int value);
  void onClassSelectionChanged();
  void onProbabilityClassChanged(int index);
  void onOpacityChanged(int value);
  void onPaintTool();
  void onEraseTool();
  void onPanTool();
  void onTraceTool();
  void onResetZoom();
  void onAddToClass1();
  void onAddToClass2();
  void onRemoveSelectedClass1Trace();
  void onRemoveSelectedClass2Trace();

private:
  void buildUi();
  void connectSignals();
  void initializeDefaultClasses();
  void refreshClassList();
  void updateUiState();
  void rebuildFeatureStack();
  void updateViewer();
  void updateProbabilityView();
  void repaintAnnotationPreview();
  void applyBrushStroke(const QPoint& imagePos, int radius, bool erase);
  bool ensureImageLoaded(const QString& actionName);
  bool ensureModelReady(const QString& actionName);
  void logMessage(const QString& message);
  void activateLabelShortcut(int classIndex, const QString& semanticDescription);
  void ensureAnnotationStorage();
  void rebuildMasksFromAnnotations();
  void updateTraceLists();
  void setPendingTrace(const QPolygon& trace);
  void addPendingTraceToClass(int classIndex, const QString& semanticDescription);
  void removeSelectedTraceFromClass(int classIndex);

  bool loadImageFile(const QString& path);
  bool saveTrainingData(const QString& path);
  bool loadTrainingData(const QString& path);

  static QImage cvMatToQImage(const cv::Mat& mat);
  static cv::Mat qImageToCvMat(const QImage& image);

  SegmentationView* view_ = nullptr;
  QListWidget* classList_ = nullptr;
  QLabel* infoLabel_ = nullptr;
  QLabel* probabilityLabel_ = nullptr;
  QLabel* modelLabel_ = nullptr;
  QLabel* imagePathLabel_ = nullptr;
  QTextEdit* logEdit_ = nullptr;
  QSpinBox* brushSizeSpin_ = nullptr;
  QSlider* opacitySlider_ = nullptr;
  QComboBox* probabilityCombo_ = nullptr;
  QComboBox* classifierCombo_ = nullptr;
  QCheckBox* overlayCheck_ = nullptr;
  QCheckBox* contourCheck_ = nullptr;

  QPushButton* trainButton_ = nullptr;
  QPushButton* applyButton_ = nullptr;
  QPushButton* createResultButton_ = nullptr;
  QPushButton* probabilityButton_ = nullptr;
  QPushButton* addToClass1Button_ = nullptr;
  QPushButton* addToClass2Button_ = nullptr;
  QPushButton* removeClass1TraceButton_ = nullptr;
  QPushButton* removeClass2TraceButton_ = nullptr;
  QListWidget* class1TraceList_ = nullptr;
  QListWidget* class2TraceList_ = nullptr;

  cv::Mat originalImage_;
  cv::Mat featureStack_;
  cv::Mat labelResult_;
  cv::Mat overlayImage_;
  cv::Mat probabilityImage_;
  std::vector<cv::Mat> classMasks_;
  std::vector<cv::Mat> classBrushMasks_;
  std::vector<std::vector<QPolygon>> classTracePolygons_;
  QPolygon pendingTrace_;
  std::vector<SegmentationClassInfo> classes_;
  SegmentationFeatureSettings featureSettings_;
  SegmentationClassifier model_;
  SegmentationClassifierSettings classifierSettings_;
  SegmentationTrainingStats trainingStats_;
  QString imagePath_;
  bool showOverlay_ = true;
  double overlayOpacity_ = 0.45;
};
