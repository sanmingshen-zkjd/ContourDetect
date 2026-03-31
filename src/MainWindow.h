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
#include <QGroupBox>
#include <QFutureWatcher>

#include <opencv2/core.hpp>

#include <vector>

#include "SegmentationEngine.h"

class QCustomPlot;
class SegmentationView;
class QToolButton;

struct TraceRegion {
  QString name;
  QPolygon polygon;
};

struct AnnotationSnapshot {
  std::vector<cv::Mat> classBrushMasks;
  std::vector<std::vector<TraceRegion>> classTraceRegions;
  std::vector<QPolygon> exclusionRegions;
};

struct ProjectImageEntry {
  QString imagePath;
  QString workingDataPath;
};

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow() override = default;

private slots:
  void onOpenImage();
  void onTrainClassifier();
  void onStopTraining();
  void onApplyClassifier();
  void onToggleOverlay();
  void onCreateResult();
  void onGetProbability();
  void onPlotResult();
  void onEvaluateModel();
  void onSuggestLabels();
  void onSaveClassifier();
  void onLoadClassifier();
  void onSaveData();
  void onLoadData();
  void onCreateNewClass();
  void onSettings();
  void onExportRoiMacro();
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
  void onAddTraceToSelectedClass();
  void onAddExclusionRoi();
  void onClearExclusionRois();
  void onRemoveSelectedTrace();
  void onRenameSelectedTrace();
  void onClearTracesForSelectedClass();
  void onUndo();
  void onRedo();
  void onSliceChanged(int value);
  void onNewProject();
  void onOpenProject();
  void onSaveProject();
  void onAddImageToProject();
  void onRemoveProjectImage();
  void onProjectSelectionChanged();

private:
  void buildUi();
  void connectSignals();
  void initializeDefaultClasses();
  void refreshClassList();
  void updateUiState();
  void rebuildFeatureStack();
  void updateViewer();
  bool updateProbabilityView(bool showProgress = false, const QString& title = QString());
  bool computeProbabilityOutputs(const QString& title,
                                 int classIndex,
                                 cv::Mat* probabilityImage,
                                 cv::Mat* fullProbabilities = nullptr);
  void repaintAnnotationPreview();
  void applyBrushStroke(const QPoint& imagePos, int radius, bool erase);
  bool ensureImageLoaded(const QString& actionName);
  bool ensureModelReady(const QString& actionName);
  void logMessage(const QString& message);
  void activateLabelShortcut(int classIndex, const QString& semanticDescription);
  void ensureAnnotationStorage();
  void rebuildMasksFromAnnotations();
  void clearAnnotationsForCurrentImage();
  void clearInferenceOutputs();
  void updateTraceLists();
  void setPendingTrace(const QPolygon& trace);
  void addPendingTraceToSelectedClass();
  void removeSelectedTraceFromClass();
  int currentSelectedClassIndex() const;
  bool applyClassifierToImage(const cv::Mat& image, const QString& path, bool resetAnnotations);
  bool loadImageFile(const QString& path);
  bool saveTrainingData(const QString& path);
  bool loadTrainingData(const QString& path);
  bool saveTrainingDataJson(const QString& path);
  bool loadTrainingDataJson(const QString& path);
  bool saveTrainingDataArff(const QString& path);
  bool loadTrainingDataArff(const QString& path);
  bool saveProject(const QString& path);
  bool loadProject(const QString& path);
  void switchToProjectImage(int index);
  void persistCurrentProjectImageState();
  QString ensureEntryWorkingPath(int index);
  void updateProjectList();
  void ensureSliceStorage();
  void saveCurrentSliceState();
  void restoreSliceState(int sliceIndex);
  void pushUndoSnapshot();
  void restoreSnapshot(const AnnotationSnapshot& snapshot);
  void trimUndoHistory();
  QString defaultTraceNameForClass(int classIndex) const;
  void clearImportedTrainingData();

  static QImage cvMatToQImage(const cv::Mat& mat);
  static cv::Mat qImageToCvMat(const QImage& image);

  SegmentationView* view_ = nullptr;
  QListWidget* classList_ = nullptr;
  QLabel* infoLabel_ = nullptr;
  QLabel* probabilityLabel_ = nullptr;
  QLabel* modelLabel_ = nullptr;
  QLabel* imagePathLabel_ = nullptr;
  QLabel* sliceLabel_ = nullptr;
  QTextEdit* logEdit_ = nullptr;
  QSpinBox* brushSizeSpin_ = nullptr;
  QSlider* opacitySlider_ = nullptr;
  QSlider* sliceSlider_ = nullptr;
  QComboBox* probabilityCombo_ = nullptr;
  QComboBox* classifierCombo_ = nullptr;
  QCheckBox* overlayCheck_ = nullptr;
  QCheckBox* contourCheck_ = nullptr;

  QPushButton* trainButton_ = nullptr;
  QPushButton* stopTrainingButton_ = nullptr;
  QPushButton* applyButton_ = nullptr;
  QPushButton* createResultButton_ = nullptr;
  QPushButton* probabilityButton_ = nullptr;
  QPushButton* addTraceButton_ = nullptr;
  QPushButton* addExclusionRoiButton_ = nullptr;
  QPushButton* clearExclusionRoiButton_ = nullptr;
  QPushButton* removeTraceButton_ = nullptr;
  QPushButton* renameTraceButton_ = nullptr;
  QPushButton* clearTraceButton_ = nullptr;
  QPushButton* undoButton_ = nullptr;
  QPushButton* redoButton_ = nullptr;
  QPushButton* evaluateButton_ = nullptr;
  QPushButton* suggestButton_ = nullptr;
  QPushButton* exportRoiMacroButton_ = nullptr;
  QListWidget* traceList_ = nullptr;
  QListWidget* projectList_ = nullptr;
  QGroupBox* traceGroup_ = nullptr;

  cv::Mat originalImage_;
  cv::Mat featureStack_;
  cv::Mat labelResult_;
  cv::Mat overlayImage_;
  cv::Mat probabilityImage_;
  std::vector<cv::Mat> classMasks_;
  std::vector<cv::Mat> classBrushMasks_;
  std::vector<std::vector<TraceRegion>> classTraceRegions_;
  std::vector<QPolygon> exclusionRegions_;
  cv::Mat exclusionMask_;
  QPolygon pendingTrace_;
  std::vector<SegmentationClassInfo> classes_;
  SegmentationFeatureSettings featureSettings_;
  SegmentationClassifier model_;
  SegmentationClassifierSettings classifierSettings_;
  SegmentationTrainingStats trainingStats_;
  QString imagePath_;
  bool showOverlay_ = true;
  double overlayOpacity_ = 0.45;
  bool trainingInProgress_ = false;
  bool stopTrainingRequested_ = false;

  std::vector<cv::Mat> imageVolume_;
  std::vector<cv::Mat> featureVolume_;
  int currentSliceIndex_ = 0;
  std::vector<AnnotationSnapshot> sliceAnnotationStates_;
  std::vector<cv::Mat> sliceLabelResults_;
  std::vector<cv::Mat> sliceProbabilityResults_;
  std::vector<AnnotationSnapshot> undoStack_;
  std::vector<AnnotationSnapshot> redoStack_;

  QString projectPath_;
  std::vector<ProjectImageEntry> projectImages_;
  int currentProjectImageIndex_ = -1;
  bool switchingProjectSelection_ = false;
  cv::Mat importedTrainingSamples_;
  cv::Mat importedTrainingLabels_;
  QString importedTrainingSource_;
  bool probabilityTaskRunning_ = false;
  bool trainingTaskRunning_ = false;
};
