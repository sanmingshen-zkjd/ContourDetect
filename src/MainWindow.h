#pragma once
#include <QMainWindow>
#include <QThread>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QDockWidget>
#include <QTimer>
#include <QLabel>
#include <QTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QTabWidget>
#include <QTabBar>
#include <QListWidget>
#include <QCheckBox>
#include <QComboBox>
#include <QStatusBar>
#include <QGroupBox>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QSettings>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QGraphicsLineItem>
#include <QGraphicsPolygonItem>
#include <QPolygonF>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QResizeEvent>
#include <QSlider>
#include <QProgressBar>
#include <QTableWidget>
#include <QToolButton>
#include <QLineEdit>
#include <QStringList>
#include <QVector>
#include <QColor>
#include <functional>
#include <limits>

#include <opencv2/opencv.hpp>
#include "Core.h"
#include "SolveWorker.h"
#include "Types.h"
#include <QMutex>

struct InputSource {
  bool is_cam=false;
  bool is_image_seq=false;
  int cam_id=-1;
  QString video_path;
  QString seq_dir;
  QStringList seq_files;
  int seq_idx=0;
  int mode_owner=0; // 0=Calibration tab, 1=Tracking tab
  cv::VideoCapture cap;
};

class CaptureWorker;
class SolveWorker;
class QCustomPlot;

class ImageViewer : public QGraphicsView {
public:
  enum ToolMode { PanTool=0, PointTool=1, LineTool=2, PolygonTool=3 };
  explicit ImageViewer(QWidget* parent=nullptr);
  void setImage(const QImage& img);
  void setToolMode(ToolMode mode);
  void zoomIn();
  void zoomOut();
  void resetView();
  void clearAnnotations();
  void setLineCreatedCallback(const std::function<void(double)>& cb);
  void setLineDoubleClickCallback(const std::function<void(double)>& cb);
  void setLineValueEditedCallback(const std::function<void(double,double)>& cb);
  void setPolygonFinishedCallback(const std::function<void(const QPolygonF&)>& cb);
  void setRegionPolygons(const std::vector<QPolygonF>& polygons, const std::vector<bool>& includes, int highlightedIndex=-1);
  void setRegionEditIndex(int index);
  void setRegionEditedCallback(const std::function<void(int, const QPolygonF&)>& cb);
  void applySelectedLineStyle(const QString& name, const QColor& color, int width);
  void setAnnotationsVisible(bool visible);
  void setPointClickCallback(const std::function<void(const QPointF&)>& cb);
  void clearAllLines();
  double selectedLineLength() const;
  double anyLineLength() const;

protected:
  void wheelEvent(QWheelEvent* e) override;
  void mousePressEvent(QMouseEvent* e) override;
  void mouseMoveEvent(QMouseEvent* e) override;
  void mouseDoubleClickEvent(QMouseEvent* e) override;
  void mouseReleaseEvent(QMouseEvent* e) override;
  void resizeEvent(QResizeEvent* e) override;

private:
  void applyZoom(double factor);

  QGraphicsScene scene_;
  QGraphicsPixmapItem* pixmapItem_ = nullptr;
  ToolMode toolMode_ = PanTool;
  double zoomFactor_ = 1.0;
  bool lineDrawing_ = false;
  QPointF lineStart_;
  QGraphicsLineItem* previewLine_ = nullptr;
  bool polygonDrawing_ = false;
  QPolygonF polygonPoints_;
  QGraphicsPolygonItem* previewPolygon_ = nullptr;
  std::vector<QGraphicsPolygonItem*> regionPolygonItems_;
  std::vector<QGraphicsEllipseItem*> regionEditHandles_;
  int editingRegionIndex_ = -1;
  int highlightedRegionIndex_ = -1;
  bool draggingRegionHandle_ = false;
  int draggingHandleIndex_ = -1;
  std::function<void(double)> onLineCreated_;
  std::function<void(double)> onLineDoubleClick_;
  std::function<void(double,double)> onLineValueEdited_;
  std::function<void(const QPolygonF&)> onPolygonFinished_;
  std::function<void(int, const QPolygonF&)> onRegionEdited_;
  std::function<void(const QPointF&)> onPointClicked_;
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
  struct MeasureRow { double disp=0, speed=0, accel=0, area=0, perim=0, major=0, minor=0, circ=0; qint64 key=0; };
  struct TargetMeasureRow { int id=-1; MeasureRow m; };

public:
  MainWindow(const std::vector<InputSource>& sources,
             int board_w, int board_h, double square_m,
             QWidget* parent=nullptr);
  ~MainWindow();

private slots:
  void onTick();

  // Sources actions
  void onAddCamera();
  void onAddVideo();
  void onAddImageSequence();
  void onCaptureNow();
  void onRemoveSource();
  void onPauseResumeSelected();
  void onPlayAll();
  void onStopAll();
  void onStepPrevFrame();
  void onStepNextFrame();
  void onToolPan();
  void onToolPoint();
  void onToolLine();
  void onZoomIn();
  void onZoomOut();
  void onResetView();
  void onClearAnnotations();
  void onProgressSliderReleased();
  void onFrameJumpReturnPressed();

  // Calibration actions
  void onGrabFrame();
  void onResetFrames();
  void onComputeCalibration();
  void onRecomputeCalibrationSelected();
  void onSaveCalibrationYaml();

  // Tracking actions
  void onLoadTagMap();
  void onLoadCalibYaml();
  //void onTogglePose(bool on);
  void onDetectAllTrackingFrames();
  void onExportTrajectory();
  void onSaveProject();
  void onLoadProject();
  void onSaveLayout();
  void onRestoreLayout();
  void onToggleDocks(bool on);

  void onFramesFromWorker(FramePack frames, qint64 capture_ts_ms);
  void onPoseFromWorker(const PoseResult& res);

  // UI mode
  void onModeCalibration();
  void onModeTracking();
  void onModeCapture();
  void onModeTabChanged(int idx);
  void onPreprocessParamsChanged();
  void onPreprocessAuto();
  void onObjectThresholdParamsChanged();
  void onStartScaleLine();
  void onDeleteScaleLine();
  void onApplyScaleFromInput();
  void onTryAllGlobalMethods();
  void onTryAllLocalMethods();
  void onShowBinaryPreviewPopup();
  void onBinaryPreviewPrevFrame();
  void onBinaryPreviewNextFrame();
  void onBinaryPreviewSnap();
  void onAddMaskRegion();
  void onAddDetectRegion();
  void onStartAddMaskContour();
  void onStartModifyMaskContour();
  void onStartDeleteMaskContour();
  void onContourMaskPointClicked(const QPointF& p);
  void onToggleMaskEditEnabled(bool on);
  void onDeleteRegion();
  void onRegionTableSelectionChanged();
  void onSelectBinaryOp();
  void onUndoBinaryOp();
  void onAnalyzeParticles();
  void onToggleTrackBinary();

private:
  void buildUI();
  void logLine(const QString& s);
  bool openAllSources();
  void closeAllSources();
  void rebuildCalibratorFromUI(bool reset=true);
  void refreshSourceList();
  void rebuildSourceDocks();
  void updateSourceDocks(const std::vector<cv::Mat>& frames);
  void rebuildSourceViews();
  void updateSourceViews(const std::vector<cv::Mat>& frames);
  std::vector<int> activeSourceIndices() const;
  void stopCaptureBlocking();
  void updatePlaybackParams();
  void stepAllVideos(int delta);
  void updateProgressUI(int64_t frame, int64_t endFrame);
  int videoSourceCount() const;
  void setSourceEnabled(int idx, bool enabled);
  bool readFrames(std::vector<cv::Mat>& frames);
  void updateFpsStats(double dt_ms);
  QJsonObject toProjectJson() const;
  bool fromProjectJson(const QJsonObject& o);

  static QImage matToQImage(const cv::Mat& bgr);
  static cv::Mat makeMosaic(const std::vector<cv::Mat>& imgs, int cols=2);

  void overlayCalibration(std::vector<cv::Mat>& vis, const std::vector<cv::Mat>& frames);
  void overlayTracking(std::vector<cv::Mat>& vis, const std::vector<cv::Mat>& frames);

  void updateStatus();
  void updateStepAvailability();
  bool hasAnySourceInCurrentMode();
  cv::Mat applyPreprocess(const cv::Mat& src) const;
  void updateScaleStatus(double pxLen);
  cv::Mat makeObjectBinaryMask(const cv::Mat& src, int* outGlobalThreshold=nullptr) const;
  cv::Mat applyBinaryProcessOps(const cv::Mat& binMask) const;
  cv::Mat makeObjectBinaryPreview(const cv::Mat& src, int* outGlobalThreshold=nullptr) const;
  std::vector<std::vector<cv::Point>> detectBinaryContours(const cv::Mat& src, int* outGlobalThreshold=nullptr) const;
  bool runCalibrationOnPairs(const std::vector<int>& pairIndices, bool updateTable);
  void refreshTrajectoryPlot();
  void updateMeasurementFromFrame(const cv::Mat& preprocessedFrame);
  void updateHistogramPlot();
  void refreshRegionTable();
  void refreshRegionOverlays(int highlightedIndex=-1, int editingIndex=-1);
  void updateLeftVisualDashboard();
  void updateDetectParticlePanel();
  void fitAllVisualPlots();
  void updateBinaryPreviewPopupFrame();
  void refreshBinaryPreviewPopupPixmaps();
  void refreshVisualizationImagePixmap();
  double metricValueForHist(const MeasureRow& r, const QString& metric) const;
  bool passesConfirmedHistogramRules(const MeasureRow& r) const;
  void configureHistogramEditorsForMetric(const QString& metric);
  void rebuildMeasurementSeriesFromCurrentSource(bool showProgress=false);
  void savePlotAsBmp(QCustomPlot* plot, const QString& nameHint);
  void onExportTableCsv();
  void onCaptureVisualSnapshot();
  void onExportVisualMp4();
  void onAddVisualizationChart();
  void onVisualizationPlotContextMenu(const QPoint& pos);
  void ensureWindowInVisibleArea();
  bool chooseVisualizationDataTypes(QVector<int>& components, QString& labelOut);

private:
  // Inputs
  std::vector<InputSource> sources_;
  std::vector<bool> source_enabled_;
  std::vector<cv::Mat> last_frames_;
  int num_cams_=0;

  // Calib params
  int board_w_=0, board_h_=0;
  double square_=0.0;

  // Core
  std::unique_ptr<MultiCamCalibrator> calibrator_;
  std::vector<CameraModel> cams_;
  bool calib_loaded_=false;

  std::unordered_map<uint64_t, Eigen::Vector3d> tag_corner_map_;
  bool tagmap_loaded_=false;
  QString tagmap_path_;
  QString calib_path_;

  // Pose state
 // bool pose_on_=false;
  Eigen::Matrix3d R_wr_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_wr_ = Eigen::Vector3d::Zero();
  int last_inliers_=0;
  struct TrajRow { qint64 t_ms; Eigen::Vector3d t; Eigen::Vector3d aa; int inliers; };
  std::vector<TrajRow> traj_;
  int ransac_iters_=280;
  double inlier_thresh_px_=3.0;
  int tag_dict_id_=cv::aruco::DICT_APRILTAG_36h11;
  double fps_=0.0;
  qint64 last_tick_ms_=0;

  // Mode
  enum Mode { CAPTURE=0, CALIB=1, TRACK=2 } mode_=CAPTURE;

  // UI widgets
  QTabWidget* modeTabs_=nullptr;
  QWidget* viewsHost_=nullptr;
  QGridLayout* viewsGrid_=nullptr;
  std::vector<ImageViewer*> sourceViews_;
  std::vector<int> active_view_source_indices_;
  QTextEdit* log_=nullptr;

  // Per-source dock views
  bool show_docks_ = false; // default OFF to avoid duplicate display
  std::vector<QDockWidget*> camDocks_;
  std::vector<QLabel*> camLabels_;

  // Menu actions
  QAction* actSaveProject_=nullptr;
  QAction* actLoadProject_=nullptr;
  QAction* actSaveLayout_=nullptr;
  QAction* actRestoreLayout_=nullptr;
  QAction* actExportTraj_=nullptr;
  QAction* actToggleDocks_=nullptr;


  // Sources panel
  QToolButton* btnAddCam_=nullptr;
  QToolButton* btnAddVideo_=nullptr;
  QToolButton* btnAddImgSeq_=nullptr;
  QPushButton* btnCaptureNow_=nullptr;
  QToolButton* btnRemoveSource_=nullptr;
  QToolButton* btnPlayAll_=nullptr;
  QToolButton* btnStopAll_=nullptr;
  QToolButton* btnStepPrev_=nullptr;
  QToolButton* btnStepNext_=nullptr;
  QDoubleSpinBox* spSourceFps_=nullptr;
  QLabel* lblSourcePath_=nullptr;

  QToolButton* btnToolPan_=nullptr;
  QToolButton* btnToolPoint_=nullptr;
  QToolButton* btnToolLine_=nullptr;
  QToolButton* btnZoomIn_=nullptr;
  QToolButton* btnZoomOut_=nullptr;
  QToolButton* btnResetView_=nullptr;
  QToolButton* btnClearAnno_=nullptr;
  QLabel* lblLineState_=nullptr;
  QSlider* progressSlider_=nullptr;
  QLineEdit* editCurFrame_=nullptr;
  QLabel* lblTotalFrame_=nullptr;
  QLabel* lblResolution_=nullptr;
  //QCheckBox* chkSyncPlay_=nullptr;
  //QLabel* lblPlayState_=nullptr;
  QTabBar* sideModeTabs_=nullptr;
  QTabBar* stepTabs_=nullptr;
  bool stepDone_[4] = {false, false, false, false};
  bool pre_scale_line_drawn_ = false;
  bool pre_scale_calculated_ = false;
  QToolButton* btnFileMenu_=nullptr;
  QTabWidget* actionTabs_=nullptr;

  // Calibration tab
  QSpinBox* spBoardW_=nullptr;
  QSpinBox* spBoardH_=nullptr;
  QDoubleSpinBox* spSquare_=nullptr;
  QComboBox* cbCalibMethod_=nullptr;
  QPushButton* btnGrab_=nullptr;
  QPushButton* btnReset_=nullptr;
  QPushButton* btnComputeCalib_=nullptr;
  QPushButton* btnRecomputeCalib_=nullptr;
  QPushButton* btnSaveCalib_=nullptr;
  QProgressBar* calibProgressBar_=nullptr;
  QLabel* lblCalibProgress_=nullptr;
  QTableWidget* calibErrorTable_=nullptr;
  QLabel* lblCaptured_=nullptr;

  // Preprocess tab
  QComboBox* cbPreColor_=nullptr;
  QSlider* slBrightness_=nullptr;
  QSlider* slContrast_=nullptr;
  QSpinBox* spBrightness_=nullptr;
  QSpinBox* spContrast_=nullptr;
  QComboBox* cbLineColor_=nullptr;
  QSpinBox* spLineWidth_=nullptr;
  QPushButton* btnStartScaleLine_=nullptr;
  QCheckBox* chkShowLines_=nullptr;
  QPushButton* btnDeleteScaleLine_=nullptr;
  QLineEdit* editPhysicalMm_=nullptr;
  QPushButton* btnCalcScale_=nullptr;
  QGroupBox* gbLineProps_=nullptr;
  QLabel* lblPreprocessHint_=nullptr;
  QPushButton* btnPreAuto_=nullptr;
  QLabel* lblScaleInfo_=nullptr;
  QPushButton* btnAddMaskRegion_=nullptr;
  QPushButton* btnAddDetectRegion_=nullptr;
  QPushButton* btnDeleteRegion_=nullptr;
  QTableWidget* tblRegions_=nullptr;

  // ObjectDefine tab
  QComboBox* cbThreshType_=nullptr;
  QComboBox* cbGlobalMethod_=nullptr;
  QComboBox* cbLocalMethod_=nullptr;
  QSlider* slObjectThresh_=nullptr;
  QSpinBox* spObjectThresh_=nullptr;
  QCheckBox* chkInvertBinary_=nullptr;
  QPushButton* btnBinaryPreviewPopup_=nullptr;
  QSpinBox* spLocalBlockSize_=nullptr;
  QDoubleSpinBox* spLocalK_=nullptr;
  QLabel* lblBinaryPreview_=nullptr;
  QDialog* binaryPreviewDialog_=nullptr;
  QLabel* lblBinaryPreviewOrig_=nullptr;
  QLabel* lblBinaryPreviewPopup_=nullptr;
  QPushButton* btnBinaryPreviewPrev_=nullptr;
  QPushButton* btnBinaryPreviewNext_=nullptr;
  QPushButton* btnBinaryPreviewPlay_=nullptr;
  QPushButton* btnBinaryPreviewSnap_=nullptr;
  QComboBox* cbBinaryPreviewMorphOp_=nullptr;
  QComboBox* cbBinaryPreviewContourMode_=nullptr;
  QComboBox* cbBinaryPreviewFitMethod_=nullptr;
  QPushButton* btnBinaryPreviewMorphUndo_=nullptr;
  QLabel* lblBinaryPreviewMorphOps_=nullptr;
  QPushButton* btnTryAllGlobal_=nullptr;
  QPushButton* btnTryAllLocal_=nullptr;
  QComboBox* cbBinaryOp_=nullptr;
  QPushButton* btnUndoBinaryOp_=nullptr;
  QPushButton* btnAnalyzeParticles_=nullptr;
  QCheckBox* chkEnableMaskEdit_=nullptr;
  QPushButton* btnAddMaskContour_=nullptr;
  QPushButton* btnModifyMaskContour_=nullptr;
  QPushButton* btnDeleteMaskContour_=nullptr;
  QPushButton* btnTrackBinary_=nullptr;
  QLabel* lblBinaryOps_=nullptr;
  QLabel* lblDetectParticleCount_=nullptr;
  QTableWidget* tblDetectParticles_=nullptr;

  // Tracking tab
  QPushButton* btnLoadTag_=nullptr;
  QPushButton* btnExportTraj_=nullptr;
  QSpinBox* spRansacIters_=nullptr;
  QDoubleSpinBox* spInlierThresh_=nullptr;
  QComboBox* cbTagDict_=nullptr;
  QLabel* lblFps_=nullptr;
  QLabel* lblLatency_=nullptr;
  QPushButton* btnLoadYaml_=nullptr;
  QLabel* lblTagPath_=nullptr;
  QLabel* lblYamlPath_=nullptr;
  QCheckBox* chkPose_=nullptr;
  QPushButton* btnDetectAll_=nullptr;
  QLabel* lblPose_=nullptr;
  QLabel* lblInliers_=nullptr;
  QLabel* visImageLabel_=nullptr;
  QCustomPlot* lblTrajPosPlot_=nullptr;
  QCustomPlot* lblTrajAngPlot_=nullptr;
  QCustomPlot* plotArea_=nullptr;
  QCustomPlot* plotPerimeter_=nullptr;
  QCustomPlot* plotCircularity_=nullptr;
  QCustomPlot* plotAccel_=nullptr;
  QTableWidget* tblMeasurements_=nullptr;
  QComboBox* cbHistMetric_=nullptr;
  QDoubleSpinBox* spHistMin_=nullptr;
  QDoubleSpinBox* spHistMax_=nullptr;
  QCustomPlot* plotHistogram_=nullptr;
  QPushButton* btnHistApply_=nullptr;
  QPushButton* btnAddVisChart_=nullptr;
  QVBoxLayout* visChartsLayout_=nullptr;
  struct VisChartConfig {
    QCustomPlot* plot = nullptr;
    QComboBox* selector = nullptr;
    QVector<int> components;
    QString yLabel;
    bool removable = false;
  };
  std::vector<VisChartConfig> visCharts_;

  // Settings
  QSettings settings_;

  // Threads
  QThread captureThread_;
  QThread solveThread_;
  CaptureWorker* captureWorker_=nullptr;
  SolveWorker* solveWorker_=nullptr;
  QMutex sources_mutex_;
  QMutex frames_mutex_;
  qint64 last_capture_ts_ms_=0;
  bool playback_running_=false;
  bool sync_play_=true;
  int64_t play_frame_=0;
  int64_t play_end_frame_=0;
  double play_fps_=30.0;
  int ui_frame_skip_=0;
  int ui_overlay_div_=4; // run heavy overlay every N UI ticks
  double mm_per_pixel_ = 0.0;
  bool object_thresh_manual_ = false;
  std::vector<QString> binary_ops_pipeline_;
  std::vector<std::vector<cv::Point>> analyzed_contours_;
  enum ContourMaskMode { ContourMaskNone=0, ContourMaskAdd=1, ContourMaskModify=2 };
  int contour_mask_mode_ = ContourMaskNone;
  int contour_mask_selected_index_ = -1;
  int64_t contour_mask_frame_key_ = -1;
  std::vector<std::vector<cv::Point>> contour_mask_working_contours_;
  std::vector<cv::Point> contour_mask_add_path_;
  std::vector<cv::Point> contour_mask_add_snap_points_;
  bool track_binary_enabled_ = false;
  int next_track_id_ = 1;
  std::unordered_map<int, cv::Point2f> tracked_centroids_;
  std::unordered_map<int, std::vector<cv::Point>> tracked_contours_;
  std::vector<MeasureRow> meas_rows_;
  std::vector<TargetMeasureRow> target_meas_rows_;
  qint64 last_meas_key_ = std::numeric_limits<qint64>::min();
  cv::Point2f last_ctr_{0,0};
  cv::Point2f first_ctr_{0,0};
  double last_speed_ = 0.0;
  bool measurements_frozen_ = false;
  struct RegionSpec { int id=0; bool include=true; QPolygonF poly; };
  std::vector<RegionSpec> regions_;
  int next_region_id_ = 1;
  int drawing_region_type_ = 0; // 0 none, 1 include, 2 exclude
  int editing_region_index_ = -1;
  std::vector<std::vector<std::vector<cv::Point>>> analyzed_contours_by_frame_;
  struct AnalyzedContourMeasure { std::vector<cv::Point> contour; MeasureRow m; bool enabled=true; };
  std::vector<std::vector<AnalyzedContourMeasure>> analyzed_measures_by_frame_;
  struct TrackedContour { int id=-1; std::vector<cv::Point> contour; cv::Point2f centroid{0,0}; };
  std::vector<std::vector<TrackedContour>> tracked_contours_by_frame_;

  struct CalibrationPair {
    int frame_id = -1;
    cv::Mat left;
    cv::Mat right;
  };
  std::vector<CalibrationPair> calib_pairs_;
  std::vector<double> calib_pair_rmse_;
  bool has_computed_calib_ = false;

  // Tracking detect-all overlay cache: source index -> frame index -> visualized frame
  std::unordered_map<int, std::unordered_map<int64_t, cv::Mat>> detect_overlay_cache_;

  QWidget* visualDashHost_=nullptr;
  QGridLayout* visualDashGrid_=nullptr;
  QLabel* leftVisImage_=nullptr;
  QCustomPlot* leftDispPlot_=nullptr;
  QCustomPlot* leftSpeedPlot_=nullptr;
  QCustomPlot* leftAreaPlot_=nullptr;
  QCustomPlot* leftPerimPlot_=nullptr;
  QCustomPlot* leftCircPlot_=nullptr;
  QComboBox* cbDispMetric_=nullptr;
  QComboBox* cbSpeedMetric_=nullptr;
  QComboBox* cbAreaMetric_=nullptr;
  QComboBox* cbPerimMetric_=nullptr;
  QComboBox* cbCircMetric_=nullptr;
  QLabel* lblTargetFilter_=nullptr;
  QComboBox* cbTargetFilter_=nullptr;
  bool updating_left_visual_dashboard_=false;
  bool left_visual_refresh_queued_=false;
  QLabel* lblXAxisMode_=nullptr;
  QComboBox* cbXAxisMode_=nullptr;
  QPushButton* btnCaptureVisual_=nullptr;
  QPushButton* btnExportTableCsv_=nullptr;
  QPushButton* btnExportMp4_=nullptr;
  QPushButton* btnFitAllPlots_=nullptr;
  QPushButton* btnVisualAuto_=nullptr;
  QPushButton* btnToggleTable_=nullptr;
  QPushButton* btnTileLayout_=nullptr;
  QTableWidget* leftMeasureTable_=nullptr;
  int selected_target_id_ = -1;
  QSpinBox* spSmoothMedianWindow_=nullptr;
  QDoubleSpinBox* spSmoothAlphaSpeed_=nullptr;
  QDoubleSpinBox* spSmoothAlphaAccel_=nullptr;
  int selected_target_frame_ = -1;
  bool pending_visual_fit_ = false;
  double binary_preview_zoom_ = 1.0;
  QPointF binary_preview_center_ = QPointF(0.5, 0.5);
  bool binary_preview_dragging_ = false;
  QPoint binary_preview_last_pos_;
  QImage binary_preview_orig_img_;
  QImage binary_preview_mask_img_;
  double visual_image_zoom_ = 1.0;
  QPointF visual_image_center_ = QPointF(0.5, 0.5);
  bool visual_image_dragging_ = false;
  QPoint visual_image_last_pos_;
  bool visual_image_auto_mode_ = true;
  QImage visual_image_base_;
  QTimer binary_preview_play_timer_;

  bool eventFilter(QObject* watched, QEvent* event) override;

  // Timer (UI refresh)
  std::map<QString, std::pair<double,double>> confirmed_hist_rules_;
  bool suppress_histogram_updates_ = false;

  QTimer timer_;
};
