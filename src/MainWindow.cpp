
#include "MainWindow.h"
#include "CaptureWorker.h"
#include "SolveWorker.h"
#include "qcustomplot.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QDockWidget>
#include <QFileDialog>
#include <QInputDialog>
#include <QDateTime>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QScrollArea>
#include <QStackedWidget>
#include <QApplication>
#include <QScreen>
#include <QHeaderView>
#include <QTabBar>
#include <QStyle>
#include <QFrame>
#include <QIntValidator>
#include <QDialog>
#include <QDialogButtonBox>
#include <QCheckBox>
#include <QProgressDialog>
#include <QFormLayout>
#include <QMenu>
#include <QTextDocument>
#include <QRegularExpression>
#include <QTextStream>
#include <functional>
#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>

static QString nowStr() {
  return QDateTime::currentDateTime().toString("hh:mm:ss");
}

static QStringList kImageNameFilters() {
  return {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"};
}

static cv::Mat imreadUnicodePath(const QString& filePath, int flags=cv::IMREAD_COLOR) {
  QFile f(filePath);
  if (!f.open(QIODevice::ReadOnly)) return cv::Mat();
  QByteArray bytes = f.readAll();
  if (bytes.isEmpty()) return cv::Mat();
  std::vector<uchar> buf(bytes.begin(), bytes.end());
  return cv::imdecode(buf, flags);
}

static bool openVideoCaptureUnicode(cv::VideoCapture& cap, const QString& path) {
  cap.release();
  const QString nativePath = QDir::toNativeSeparators(path);
  const QByteArray utf8 = nativePath.toUtf8();
  if (cap.open(utf8.constData())) return true;
  const QByteArray local = QFile::encodeName(nativePath);
  if (cap.open(local.constData())) return true;
  if (cap.open(nativePath.toStdString())) return true;
  return false;
}

static int stepToTabIndex(int step) { return step * 2; }
static int tabIndexToStep(int tabIndex) { return tabIndex / 2; }
static bool isArrowTab(int tabIndex) { return (tabIndex % 2) == 1; }

class ClickJumpSlider : public QSlider {
public:
  explicit ClickJumpSlider(Qt::Orientation orientation, QWidget* parent=nullptr)
      : QSlider(orientation, parent) {}

protected:
  void mousePressEvent(QMouseEvent* event) override {
    if (event->button() == Qt::LeftButton) {
      const int minV = minimum();
      const int maxV = maximum();
      int next = minV;
      if (orientation() == Qt::Horizontal) {
        const int w = width();
        if (w > 1) next = QStyle::sliderValueFromPosition(minV, maxV, event->pos().x(), w - 1);
      } else {
        const int h = height();
        if (h > 1) next = QStyle::sliderValueFromPosition(minV, maxV, h - 1 - event->pos().y(), h - 1);
      }
      setValue(next);
      event->accept();
      return;
    }
    QSlider::mousePressEvent(event);
  }
};

class TitleBarWidget : public QWidget {
public:
  explicit TitleBarWidget(QWidget* parent=nullptr) : QWidget(parent) {}

  std::function<void(const QPoint&)> onDragMove;
  std::function<void()> onToggleMaxRestore;

protected:
  void mousePressEvent(QMouseEvent* event) override {
    if (event->button() == Qt::LeftButton) {
      dragging_ = true;
      dragOffset_ = event->globalPos() - window()->frameGeometry().topLeft();
      event->accept();
      return;
    }
    QWidget::mousePressEvent(event);
  }

  void mouseMoveEvent(QMouseEvent* event) override {
    if (dragging_ && (event->buttons() & Qt::LeftButton) && onDragMove) {
      onDragMove(event->globalPos() - dragOffset_);
      event->accept();
      return;
    }
    QWidget::mouseMoveEvent(event);
  }

  void mouseReleaseEvent(QMouseEvent* event) override {
    dragging_ = false;
    QWidget::mouseReleaseEvent(event);
  }

  void mouseDoubleClickEvent(QMouseEvent* event) override {
    if (event->button() == Qt::LeftButton && onToggleMaxRestore) {
      onToggleMaxRestore();
      event->accept();
      return;
    }
    QWidget::mouseDoubleClickEvent(event);
  }

private:
  bool dragging_ = false;
  QPoint dragOffset_;
};



class ThumbnailLabel : public QLabel {
public:
  explicit ThumbnailLabel(int idx, QWidget* parent=nullptr) : QLabel(parent), index_(idx) {
    setCursor(Qt::PointingHandCursor);
    setSelected(false);
  }
  std::function<void(int)> onClicked;
  std::function<void(int)> onDoubleClick;
  void setSelected(bool on) {
    selected_ = on;
    setStyleSheet(selected_ ? "border:2px solid #22c55e;" : "border:1px solid #4b5563;");
  }
protected:
  void mousePressEvent(QMouseEvent* e) override {
    if (e->button() == Qt::LeftButton && onClicked) {
      onClicked(index_);
      e->accept();
      return;
    }
    QLabel::mousePressEvent(e);
  }
  void mouseDoubleClickEvent(QMouseEvent* e) override {
    if (e->button() == Qt::LeftButton && onDoubleClick) {
      onDoubleClick(index_);
      e->accept();
      return;
    }
    QLabel::mouseDoubleClickEvent(e);
  }
private:
  int index_ = -1;
  bool selected_ = false;
};

class NamedLineItem : public QGraphicsLineItem {
public:
  NamedLineItem(const QLineF& line, const QString& name, const QColor& color, int width, QGraphicsItem* parent=nullptr)
      : QGraphicsLineItem(line, parent), name_(name), color_(color), width_(std::max(1, width)) {
    setFlags(QGraphicsItem::ItemIsSelectable | QGraphicsItem::ItemIsFocusable);
    applyStyle();
    label_ = new QGraphicsTextItem(this);
    label_->setDefaultTextColor(color_);
    label_->setPlainText(name_);
    label_->setTextInteractionFlags(Qt::NoTextInteraction);
    updateLabelPos();
    QObject::connect(label_->document(), &QTextDocument::contentsChanged, [this]() {
      QString t = label_->toPlainText().trimmed();
      if (!t.isEmpty()) name_ = t;

      // Regular expression to match numbers (integer or decimal)
      QRegularExpression re("([-+]?[0-9]*\\.?[0-9]+)");
      auto m = re.match(t);

      if (m.hasMatch() && onValueEdited_) {
        bool ok = false;
        double v = m.captured(1).toDouble(&ok);
        if (ok) {
          onValueEdited_(this->line().length(), v);
        }
      }
    });
  }

  void setMeta(const QString& name, const QColor& color, int width) {
    if (!name.isEmpty()) { name_ = name; label_->setPlainText(name_); }
    color_ = color;
    width_ = std::max(1, width);
    applyStyle();
    label_->setDefaultTextColor(color_);
    updateLabelPos();
  }
  void enableInlineEdit() {
    updateLabelPos();
    label_->setTextInteractionFlags(Qt::TextEditorInteraction);
    label_->setFocus(Qt::MouseFocusReason);
  }
  void finishInlineEdit() { label_->setTextInteractionFlags(Qt::NoTextInteraction); }
  void setValueEditedCallback(const std::function<void(double,double)>& cb) { onValueEdited_ = cb; }
  void updateLabelPos() {
    QPointF mid = (line().p1() + line().p2()) * 0.5;
    label_->setPos(mid + QPointF(4, -18));
  }

private:
  void applyStyle() {
    QPen p(color_, width_);
    if (isSelected()) p.setStyle(Qt::DashLine);
    setPen(p);
  }
  QPainterPath shape() const override {
    QPainterPath p;
    p.moveTo(line().p1());
    p.lineTo(line().p2());
    QPainterPathStroker stroker;
    stroker.setWidth(std::max(10.0, (double)width_ + 10.0));
    return stroker.createStroke(p);
  }

  QVariant itemChange(GraphicsItemChange change, const QVariant &value) override {
    if (change == QGraphicsItem::ItemSelectedHasChanged) applyStyle();
    return QGraphicsLineItem::itemChange(change, value);
  }

  QString name_;
  QColor color_;
  int width_ = 2;
  QGraphicsTextItem* label_ = nullptr;
  std::function<void(double,double)> onValueEdited_;
};

ImageViewer::ImageViewer(QWidget* parent) : QGraphicsView(parent) {
  setScene(&scene_);
  setRenderHint(QPainter::Antialiasing, true);
  setBackgroundBrush(QColor(17,17,17));
  setFrameShape(QFrame::StyledPanel);
  setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
  setResizeAnchor(QGraphicsView::AnchorViewCenter);
  setDragMode(QGraphicsView::ScrollHandDrag);
  pixmapItem_ = scene_.addPixmap(QPixmap());
}

void ImageViewer::setImage(const QImage& img) {
  if (img.isNull()) {
    pixmapItem_->setPixmap(QPixmap());
    return;
  }
  pixmapItem_->setPixmap(QPixmap::fromImage(img));
  scene_.setSceneRect(pixmapItem_->boundingRect());
  if (zoomFactor_ == 1.0) {
    resetTransform();
    fitInView(pixmapItem_, Qt::KeepAspectRatio);
  }
}

void ImageViewer::setToolMode(ToolMode mode) {
  toolMode_ = mode;
  if (toolMode_ == PanTool) {
    setCursor(Qt::OpenHandCursor);
    setDragMode(QGraphicsView::ScrollHandDrag);
  } else {
    setCursor(Qt::CrossCursor);
    setDragMode(QGraphicsView::NoDrag);
  }
  if (!lineDrawing_ && previewLine_) {
    scene_.removeItem(previewLine_);
    delete previewLine_;
    previewLine_ = nullptr;
  }
}

void ImageViewer::zoomIn() { applyZoom(1.15); }
void ImageViewer::zoomOut() { applyZoom(1.0/1.15); }

void ImageViewer::resetView() {
  resetTransform();
  zoomFactor_ = 1.0;
  if (!pixmapItem_->pixmap().isNull()) fitInView(pixmapItem_, Qt::KeepAspectRatio);
}

void ImageViewer::clearAnnotations() {
  const auto items = scene_.items();
  for (auto* it : items) {
    if (it == pixmapItem_) continue;
    scene_.removeItem(it);
    delete it;
  }
  previewLine_ = nullptr;
  lineDrawing_ = false;
}


void ImageViewer::setLineCreatedCallback(const std::function<void(double)>& cb) { onLineCreated_ = cb; }
void ImageViewer::setLineDoubleClickCallback(const std::function<void(double)>& cb) { onLineDoubleClick_ = cb; }
void ImageViewer::setLineValueEditedCallback(const std::function<void(double,double)>& cb) { onLineValueEdited_ = cb; }

void ImageViewer::applySelectedLineStyle(const QString& name, const QColor& color, int width) {
  bool applied = false;
  const auto selected = scene_.selectedItems();
  for (auto* it : selected) {
    if (auto* li = dynamic_cast<NamedLineItem*>(it)) {
      li->setMeta(name, color, width);
      applied = true;
    }
  }
  if (applied) return;
  // Single-line workflow fallback: apply to the first line if none is explicitly selected.
  const auto items = scene_.items();
  for (auto* it : items) {
    if (auto* li = dynamic_cast<NamedLineItem*>(it)) {
      li->setMeta(name, color, width);
      break;
    }
  }
}

void ImageViewer::setAnnotationsVisible(bool visible) {
  const auto items = scene_.items();
  for (auto* it : items) {
    if (it == pixmapItem_) continue;
    it->setVisible(visible);
  }
}

void ImageViewer::clearAllLines() {
  const auto items = scene_.items();
  for (auto* it : items) {
    if (it == pixmapItem_ || it == previewLine_) continue;
    if (dynamic_cast<QGraphicsLineItem*>(it)) {
      scene_.removeItem(it);
      delete it;
    }
  }
}

double ImageViewer::selectedLineLength() const {
  const auto items = scene_.selectedItems();
  for (auto* it : items) {
    if (auto* li = dynamic_cast<QGraphicsLineItem*>(it)) return li->line().length();
  }
  return 0.0;
}

double ImageViewer::anyLineLength() const {
  const auto items = scene_.items();
  for (auto* it : items) {
    if (auto* li = dynamic_cast<QGraphicsLineItem*>(it)) return li->line().length();
  }
  return 0.0;
}

void ImageViewer::applyZoom(double factor) {
  zoomFactor_ *= factor;
  zoomFactor_ = std::max(0.1, std::min(zoomFactor_, 20.0));
  scale(factor, factor);
}

void ImageViewer::wheelEvent(QWheelEvent* e) {
  if (e->angleDelta().y() > 0) zoomIn();
  else zoomOut();
}

void ImageViewer::mousePressEvent(QMouseEvent* e) {
  if (toolMode_ == PanTool) {
    QGraphicsView::mousePressEvent(e);
    return;
  }
  if (e->button() != Qt::LeftButton || pixmapItem_->pixmap().isNull()) return;

  QPointF p = mapToScene(e->pos());
  if (!sceneRect().contains(p)) return;

  if (toolMode_ == PointTool) {
    scene_.addEllipse(p.x()-3, p.y()-3, 6, 6, QPen(QColor(255,80,80), 2), QBrush(QColor(255,80,80)));
    setToolMode(PanTool);
    return;
  }

  if (toolMode_ == LineTool) {
    if (!lineDrawing_) {
      clearAllLines(); // keep only one scale line
      lineDrawing_ = true;
      lineStart_ = p;
      previewLine_ = scene_.addLine(QLineF(lineStart_, lineStart_), QPen(QColor(80,220,255), 2));
    } else {
      auto* finalLine = new NamedLineItem(QLineF(lineStart_, p), "Line", QColor(80,220,255), 2);
      finalLine->setValueEditedCallback(onLineValueEdited_);
      scene_.addItem(finalLine);
      if (onLineCreated_) onLineCreated_(finalLine->line().length());
      if (previewLine_) {
        scene_.removeItem(previewLine_);
        delete previewLine_;
        previewLine_ = nullptr;
      }
      lineDrawing_ = false;
      setToolMode(PanTool);
    }
  }
}

void ImageViewer::mouseMoveEvent(QMouseEvent* e) {
  if (toolMode_ == LineTool && lineDrawing_ && previewLine_) {
    QPointF p = mapToScene(e->pos());
    previewLine_->setLine(QLineF(lineStart_, p));
    return;
  }
  QGraphicsView::mouseMoveEvent(e);
}


void ImageViewer::mouseDoubleClickEvent(QMouseEvent* e) {
  if (e->button() != Qt::LeftButton) {
    QGraphicsView::mouseDoubleClickEvent(e);
    return;
  }
  QPointF p = mapToScene(e->pos());
  if (QGraphicsItem* it = scene_.itemAt(p, transform())) {
    if (it != pixmapItem_) {
      if (auto* li = dynamic_cast<NamedLineItem*>(it)) {
        li->setSelected(true);
        li->enableInlineEdit();
        if (onLineDoubleClick_) onLineDoubleClick_(li->line().length());
        e->accept();
        return;
      }
      if (auto* txt = dynamic_cast<QGraphicsTextItem*>(it)) {
        if (auto* parentLine = dynamic_cast<NamedLineItem*>(txt->parentItem())) {
          parentLine->setSelected(true);
          parentLine->enableInlineEdit();
          e->accept();
          return;
        }
      }
    }
  }
  QGraphicsView::mouseDoubleClickEvent(e);
}

void ImageViewer::resizeEvent(QResizeEvent* e) {
  QGraphicsView::resizeEvent(e);
  if (zoomFactor_ == 1.0 && pixmapItem_ && !pixmapItem_->pixmap().isNull()) {
    resetTransform();
    fitInView(pixmapItem_, Qt::KeepAspectRatio);
  }
}

MainWindow::MainWindow(const std::vector<InputSource>& sources,
                       int board_w, int board_h, double square_m,
                       QWidget* parent)
  : QMainWindow(parent),
    sources_(sources),
    num_cams_((int)sources.size()),
    board_w_(board_w),
    board_h_(board_h),
    square_(square_m),
    settings_("YourCompany", "Multi6DTracker")
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::Window);
    setWindowTitle("MonoMeasure");
    if (QScreen* screen = QGuiApplication::primaryScreen()) {
      setGeometry(screen->availableGeometry());
    } else {
      resize(1280, 800);
    }

    calibrator_.reset(new MultiCamCalibrator(std::max(1,num_cams_), cv::Size(board_w_, board_h_), square_));

    source_enabled_.assign(std::max(0,num_cams_), true);
    last_frames_.resize(std::max(0,num_cams_));
    buildUI();
    connect(&timer_, &QTimer::timeout, this, &MainWindow::onTick);
    timer_.start(33);

    // Start capture/solve threads
    captureWorker_ = new CaptureWorker(&sources_, &source_enabled_, &last_frames_, &sources_mutex_, 33);
    captureWorker_->moveToThread(&captureThread_);
    connect(&captureThread_, &QThread::started, captureWorker_, &CaptureWorker::start);
    connect(this, &MainWindow::destroyed, captureWorker_, &CaptureWorker::stop);

    // 单目测量版本：移除测量求解线程，仅保留采集与预处理。

    // Wire signals
    connect(captureWorker_, &CaptureWorker::framesReady, this, &MainWindow::onFramesFromWorker, Qt::QueuedConnection);

    captureThread_.start();
}

MainWindow::~MainWindow() 
{
    timer_.stop();
    // Stop workers/threads
    if (captureWorker_) 
        captureWorker_->stop();
    captureThread_.quit();
    captureThread_.wait();
    solveThread_.quit();
    solveThread_.wait();
    delete solveWorker_;
    solveWorker_ = nullptr;
    delete captureWorker_;
    captureWorker_ = nullptr;

    for (auto& s : sources_) 
    {
        if (s.cap.isOpened()) 
            s.cap.release();
    }
}

void MainWindow::buildUI() {
    const QScreen* screen = QGuiApplication::primaryScreen();
    const qreal dpiScaleRaw = screen ? (screen->logicalDotsPerInch() / 96.0) : 1.0;
    const qreal dpiScale = std::clamp(dpiScaleRaw, 0.90, 1.40);
    const qreal geomScale = screen
        ? std::clamp(std::sqrt((qreal)screen->availableGeometry().width() * (qreal)screen->availableGeometry().height()
                               / (1920.0 * 1080.0)), 0.90, 1.20)
        : 1.0;
    const qreal uiScale = std::clamp(dpiScale * geomScale, 0.90, 1.60);

    // Font scaling is intentionally gentler than control scaling:
    // at 1920x1080 and 100% DPI this yields ~12pt.
    const qreal fontScale = std::clamp(0.75 + 0.20 * dpiScale + 0.05 * geomScale, 0.95, 1.20);
    QFont baseFont = font();
    baseFont.setPointSizeF(12.0 * fontScale);
    setFont(baseFont);

    // Top title bar spans the full QMainWindow width (including dock area)
    TitleBarWidget* titleBar = new TitleBarWidget(this);
    titleBar->setObjectName("customTitleBar");
    titleBar->setFixedHeight((int)std::round(34 * uiScale));
    QHBoxLayout* titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins((int)std::round(10 * uiScale), 0, 0, 0);
    titleLayout->setSpacing((int)std::round(6 * uiScale));

    QLabel* titleText = new QLabel("MonoMeasure", titleBar);
    titleText->setStyleSheet("color:#c7d2df;font-weight:600;");
    titleLayout->addWidget(titleText);
    titleLayout->addStretch(1);

    auto makeTitleBtn = [titleBar, uiScale](const QString& text, const QString& objName) {
      QToolButton* b = new QToolButton(titleBar);
      b->setObjectName(objName);
      b->setText(text);
      b->setFixedSize((int)std::round(46 * uiScale), (int)std::round(34 * uiScale));
      return b;
    };

    btnFileMenu_ = new QToolButton(titleBar);
    btnFileMenu_->setObjectName("fileMenuBtn");
    btnFileMenu_->setText("File");
    btnFileMenu_->setPopupMode(QToolButton::InstantPopup);
    btnFileMenu_->setToolButtonStyle(Qt::ToolButtonTextOnly);
    btnFileMenu_->setMinimumWidth((int)std::round(70 * uiScale));
    btnFileMenu_->setToolTip("Software info, Save Project, Load Project");

    QToolButton* btnMin = makeTitleBtn("-", "titleMinBtn");
    QToolButton* btnMax = makeTitleBtn("[]", "titleMaxBtn");
    QToolButton* btnClose = makeTitleBtn("X", "titleCloseBtn");
    titleLayout->addWidget(btnFileMenu_);
    titleLayout->addSpacing((int)std::round(4 * uiScale));
    titleLayout->addWidget(btnMin);
    titleLayout->addWidget(btnMax);
    titleLayout->addWidget(btnClose);

    titleBar->setStyleSheet(
      "#customTitleBar{background:#1f232b;border-bottom:1px solid #3a4250;}"
      "QToolButton{background:#2b3340;color:#cfd8e3;border:none;border-left:1px solid #4b586d;border-radius:0;font-size:12px;}#fileMenuBtn{border:1px solid #4b586d;border-radius:4px;padding:0 10px;}"
      "QToolButton:hover{background:#374255;}"
      "QToolButton#titleCloseBtn:hover{background:#b42318;color:#ffffff;}");

    connect(btnMin, &QToolButton::clicked, this, &QWidget::showMinimized);
    connect(btnClose, &QToolButton::clicked, this, &QWidget::close);
    connect(btnMax, &QToolButton::clicked, this, [this]() {
      isMaximized() ? showNormal() : showMaximized();
    });
    titleBar->onToggleMaxRestore = [this]() {
      isMaximized() ? showNormal() : showMaximized();
    };
    titleBar->onDragMove = [this](const QPoint& p) {
      if (!isMaximized()) move(p);
    };
    setMenuWidget(titleBar);

    QWidget* central = new QWidget(this);
    QHBoxLayout* root = new QHBoxLayout();
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    QWidget* sideBar = new QWidget(central);
    sideBar->setObjectName("leftSidebar");
    sideBar->setFixedWidth((int)std::round(72 * uiScale));
    QVBoxLayout* sv = new QVBoxLayout(sideBar);
    sv->setContentsMargins((int)std::round(6 * uiScale), (int)std::round(8 * uiScale), (int)std::round(6 * uiScale), (int)std::round(8 * uiScale));
    sv->setSpacing((int)std::round(10 * uiScale));

    QLabel* appTitle = new QLabel("MonoMeasure", sideBar);
    appTitle->setObjectName("sidebarTitle");
    appTitle->setWordWrap(true);
    appTitle->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    sv->addWidget(appTitle);

    QFrame* titleSep = new QFrame(sideBar);
    titleSep->setFrameShape(QFrame::HLine);
    titleSep->setFrameShadow(QFrame::Plain);
    sv->addWidget(titleSep);

    // 左侧步骤TAB已删除，改为顶部步骤引导。
    sv->addStretch(1);

    QFrame* bottomSep = new QFrame(sideBar);
    bottomSep->setFrameShape(QFrame::HLine);
    bottomSep->setFrameShadow(QFrame::Plain);
    sv->addWidget(bottomSep);
    sideBar->setStyleSheet(
      "#leftSidebar{background:#1f232b;border-right:1px solid #4a5568;}"
      "#sidebarTitle{font-size:11px;font-weight:600;color:#b8c4d6;padding:4px 4px;line-height:1.1;}"
      "QToolButton{background:#2b3340;border:1px solid #4b586d;border-radius:6px;padding:6px 6px;color:#eef2f8;text-align:left;}"
      "QToolButton::menu-indicator{subcontrol-position:right center;right:8px;}"
      "QFrame{color:#394150;background:#394150;}");

    statusBar()->setStyleSheet("QStatusBar{border-top:1px solid #3a4250;}QStatusBar::item{border:none;}");

    QWidget* mainPane = new QWidget(central);
    QVBoxLayout* v = new QVBoxLayout(mainPane);
    v->setSpacing(10);

    // Top step guide
    stepTabs_ = new QTabBar(central);
    stepTabs_->addTab("1 Source");
    stepTabs_->addTab("");
    stepTabs_->addTab("2 PreProcess");
    stepTabs_->addTab("");
    stepTabs_->addTab("3 ObjectDefine");
    stepTabs_->addTab("");
    stepTabs_->addTab("4 Visual");
    stepTabs_->setExpanding(false);
    stepTabs_->setDocumentMode(true);
    stepTabs_->setCurrentIndex(stepToTabIndex(0));
    for (int i=0;i<4;++i) stepDone_[i] = false;
    stepTabs_->setStyleSheet(
      "QTabBar::tab{padding:4px 10px;margin-right:8px;min-width:112px;background:#2d333b;color:#dbe5f1;border:1px solid #485468;border-radius:6px;}"
      "QTabBar::tab:selected{background:#3b82f6;color:#ffffff;font-weight:700;}"
      "QTabBar::tab:hover:!selected{background:#374151;}"
      "QTabBar::tab:disabled{background:#1f232b;color:#6b7280;border:1px solid #394150;}"
      "QTabBar::tab:nth-child(2),QTabBar::tab:nth-child(4),QTabBar::tab:nth-child(6){background:transparent;border:none;color:transparent;padding:0px;margin:0 6px;min-width:30px;}"
      "QTabBar::tab:nth-child(2):disabled,QTabBar::tab:nth-child(4):disabled,QTabBar::tab:nth-child(6):disabled{background:transparent;border:none;color:transparent;}");
    auto mkArrowBtn = [this]() {
      auto* b = new QToolButton(stepTabs_);
      b->setIcon(style()->standardIcon(QStyle::SP_ArrowRight));
      b->setIconSize(QSize(24, 24));
      b->setFixedSize(QSize(28, 28));
      b->setAutoRaise(true);
      b->setToolButtonStyle(Qt::ToolButtonIconOnly);
      b->setEnabled(false);
      b->setStyleSheet("QToolButton{border:none;background:transparent;padding:0px;margin:0px;}QToolButton:disabled{border:none;background:transparent;}");
      return b;
    };
    stepTabs_->setTabButton(1, QTabBar::LeftSide, mkArrowBtn());
    stepTabs_->setTabButton(3, QTabBar::LeftSide, mkArrowBtn());
    stepTabs_->setTabButton(5, QTabBar::LeftSide, mkArrowBtn());
    stepTabs_->setTabEnabled(1, false);
    stepTabs_->setTabEnabled(3, false);
    stepTabs_->setTabEnabled(5, false);
    v->addWidget(stepTabs_);

    QHBoxLayout* topSourceBar = new QHBoxLayout();
    btnAddCam_ = new QToolButton(central);
    btnAddVideo_ = new QToolButton(central);
    btnAddImgSeq_ = new QToolButton(central);
    btnRemoveSource_ = new QToolButton(central);
    btnAddCam_->setText("AddCamera");
    btnAddVideo_->setText("AddVideo");
    btnAddImgSeq_->setText("AddImageSeq");
    btnRemoveSource_->setText("Remove");
    btnAddCam_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    btnAddVideo_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    btnAddImgSeq_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    btnRemoveSource_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    btnAddCam_->setIcon(style()->standardIcon(QStyle::SP_ComputerIcon));
    btnAddVideo_->setIcon(style()->standardIcon(QStyle::SP_FileIcon));
    btnAddImgSeq_->setIcon(style()->standardIcon(QStyle::SP_DirIcon));
    btnRemoveSource_->setIcon(style()->standardIcon(QStyle::SP_TrashIcon));
    topSourceBar->addStretch(1);
    v->addLayout(topSourceBar);

    QFrame* topSep = new QFrame(central);
    topSep->setFrameShape(QFrame::HLine);
    topSep->setFrameShadow(QFrame::Sunken);
    v->addWidget(topSep);

    btnPlayAll_ = new QToolButton(central);
    btnStopAll_ = new QToolButton(central);
    btnStepPrev_ = new QToolButton(central);
    btnStepNext_ = new QToolButton(central);

    btnPlayAll_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    btnPlayAll_->setCheckable(true);
    btnStopAll_->setIcon(style()->standardIcon(QStyle::SP_MediaStop));
    btnStepPrev_->setIcon(style()->standardIcon(QStyle::SP_MediaSkipBackward));
    btnStepNext_->setIcon(style()->standardIcon(QStyle::SP_MediaSkipForward));
    btnPlayAll_->setToolTip("Play/Pause");
    btnStopAll_->setToolTip("Stop");
    btnStepPrev_->setToolTip("Prev Frame");
    btnStepNext_->setToolTip("Next Frame");

    // Per-view toolbars are created in rebuildSourceViews(), one toolbar per player.

    auto* leftStack = new QStackedWidget(central);

    viewsHost_ = new QWidget(this);
    viewsGrid_ = new QGridLayout(viewsHost_);
    viewsGrid_->setContentsMargins(0,0,0,0);
    viewsGrid_->setSpacing(8);
    viewsHost_->setMinimumSize((int)std::round(960 * uiScale), (int)std::round(540 * uiScale));
    rebuildSourceViews();

    visualDashHost_ = new QWidget(this);
    visualDashGrid_ = new QGridLayout(visualDashHost_);
    visualDashGrid_->setContentsMargins(2,2,2,2);
    visualDashGrid_->setSpacing(4);
    leftVisImage_ = new QLabel(visualDashHost_);
    const int visH = (int)std::round(240 * uiScale);
    leftVisImage_->setMinimumSize((int)std::round(320 * uiScale), visH);
    leftVisImage_->setMaximumHeight(visH);
    leftVisImage_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    leftVisImage_->setAlignment(Qt::AlignCenter);
    leftVisImage_->setStyleSheet("background:#111827;border:1px solid #374151;color:#cbd5e1;");
    leftVisImage_->setText("Image view");
    auto mkPlot=[&](const QString& y){ QCustomPlot* p=new QCustomPlot(visualDashHost_); p->setMinimumHeight(visH); p->setMaximumHeight(visH); p->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed); p->setBackground(QBrush(QColor(17,24,39))); p->xAxis->setBasePen(QPen(QColor(148,163,184))); p->yAxis->setBasePen(QPen(QColor(148,163,184))); p->xAxis->setTickPen(QPen(QColor(148,163,184))); p->yAxis->setTickPen(QPen(QColor(148,163,184))); p->xAxis->setSubTickPen(QPen(QColor(100,116,139))); p->yAxis->setSubTickPen(QPen(QColor(100,116,139))); p->xAxis->setTickLabelColor(QColor(226,232,240)); p->yAxis->setTickLabelColor(QColor(226,232,240)); p->xAxis->setLabelColor(QColor(226,232,240)); p->yAxis->setLabelColor(QColor(226,232,240)); p->xAxis->setUpperEnding(QCPLineEnding::esSpikeArrow); p->yAxis->setUpperEnding(QCPLineEnding::esSpikeArrow); p->addGraph(); p->graph(0)->setPen(QPen(QColor(96,165,250), 2.0)); p->addGraph(); p->graph(1)->setPen(QPen(QColor(239,68,68), 1.6)); p->xAxis->setLabel("frame"); p->yAxis->setLabel(y); return p; };
    QStringList metricItems = {"Displacement","Speed","Acceleration","Area","Perimeter","Major Axis","Minor Axis","Circularity"};
    auto mkPlotCard=[&](QCustomPlot* p, QComboBox*& cb, const QString& shotName){
      QWidget* card = new QWidget(visualDashHost_);
      QVBoxLayout* cv = new QVBoxLayout(card); cv->setContentsMargins(0,0,0,0); cv->setSpacing(2);
      QHBoxLayout* top = new QHBoxLayout(); top->setContentsMargins(0,0,0,0);
      cb = new QComboBox(card); cb->addItems(metricItems); cb->setMinimumWidth((int)std::round(140*uiScale));
      QPushButton* btnShot = new QPushButton("Shot", card);
      connect(cb, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ updateLeftVisualDashboard(); });
      connect(btnShot, &QPushButton::clicked, this, [this, p, shotName](){ savePlotAsBmp(p, shotName); });
      top->addWidget(cb, 1); top->addWidget(btnShot, 0);
      cv->addLayout(top); cv->addWidget(p, 1);
      return card;
    };
    leftDispPlot_=mkPlot("Displacement");
    leftSpeedPlot_=mkPlot("Speed");
    leftAreaPlot_=mkPlot("Area");
    leftPerimPlot_=mkPlot("Perimeter");
    leftCircPlot_=mkPlot("Circularity");
    visualDashGrid_->addWidget(leftVisImage_,0,0);
    visualDashGrid_->addWidget(mkPlotCard(leftDispPlot_, cbDispMetric_, "disp"),0,1);
    visualDashGrid_->addWidget(mkPlotCard(leftSpeedPlot_, cbSpeedMetric_, "speed"),0,2);
    visualDashGrid_->addWidget(mkPlotCard(leftAreaPlot_, cbAreaMetric_, "area"),1,0);
    visualDashGrid_->addWidget(mkPlotCard(leftPerimPlot_, cbPerimMetric_, "perimeter"),1,1);
    visualDashGrid_->addWidget(mkPlotCard(leftCircPlot_, cbCircMetric_, "circularity"),1,2);
    if (cbDispMetric_) cbDispMetric_->setCurrentIndex(0);
    if (cbSpeedMetric_) cbSpeedMetric_->setCurrentIndex(1);
    if (cbAreaMetric_) cbAreaMetric_->setCurrentIndex(3);
    if (cbPerimMetric_) cbPerimMetric_->setCurrentIndex(4);
    if (cbCircMetric_) cbCircMetric_->setCurrentIndex(7);
    leftMeasureTable_ = new QTableWidget(visualDashHost_);
    leftMeasureTable_->setColumnCount(9);
    leftMeasureTable_->setHorizontalHeaderLabels({"Disp","Speed","Accel","Area","Perimeter","Major","Minor","Circularity","Scale(mm/px)"});
    leftMeasureTable_->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    leftMeasureTable_->setSelectionBehavior(QAbstractItemView::SelectRows);
    leftMeasureTable_->setSelectionMode(QAbstractItemView::SingleSelection);
    leftMeasureTable_->setMaximumHeight((int)std::round(220 * uiScale));
    leftMeasureTable_->setStyleSheet("QTableWidget{background:#1f2937;color:#e5e7eb;gridline-color:#4b5563;selection-background-color:#2563eb;selection-color:#ffffff;}QHeaderView::section{background:#334155;color:#f8fafc;border:1px solid #475569;padding:5px;font-weight:600;}");
    QFrame* visSep = new QFrame(visualDashHost_);
    visSep->setFrameShape(QFrame::HLine);
    visSep->setFrameShadow(QFrame::Plain);
    visSep->setStyleSheet("color:#475569;background:#475569;");
    QHBoxLayout* visActions = new QHBoxLayout();
    btnCaptureVisual_ = new QPushButton("Capture Window", visualDashHost_);
    btnExportTableCsv_ = new QPushButton("Export CSV", visualDashHost_);
    btnExportMp4_ = new QPushButton("Export MP4", visualDashHost_);
    connect(btnCaptureVisual_, &QPushButton::clicked, this, &MainWindow::onCaptureVisualSnapshot);
    connect(btnExportTableCsv_, &QPushButton::clicked, this, &MainWindow::onExportTableCsv);
    connect(btnExportMp4_, &QPushButton::clicked, this, &MainWindow::onExportVisualMp4);
    visActions->addWidget(btnCaptureVisual_);
    visActions->addWidget(btnExportTableCsv_);
    visActions->addWidget(btnExportMp4_);
    visActions->addStretch(1);
    QWidget* visActionHost = new QWidget(visualDashHost_);
    visActionHost->setLayout(visActions);
    visualDashGrid_->addWidget(visActionHost,2,0,1,3);
    visualDashGrid_->addWidget(visSep,3,0,1,3);
    visualDashGrid_->addWidget(leftMeasureTable_,4,0,1,3);

    leftStack->addWidget(viewsHost_);
    leftStack->addWidget(visualDashHost_);
    leftStack->setCurrentWidget(viewsHost_);
    v->addWidget(leftStack, 1);

    QHBoxLayout* playProgressRow = new QHBoxLayout();
    playProgressRow->addWidget(btnPlayAll_);
    playProgressRow->addWidget(btnStopAll_);
    playProgressRow->addWidget(btnStepPrev_);
    playProgressRow->addWidget(btnStepNext_);
    playProgressRow->addSpacing(10);
    progressSlider_ = new ClickJumpSlider(Qt::Horizontal, central);
    progressSlider_->setRange(0, 0);
    progressSlider_->setSingleStep(1);
    progressSlider_->setPageStep(30);
    playProgressRow->addWidget(progressSlider_, 1);
    editCurFrame_ = new QLineEdit("0", central);
    editCurFrame_->setFixedWidth(70);
    editCurFrame_->setAlignment(Qt::AlignCenter);
    editCurFrame_->setValidator(new QIntValidator(0, 9999999, editCurFrame_));
    lblTotalFrame_ = new QLabel("/ 0", central);
    playProgressRow->addWidget(editCurFrame_);
    playProgressRow->addWidget(lblTotalFrame_);
    v->addLayout(playProgressRow);

    root->addWidget(mainPane, 1);
    central->setLayout(root);
    setCentralWidget(central);
    //menuBar()->hide();

    connect(btnAddCam_, &QToolButton::clicked, this, &MainWindow::onAddCamera);
    connect(btnAddVideo_, &QToolButton::clicked, this, &MainWindow::onAddVideo);
    connect(btnAddImgSeq_, &QToolButton::clicked, this, &MainWindow::onAddImageSequence);
    connect(btnRemoveSource_, &QToolButton::clicked, this, &MainWindow::onRemoveSource);
    connect(btnPlayAll_, &QToolButton::clicked, this, &MainWindow::onPlayAll);
    connect(btnStopAll_, &QToolButton::clicked, this, &MainWindow::onStopAll);
    connect(btnStepPrev_, &QToolButton::clicked, this, &MainWindow::onStepPrevFrame);
    connect(btnStepNext_, &QToolButton::clicked, this, &MainWindow::onStepNextFrame);
    connect(progressSlider_, &QSlider::sliderReleased, this, &MainWindow::onProgressSliderReleased);
    connect(progressSlider_, &QSlider::valueChanged, this, [this](int) {
      if (!progressSlider_ || progressSlider_->isSliderDown()) return;
      onProgressSliderReleased();
    });
    connect(editCurFrame_, &QLineEdit::returnPressed, this, &MainWindow::onFrameJumpReturnPressed);
    QMenu* fileMenu = new QMenu(btnFileMenu_);
    QAction* actAbout = fileMenu->addAction("Software Info");
    fileMenu->addSeparator();
    actSaveProject_ = fileMenu->addAction("Save Project...");
    actLoadProject_ = fileMenu->addAction("Load Project...");
    btnFileMenu_->setMenu(fileMenu);
    connect(actAbout, &QAction::triggered, this, [this](){
      QMessageBox::information(this, "Software Info",
                               "MonoMeasure\n\nCapture / Preprocess / Tracking workflow.");
    });
    connect(actSaveProject_, &QAction::triggered, this, &MainWindow::onSaveProject);
    connect(actLoadProject_, &QAction::triggered, this, &MainWindow::onLoadProject);

    // Right actions panel (embedded in central layout to keep top bar full-width)
    QWidget* dock = new QWidget(central);
    dock->setObjectName("actionsPanel");
    dock->setMinimumWidth(320);
    dock->setMaximumWidth(420);

    QWidget* dockw = new QWidget(dock);
    QVBoxLayout* dv = new QVBoxLayout(dockw);

    actionTabs_ = new QTabWidget(dockw);

    // Capture tab
    QWidget* tabCap = new QWidget(actionTabs_);
    QVBoxLayout* capv = new QVBoxLayout(tabCap);
    capv->addWidget(new QLabel("Live capture source management", tabCap));
    btnCaptureNow_ = new QPushButton("Capture", tabCap);
    capv->addWidget(btnAddCam_);
    capv->addWidget(btnAddVideo_);
    capv->addWidget(btnAddImgSeq_);
    capv->addWidget(btnRemoveSource_);
    capv->addWidget(btnCaptureNow_);
    capv->addStretch(1);
    connect(btnCaptureNow_, &QPushButton::clicked, this, &MainWindow::onCaptureNow);

    // Preprocess tab
    QWidget* tabCal = new QWidget(actionTabs_);
    QVBoxLayout* calv = new QVBoxLayout(tabCal);

    calv->addWidget(new QLabel("Step 1: Channel", tabCal));
    QGroupBox* gbColor = new QGroupBox("Color", tabCal);
    QVBoxLayout* colorLayout = new QVBoxLayout(gbColor);
    cbPreColor_ = new QComboBox(gbColor);
    cbPreColor_->addItem("Black & White");
    cbPreColor_->addItem("Color");
    cbPreColor_->setCurrentIndex(1);
    colorLayout->addWidget(cbPreColor_);
    gbColor->setLayout(colorLayout);

    calv->addWidget(new QLabel("Step 2: B&C", tabCal));
    QGroupBox* gbBC = new QGroupBox("Brightness / Contrast", tabCal);
    QGridLayout* bcLayout = new QGridLayout(gbBC);
    QLabel* lblB = new QLabel("Brightness", gbBC);
    QLabel* lblC = new QLabel("Contrast", gbBC);
    slBrightness_ = new QSlider(Qt::Horizontal, gbBC);
    slContrast_ = new QSlider(Qt::Horizontal, gbBC);
    spBrightness_ = new QSpinBox(gbBC);
    spContrast_ = new QSpinBox(gbBC);

    slBrightness_->setRange(0, 255);
    slContrast_->setRange(0, 255);
    spBrightness_->setRange(0, 255);
    spContrast_->setRange(0, 255);

    slBrightness_->setValue(128);
    slBrightness_->setMinimumWidth(220);
    slContrast_->setValue(128);
    slContrast_->setMinimumWidth(220);
    spBrightness_->setValue(128);
    spContrast_->setValue(128);

    bcLayout->addWidget(lblB, 0, 0);
    bcLayout->addWidget(slBrightness_, 0, 1);
    bcLayout->addWidget(spBrightness_, 0, 2);
    bcLayout->addWidget(lblC, 1, 0);
    bcLayout->addWidget(slContrast_, 1, 1);
    bcLayout->addWidget(spContrast_, 1, 2);
    gbBC->setLayout(bcLayout);

    btnPreAuto_ = new QPushButton("Auto", tabCal);
    lblPreprocessHint_ = new QLabel("Apply color mode + brightness/contrast to the live view.", tabCal);
    lblCaptured_ = new QLabel("Captured: 0", tabCal);
    lblScaleInfo_ = new QLabel("Scale: not calibrated", tabCal);

    calv->addWidget(gbColor);
    calv->addWidget(gbBC);
    calv->addWidget(btnPreAuto_);
    calv->addWidget(lblPreprocessHint_);

    calv->addWidget(new QLabel("Step 3: Scale Calibration", tabCal));
    QHBoxLayout* scaleCtl = new QHBoxLayout();
    btnStartScaleLine_ = new QPushButton("Draw Scale Line", tabCal);
    btnDeleteScaleLine_ = new QPushButton("Delete Scale Line", tabCal);
    chkShowLines_ = new QCheckBox("Show Lines", tabCal);
    chkShowLines_->setChecked(false);
    scaleCtl->addWidget(btnStartScaleLine_);
    scaleCtl->addWidget(btnDeleteScaleLine_);
    scaleCtl->addWidget(chkShowLines_);
    scaleCtl->addStretch(1);
    calv->addLayout(scaleCtl);

    gbLineProps_ = new QGroupBox("Line Properties", tabCal);
    QGridLayout* lp = new QGridLayout(gbLineProps_);
    cbLineColor_ = new QComboBox(gbLineProps_);
    cbLineColor_->addItem("Cyan", QColor(80,220,255));
    cbLineColor_->addItem("Red", QColor(255,80,80));
    cbLineColor_->addItem("Green", QColor(80,255,120));
    cbLineColor_->addItem("Yellow", QColor(255,220,80));
    spLineWidth_ = new QSpinBox(gbLineProps_);
    spLineWidth_->setRange(1, 12);
    spLineWidth_->setValue(2);
    editPhysicalMm_ = new QLineEdit("100.0", gbLineProps_);
    btnCalcScale_ = new QPushButton("Calculate", gbLineProps_);
    lp->addWidget(new QLabel("Color", gbLineProps_), 0, 0);
    lp->addWidget(cbLineColor_, 0, 1);
    lp->addWidget(new QLabel("Width", gbLineProps_), 1, 0);
    lp->addWidget(spLineWidth_, 1, 1);
    lp->addWidget(new QLabel("Physical distance (mm)", gbLineProps_), 2, 0);
    lp->addWidget(editPhysicalMm_, 2, 1);
    lp->addWidget(btnCalcScale_, 3, 0, 1, 2);
    gbLineProps_->setLayout(lp);
    gbLineProps_->setVisible(false);

    calv->addWidget(gbLineProps_);
    calv->addWidget(lblScaleInfo_);
    calv->addWidget(lblCaptured_);
    calv->addStretch(1);

    // If board params change, rebuild calibrator and reset captures

    connect(cbPreColor_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onPreprocessParamsChanged);
    connect(slBrightness_, &QSlider::valueChanged, this, [this](int v){ if (spBrightness_ && spBrightness_->value()!=v) spBrightness_->setValue(v); onPreprocessParamsChanged(); });
    connect(slContrast_, &QSlider::valueChanged, this, [this](int v){ if (spContrast_ && spContrast_->value()!=v) spContrast_->setValue(v); onPreprocessParamsChanged(); });
    connect(spBrightness_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v){ if (slBrightness_ && slBrightness_->value()!=v) slBrightness_->setValue(v); onPreprocessParamsChanged(); });
    connect(spContrast_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v){ if (slContrast_ && slContrast_->value()!=v) slContrast_->setValue(v); onPreprocessParamsChanged(); });
    connect(btnPreAuto_, &QPushButton::clicked, this, &MainWindow::onPreprocessAuto);
    connect(btnStartScaleLine_, &QPushButton::clicked, this, &MainWindow::onStartScaleLine);
    connect(btnDeleteScaleLine_, &QPushButton::clicked, this, &MainWindow::onDeleteScaleLine);
    connect(btnCalcScale_, &QPushButton::clicked, this, &MainWindow::onApplyScaleFromInput);
    connect(cbLineColor_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ QColor c = cbLineColor_->currentData().value<QColor>(); int w = spLineWidth_?spLineWidth_->value():2; for (auto* v: sourceViews_) if(v) v->applySelectedLineStyle("Line", c, w); });
    connect(spLineWidth_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int){ QColor c = cbLineColor_?cbLineColor_->currentData().value<QColor>():QColor(80,220,255); int w = spLineWidth_?spLineWidth_->value():2; for (auto* v: sourceViews_) if(v) v->applySelectedLineStyle("Line", c, w); });
    connect(chkShowLines_, &QCheckBox::toggled, this, [this](bool on){ for (auto* v: sourceViews_) if (v) v->setAnnotationsVisible(on); });

    // ObjectDefine + Visual pages
    QWidget* tabObj = new QWidget(actionTabs_);
    QVBoxLayout* trkMainLayout = new QVBoxLayout(tabObj);
    QWidget* tabVis = new QWidget(actionTabs_);
    QVBoxLayout* trkVisRoot = new QVBoxLayout(tabVis);
    QScrollArea* visScrollArea = new QScrollArea(tabVis);
    visScrollArea->setWidgetResizable(true);
    visScrollArea->setFrameShape(QFrame::NoFrame);
    QWidget* visContainer = new QWidget(visScrollArea);
    visChartsLayout_ = new QVBoxLayout(visContainer);
    visScrollArea->setWidget(visContainer);
    trkVisRoot->addWidget(visScrollArea);

    QGroupBox* gbThresh = new QGroupBox("Threshold", tabObj);
    QGridLayout* tg = new QGridLayout(gbThresh);

    cbThreshType_ = new QComboBox(gbThresh);
    cbThreshType_->addItem("Auto Threshold (Global)");
    cbThreshType_->addItem("Auto Local Threshold");

    cbGlobalMethod_ = new QComboBox(gbThresh);
    for (const QString& m : {"Default","Huang","Intermodes","IsoData","Li","MaxEntropy","Mean","MinError(I)","Minimum","Moments","Otsu","Percentile","RenyiEntropy","Shanbhag","Triangle","Yen"}) {
      cbGlobalMethod_->addItem(m);
    }

    cbLocalMethod_ = new QComboBox(gbThresh);
    for (const QString& m : {"Bernsen","Contrast","Mean","Median","MidGrey","Niblack","Otsu","Phansalkar","Sauvola","Gaussian"}) {
      cbLocalMethod_->addItem(m);
    }

    slObjectThresh_ = new QSlider(Qt::Horizontal, gbThresh);
    spObjectThresh_ = new QSpinBox(gbThresh);
    slObjectThresh_->setRange(0, 255);
    spObjectThresh_->setRange(0, 255);
    slObjectThresh_->setValue(128);
    spObjectThresh_->setValue(128);

    chkInvertBinary_ = new QCheckBox("Invert", gbThresh);
    spLocalBlockSize_ = new QSpinBox(gbThresh);
    spLocalBlockSize_->setRange(3, 99);
    spLocalBlockSize_->setSingleStep(2);
    spLocalBlockSize_->setValue(31);
    spLocalK_ = new QDoubleSpinBox(gbThresh);
    spLocalK_->setRange(-50.0, 50.0);
    spLocalK_->setDecimals(2);
    spLocalK_->setValue(5.0);

    tg->addWidget(new QLabel("Type", gbThresh), 0, 0);
    tg->addWidget(cbThreshType_, 0, 1, 1, 3);
    QLabel* lblGlobalMethod = new QLabel("Global method", gbThresh);
    QLabel* lblLocalMethod = new QLabel("Local method", gbThresh);
    QLabel* lblLocalBlock = new QLabel("Local block size", gbThresh);
    QLabel* lblLocalK = new QLabel("Local k/C", gbThresh);
    tg->addWidget(lblGlobalMethod, 1, 0);
    cbGlobalMethod_->setMinimumWidth(170);
    cbLocalMethod_->setMinimumWidth(170);
    tg->addWidget(cbGlobalMethod_, 1, 1);
    tg->setColumnMinimumWidth(2, 36);
    btnTryAllGlobal_ = new QPushButton("Try All", gbThresh);
    tg->addWidget(btnTryAllGlobal_, 1, 3);
    tg->addWidget(lblLocalMethod, 2, 0);
    tg->addWidget(cbLocalMethod_, 2, 1);
    btnTryAllLocal_ = new QPushButton("Try All", gbThresh);
    tg->addWidget(btnTryAllLocal_, 2, 3);
    tg->addWidget(lblLocalBlock, 3, 0);
    tg->addWidget(spLocalBlockSize_, 3, 1);
    tg->addWidget(lblLocalK, 3, 2);
    tg->addWidget(spLocalK_, 3, 3);
    tg->addWidget(new QLabel("Threshold", gbThresh), 4, 0);
    tg->addWidget(slObjectThresh_, 4, 1, 1, 2);
    tg->addWidget(spObjectThresh_, 4, 3);
    gbThresh->setLayout(tg);

    auto updateMethodUi = [this, lblGlobalMethod, lblLocalMethod, lblLocalBlock, lblLocalK]() {
      const bool local = cbThreshType_ && cbThreshType_->currentIndex() == 1;
      if (lblGlobalMethod) lblGlobalMethod->setVisible(!local);
      if (cbGlobalMethod_) cbGlobalMethod_->setVisible(!local);
      if (btnTryAllGlobal_) btnTryAllGlobal_->setVisible(!local);
      if (lblLocalMethod) lblLocalMethod->setVisible(local);
      if (cbLocalMethod_) cbLocalMethod_->setVisible(local);
      if (btnTryAllLocal_) btnTryAllLocal_->setVisible(local);
      if (lblLocalBlock) lblLocalBlock->setVisible(local);
      if (spLocalBlockSize_) spLocalBlockSize_->setVisible(local);
      if (lblLocalK) lblLocalK->setVisible(local);
      if (spLocalK_) spLocalK_->setVisible(local);
    };
    updateMethodUi();

    trkMainLayout->addWidget(gbThresh);

    QHBoxLayout* prevTitle = new QHBoxLayout();
    prevTitle->addWidget(new QLabel("Binary Preview", tabObj));
    prevTitle->addStretch(1);
    prevTitle->addWidget(chkInvertBinary_);
    trkMainLayout->addLayout(prevTitle);
    lblBinaryPreview_ = new QLabel(tabObj);
    lblBinaryPreview_->setFixedSize(320, 200);
    lblBinaryPreview_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    lblBinaryPreview_->setAlignment(Qt::AlignCenter);
    lblBinaryPreview_->setStyleSheet("background:#111;border:1px solid #3a4250;color:#9aa7b8;");
    lblBinaryPreview_->setText("No binary preview");
    trkMainLayout->addWidget(lblBinaryPreview_);

    QGroupBox* gbBinaryProc = new QGroupBox("Process(Binary)", tabObj);
    QGridLayout* bp = new QGridLayout(gbBinaryProc);
    cbBinaryOp_ = new QComboBox(gbBinaryProc);
    for (const QString& op : {"Erode","Dilate","Open","Close","Fill Holes","Watershed","Skeletonize","Outline","Clear Border"}) cbBinaryOp_->addItem(op);
    btnUndoBinaryOp_ = new QPushButton("Undo", gbBinaryProc);
    lblBinaryOps_ = new QLabel("Pipeline: (none)", gbBinaryProc);
    lblBinaryOps_->setWordWrap(true);
    btnAnalyzeParticles_ = new QPushButton("Analyze Particles", gbBinaryProc);
    btnTrackBinary_ = new QPushButton("Track", gbBinaryProc);
    btnTrackBinary_->setCheckable(true);
    bp->addWidget(new QLabel("Operation", gbBinaryProc), 0, 0);
    bp->addWidget(cbBinaryOp_, 0, 1);
    bp->addWidget(btnUndoBinaryOp_, 0, 2);
    bp->addWidget(lblBinaryOps_, 1, 0, 1, 4);
    bp->addWidget(btnAnalyzeParticles_, 2, 0, 1, 2);
    bp->addWidget(btnTrackBinary_, 2, 2, 1, 2);
    gbBinaryProc->setLayout(bp);
    trkMainLayout->addWidget(gbBinaryProc);

    QGroupBox* gbHist = new QGroupBox("Statistics Histogram", tabObj);
    QGridLayout* hg = new QGridLayout(gbHist);
    cbHistMetric_ = new QComboBox(gbHist);
    cbHistMetric_->addItems({"Area","Perimeter","Circularity","Speed","Displacement","MajorAxis","MinorAxis"});
    spHistMin_ = new QDoubleSpinBox(gbHist);
    spHistMax_ = new QDoubleSpinBox(gbHist);
    spHistMin_->setRange(-1e9, 1e9); spHistMax_->setRange(-1e9, 1e9);
    spHistMin_->setValue(0.0); spHistMax_->setValue(1e6);
    plotHistogram_ = new QCustomPlot(gbHist);
    plotHistogram_->setMinimumHeight(180);
    plotHistogram_->addGraph();
    hg->addWidget(new QLabel("Metric", gbHist), 0, 0);
    hg->addWidget(cbHistMetric_, 0, 1);
    hg->addWidget(new QLabel("Min", gbHist), 0, 2);
    hg->addWidget(spHistMin_, 0, 3);
    hg->addWidget(new QLabel("Max", gbHist), 0, 4);
    hg->addWidget(spHistMax_, 0, 5);
    hg->addWidget(plotHistogram_, 1, 0, 1, 6);
    gbHist->setLayout(hg);
    trkMainLayout->addWidget(gbHist);
    trkMainLayout->addStretch(1);


    visCharts_.clear();
    visChartsLayout_->addWidget(new QLabel("Visual dashboard is shown in the left player area.", tabVis));
    visChartsLayout_->addStretch(1);

    connect(cbThreshType_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, updateMethodUi](int){ object_thresh_manual_ = false; updateMethodUi(); onObjectThresholdParamsChanged(); });
    connect(cbGlobalMethod_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ object_thresh_manual_ = false; onObjectThresholdParamsChanged(); });
    connect(btnTryAllGlobal_, &QPushButton::clicked, this, &MainWindow::onTryAllGlobalMethods);
    connect(btnTryAllLocal_, &QPushButton::clicked, this, &MainWindow::onTryAllLocalMethods);
    connect(cbBinaryOp_, QOverload<int>::of(&QComboBox::activated), this, &MainWindow::onSelectBinaryOp);
    connect(btnUndoBinaryOp_, &QPushButton::clicked, this, &MainWindow::onUndoBinaryOp);
    connect(btnAnalyzeParticles_, &QPushButton::clicked, this, &MainWindow::onAnalyzeParticles);
    connect(btnTrackBinary_, &QPushButton::clicked, this, &MainWindow::onToggleTrackBinary);
    connect(cbLocalMethod_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ object_thresh_manual_ = false; onObjectThresholdParamsChanged(); });
    connect(chkInvertBinary_, &QCheckBox::toggled, this, &MainWindow::onObjectThresholdParamsChanged);
    connect(slObjectThresh_, &QSlider::valueChanged, this, [this](int v){ object_thresh_manual_ = true; if (spObjectThresh_ && spObjectThresh_->value()!=v) spObjectThresh_->setValue(v); onObjectThresholdParamsChanged(); });
    connect(spObjectThresh_, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v){ object_thresh_manual_ = true; if (slObjectThresh_ && slObjectThresh_->value()!=v) slObjectThresh_->setValue(v); onObjectThresholdParamsChanged(); });
    connect(spLocalBlockSize_, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::onObjectThresholdParamsChanged);
    connect(spLocalK_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::onObjectThresholdParamsChanged);
    connect(cbHistMetric_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ updateHistogramPlot(); });
    connect(spHistMin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double){ updateHistogramPlot(); });
    connect(spHistMax_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double){ updateHistogramPlot(); });


    actionTabs_->addTab(tabCap, "Source");
    actionTabs_->addTab(tabCal, "PreProcess");
    actionTabs_->addTab(tabObj, "ObjectDefine");
    actionTabs_->addTab(tabVis, "Visual");
    if (actionTabs_->tabBar()) actionTabs_->tabBar()->hide();

    connect(stepTabs_, &QTabBar::currentChanged, this, [this, leftStack, dock, root](int idx){
      const QSize keepSize = this->size();
      if (isArrowTab(idx)) {
        int fallback = stepToTabIndex(std::max(0, std::min(3, tabIndexToStep(std::max(0, idx-1)))));
        if (stepTabs_ && stepTabs_->currentIndex() != fallback) stepTabs_->setCurrentIndex(fallback);
        return;
      }
      const int stepIdx = std::max(0, std::min(3, tabIndexToStep(idx)));
      if (actionTabs_) actionTabs_->setCurrentIndex(std::max(0, std::min(stepIdx, actionTabs_->count()-1)));
      const bool visual = (stepIdx == 3);
      if (dock) dock->setVisible(!visual);
      if (actionTabs_) actionTabs_->setVisible(!visual);
      if (log_) log_->setVisible(!visual);
      if (leftStack) leftStack->setCurrentWidget(visual ? visualDashHost_ : viewsHost_);
      if (stepIdx == 0) onModeCapture();
      else if (stepIdx == 1) onModeCalibration();
      else if (stepIdx == 2) onModeTracking();
      else {
        mode_ = CAPTURE;
        updateLeftVisualDashboard();
        logLine("Switched to Visual mode.");
      }
      if (stepIdx >= 0 && stepIdx < 4) stepDone_[stepIdx] = true;
      updateStepAvailability();
      if (root) {
        root->setStretch(0, visual ? 1 : 5);
        root->setStretch(1, visual ? 0 : 2);
      }
      if (this->size() != keepSize) this->resize(keepSize);
    });

    dv->addWidget(actionTabs_);

    // Log window
    log_ = new QTextEdit(dockw);
    log_->setReadOnly(true);
    log_->setMinimumHeight(120);
    dv->addWidget(log_);

    dockw->setLayout(dv);
    QVBoxLayout* panelLayout = new QVBoxLayout(dock);
    panelLayout->setContentsMargins(0,0,0,0);
    panelLayout->addWidget(dockw);
    dock->setLayout(panelLayout);
    dock->setStyleSheet("#actionsPanel{background:#222831;border-left:1px solid #3a4250;}");

    root->addWidget(dock);

    lblResolution_ = new QLabel("Resolution: -", this);
    statusBar()->addPermanentWidget(lblResolution_);

    refreshTrajectoryPlot();
    onModeCapture();
    updateStepAvailability();
    statusBar()->showMessage("Ready");
    refreshSourceList();
    // Per-source docks are OFF by default to avoid duplicate display.
    logLine("App started. Configure sources and chessboard in the UI.");
}

void MainWindow::logLine(const QString& s) {
    log_->append(QString("[%1] %2").arg(nowStr(), s));
}

bool MainWindow::openAllSources() {
  closeAllSources();
  bool okAll = true;
  for (auto& s : sources_) {
    if (s.is_image_seq) {
      if (s.seq_files.isEmpty()) okAll = false;
      continue;
    }
    if (s.is_cam) s.cap.open(s.cam_id);
    else openVideoCaptureUnicode(s.cap, s.video_path);
    if (!s.cap.isOpened()) okAll = false;
  }
  return okAll;
}

void MainWindow::closeAllSources() {
  for (auto& s : sources_) {
    if (s.cap.isOpened()) s.cap.release();
  }
}

void MainWindow::refreshSourceList() {
  QStringList status;
  QMutexLocker locker(&sources_mutex_);
  for (int i=0;i<(int)sources_.size();++i) {
    const auto& s = sources_[i];
    QString label;
    if (s.is_cam) label = QString("Camera %1").arg(s.cam_id);
    else if (s.is_image_seq) label = QString("ImgSeq:%1").arg(QFileInfo(s.seq_dir).fileName());
    else label = QFileInfo(s.video_path).fileName();
    bool en = (i < (int)source_enabled_.size()) ? source_enabled_[i] : true;
    QString owner = "{Capture}";
    if (s.mode_owner == (int)CALIB) owner = "{Calib}";
    else if (s.mode_owner == (int)TRACK) owner = "{Track}";
    label += owner;
    label += en ? "[RUN]" : "[PAUSED]";
    status << label;
  }
  Q_UNUSED(status); // status bar text is centralized in updateStatus().
}

std::vector<int> MainWindow::activeSourceIndices() const {
  // Keep the main image view stable across Source / PreProcess / ObjectDefine steps.
  std::vector<int> idx;
  idx.reserve(sources_.size());
  for (int i=0;i<(int)sources_.size();++i) idx.push_back(i);
  return idx;
}

void MainWindow::rebuildSourceViews() {
  if (!viewsGrid_) return;

  while (QLayoutItem* item = viewsGrid_->takeAt(0)) {
    if (item->widget()) item->widget()->deleteLater();
    delete item;
  }
  sourceViews_.clear();

  active_view_source_indices_ = activeSourceIndices();
  const int n = std::min(2, (int)active_view_source_indices_.size());
  if (n <= 0) {
    QLabel* hint = new QLabel("No sources. Click AddVideo to create a viewer.", viewsHost_);
    hint->setAlignment(Qt::AlignCenter);
    hint->setStyleSheet("background-color:#111; border:1px solid #333; color:#ddd;");
    hint->setMinimumSize(960, 540);
    viewsGrid_->addWidget(hint, 0, 0);
    return;
  }

  int cols = (n <= 1) ? 1 : 2;

  for (int i=0; i<n; ++i) {
    QWidget* panel = new QWidget(viewsHost_);
    panel->setStyleSheet("background:#1f2328; border:1px solid #3f4650;");
    QVBoxLayout* pv = new QVBoxLayout(panel);
    pv->setContentsMargins(4,4,4,4);
    pv->setSpacing(4);

    //QToolButton* panBtn = new QToolButton(panel);
    QToolButton* pointBtn = new QToolButton(panel);
    QToolButton* lineBtn = new QToolButton(panel);
    QToolButton* zoomInBtn = new QToolButton(panel);
    QToolButton* zoomOutBtn = new QToolButton(panel);
    QToolButton* resetBtn = new QToolButton(panel);
   // QToolButton* clearBtn = new QToolButton(panel);

    QSize btnIconSize(24, 24);
    pointBtn->setIcon(QIcon(":/icons/Point.png"));
    pointBtn->setIconSize(btnIconSize);
    lineBtn->setIcon(QIcon(":/icons/Line.png"));
    lineBtn->setIconSize(btnIconSize);
    zoomInBtn->setIcon(QIcon(":/icons/Zoom-.png"));
    zoomInBtn->setIconSize(btnIconSize);
    zoomOutBtn->setIcon(QIcon(":/icons/Zoom+.png"));
    zoomOutBtn->setIconSize(btnIconSize);
    resetBtn->setIcon(QIcon(":/icons/Auto.png"));
    resetBtn->setIconSize(btnIconSize);


   // panBtn->setCheckable(true);
    pointBtn->setCheckable(true);
    lineBtn->setCheckable(true);
    //panBtn->setChecked(true);

    //panBtn->setText("Pan");
    pointBtn->setText("Point");
    lineBtn->setText("Line");
    zoomInBtn->setText("Zoom+");
    zoomOutBtn->setText("Zoom-");
    resetBtn->setText("Reset");
   // clearBtn->setText("Clear");

   // panBtn->setToolTip("Pan view");
    pointBtn->setToolTip("Draw point");
    lineBtn->setToolTip("Draw line");
    zoomInBtn->setToolTip("Zoom in");
    zoomOutBtn->setToolTip("Zoom out");
    resetBtn->setToolTip("Reset view");
   // clearBtn->setToolTip("Clear drawings");

    QWidget* canvas = new QWidget(panel);
    QGridLayout* overlay = new QGridLayout(canvas);
    overlay->setContentsMargins(0,0,0,0);
    overlay->setSpacing(0);

    QWidget* topBar = new QWidget(canvas);
    topBar->setAttribute(Qt::WA_StyledBackground, false);
    topBar->setStyleSheet("background: transparent; border: none;");
    topBar->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    QHBoxLayout* tb = new QHBoxLayout(topBar);
    tb->setContentsMargins(6,0,6,6);
    tb->setSpacing(2);
    for (QToolButton* b : {pointBtn, lineBtn, zoomInBtn, zoomOutBtn, resetBtn}) {
      b->setAutoRaise(true);
      b->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
      tb->addWidget(b);
    }
    tb->addStretch(1);

    ImageViewer* v = new ImageViewer(canvas);
    v->setMinimumSize(480, 270);
    v->setToolMode(ImageViewer::PanTool);
    v->setLineCreatedCallback([this, v](double pxLen){ if (gbLineProps_) gbLineProps_->setVisible(true); v->setToolMode(ImageViewer::PanTool); for (auto* ov: sourceViews_) if (ov!=v && ov) ov->setToolMode(ImageViewer::PanTool); pre_scale_line_drawn_ = true; updateScaleStatus(pxLen); updateStepAvailability(); });
    v->setLineDoubleClickCallback([this](double pxLen){ updateScaleStatus(pxLen); });
    v->setLineValueEditedCallback([this](double pxLen, double mm){
      if (pxLen <= 1e-9 || mm <= 0.0) return;
      mm_per_pixel_ = mm / pxLen;
      updateScaleStatus(pxLen);
      logLine(QString("Scale calibrated(inline): %1 mm / %2 px = %3 mm/px")
              .arg(mm,0,'f',6).arg(pxLen,0,'f',3).arg(mm_per_pixel_,0,'f',9));
    });
    v->setAnnotationsVisible(chkShowLines_ ? chkShowLines_->isChecked() : false);
    //connect(panBtn, &QToolButton::clicked, this, [v, panBtn, pointBtn, lineBtn]() {
    //  panBtn->setChecked(true); pointBtn->setChecked(false); lineBtn->setChecked(false);
    //  v->setToolMode(ImageViewer::PanTool);
    //});
    connect(pointBtn, &QToolButton::clicked, this, [v, pointBtn, lineBtn]() {
      pointBtn->setChecked(true); lineBtn->setChecked(false);
      v->setToolMode(ImageViewer::PointTool);
    });
    connect(lineBtn, &QToolButton::clicked, this, [v, pointBtn, lineBtn]() {
      pointBtn->setChecked(false); lineBtn->setChecked(true);
      v->setToolMode(ImageViewer::LineTool);
    });
    connect(zoomInBtn, &QToolButton::clicked, this, [v, pointBtn, lineBtn]() {
      pointBtn->setChecked(false); lineBtn->setChecked(false);
      v->setToolMode(ImageViewer::PanTool);
      v->zoomIn();
    });
    connect(zoomOutBtn, &QToolButton::clicked, this, [v, pointBtn, lineBtn]() {
     pointBtn->setChecked(false); lineBtn->setChecked(false);
      v->setToolMode(ImageViewer::PanTool);
      v->zoomOut();
    });
    connect(resetBtn, &QToolButton::clicked, this, [v, pointBtn, lineBtn]() {
      pointBtn->setChecked(false); lineBtn->setChecked(false);
      v->setToolMode(ImageViewer::PanTool);
      v->resetView();
    });
    //connect(clearBtn, &QToolButton::clicked, this, [v, panBtn, pointBtn, lineBtn]() {
    //  panBtn->setChecked(true); pointBtn->setChecked(false); lineBtn->setChecked(false);
    //  v->setToolMode(ImageViewer::PanTool);
    //  v->clearAnnotations();
    //});

    overlay->addWidget(v, 0, 0);
    overlay->addWidget(topBar, 0, 0, Qt::AlignBottom | Qt::AlignLeft);
    overlay->setRowStretch(0, 1);
    overlay->setColumnStretch(0, 1);
    topBar->raise();
    pv->addWidget(canvas, 1);

    sourceViews_.push_back(v);
    int r = i / cols;
    int c = i % cols;
    viewsGrid_->addWidget(panel, r, c);
  }
}

void MainWindow::updateSourceViews(const std::vector<cv::Mat>& frames) {
  int n = std::min((int)sourceViews_.size(), std::min(2, (int)active_view_source_indices_.size()));
  QStringList resText;
  for (int i=0; i<n; ++i) {
    int srcIdx = active_view_source_indices_[i];
    if (!sourceViews_[i] || srcIdx < 0 || srcIdx >= (int)frames.size()) continue;
    if (frames[srcIdx].empty()) {
      sourceViews_[i]->setImage(QImage());
      continue;
    }
    resText << QString("S%1: %2x%3").arg(i+1).arg(frames[srcIdx].cols).arg(frames[srcIdx].rows);
    sourceViews_[i]->setImage(matToQImage(frames[srcIdx]));
  }
  for (int i=n; i<(int)sourceViews_.size(); ++i) {
    if (sourceViews_[i]) sourceViews_[i]->setImage(QImage());
  }
  if (lblResolution_) {
    lblResolution_->setText(resText.isEmpty() ? "Resolution: -" : QString("Resolution: %1").arg(resText.join(" | ")));
  }
}

void MainWindow::rebuildCalibratorFromUI(bool reset) {
  board_w_ = spBoardW_ ? spBoardW_->value() : board_w_;
  board_h_ = spBoardH_ ? spBoardH_->value() : board_h_;
  square_  = spSquare_ ? spSquare_->value() : square_;

  if (reset) {
    calibrator_.reset(new MultiCamCalibrator(std::max(1, (int)sources_.size()),
                                             cv::Size(board_w_, board_h_), square_));
    calib_pairs_.clear();
    calib_pair_rmse_.clear();
    has_computed_calib_ = false;
    if (btnSaveCalib_) btnSaveCalib_->setEnabled(false);
    if (calibErrorTable_) calibErrorTable_->setRowCount(0);
    if (calibProgressBar_) calibProgressBar_->setValue(0);
    if (lblCalibProgress_) lblCalibProgress_->setText("Progress: idle");
    logLine(QString("Chessboard params updated: %1x%2 square=%3 m (captures reset)")
            .arg(board_w_).arg(board_h_).arg(square_,0,'f',6));
    last_inliers_ = 0;
  }
}

// ---------------- Sources actions ----------------
void MainWindow::onAddCamera() {
  detect_overlay_cache_.clear();
  bool ok=false;
  int camId = QInputDialog::getInt(this, "Add Camera", "Camera index:", 0, 0, 64, 1, &ok);
  if (!ok) return;
  if (mode_ != CAPTURE) {
    QMessageBox::information(this, "Add Camera", "Please switch to Capture tab to add camera sources.");
    return;
  }
  if (!sources_.empty()) {
    QMessageBox::information(this, "Add Camera", "Monocular mode supports only 1 source.");
    return;
  }

  InputSource s;
  s.is_cam = true;
  s.cam_id = camId;
  s.mode_owner = (int)CAPTURE;

  // Try open immediately
  s.cap.open(camId);
  if (!s.cap.isOpened()) {
    QMessageBox::warning(this, "Camera", "Failed to open camera. Try another index or close other apps.");
    logLine(QString("Failed to open camera %1").arg(camId));
  } else {
    logLine(QString("Added camera %1").arg(camId));
  }

  timer_.stop();
  if (captureWorker_) QMetaObject::invokeMethod(captureWorker_, "stop", Qt::BlockingQueuedConnection);
  sources_.push_back(std::move(s));
  num_cams_ = (int)sources_.size();
  // Do NOT auto-play on import
  if ((int)source_enabled_.size() < num_cams_) source_enabled_.resize(num_cams_, true);
  source_enabled_[num_cams_-1] = true;
  if ((int)last_frames_.size() < num_cams_) last_frames_.resize(num_cams_);
  cv::Mat firstFrame;
  if (!sources_.empty() && sources_.back().cap.isOpened()) {
    sources_.back().cap.read(firstFrame);
  }
  last_frames_[num_cams_ - 1] = firstFrame.clone();
  source_enabled_.assign(std::max(0,num_cams_), true);
  // last_frames_ will be populated by CaptureWorker when frames arrive.
  last_frames_.resize(std::max(0,num_cams_));
  // Rebuild calibrator with new cam count; reset captures
  calibrator_.reset(new MultiCamCalibrator(std::max(1,num_cams_), cv::Size(board_w_, board_h_), square_));
  refreshSourceList();
  if (show_docks_) rebuildSourceDocks();
  rebuildSourceViews();
  updateSourceViews(last_frames_);
  updateStatus();
  updateStepAvailability();
}

void MainWindow::onAddVideo() 
{
    detect_overlay_cache_.clear();
    QString last = settings_.value("lastVideoDir", "").toString();
    QString path = QFileDialog::getOpenFileName(this, "Add Video", last,"Video (*.mp4 *.avi *.mov *.mkv);;All (*.*)");
    if (path.isEmpty()) return;

    if (!sources_.empty()) {
      QMessageBox::information(this, "Add Video", "Monocular mode supports only 1 source.");
      return;
    }

    InputSource s;
    s.is_cam = false;
    s.mode_owner = (int)mode_;
    s.video_path = path;
    openVideoCaptureUnicode(s.cap, path);
    if (!s.cap.isOpened()) 
    {
        QMessageBox::warning(this, "Video", "Failed to open video file.");
        logLine(QString("Failed to open video: %1").arg(path));
        return;
    }
    settings_.setValue("lastVideoDir", QFileInfo(path).absolutePath());

    timer_.stop();
    if (captureWorker_) 
        QMetaObject::invokeMethod(captureWorker_, "stop", Qt::BlockingQueuedConnection);
    sources_.push_back(std::move(s));
    num_cams_ = (int)sources_.size();
    // Do NOT auto-play on import
    if ((int)source_enabled_.size() < num_cams_) 
        source_enabled_.resize(num_cams_, true);
    source_enabled_[num_cams_-1] = false;
    if ((int)last_frames_.size() < num_cams_) 
        last_frames_.resize(num_cams_);
 
    cv::Mat firstFrame;
    if (!sources_.empty() && sources_.back().cap.isOpened()) {
      sources_.back().cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      sources_.back().cap.read(firstFrame);
      sources_.back().cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
    last_frames_[num_cams_ - 1] = firstFrame.clone();

    source_enabled_.assign(std::max(0,num_cams_), true);
    calibrator_.reset(new MultiCamCalibrator(std::max(1,num_cams_), cv::Size(board_w_, board_h_), square_));
    refreshSourceList();
    if (show_docks_) rebuildSourceDocks();
    rebuildSourceViews();
    updateSourceViews(last_frames_);
    updateStatus();
    updateStepAvailability();
    logLine(QString("Added video: %1").arg(path));
}


void MainWindow::onAddImageSequence()
{
    detect_overlay_cache_.clear();
    QString last = settings_.value("lastImageSeqDir", "").toString();
    QString dir = QFileDialog::getExistingDirectory(this, "Add Image Sequence Folder", last);
    if (dir.isEmpty()) return;

    QDir qdir(dir);
    QFileInfoList files = qdir.entryInfoList(kImageNameFilters(), QDir::Files, QDir::Name);
    if (files.isEmpty()) {
      QMessageBox::warning(this, "Image Sequence", "No image files found in selected folder.");
      return;
    }

    if (!sources_.empty()) {
      QMessageBox::information(this, "Add Image Sequence", "Monocular mode supports only 1 source.");
      return;
    }

    InputSource s;
    s.is_cam = false;
    s.is_image_seq = true;
    s.mode_owner = (int)mode_;
    s.seq_dir = dir;
    s.video_path = dir;
    s.seq_idx = 0;
    for (const auto& fi : files) s.seq_files.push_back(fi.absoluteFilePath());

    cv::Mat firstFrame = imreadUnicodePath(s.seq_files.front(), cv::IMREAD_COLOR);
    if (firstFrame.empty()) {
      QMessageBox::warning(this, "Image Sequence", "Failed to read first image in selected folder.");
      return;
    }

    settings_.setValue("lastImageSeqDir", dir);

    timer_.stop();
    if (captureWorker_) QMetaObject::invokeMethod(captureWorker_, "stop", Qt::BlockingQueuedConnection);

    sources_.push_back(std::move(s));
    num_cams_ = (int)sources_.size();
    if ((int)source_enabled_.size() < num_cams_) source_enabled_.resize(num_cams_, true);
    source_enabled_[num_cams_ - 1] = true;
    if ((int)last_frames_.size() < num_cams_) last_frames_.resize(num_cams_);
    last_frames_[num_cams_ - 1] = firstFrame.clone();

    source_enabled_.assign(std::max(0, num_cams_), true);
    calibrator_.reset(new MultiCamCalibrator(std::max(1, num_cams_), cv::Size(board_w_, board_h_), square_));
    refreshSourceList();
    if (show_docks_) rebuildSourceDocks();
    rebuildSourceViews();
    updateSourceViews(last_frames_);
    updateStatus();
    updateStepAvailability();
    logLine(QString("Added image sequence: %1 (%2 frames)").arg(dir).arg(files.size()));
}

void MainWindow::onRemoveSource() {
  detect_overlay_cache_.clear();
  int removed = 0;
  timer_.stop();
  if (captureWorker_) QMetaObject::invokeMethod(captureWorker_, "stop", Qt::BlockingQueuedConnection);

  for (int i=(int)sources_.size()-1; i>=0; --i) {
    if (sources_[i].mode_owner != (int)mode_) continue;
    if (sources_[i].cap.isOpened()) sources_[i].cap.release();
    sources_.erase(sources_.begin() + i);
    ++removed;
  }
  if (removed <= 0) return;

  // Source removal must reset all Visual/ObjectDefine measurement outputs.
  meas_rows_.clear();
  pre_scale_line_drawn_ = false;
  pre_scale_calculated_ = false;
  stepDone_[1] = false;
  stepDone_[2] = false;
  stepDone_[3] = false;
  last_meas_key_ = std::numeric_limits<qint64>::min();
  last_ctr_ = cv::Point2f(0, 0);
  last_speed_ = 0.0;
  measurements_frozen_ = false;
  track_binary_enabled_ = false;
  next_track_id_ = 1;
  tracked_centroids_.clear();
  tracked_contours_.clear();
  if (btnTrackBinary_) btnTrackBinary_->setChecked(false);
  refreshTrajectoryPlot();
  updateLeftVisualDashboard();

  num_cams_ = (int)sources_.size();
  source_enabled_.assign(std::max(0,num_cams_), true);
  // last_frames_ will be populated by CaptureWorker when frames arrive.
  last_frames_.resize(std::max(0,num_cams_));

  calibrator_.reset(new MultiCamCalibrator(std::max(1,num_cams_), cv::Size(board_w_, board_h_), square_));
  refreshSourceList();
  if (show_docks_) rebuildSourceDocks();
  rebuildSourceViews();
  updateSourceViews(last_frames_);
  updateStatus();
  updateStepAvailability();
  logLine(QString("Removed %1 source(s) in current mode.").arg(removed));
}

void MainWindow::onModeCalibration() {
  // Keep image/render pipeline on capture source set; only switch right-side params.
  mode_ = CAPTURE;
  if (btnAddCam_) btnAddCam_->setVisible(false);
  if (btnAddVideo_) btnAddVideo_->setVisible(true);
  if (btnAddImgSeq_) btnAddImgSeq_->setVisible(true);
  if (actionTabs_ && actionTabs_->currentIndex()!=1) actionTabs_->setCurrentIndex(1);
  if (stepTabs_ && stepTabs_->currentIndex()!=stepToTabIndex(1)) stepTabs_->setCurrentIndex(stepToTabIndex(1));
  logLine("Switched to Preprocess mode.");
}

void MainWindow::onModeTracking() {
  // Keep image/render pipeline stable across Source/PreProcess/ObjectDefine.
  mode_ = CAPTURE;
  if (btnAddCam_) btnAddCam_->setVisible(false);
  if (btnAddVideo_) btnAddVideo_->setVisible(true);
  if (btnAddImgSeq_) btnAddImgSeq_->setVisible(true);
  if (actionTabs_ && actionTabs_->currentIndex()!=2) actionTabs_->setCurrentIndex(2);
  if (stepTabs_ && stepTabs_->currentIndex()!=stepToTabIndex(2)) stepTabs_->setCurrentIndex(stepToTabIndex(2));
  logLine("Switched to ObjectDefine mode.");
}

void MainWindow::onModeCapture() {
  mode_ = CAPTURE;
  if (btnAddCam_) btnAddCam_->setVisible(true);
  if (btnAddVideo_) btnAddVideo_->setVisible(true);
  if (btnAddImgSeq_) btnAddImgSeq_->setVisible(true);
  if (actionTabs_ && actionTabs_->currentIndex()!=0) actionTabs_->setCurrentIndex(0);
  if (stepTabs_ && stepTabs_->currentIndex()!=stepToTabIndex(0)) stepTabs_->setCurrentIndex(stepToTabIndex(0));
  logLine("Switched to Source mode.");
}

void MainWindow::onCaptureNow() {
  stopCaptureBlocking();
  {
    QMutexLocker srcLock(&sources_mutex_);
    if ((int)last_frames_.size() < (int)sources_.size()) last_frames_.resize(sources_.size());
    for (int i=0;i<(int)sources_.size();++i) {
      auto& s = sources_[i];
      if (s.mode_owner != (int)CAPTURE) continue;
      cv::Mat f;
      if (s.is_image_seq) {
        if (!s.seq_files.isEmpty()) {
          int idx = std::max(0, std::min(s.seq_idx, s.seq_files.size()-1));
          f = imreadUnicodePath(s.seq_files[idx], cv::IMREAD_COLOR);
        }
      } else if (s.cap.isOpened()) {
        s.cap.read(f);
      }
      if (!f.empty()) last_frames_[i] = f;
    }
  }
  updateSourceViews(last_frames_);
  logLine("Capture snapshot updated.");
}

void MainWindow::onModeTabChanged(int idx) {
  if (!stepTabs_) return;
  updateStepAvailability();
  const int tabIdx = stepToTabIndex(std::max(0, std::min(3, idx)));
  int target = tabIdx;
  if (target < 0 || target >= stepTabs_->count() || !stepTabs_->isTabEnabled(target)) {
    target = stepTabs_->currentIndex();
  }
  if (target >= 0 && target < stepTabs_->count() && stepTabs_->currentIndex()!=target) {
    stepTabs_->setCurrentIndex(target);
  }
}

bool MainWindow::hasAnySourceInCurrentMode() {
  QMutexLocker srcLock(&sources_mutex_);
  for (const auto& s : sources_) {
    if (s.mode_owner == (int)mode_) return true;
  }
  return false;
}

void MainWindow::updateStepAvailability() {
  if (!stepTabs_ || stepTabs_->count() < 7) return;
  const bool sourceDone = hasAnySourceInCurrentMode();
  stepDone_[0] = sourceDone;
  stepDone_[1] = pre_scale_line_drawn_ && pre_scale_calculated_;
  stepDone_[2] = stepDone_[2] && !meas_rows_.empty();

  // Progressive unlock: step N is clickable only when step N-1 is done.
  const bool en0 = true;
  const bool en1 = stepDone_[0];
  const bool en2 = en1 && stepDone_[1];
  const bool en3 = en2 && stepDone_[2];
  stepTabs_->setTabEnabled(stepToTabIndex(0), en0);
  stepTabs_->setTabEnabled(stepToTabIndex(1), en1);
  stepTabs_->setTabEnabled(stepToTabIndex(2), en2);
  stepTabs_->setTabEnabled(stepToTabIndex(3), en3);
  stepTabs_->setTabEnabled(1, false);
  stepTabs_->setTabEnabled(3, false);
  stepTabs_->setTabEnabled(5, false);

  int curStep = tabIndexToStep(stepTabs_->currentIndex());
  if (curStep == 3 && !en3) curStep = 2;
  if (curStep == 2 && !en2) curStep = 1;
  if (curStep == 1 && !en1) curStep = 0;
  const int curTab = stepToTabIndex(curStep);
  if (curTab != stepTabs_->currentIndex()) stepTabs_->setCurrentIndex(curTab);
}

void MainWindow::onPreprocessParamsChanged() {
  logLine(QString("Preprocess changed: color=%1, brightness=%2, contrast=%3")
          .arg(cbPreColor_ && cbPreColor_->currentIndex()==0 ? "B/W" : "Color")
          .arg(slBrightness_ ? slBrightness_->value() : 128)
          .arg(slContrast_ ? slContrast_->value() : 128));

  std::vector<cv::Mat> frames;
  {
    QMutexLocker locker(&frames_mutex_);
    frames = last_frames_;
  }
  if (frames.empty()) return;
  for (auto& f : frames) {
    if (!f.empty()) f = applyPreprocess(f);
  }
  updateSourceViews(frames);
}

void MainWindow::onObjectThresholdParamsChanged() {
  if (track_binary_enabled_ && measurements_frozen_) rebuildMeasurementSeriesFromCurrentSource(false);
  std::vector<cv::Mat> frames;
  {
    QMutexLocker locker(&frames_mutex_);
    frames = last_frames_;
  }
  if (frames.empty()) return;
  for (const auto& f : frames) {
    if (f.empty()) continue;
    int autoT = spObjectThresh_ ? spObjectThresh_->value() : 128;
    cv::Mat bin = makeObjectBinaryPreview(applyPreprocess(f), &autoT);
    if (!object_thresh_manual_ && slObjectThresh_ && cbThreshType_ && cbThreshType_->currentIndex()==0) {
      if (slObjectThresh_->value()!=autoT) slObjectThresh_->setValue(autoT);
      if (spObjectThresh_ && spObjectThresh_->value()!=autoT) spObjectThresh_->setValue(autoT);
    }
    if (!bin.empty() && lblBinaryPreview_) {
      QImage qi = matToQImage(bin);
      lblBinaryPreview_->setPixmap(QPixmap::fromImage(qi).scaled(lblBinaryPreview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    break;
  }
}

cv::Mat MainWindow::makeObjectBinaryMask(const cv::Mat& src, int* outGlobalThreshold) const {
  if (src.empty()) return cv::Mat();
  cv::Mat gray;
  if (src.channels()==3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  else if (src.channels()==4) cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
  else gray = src.clone();

  auto clampT = [](int t){ return std::max(0, std::min(255, t)); };
  auto hist256 = [&](const cv::Mat& g) {
    std::vector<double> h(256, 0.0);
    for (int y=0; y<g.rows; ++y) {
      const uchar* r = g.ptr<uchar>(y);
      for (int x=0; x<g.cols; ++x) h[r[x]] += 1.0;
    }
    return h;
  };
  auto sumHist = [](const std::vector<double>& h){ double s=0; for(double v:h) s+=v; return s; };

  cv::Mat bin;
  const bool local = cbThreshType_ && cbThreshType_->currentIndex() == 1;
  if (!local) {
    int thr = spObjectThresh_ ? spObjectThresh_->value() : 128;
    QString gm = cbGlobalMethod_ ? cbGlobalMethod_->currentText() : QString("Otsu");

    if (gm == "Otsu") {
      thr = (int)std::round(cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU));
    } else if (gm == "Triangle") {
      thr = (int)std::round(cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_TRIANGLE));
    } else if (gm == "Mean" || gm == "Percentile") {
      cv::Scalar m = cv::mean(gray);
      thr = (gm == "Mean") ? (int)std::round(m[0]) : (int)std::round(0.5 * 255.0);
      cv::threshold(gray, bin, thr, 255, cv::THRESH_BINARY);
    } else if (gm == "IsoData" || gm == "Default") {
      cv::Scalar m = cv::mean(gray);
      double t = m[0], prev = -1;
      for (int i=0;i<64 && std::abs(t-prev)>0.5;i++) {
        prev = t;
        cv::Mat a,b;
        cv::threshold(gray, a, t, 255, cv::THRESH_BINARY_INV);
        cv::threshold(gray, b, t, 255, cv::THRESH_BINARY);
        double m1 = cv::mean(gray, a)[0];
        double m2 = cv::mean(gray, b)[0];
        t = (m1 + m2) * 0.5;
      }
      thr = clampT((int)std::round(t));
      cv::threshold(gray, bin, thr, 255, cv::THRESH_BINARY);
    } else if (gm == "Li") {
      cv::Scalar m = cv::mean(gray);
      double t = std::max(1.0, m[0]);
      for (int i=0;i<64;i++) {
        cv::Mat a,b;
        cv::threshold(gray, a, t, 255, cv::THRESH_BINARY_INV);
        cv::threshold(gray, b, t, 255, cv::THRESH_BINARY);
        double m1 = std::max(1e-6, cv::mean(gray, a)[0]);
        double m2 = std::max(1e-6, cv::mean(gray, b)[0]);
        double nt = (m1 - m2) / (std::log(m1) - std::log(m2));
        if (!std::isfinite(nt) || std::abs(nt - t) < 0.5) break;
        t = nt;
      }
      thr = clampT((int)std::round(t));
      cv::threshold(gray, bin, thr, 255, cv::THRESH_BINARY);
    } else if (gm == "Yen" || gm == "MaxEntropy" || gm == "RenyiEntropy" || gm == "Shanbhag" || gm == "Huang" || gm == "Intermodes" || gm == "MinError(I)" || gm == "Minimum" || gm == "Moments") {
      // Entropy-family and legacy methods: approximation via entropy-max threshold.
      auto h = hist256(gray);
      double total = sumHist(h);
      if (total <= 0) total = 1;
      for (double& v : h) v /= total;
      std::vector<double> P1(256,0), P2(256,0), S1(256,0), S2(256,0);
      P1[0]=h[0]; S1[0]=(h[0]>0? -h[0]*std::log(h[0]):0);
      for(int i=1;i<256;i++){ P1[i]=P1[i-1]+h[i]; S1[i]=S1[i-1]+(h[i]>0?-h[i]*std::log(h[i]):0); }
      P2[255]=h[255]; S2[255]=(h[255]>0? -h[255]*std::log(h[255]):0);
      for(int i=254;i>=0;i--){ P2[i]=P2[i+1]+h[i]; S2[i]=S2[i+1]+(h[i]>0?-h[i]*std::log(h[i]):0); }
      double best=-1e9; int bestT=thr;
      for(int t=1;t<255;t++){
        if (P1[t] <= 1e-12 || P2[t+1] <= 1e-12) continue;
        double H = std::log(P1[t]) + std::log(P2[t+1]) + S1[t]/P1[t] + S2[t+1]/P2[t+1];
        if (H>best){ best=H; bestT=t; }
      }
      thr = bestT;
      cv::threshold(gray, bin, thr, 255, cv::THRESH_BINARY);
    } else {
      cv::threshold(gray, bin, thr, 255, cv::THRESH_BINARY);
    }

    const int autoThr = thr;
    if (object_thresh_manual_ && spObjectThresh_) {
      thr = spObjectThresh_->value();
      cv::threshold(gray, bin, thr, 255, cv::THRESH_BINARY);
    }
    if (outGlobalThreshold) *outGlobalThreshold = autoThr;
  } else {
    int block = spLocalBlockSize_ ? spLocalBlockSize_->value() : 31;
    if (block % 2 == 0) block += 1;
    block = std::max(3, block);
    double c = spLocalK_ ? spLocalK_->value() : 5.0;
    QString lm = cbLocalMethod_ ? cbLocalMethod_->currentText() : QString("Mean");

    if (lm == "Gaussian") {
      cv::adaptiveThreshold(gray, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block, c);
    } else if (lm == "Mean") {
      cv::adaptiveThreshold(gray, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, block, c);
    } else if (lm == "Median") {
      cv::Mat med;
      cv::medianBlur(gray, med, std::max(3, block|1));
      cv::subtract(med, cv::Scalar(c), med);
      cv::compare(gray, med, bin, cv::CMP_GT);
    } else if (lm == "MidGrey") {
      cv::Mat mn,mx;
      cv::erode(gray, mn, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(block, block)));
      cv::dilate(gray, mx, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(block, block)));
      cv::Mat mid; cv::addWeighted(mn,0.5,mx,0.5,0,mid);
      cv::subtract(mid, cv::Scalar(c), mid);
      cv::compare(gray, mid, bin, cv::CMP_GT);
    } else if (lm == "Bernsen" || lm == "Contrast") {
      cv::Mat mn,mx;
      cv::erode(gray, mn, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(block, block)));
      cv::dilate(gray, mx, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(block, block)));
      cv::Mat contrast; cv::subtract(mx,mn,contrast);
      cv::Mat mid; cv::addWeighted(mn,0.5,mx,0.5,0,mid);
      cv::compare(gray, mid, bin, cv::CMP_GT);
      if (lm == "Contrast") {
        cv::Mat low; cv::threshold(contrast, low, c, 255, cv::THRESH_BINARY_INV);
        bin.setTo(0, low);
      }
    } else {
      // Niblack / Sauvola / Phansalkar / Otsu(local) fallback to Mean-style local thresholding.
      cv::adaptiveThreshold(gray, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, block, c);
    }

    if (outGlobalThreshold) *outGlobalThreshold = spObjectThresh_ ? spObjectThresh_->value() : 128;
  }

  if (chkInvertBinary_ && chkInvertBinary_->isChecked()) cv::bitwise_not(bin, bin);
  return applyBinaryProcessOps(bin);
}

cv::Mat MainWindow::applyBinaryProcessOps(const cv::Mat& binMask) const {
  if (binMask.empty()) return cv::Mat();
  cv::Mat out = binMask.clone();
  auto fillHoles = [](cv::Mat& m) {
    cv::Mat flood = m.clone();
    cv::floodFill(flood, cv::Point(0,0), cv::Scalar(255));
    cv::bitwise_not(flood, flood);
    cv::bitwise_or(m, flood, m);
  };
  auto skeletonize = [](const cv::Mat& src){
    cv::Mat skel(src.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat img = src.clone();
    cv::Mat temp, eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
    bool done;
    do {
      cv::erode(img, eroded, element);
      cv::dilate(eroded, temp, element);
      cv::subtract(img, temp, temp);
      cv::bitwise_or(skel, temp, skel);
      eroded.copyTo(img);
      done = (cv::countNonZero(img) == 0);
    } while(!done);
    return skel;
  };
  auto clearBorder = [](cv::Mat& m) {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(m.clone(), cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& c : cs) {
      cv::Rect r = cv::boundingRect(c);
      if (r.x <= 0 || r.y <= 0 || r.br().x >= m.cols-1 || r.br().y >= m.rows-1) {
        cv::drawContours(m, std::vector<std::vector<cv::Point>>{c}, -1, cv::Scalar(0), cv::FILLED);
      }
    }
  };
  for (const auto& op : binary_ops_pipeline_) {
    if (op == "Erode") cv::erode(out, out, cv::Mat(), cv::Point(-1,-1), 1);
    else if (op == "Dilate") cv::dilate(out, out, cv::Mat(), cv::Point(-1,-1), 1);
    else if (op == "Open") cv::morphologyEx(out, out, cv::MORPH_OPEN, cv::Mat());
    else if (op == "Close") cv::morphologyEx(out, out, cv::MORPH_CLOSE, cv::Mat());
    else if (op == "Fill Holes") fillHoles(out);
    else if (op == "Skeletonize") out = skeletonize(out);
    else if (op == "Outline") { cv::Mat e; cv::erode(out, e, cv::Mat()); cv::subtract(out, e, out); }
    else if (op == "Clear Border") clearBorder(out);
    else if (op == "Watershed") {
      cv::Mat dist; cv::distanceTransform(out, dist, cv::DIST_L2, 5);
      cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
      cv::Mat peaks; cv::threshold(dist, peaks, 0.4, 1.0, cv::THRESH_BINARY);
      peaks.convertTo(peaks, CV_8U, 255);
      cv::Mat markers; cv::connectedComponents(peaks, markers);
      cv::Mat color; cv::cvtColor(out, color, cv::COLOR_GRAY2BGR);
      cv::watershed(color, markers);
      out.setTo(0);
      out.setTo(255, markers > 1);
    }
  }
  return out;
}

cv::Mat MainWindow::makeObjectBinaryPreview(const cv::Mat& src, int* outGlobalThreshold) const {
  cv::Mat mask = makeObjectBinaryMask(src, outGlobalThreshold);
  if (mask.empty()) return cv::Mat();
  cv::Mat bgr; cv::cvtColor(mask, bgr, cv::COLOR_GRAY2BGR);
  return bgr;
}

std::vector<std::vector<cv::Point>> MainWindow::detectBinaryContours(const cv::Mat& src, int* outGlobalThreshold) const {
  cv::Mat mask = makeObjectBinaryMask(src, outGlobalThreshold);
  std::vector<std::vector<cv::Point>> contours;
  if (!mask.empty()) cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  return contours;
}

void MainWindow::onSelectBinaryOp() {
  if (!cbBinaryOp_) return;
  const QString op = cbBinaryOp_->currentText();
  if (op.isEmpty()) return;
  binary_ops_pipeline_.push_back(op);
  if (lblBinaryOps_) {
    QStringList ops; for (const auto& o : binary_ops_pipeline_) ops << o;
    lblBinaryOps_->setText(QString("Pipeline: %1").arg(ops.join(" -> ")));
  }
  onObjectThresholdParamsChanged();
}

void MainWindow::onUndoBinaryOp() {
  if (!binary_ops_pipeline_.empty()) binary_ops_pipeline_.pop_back();
  if (lblBinaryOps_) {
    if (binary_ops_pipeline_.empty()) lblBinaryOps_->setText("Pipeline: (none)");
    else { QStringList ops; for (const auto& o : binary_ops_pipeline_) ops << o; lblBinaryOps_->setText(QString("Pipeline: %1").arg(ops.join(" -> "))); }
  }
  onObjectThresholdParamsChanged();
}

void MainWindow::onAnalyzeParticles() {
  std::vector<cv::Mat> frames;
  { QMutexLocker locker(&frames_mutex_); frames = last_frames_; }
  analyzed_contours_.clear();
  std::vector<cv::Mat> vis = frames;
  for (auto& f : vis) if (!f.empty()) f = applyPreprocess(f);
  for (auto& f : vis) {
    if (f.empty()) continue;
    analyzed_contours_ = detectBinaryContours(f); // includes current threshold + invert + binary ops
    if (!analyzed_contours_.empty()) {
      cv::drawContours(f, analyzed_contours_, -1, cv::Scalar(0,255,0), 2);
    }
    break;
  }
  updateSourceViews(vis);
  logLine(QString("Analyze Particles: %1 regions").arg((int)analyzed_contours_.size()));
}

void MainWindow::onToggleTrackBinary() {
  track_binary_enabled_ = btnTrackBinary_ && btnTrackBinary_->isChecked();
  tracked_centroids_.clear();
  tracked_contours_.clear();
  next_track_id_ = 1;
  if (track_binary_enabled_) {
    // Build full frame range first so Track can compute all-frame measurements
    // immediately, without requiring manual playback.
    updatePlaybackParams();
    rebuildMeasurementSeriesFromCurrentSource(true);
    if (!meas_rows_.empty()) stepDone_[2] = true;
  }
  else measurements_frozen_ = false;
  updateStepAvailability();
  logLine(track_binary_enabled_ ? "Binary contour tracking enabled." : "Binary contour tracking disabled.");
  onTick();
}

void MainWindow::onStartScaleLine() {
  if (chkShowLines_ && !chkShowLines_->isChecked()) chkShowLines_->setChecked(true);
  for (auto* v : sourceViews_) {
    if (!v) continue;
    v->clearAllLines(); // only one line allowed
    v->setAnnotationsVisible(true);
    v->setToolMode(ImageViewer::LineTool);
  }
  if (gbLineProps_) gbLineProps_->setVisible(true);
  logLine("Scale line drawing mode enabled (single line).");
}

void MainWindow::onDeleteScaleLine() {
  for (auto* v : sourceViews_) {
    if (v) v->clearAllLines();
  }
  mm_per_pixel_ = 0.0;
  if (lblScaleInfo_) lblScaleInfo_->setText("Scale: not calibrated");
  logLine("Scale line deleted.");
}

void MainWindow::onApplyScaleFromInput() {
  double px = 0.0;
  for (auto* v : sourceViews_) {
    if (!v) continue;
    px = std::max(px, v->anyLineLength());
  }
  if (px <= 1e-9) {
    QMessageBox::information(this, "Scale", "Please draw one scale line first.");
    return;
  }
  double mm = 0.0;
  if (editPhysicalMm_) mm = editPhysicalMm_->text().toDouble();
  if (mm <= 0.0) {
    QMessageBox::information(this, "Scale", "Please input physical distance (mm) > 0.");
    return;
  }
  mm_per_pixel_ = mm / px;
  pre_scale_calculated_ = true;
  stepDone_[1] = pre_scale_line_drawn_ && pre_scale_calculated_;
  updateStepAvailability();
  updateScaleStatus(px);
  logLine(QString("Scale calibrated: %1 mm / %2 px = %3 mm/px")
          .arg(mm,0,'f',6).arg(px,0,'f',3).arg(mm_per_pixel_,0,'f',9));
}

void MainWindow::onTryAllGlobalMethods() {
  std::vector<cv::Mat> frames;
  {
    QMutexLocker locker(&frames_mutex_);
    frames = last_frames_;
  }
  cv::Mat src;
  for (auto& f: frames) { if (!f.empty()) { src = applyPreprocess(f); break; } }
  if (src.empty()) {
    QMessageBox::information(this, "Try All", "No image available.");
    return;
  }

  QString oldMethod = cbGlobalMethod_ ? cbGlobalMethod_->currentText() : QString();
  const int n = cbGlobalMethod_ ? cbGlobalMethod_->count() : 0;

  QDialog* dlg = new QDialog(this);
  dlg->setWindowTitle("Try All Global Methods");
  QVBoxLayout* outer = new QVBoxLayout(dlg);
  QGridLayout* grid = new QGridLayout();
  auto tiles = std::make_shared<std::vector<ThumbnailLabel*>>();
  const int thumbW = 220, thumbH = 140, cols = 4;

  for (int i = 0; i < n; ++i) {
    if (cbGlobalMethod_) cbGlobalMethod_->setCurrentIndex(i);
    cv::Mat b = makeObjectBinaryPreview(src);
    if (b.empty()) continue;
    cv::resize(b, b, cv::Size(thumbW, thumbH));
    const QString methodName = cbGlobalMethod_->itemText(i);
    cv::putText(b, methodName.toStdString(), cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,255), 2, cv::LINE_AA);

    auto* tile = new ThumbnailLabel(i, dlg);
    tile->setFixedSize(thumbW, thumbH);
    tile->setPixmap(QPixmap::fromImage(matToQImage(b)));
    tile->setScaledContents(true);
    tile->setToolTip(methodName + "\nDouble-click to use this method");
    tiles->push_back(tile);
    tile->onClicked = [tiles, tile](int) {
      for (auto* t : *tiles) if (t) t->setSelected(false);
      if (tile) tile->setSelected(true);
    };
    tile->onDoubleClick = [this, dlg](int idx) {
      if (cbGlobalMethod_ && idx >= 0 && idx < cbGlobalMethod_->count()) cbGlobalMethod_->setCurrentIndex(idx);
      dlg->accept();
    };
    grid->addWidget(tile, i / cols, i % cols);
  }

  if (cbGlobalMethod_) {
    int idx = cbGlobalMethod_->findText(oldMethod);
    if (idx >= 0) cbGlobalMethod_->setCurrentIndex(idx);
  }

  outer->addLayout(grid);
  dlg->exec();
}

void MainWindow::onTryAllLocalMethods() {
  std::vector<cv::Mat> frames;
  {
    QMutexLocker locker(&frames_mutex_);
    frames = last_frames_;
  }
  cv::Mat src;
  for (auto& f: frames) { if (!f.empty()) { src = applyPreprocess(f); break; } }
  if (src.empty()) {
    QMessageBox::information(this, "Try All", "No image available.");
    return;
  }

  QString oldMethod = cbLocalMethod_ ? cbLocalMethod_->currentText() : QString();
  int oldType = cbThreshType_ ? cbThreshType_->currentIndex() : 1;
  if (cbThreshType_) cbThreshType_->setCurrentIndex(1);
  const int n = cbLocalMethod_ ? cbLocalMethod_->count() : 0;

  QDialog* dlg = new QDialog(this);
  dlg->setWindowTitle("Try All Local Methods");
  QVBoxLayout* outer = new QVBoxLayout(dlg);
  QGridLayout* grid = new QGridLayout();
  auto tiles = std::make_shared<std::vector<ThumbnailLabel*>>();
  const int thumbW = 220, thumbH = 140, cols = 4;

  for (int i = 0; i < n; ++i) {
    if (cbLocalMethod_) cbLocalMethod_->setCurrentIndex(i);
    cv::Mat b = makeObjectBinaryPreview(src);
    if (b.empty()) continue;
    cv::resize(b, b, cv::Size(thumbW, thumbH));
    const QString methodName = cbLocalMethod_->itemText(i);
    cv::putText(b, methodName.toStdString(), cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,255), 2, cv::LINE_AA);

    auto* tile = new ThumbnailLabel(i, dlg);
    tile->setFixedSize(thumbW, thumbH);
    tile->setPixmap(QPixmap::fromImage(matToQImage(b)));
    tile->setScaledContents(true);
    tile->setToolTip(methodName + "\nDouble-click to use this method");
    tiles->push_back(tile);
    tile->onClicked = [tiles, tile](int) {
      for (auto* t : *tiles) if (t) t->setSelected(false);
      if (tile) tile->setSelected(true);
    };
    tile->onDoubleClick = [this, dlg](int idx) {
      if (cbLocalMethod_ && idx >= 0 && idx < cbLocalMethod_->count()) cbLocalMethod_->setCurrentIndex(idx);
      dlg->accept();
    };
    grid->addWidget(tile, i / cols, i % cols);
  }

  if (cbLocalMethod_) {
    int idx = cbLocalMethod_->findText(oldMethod);
    if (idx >= 0) cbLocalMethod_->setCurrentIndex(idx);
  }
  if (cbThreshType_) cbThreshType_->setCurrentIndex(oldType);

  outer->addLayout(grid);
  dlg->exec();
}

void MainWindow::onPreprocessAuto() {
  cv::Mat frame;
  {
    QMutexLocker locker(&frames_mutex_);
    for (const auto& f : last_frames_) {
      if (!f.empty()) { frame = f; break; }
    }
  }
  if (frame.empty()) {
    QMessageBox::information(this, "Auto", "No image available for auto adjustment.");
    return;
  }

  cv::Mat gray;
  if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  else gray = frame;

  const int histSize = 256;
  float range[] = {0, 256};
  const float* histRange = {range};
  cv::Mat hist;
  cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

  const double total = static_cast<double>(gray.total());
  const double clip = total * 0.0035; // small clipping, similar to ImageJ Auto

  int lo = 0, hi = 255;
  double acc = 0.0;
  for (int i = 0; i < 256; ++i) {
    acc += hist.at<float>(i);
    if (acc >= clip) { lo = i; break; }
  }
  acc = 0.0;
  for (int i = 255; i >= 0; --i) {
    acc += hist.at<float>(i);
    if (acc >= clip) { hi = i; break; }
  }
  if (hi <= lo) { lo = 0; hi = 255; }

  // Basic brightness/contrast model: dst = alpha * src + beta
  const double alpha = 255.0 / std::max(1, hi - lo);
  const double beta = -alpha * lo;

  int contrast = (int)std::round(alpha * 128.0);
  int brightness = (int)std::round(beta + 128.0);
  contrast = std::max(0, std::min(255, contrast));
  brightness = std::max(0, std::min(255, brightness));

  if (slContrast_) slContrast_->setValue(contrast);
  if (slBrightness_) slBrightness_->setValue(brightness);

  logLine(QString("Auto brightness/contrast: lo=%1 hi=%2 => alpha=%3 beta=%4, contrast=%5 brightness=%6")
          .arg(lo).arg(hi)
          .arg(alpha, 0, 'f', 4)
          .arg(beta, 0, 'f', 4)
          .arg(contrast)
          .arg(brightness));
}

void MainWindow::updateScaleStatus(double pxLen) {
  if (!lblScaleInfo_) return;
  if (mm_per_pixel_ > 0.0) {
    lblScaleInfo_->setText(QString("Scale: %1 mm/px (line: %2 px, %3 mm)")
                           .arg(mm_per_pixel_,0,'f',9)
                           .arg(pxLen,0,'f',2)
                           .arg(pxLen * mm_per_pixel_,0,'f',4));
  } else {
    lblScaleInfo_->setText(QString("Scale: line length %1 px (double click line to set real distance)").arg(pxLen,0,'f',2));
  }
}

bool MainWindow::readFrames(std::vector<cv::Mat>& frames) {
  frames.clear();
  QMutexLocker locker(&frames_mutex_);
  if (last_frames_.empty()) return false;
  for (const auto& f : last_frames_) {
    if (f.empty()) return false;
    frames.push_back(f);
  }
  return !frames.empty();
}

QImage MainWindow::matToQImage(const cv::Mat& bgr) {
  cv::Mat rgb;
  if (bgr.channels()==3) cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  else if (bgr.channels()==4) cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGBA);
  else cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);

  return QImage((const uchar*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
}

cv::Mat MainWindow::makeMosaic(const std::vector<cv::Mat>& imgs, int cols) {
  if (imgs.empty()) return cv::Mat();
  cols = std::max(1, cols);
  int rows = (int)std::ceil((double)imgs.size() / (double)cols);

  // Find first non-empty to determine cell size
  int cell_w = 0, cell_h = 0;
  for (const auto& im : imgs) {
    if (!im.empty()) { cell_w = im.cols; cell_h = im.rows; break; }
  }
  if (cell_w <= 0 || cell_h <= 0) return cv::Mat();

  cv::Mat mosaic(cell_h * rows, cell_w * cols, CV_8UC3, cv::Scalar(16,16,16));

  for (int i = 0; i < (int)imgs.size(); ++i) {
    int r = i / cols;
    int c = i % cols;
    cv::Rect roi(c * cell_w, r * cell_h, cell_w, cell_h);

    cv::Mat src = imgs[i];
    cv::Mat rgb;
    if (src.empty()) {
      rgb = cv::Mat(cell_h, cell_w, CV_8UC3, cv::Scalar(16,16,16));
    } else {
      if (src.channels() == 1) cv::cvtColor(src, rgb, cv::COLOR_GRAY2BGR);
      else if (src.channels() == 3) rgb = src;
      else cv::cvtColor(src, rgb, cv::COLOR_BGRA2BGR);

      if (rgb.cols != cell_w || rgb.rows != cell_h) {
        cv::resize(rgb, rgb, cv::Size(cell_w, cell_h), 0, 0, cv::INTER_AREA);
      }
    }

    rgb.copyTo(mosaic(roi));
  }
  return mosaic;
}

void MainWindow::overlayCalibration(std::vector<cv::Mat>& vis, const std::vector<cv::Mat>& frames) {
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<bool> ok;
  calibrator_->detectAndMaybeStore(frames, false, &corners, &ok);

  for (int i=0;i<num_cams_;++i) {
    if (ok[i]) {
      cv::drawChessboardCorners(vis[i], cv::Size(board_w_, board_h_), corners[i], true);
    }
    cv::putText(vis[i], "cam"+std::to_string(i), cv::Point(15,30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2, cv::LINE_AA);
  }
}

void MainWindow::overlayTracking(std::vector<cv::Mat>& vis, const std::vector<cv::Mat>& frames) {
  AprilTagDetections det;
  std::vector<Observation> obs;

  if (tagmap_loaded_) {
    buildObservationsFromFrames(frames, tag_corner_map_, obs, &det, tag_dict_id_);
    for (int i=0;i<num_cams_;++i) {
      if (i < (int)det.ids_per_cam.size())
        cv::aruco::drawDetectedMarkers(vis[i], det.corners_per_cam[i], det.ids_per_cam[i]);
      cv::putText(vis[i], "cam"+std::to_string(i), cv::Point(15,30),
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2, cv::LINE_AA);
    }

  } else {
    for (int i=0;i<num_cams_;++i) {
      cv::putText(vis[i], "Load tag map to start tracking", cv::Point(15,60),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2, cv::LINE_AA);
    }
  }
}

cv::Mat MainWindow::applyPreprocess(const cv::Mat& src) const {
  if (src.empty()) return src;
  cv::Mat out = src.clone();

  if (cbPreColor_ && cbPreColor_->currentIndex() == 0) {
    cv::Mat gray;
    cv::cvtColor(out, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, out, cv::COLOR_GRAY2BGR);
  }

  const int brightness = slBrightness_ ? slBrightness_->value() : 128;
  const int contrastSlider = slContrast_ ? slContrast_->value() : 128;
  // Basic formula: dst = alpha * src + beta, slider range [0,255]
  const double alpha = static_cast<double>(contrastSlider) / 128.0; // 128 => neutral
  const double beta = static_cast<double>(brightness - 128);        // 128 => neutral
  out.convertTo(out, -1, alpha, beta);
  return out;
}

void MainWindow::updateStatus() {
  updateStepAvailability();
  if (lblCaptured_) lblCaptured_->setText(QString("Captured: %1").arg(calibrator_->captured()));
  if (lblInliers_) lblInliers_->setText(QString("Inliers: %1").arg(last_inliers_));

  QString m = "Source";
  if (stepTabs_) {
    const int si = tabIndexToStep(stepTabs_->currentIndex());
    if (si == 1) m = "PreProcess";
    else if (si == 2) m = "ObjectDefine";
    else if (si == 3) m = "Visual";
  }
  statusBar()->showMessage(QString("Mode: %1 | Source: %2 | Captured: %3 | Inliers: %4 | FPS: %5")
    .arg(m)
    .arg((int)sources_.size())
    .arg(calibrator_->captured())
    .arg(last_inliers_)
    .arg(fps_, 0, 'f', 1));
}

void MainWindow::onTick() {
  // UI refresh at a steady rate; frames arrive from CaptureWorker
  if (sources_.empty()) {
    updateSourceViews(std::vector<cv::Mat>());
    updateStatus();
    return;
  }
  std::vector<cv::Mat> frames;
  {
    QMutexLocker locker(&frames_mutex_);
    if (last_frames_.empty()) return;
    frames = last_frames_;
  }

  // Note: some sources may not have frames yet; mosaic will show placeholders.

  std::vector<cv::Mat> vis = frames;
  for (auto& f : vis) {
    if (!f.empty()) f = applyPreprocess(f);
  }
  std::vector<cv::Mat> preVis = vis;
  // If sources count changed, rebuild calibrator to match to avoid crash
  if (mode_==CALIB) {
    int n = (int)frames.size();
    if (n <= 0) { updateStatus(); return; }
    if (num_cams_ != n) {
      num_cams_ = n;
      calibrator_.reset(new MultiCamCalibrator(num_cams_, cv::Size(board_w_, board_h_), square_));
      //logLine(QString(\"Sources changed -> rebuild calibrator (num=%1)\").arg(num_cams_));
    }
  }
  // Tracking overlay is triggered by the explicit Detect button to avoid
  // repeatedly accumulating visual detections on static frames.

  if (track_binary_enabled_) {
    const float maxDist = 60.0f;
    for (size_t i = 0; i < vis.size(); ++i) {
      if (vis[i].empty()) continue;
      auto contours = detectBinaryContours(vis[i]);
      std::unordered_map<int, cv::Point2f> newCenters;
      std::unordered_map<int, std::vector<cv::Point>> newContours;
      for (const auto& c : contours) {
        if (c.empty()) continue;
        cv::Moments mm = cv::moments(c);
        if (std::abs(mm.m00) < 1e-6) continue;
        cv::Point2f ctr((float)(mm.m10/mm.m00), (float)(mm.m01/mm.m00));
        int bestId = -1; float bestD = maxDist;
        for (const auto& kv : tracked_centroids_) {
          float d = cv::norm(ctr - kv.second);
          if (d < bestD) { bestD = d; bestId = kv.first; }
        }
        if (bestId < 0) bestId = next_track_id_++;
        newCenters[bestId] = ctr;
        newContours[bestId] = c;
      }
      tracked_centroids_ = std::move(newCenters);
      tracked_contours_ = std::move(newContours);
      for (const auto& kv : tracked_contours_) {
        cv::drawContours(vis[i], std::vector<std::vector<cv::Point>>{kv.second}, -1, cv::Scalar(0,255,0), 2);
        auto itc = tracked_centroids_.find(kv.first);
        if (itc != tracked_centroids_.end()) {
          cv::putText(vis[i], std::string("ID:")+std::to_string(kv.first), itc->second + cv::Point2f(3.f,-3.f),
                      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        }
      }
    }
  } else if (!analyzed_contours_.empty()) {
    for (auto& f : vis) {
      if (f.empty()) continue;
      cv::drawContours(f, analyzed_contours_, -1, cv::Scalar(0,255,0), 2);
    }
  }

  updateSourceViews(vis);

  for (const auto& f : preVis) {
    if (f.empty()) continue;
    if (!measurements_frozen_) updateMeasurementFromFrame(f);
    if (visImageLabel_) {
      QImage qi = matToQImage(f);
      visImageLabel_->setPixmap(QPixmap::fromImage(qi).scaled(visImageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    if (leftVisImage_) {
      QImage qi = matToQImage(f);
      leftVisImage_->setPixmap(QPixmap::fromImage(qi).scaled(leftVisImage_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    break;
  }
  refreshTrajectoryPlot();
  updateLeftVisualDashboard();

  // ObjectDefine small binary preview
  if (lblBinaryPreview_) {
    for (const auto& f : preVis) {
      if (f.empty()) continue;
      int autoT = spObjectThresh_ ? spObjectThresh_->value() : 128;
      cv::Mat bin = makeObjectBinaryPreview(f, &autoT);
      if (!bin.empty()) {
        if (!object_thresh_manual_ && slObjectThresh_ && cbThreshType_ && cbThreshType_->currentIndex()==0) {
          if (slObjectThresh_->value() != autoT) slObjectThresh_->setValue(autoT);
          if (spObjectThresh_ && spObjectThresh_->value() != autoT) spObjectThresh_->setValue(autoT);
        }
        QImage qi = matToQImage(bin);
        lblBinaryPreview_->setPixmap(QPixmap::fromImage(qi).scaled(lblBinaryPreview_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
      }
      break;
    }
  }

  if (show_docks_) updateSourceDocks(frames);
  updateStatus();
}

void MainWindow::updateFpsStats(double dt_ms) {
  if (dt_ms <= 0.0) return;
  double inst = 1000.0 / dt_ms;
  // simple low-pass filter
  fps_ = (fps_<=0.0) ? inst : (0.9*fps_ + 0.1*inst);
  if (lblFps_) lblFps_->setText(QString("FPS: %1").arg(fps_, 0, 'f', 1));
}

void MainWindow::setSourceEnabled(int idx, bool enabled) {
  {
    QMutexLocker locker(&sources_mutex_);
    if (idx < 0 || idx >= (int)source_enabled_.size()) return;
    source_enabled_[idx] = enabled;
  }
  refreshSourceList();
}

QJsonObject MainWindow::toProjectJson() const {
  QJsonObject o;
  o["board_w"] = board_w_;
  o["board_h"] = board_h_;
  o["square"] = square_;
  o["ransac_iters"] = ransac_iters_;
  o["inlier_thresh_px"] = inlier_thresh_px_;
  o["tag_dict_id"] = tag_dict_id_;

  QJsonArray srcs;
  for (int i=0;i<(int)sources_.size();++i) {
    const auto& s = sources_[i];
    QJsonObject si;
    si["enabled"] = (i < (int)source_enabled_.size()) ? source_enabled_[i] : true;
    si["mode_owner"] = s.mode_owner;
    if (s.is_cam) {
      si["type"] = "cam";
      si["cam_id"] = s.cam_id;
    } else if (s.is_image_seq) {
      si["type"] = "imgseq";
      si["dir"] = s.seq_dir;
    } else {
      si["type"] = "video";
      si["path"] = s.video_path;
    }
    srcs.append(si);
  }
  o["sources"] = srcs;

  // ObjectDefine settings persistence
  if (cbThreshType_) o["obj_thresh_type"] = cbThreshType_->currentIndex();
  if (cbGlobalMethod_) o["obj_global_method"] = cbGlobalMethod_->currentText();
  if (spObjectThresh_) o["obj_threshold"] = spObjectThresh_->value();
  if (chkInvertBinary_) o["obj_invert"] = chkInvertBinary_->isChecked();
  {
    QJsonArray ops;
    for (const auto& op : binary_ops_pipeline_) ops.append(op);
    o["obj_binary_ops"] = ops;
  }
  o["obj_track_enabled"] = track_binary_enabled_;

  o["tagmap_path"] = tagmap_path_;
  o["calib_path"] = calib_path_;
  o["layout_geometry_b64"] = QString(saveGeometry().toBase64());
  o["layout_state_b64"] = QString(saveState().toBase64());
  return o;
}

bool MainWindow::fromProjectJson(const QJsonObject& o) {
  if (o.contains("board_w") && spBoardW_) spBoardW_->setValue(o["board_w"].toInt(board_w_));
  if (o.contains("board_h") && spBoardH_) spBoardH_->setValue(o["board_h"].toInt(board_h_));
  if (o.contains("square") && spSquare_) spSquare_->setValue(o["square"].toDouble(square_));
  if (o.contains("ransac_iters") && spRansacIters_) spRansacIters_->setValue(o["ransac_iters"].toInt(ransac_iters_));
  if (o.contains("inlier_thresh_px") && spInlierThresh_) spInlierThresh_->setValue(o["inlier_thresh_px"].toDouble(inlier_thresh_px_));
  if (o.contains("tag_dict_id") && cbTagDict_) {
    int did = o["tag_dict_id"].toInt(tag_dict_id_);
    for (int i=0;i<cbTagDict_->count();++i) {
      if (cbTagDict_->itemData(i).toInt() == did) { cbTagDict_->setCurrentIndex(i); break; }
    }
  }

  if (o.contains("obj_thresh_type") && cbThreshType_) cbThreshType_->setCurrentIndex(o["obj_thresh_type"].toInt(cbThreshType_->currentIndex()));
  if (o.contains("obj_global_method") && cbGlobalMethod_) {
    int idx = cbGlobalMethod_->findText(o["obj_global_method"].toString());
    if (idx >= 0) cbGlobalMethod_->setCurrentIndex(idx);
  }
  if (o.contains("obj_threshold")) {
    int t = o["obj_threshold"].toInt(spObjectThresh_ ? spObjectThresh_->value() : 128);
    if (spObjectThresh_) spObjectThresh_->setValue(t);
    if (slObjectThresh_) slObjectThresh_->setValue(t);
    object_thresh_manual_ = true;
  }
  if (o.contains("obj_invert") && chkInvertBinary_) chkInvertBinary_->setChecked(o["obj_invert"].toBool(false));
  binary_ops_pipeline_.clear();
  if (o.contains("obj_binary_ops") && o["obj_binary_ops"].isArray()) {
    for (const auto& v : o["obj_binary_ops"].toArray()) {
      if (v.isString()) binary_ops_pipeline_.push_back(v.toString());
    }
  }
  if (lblBinaryOps_) {
    if (binary_ops_pipeline_.empty()) lblBinaryOps_->setText("Pipeline: (none)");
    else { QStringList ops; for (const auto& op : binary_ops_pipeline_) ops << op; lblBinaryOps_->setText(QString("Pipeline: %1").arg(ops.join(" -> "))); }
  }
  track_binary_enabled_ = o["obj_track_enabled"].toBool(false);
  if (btnTrackBinary_) btnTrackBinary_->setChecked(track_binary_enabled_);

  // Rebuild sources
  timer_.stop();
  closeAllSources();
  sources_.clear();
  source_enabled_.clear();
  last_frames_.clear();

  if (o.contains("sources") && o["sources"].isArray()) {
    QJsonArray srcs = o["sources"].toArray();
    for (auto v : srcs) {
      if (!v.isObject()) continue;
      QJsonObject si = v.toObject();
      QString type = si["type"].toString();
      bool en = si["enabled"].toBool(true);
      int owner = si["mode_owner"].toInt(0);

      InputSource s;
      if (type == "cam") {
        s.is_cam = true;
        s.mode_owner = owner;
        s.cam_id = si["cam_id"].toInt(0);
        s.cap.open(s.cam_id);
      } else if (type == "video") {
        s.is_cam = false;
        s.mode_owner = owner;
        s.is_image_seq = false;
        s.video_path = si["path"].toString();
        openVideoCaptureUnicode(s.cap, s.video_path);
      } else if (type == "imgseq") {
        s.is_cam = false;
        s.mode_owner = owner;
        s.is_image_seq = true;
        s.seq_dir = si["dir"].toString();
        s.video_path = s.seq_dir;
        QDir qdir(s.seq_dir);
        QFileInfoList files = qdir.entryInfoList(kImageNameFilters(), QDir::Files, QDir::Name);
        for (const auto& fi : files) s.seq_files.push_back(fi.absoluteFilePath());
        s.seq_idx = 0;
      } else continue;

      sources_.push_back(std::move(s));
      source_enabled_.push_back(en);
      last_frames_.emplace_back();
    }
  }

  num_cams_ = (int)sources_.size();
  source_enabled_.assign(std::max(0,num_cams_), true);
  // last_frames_ will be populated by CaptureWorker when frames arrive.
  last_frames_.resize(std::max(0,num_cams_));
  calibrator_.reset(new MultiCamCalibrator(std::max(1,num_cams_), cv::Size(board_w_, board_h_), square_));
  refreshSourceList();
  rebuildSourceViews();

  // Bind file paths
  if (o.contains("tagmap_path")) {
    tagmap_path_ = o["tagmap_path"].toString();
    if (!tagmap_path_.isEmpty()) {
      if (QFile::exists(tagmap_path_)) {
        tagmap_loaded_ = loadTagCornersTxt(tagmap_path_.toStdString(), tag_corner_map_);
      } else {
        QMessageBox::warning(this, "Project", "Tag map file missing. Please locate it.");
        QString p = QFileDialog::getOpenFileName(this, "Locate Tag Map TXT", "", "Text (*.txt);;All (*.*)");
        if (!p.isEmpty()) { tagmap_path_=p; tagmap_loaded_=loadTagCornersTxt(p.toStdString(), tag_corner_map_); }
      }
      if (lblTagPath_) lblTagPath_->setText(QString("TagMap: %1").arg(tagmap_path_));
    }
  }
  if (o.contains("calib_path")) {
    calib_path_ = o["calib_path"].toString();
    if (!calib_path_.isEmpty()) {
      if (QFile::exists(calib_path_)) {
        calib_loaded_ = loadRigCalibYaml(calib_path_.toStdString(), cams_);
      } else {
        QMessageBox::warning(this, "Project", "Calibration YAML missing. Please locate it.");
        QString p = QFileDialog::getOpenFileName(this, "Locate Calibration YAML", "", "YAML (*.yaml *.yml);;All (*.*)");
        if (!p.isEmpty()) { calib_path_=p; calib_loaded_=loadRigCalibYaml(p.toStdString(), cams_); }
      }
      if (lblYamlPath_) lblYamlPath_->setText(QString("Calib: %1").arg(calib_path_));
    }
  }
  if (solveWorker_) solveWorker_->setStaticData(&cams_, &tag_corner_map_);

  // Restore layout if present
  if (o.contains("layout_geometry_b64")) {
    QByteArray g = QByteArray::fromBase64(o["layout_geometry_b64"].toString().toUtf8());
    if (!g.isEmpty()) restoreGeometry(g);
  }
  if (o.contains("layout_state_b64")) {
    QByteArray s = QByteArray::fromBase64(o["layout_state_b64"].toString().toUtf8());
    if (!s.isEmpty()) restoreState(s);
  }

  return true;
}

// ----------------- Calibration actions -----------------
void MainWindow::onGrabFrame() {
  std::vector<cv::Mat> frames;
  if (!readFrames(frames)) return;

  bool ok = calibrator_->detectAndMaybeStore(frames, true);
  logLine(ok ? "Grabbed chessboard frame." : "No chessboard detected.");
  updateStatus();
}

void MainWindow::onResetFrames() {
  calibrator_->reset();
  calib_pairs_.clear();
  calib_pair_rmse_.clear();
  has_computed_calib_ = false;
  if (btnSaveCalib_) btnSaveCalib_->setEnabled(false);
  if (calibErrorTable_) calibErrorTable_->setRowCount(0);
  if (calibProgressBar_) calibProgressBar_->setValue(0);
  if (lblCalibProgress_) lblCalibProgress_->setText("Progress: idle");
  logLine("Reset captured frames.");
  updateStatus();
}

void MainWindow::onComputeCalibration() {
  QMessageBox::information(this, "Preprocess", "Calibration solving is removed. Use this tab for preprocessing only.");
  logLine("Calibration solver removed in monocular preprocess mode.");
}

bool MainWindow::runCalibrationOnPairs(const std::vector<int>& pairIndices, bool updateTable) {
  if (pairIndices.empty()) {
    QMessageBox::warning(this, "Calibration", "No frame selected for calibration.");
    return false;
  }

  if (calibProgressBar_) {
    calibProgressBar_->setRange(0, (int)pairIndices.size());
    calibProgressBar_->setValue(0);
  }
  if (lblCalibProgress_) lblCalibProgress_->setText("Progress: detecting chessboard...");

  MultiCamCalibrator workCalib(2, cv::Size(board_w_, board_h_), square_);
  std::vector<cv::Size> sizes;
  bool sizesSet = false;
  int usedPairs = 0;
  std::vector<int> acceptedPairIds;
  for (int i = 0; i < (int)pairIndices.size(); ++i) {
    int id = pairIndices[i];
    if (id < 0 || id >= (int)calib_pairs_.size()) continue;
    const auto& p = calib_pairs_[id];
    std::vector<cv::Mat> pair = {p.left, p.right};
    if (!sizesSet) {
      sizes = {p.left.size(), p.right.size()};
      sizesSet = true;
    }
    if (workCalib.detectAndMaybeStore(pair, true)) {
      usedPairs++;
      acceptedPairIds.push_back(id);
    }
    if (calibProgressBar_) calibProgressBar_->setValue(i + 1);
    QApplication::processEvents();
  }

  if (usedPairs <= 0 || !sizesSet) {
    QMessageBox::warning(this, "Calibration", "No valid chessboard pairs found in selected frames.");
    return false;
  }

  if (lblCalibProgress_) lblCalibProgress_->setText("Progress: solving calibration...");
  double rms=0.0;
  std::vector<CameraModel> out;
  if (!workCalib.calibrate(sizes, out, rms)) {
    QMessageBox::warning(this, "Calibration", "Calibration failed. Ensure enough captures and paired frames with cam0.");
    logLine("Calibration failed.");
    return false;
  }

  std::vector<double> selectedRmse;
  workCalib.computeFrameReprojErrors(out, selectedRmse);
  calib_pair_rmse_.assign(calib_pairs_.size(), -1.0);
  for (int i = 0; i < (int)acceptedPairIds.size() && i < (int)selectedRmse.size(); ++i) {
    int id = acceptedPairIds[i];
    if (id >= 0 && id < (int)calib_pair_rmse_.size()) calib_pair_rmse_[id] = selectedRmse[i];
  }

  if (updateTable && calibErrorTable_) {
    for (int row = 0; row < calibErrorTable_->rowCount() && row < (int)calib_pair_rmse_.size(); ++row) {
      const double e = calib_pair_rmse_[row];
      calibErrorTable_->setItem(row, 2, new QTableWidgetItem(e >= 0.0 ? QString::number(e, 'f', 3) : "N/A"));
    }
  }

  cams_ = out;
  calib_loaded_ = true;
  has_computed_calib_ = true;
  if (btnSaveCalib_) btnSaveCalib_->setEnabled(true);
  if (solveWorker_) solveWorker_->setStaticData(&cams_, &tag_corner_map_);
  auto qualityText = [](double val)->QString {
    if (val < 0.5) return "Excellent";
    if (val < 1.0) return "Good";
    if (val < 2.0) return "Fair";
    return "Poor";
  };
  const QString quality = qualityText(rms);
  if (lblCalibProgress_) {
    lblCalibProgress_->setText(QString("Progress: done, mean RMS=%1 (%2)").arg(rms, 0, 'f', 4).arg(quality));
  }
  logLine(QString("Calibration OK. mean RMS=%1 (%2)").arg(rms, 0, 'f', 4).arg(quality));
  return true;
}

void MainWindow::onRecomputeCalibrationSelected() {
  if (calib_pairs_.empty() || !calibErrorTable_) {
    QMessageBox::information(this, "Calibration", "Please run Compute Calibration first.");
    return;
  }

  std::vector<int> selected;
  for (int row = 0; row < calibErrorTable_->rowCount(); ++row) {
    QTableWidgetItem* item = calibErrorTable_->item(row, 0);
    if (item && item->checkState() == Qt::Checked) selected.push_back(row);
  }
  runCalibrationOnPairs(selected, true);
}

void MainWindow::onSaveCalibrationYaml() {
  if (!has_computed_calib_ || cams_.empty()) {
    QMessageBox::information(this, "Save", "Please compute calibration first.");
    return;
  }

  QString savePath = QFileDialog::getSaveFileName(this, "Save rig_calib.yaml", "rig_calib.yaml", "YAML (*.yaml *.yml)");
  if (savePath.isEmpty()) return;

  calib_path_ = savePath;
  if (lblYamlPath_) lblYamlPath_->setText(QString("Calib: %1").arg(calib_path_));
  if (!saveRigCalibYaml(savePath.toStdString(), cams_)) {
    QMessageBox::warning(this, "Save", "Failed to save YAML.");
    return;
  }
  logLine(QString("Calibration YAML saved: %1").arg(savePath));
  settings_.setValue("lastCalibYaml", savePath);
}

// ----------------- Tracking actions -----------------
void MainWindow::onLoadTagMap() {
  QString last = settings_.value("lastTagMap", "tag_corners_world.txt").toString();
  QString path = QFileDialog::getOpenFileName(this, "Load tag map TXT", last, "Text (*.txt);;All (*.*)");
  if (path.isEmpty()) return;

  tagmap_path_ = path;
  tagmap_loaded_ = loadTagCornersTxt(path.toStdString(), tag_corner_map_);
  if (lblTagPath_) lblTagPath_->setText(QString("TagMap: %1").arg(tagmap_path_));
  if (!tagmap_loaded_) {
    QMessageBox::warning(this, "Tag map", "Failed to load tag map TXT.");
    logLine("Failed to load tag map.");
    return;
  }
  settings_.setValue("lastTagMap", path);
  logLine(QString("Loaded tag map: %1").arg(path));
}

void MainWindow::onLoadCalibYaml() {
  QString last = settings_.value("lastCalibYaml", "rig_calib.yaml").toString();
  QString path = QFileDialog::getOpenFileName(this, "Load calibration YAML", last, "YAML (*.yaml *.yml);;All (*.*)");
  if (path.isEmpty()) return;

  calib_path_ = path;
  calib_loaded_ = loadRigCalibYaml(path.toStdString(), cams_);
  if (lblYamlPath_) lblYamlPath_->setText(QString("Calib: %1").arg(calib_path_));
  if (solveWorker_) solveWorker_->setStaticData(&cams_, &tag_corner_map_);
  if (!calib_loaded_) {
    QMessageBox::warning(this, "Calibration", "Failed to load calibration YAML.");
    logLine("Failed to load calibration yaml.");
    return;
  }
  settings_.setValue("lastCalibYaml", path);
  logLine(QString("Loaded calib yaml: %1").arg(path));
}

//void MainWindow::onTogglePose(bool on) {
//  pose_on_ = on;
//  if (solveWorker_) solveWorker_->setParams(ransac_iters_, inlier_thresh_px_, tag_dict_id_, pose_on_);
//  logLine(QString("Pose estimation %1").arg(on ? "ON" : "OFF"));
//}

void MainWindow::onDetectAllTrackingFrames() {
  QMessageBox::information(this, "Tracking", "Measurement algorithm is removed in this monocular version.");
  logLine("Detect/Pose algorithm removed in monocular version.");
}

// ----------------- Sources: pause/resume -----------------
void MainWindow::onPauseResumeSelected() {
  bool toEnable = true;
  {
    QMutexLocker locker(&sources_mutex_);
    if ((int)source_enabled_.size() != (int)sources_.size()) source_enabled_.assign(sources_.size(), true);
    bool anyPaused = false;
    for (bool en : source_enabled_) if (!en) { anyPaused = true; break; }
    toEnable = anyPaused;
    for (int i=0;i<(int)source_enabled_.size();++i) source_enabled_[i] = toEnable;
  }
  logLine(QString("All sources %1").arg(toEnable ? "RESUMED" : "PAUSED"));
  refreshSourceList();
}

// ----------------- Tracking: export trajectory -----------------
void MainWindow::onExportTrajectory() {
  QMessageBox::information(this, "Export", "Trajectory export is unavailable because measurement algorithm is removed.");
}


// ----------------- Project config: save/load -----------------
void MainWindow::onSaveProject() {
  QString path = QFileDialog::getSaveFileName(this, "Save Project Config", "project.json", "JSON (*.json)");
  if (path.isEmpty()) return;

  QJsonObject o = toProjectJson();
  QJsonDocument doc(o);

  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
    QMessageBox::warning(this, "Save Project", "Failed to write project file.");
    return;
  }
  f.write(doc.toJson(QJsonDocument::Indented));
  f.close();
  logLine(QString("Project saved: %1").arg(path));
}

void MainWindow::onLoadProject() {
  QString path = QFileDialog::getOpenFileName(this, "Load Project Config", "", "JSON (*.json)");
  if (path.isEmpty()) return;

  QFile f(path);
  if (!f.open(QIODevice::ReadOnly)) {
    QMessageBox::warning(this, "Load Project", "Failed to open project file.");
    return;
  }
  QByteArray data = f.readAll();
  f.close();

  QJsonParseError err;
  QJsonDocument doc = QJsonDocument::fromJson(data, &err);
  if (err.error != QJsonParseError::NoError || !doc.isObject()) {
    QMessageBox::warning(this, "Load Project", "Invalid JSON.");
    return;
  }
  fromProjectJson(doc.object());
  logLine(QString("Project loaded: %1").arg(path));
}


void MainWindow::onFramesFromWorker(FramePack frames, qint64 capture_ts_ms) {
  last_capture_ts_ms_ = capture_ts_ms;
  {
    QMutexLocker locker(&frames_mutex_);
    last_frames_ = std::move(frames); // cv::Mat ref-counted
  }

  int64_t framePos = play_frame_;
  int64_t frameEnd = play_end_frame_;
  {
    QMutexLocker srcLock(&sources_mutex_);
    for (const auto& s : sources_) {
      if (s.is_cam || s.mode_owner!=(int)mode_) continue;
      if (s.is_image_seq) {
        framePos = std::max<int64_t>(0, s.seq_idx);
        if (frameEnd <= 0) frameEnd = (int64_t)s.seq_files.size();
        break;
      }
      if (!s.cap.isOpened()) continue;
      framePos = std::max<int64_t>(0, (int64_t)std::llround(s.cap.get(cv::CAP_PROP_POS_FRAMES)) - 1);
      double fc = s.cap.get(cv::CAP_PROP_FRAME_COUNT);
      if (frameEnd <= 0 && fc > 0) frameEnd = (int64_t)fc;
      break;
    }
  }
  play_frame_ = framePos;
  if (play_end_frame_ <= 0 && frameEnd > 0) play_end_frame_ = frameEnd;
  updateProgressUI(play_frame_, play_end_frame_);

  // forward to solver thread (queued)
  if (solveWorker_) {
    // handled via signal/slot wiring in constructor
  }
}

void MainWindow::onAddVisualizationChart() {
  if (!visChartsLayout_) return;

  QVector<int> comps;
  QString ylabel;
  if (!chooseVisualizationDataTypes(comps, ylabel)) return;

  QCustomPlot* plot = new QCustomPlot(actionTabs_);
  plot->setMinimumHeight(160);
  plot->setStyleSheet("background:#1d232b;border:1px solid #3a4250;color:#9fb0c4;");
  plot->xAxis->setLabel("t");
  plot->yAxis->setLabel(ylabel);
  for (int i=0;i<comps.size()+1;++i) plot->addGraph();
  plot->legend->setVisible(false);
  plot->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(plot, &QWidget::customContextMenuRequested, this, &MainWindow::onVisualizationPlotContextMenu);

  QComboBox* selector = new QComboBox(actionTabs_);
  selector->addItem(QString::fromUtf8("all"));
  selector->setToolTip(QString::fromUtf8("choose graph to show"));
  connect(selector, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int){ refreshTrajectoryPlot(); });

  int insertAt = std::max(1, visChartsLayout_->count()-1);
  visChartsLayout_->insertWidget(insertAt, selector);
  visChartsLayout_->insertWidget(insertAt+1, plot);
  visCharts_.push_back({plot, selector, comps, ylabel, true});
  refreshTrajectoryPlot();
}

bool MainWindow::chooseVisualizationDataTypes(QVector<int>& components, QString& labelOut) {
  QDialog dlg(this);
  dlg.setWindowTitle(QString::fromUtf8("choose data type"));
  QVBoxLayout* layout = new QVBoxLayout(&dlg);

  QCheckBox* c[6];
  const QString names[6] = {"x", "y", "z", "aa_x", "aa_y", "aa_z"};
  for (int i=0;i<6;++i) {
    c[i] = new QCheckBox(names[i], &dlg);
    if (i < 3) c[i]->setChecked(true);
    layout->addWidget(c[i]);
  }

  QDialogButtonBox* box = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dlg);
  connect(box, &QDialogButtonBox::accepted, &dlg, &QDialog::accept);
  connect(box, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
  layout->addWidget(box);

  if (dlg.exec() != QDialog::Accepted) return false;

  components.clear();
  QStringList labels;
  for (int i=0;i<6;++i) {
    if (c[i]->isChecked()) {
      components.push_back(i);
      labels << names[i];
    }
  }
  if (components.isEmpty()) return false;

  labelOut = labels.join("/");
  return true;
}

void MainWindow::onVisualizationPlotContextMenu(const QPoint& pos) {
  auto* plot = qobject_cast<QCustomPlot*>(sender());
  if (!plot) return;

  int idx = -1;
  for (int i=0;i<(int)visCharts_.size(); ++i) {
    if (visCharts_[i].plot == plot) { idx = i; break; }
  }
  if (idx < 0) return;

  QMenu menu(this);
  QAction* actSelect = menu.addAction(QString::fromUtf8("选择数据类型"));
  QAction* actRemove = nullptr;
  if (visCharts_[idx].removable) {
    actRemove = menu.addAction(QString::fromUtf8("删除图表"));
  }

  QAction* picked = menu.exec(plot->mapToGlobal(pos));
  if (!picked) return;

  if (picked == actSelect) {
    QVector<int> comps;
    QString ylabel;
    if (!chooseVisualizationDataTypes(comps, ylabel)) return;
    visCharts_[idx].components = comps;
    visCharts_[idx].yLabel = ylabel;
    plot->yAxis->setLabel(ylabel);

    while (plot->graphCount() > comps.size()+1) plot->removeGraph(plot->graphCount()-1);
    while (plot->graphCount() < comps.size()+1) plot->addGraph();
    refreshTrajectoryPlot();
    return;
  }

  if (actRemove && picked == actRemove) {
    QCustomPlot* p = visCharts_[idx].plot;
    QComboBox* s = visCharts_[idx].selector;
    visCharts_.erase(visCharts_.begin() + idx);
    if (visChartsLayout_) {
      if (s) visChartsLayout_->removeWidget(s);
      visChartsLayout_->removeWidget(p);
    }
    if (s) s->deleteLater();
    p->deleteLater();
    refreshTrajectoryPlot();
  }
}

void MainWindow::updateMeasurementFromFrame(const cv::Mat& preprocessedFrame) {
  if (preprocessedFrame.empty()) return;
  auto contours = detectBinaryContours(preprocessedFrame);
  if (contours.empty()) return;
  size_t best = 0;
  double bestA = 0.0;
  for (size_t i=0;i<contours.size();++i) {
    double a = std::abs(cv::contourArea(contours[i]));
    if (a > bestA) { bestA = a; best = i; }
  }
  const auto& c = contours[best];
  cv::Moments mm = cv::moments(c);
  if (std::abs(mm.m00) < 1e-9) return;
  cv::Point2f ctr((float)(mm.m10/mm.m00), (float)(mm.m01/mm.m00));
  double scale = (mm_per_pixel_ > 0.0) ? mm_per_pixel_ : 1.0;
  double area = std::abs(cv::contourArea(c)) * scale * scale;
  double perim = cv::arcLength(c, true) * scale;
  cv::RotatedRect rr = cv::minAreaRect(c);
  double major = std::max(rr.size.width, rr.size.height) * scale;
  double minor = std::min(rr.size.width, rr.size.height) * scale;
  double circ = (perim > 1e-9) ? (4.0 * std::acos(-1.0) * area / (perim * perim)) : 0.0;
  qint64 key = last_capture_ts_ms_;
  if ((progressSlider_ && progressSlider_->maximum() > 0) || play_end_frame_ > 0) {
    key = std::max<qint64>(0, play_frame_);
  }
  if (key == last_meas_key_) return;
  MeasureRow row;
  row.key = key;
  if (meas_rows_.empty()) {
    row.disp = 0.0;
    row.speed = 0.0;
    row.accel = 0.0;
  } else {
    row.disp = cv::norm(ctr - last_ctr_) * scale;
    row.speed = row.disp; // per-frame speed
    row.accel = row.speed - last_speed_;
  }
  row.area = area; row.perim = perim; row.major = major; row.minor = minor; row.circ = circ;
  last_ctr_ = ctr;
  last_speed_ = row.speed;
  last_meas_key_ = key;
  meas_rows_.push_back(row);
}

void MainWindow::updateHistogramPlot() {
  if (!plotHistogram_ || !cbHistMetric_) return;
  if (meas_rows_.empty()) { plotHistogram_->clearPlottables(); plotHistogram_->replot(); return; }
  auto pick = [&](const MeasureRow& r)->double {
    const QString m = cbHistMetric_->currentText();
    if (m == "Perimeter") return r.perim;
    if (m == "Circularity") return r.circ;
    if (m == "Speed") return r.speed;
    if (m == "Displacement") return r.disp;
    if (m == "MajorAxis") return r.major;
    if (m == "MinorAxis") return r.minor;
    return r.area;
  };
  double lo = spHistMin_ ? spHistMin_->value() : -1e9;
  double hi = spHistMax_ ? spHistMax_->value() : 1e9;
  if (hi <= lo) hi = lo + 1.0;
  const int bins = 20;
  QVector<double> x(bins), y(bins);
  for (int i=0;i<bins;++i) { x[i]=i; y[i]=0; }
  for (const auto& r : meas_rows_) {
    double v = pick(r);
    if (v < lo || v > hi) continue;
    int b = std::min(bins-1, std::max(0, (int)((v-lo)/(hi-lo)*bins)));
    y[b] += 1.0;
  }
  plotHistogram_->clearPlottables();
  QCPBars* bars = new QCPBars(plotHistogram_->xAxis, plotHistogram_->yAxis);
  bars->setWidth(0.85);
  bars->setPen(QPen(QColor(59,130,246), 1.0));
  bars->setBrush(QColor(96,165,250,180));
  bars->setData(x, y);
  plotHistogram_->xAxis->setLabel("Bins");
  plotHistogram_->yAxis->setLabel("Count");
  plotHistogram_->xAxis->setRange(-0.5, bins-0.5);
  plotHistogram_->rescaleAxes(true);
  plotHistogram_->replot();
}

void MainWindow::refreshTrajectoryPlot() {
  auto fillPlot = [&](QCustomPlot* p, auto getter, const QString& yLabel) {
    if (!p) return;
    const int n = (int)meas_rows_.size();
    QVector<double> x(n), y(n);
    for (int i=0;i<n;++i) { x[i]=i; y[i]=getter(meas_rows_[i]); }
    if (p->graphCount() == 0) p->addGraph();
    p->graph(0)->setData(x, y);
    p->xAxis->setLabel("frame");
    p->yAxis->setLabel(yLabel);
    p->rescaleAxes(true);
    p->replot();
  };
  fillPlot(lblTrajPosPlot_, [](const MeasureRow& r){ return r.disp; }, "Displacement");
  fillPlot(lblTrajAngPlot_, [](const MeasureRow& r){ return r.speed; }, "Speed");
  fillPlot(plotArea_, [](const MeasureRow& r){ return r.area; }, "Area");
  fillPlot(plotPerimeter_, [](const MeasureRow& r){ return r.perim; }, "Perimeter");
  fillPlot(plotCircularity_, [](const MeasureRow& r){ return r.circ; }, "Circularity");
  fillPlot(plotAccel_, [](const MeasureRow& r){ return r.accel; }, "Acceleration");

  if (tblMeasurements_) {
    tblMeasurements_->setRowCount((int)meas_rows_.size());
    for (int i=0;i<(int)meas_rows_.size();++i) {
      const auto& r = meas_rows_[i];
      const double sc = (mm_per_pixel_ > 0.0 ? mm_per_pixel_ : 1.0);
      auto set=[&](int c,double v){ tblMeasurements_->setItem(i,c,new QTableWidgetItem(QString::number(v,'f',4))); };
      set(0,r.disp); set(1,r.speed); set(2,r.accel); set(3,r.area); set(4,r.perim); set(5,r.major); set(6,r.minor); set(7,r.circ); set(8,sc);
    }
  }
  updateHistogramPlot();
}


void MainWindow::rebuildMeasurementSeriesFromCurrentSource(bool showProgress) {
  measurements_frozen_ = false;
  int totalFrames = progressSlider_ ? (progressSlider_->maximum() + 1) : 0;
  if (totalFrames <= 0 && play_end_frame_ > 0) totalFrames = (int)play_end_frame_;

  int srcIdx = -1;
  {
    QMutexLocker srcLock(&sources_mutex_);
    for (int i=0;i<(int)sources_.size();++i) {
      if (!sources_[i].is_cam && sources_[i].mode_owner==(int)mode_) { srcIdx = i; break; }
    }
    if (srcIdx < 0) {
      for (int i=0;i<(int)sources_.size();++i) {
        if (!sources_[i].is_cam) { srcIdx = i; break; }
      }
    }
  }
  if (srcIdx < 0) return;

  if (totalFrames <= 0) {
    QMutexLocker srcLock(&sources_mutex_);
    if (srcIdx >= 0 && srcIdx < (int)sources_.size()) {
      const auto& src = sources_[srcIdx];
      if (src.is_image_seq) totalFrames = (int)src.seq_files.size();
      else if (src.cap.isOpened()) {
        double fc = src.cap.get(cv::CAP_PROP_FRAME_COUNT);
        if (fc > 0) totalFrames = (int)fc;
      }
    }
  }
  if (totalFrames <= 0) return;

  std::unique_ptr<QProgressDialog> progress;
  if (showProgress) {
    progress.reset(new QProgressDialog("Tracking all frames...", QString(), 0, totalFrames, this));
    progress->setWindowModality(Qt::ApplicationModal);
    progress->setCancelButton(nullptr);
    progress->setMinimumDuration(0);
    progress->setValue(0);
    progress->show();
    qApp->processEvents();
  }

  std::vector<cv::Mat> frames(totalFrames);
  int savedSeq = -1;
  double savedPos = -1.0;
  {
    QMutexLocker srcLock(&sources_mutex_);
    if (srcIdx >= (int)sources_.size()) return;
    auto& src = sources_[srcIdx];
    if (src.is_cam) return;
    if (src.is_image_seq) {
      savedSeq = src.seq_idx;
      for (int i=0;i<totalFrames;++i) {
        if (i < src.seq_files.size()) frames[i] = imreadUnicodePath(src.seq_files[i], cv::IMREAD_COLOR);
      }
      src.seq_idx = std::max(0, std::min(savedSeq, std::max(0, totalFrames-1)));
    } else {
      if (!src.cap.isOpened()) return;
      savedPos = src.cap.get(cv::CAP_PROP_POS_FRAMES);
      for (int i=0;i<totalFrames;++i) {
        src.cap.set(cv::CAP_PROP_POS_FRAMES, (double)i);
        src.cap.read(frames[i]);
      }
      src.cap.set(cv::CAP_PROP_POS_FRAMES, savedPos);
    }
  }

  meas_rows_.clear();
  last_meas_key_ = std::numeric_limits<qint64>::min();
  last_ctr_ = cv::Point2f(0,0);
  last_speed_ = 0.0;

  const int64_t savedPlayFrame = play_frame_;
  if (play_end_frame_ < totalFrames) play_end_frame_ = totalFrames;
  updateProgressUI(play_frame_, play_end_frame_);

  for (int i=0;i<totalFrames;++i) {
    if (progress) {
      progress->setValue(i);
      progress->setLabelText(QString("Tracking all frames... %1/%2").arg(i+1).arg(totalFrames));
      qApp->processEvents();
    }
    if (frames[i].empty()) continue;
    cv::Mat f = applyPreprocess(frames[i]);
    play_frame_ = i;
    updateMeasurementFromFrame(f);
  }
  play_frame_ = savedPlayFrame;
  updateProgressUI(play_frame_, play_end_frame_);
  measurements_frozen_ = true;
  if (progress) {
    progress->setValue(totalFrames);
    qApp->processEvents();
  }
  updateLeftVisualDashboard();
}

void MainWindow::updateLeftVisualDashboard() {
  const bool realScale = mm_per_pixel_ > 0.0;
  const QString lenU = realScale ? "mm" : "px";
  const QString areaU = realScale ? "mm²" : "px²";

  int curIdx = -1;
  int totalFrames = progressSlider_ ? (progressSlider_->maximum() + 1) : ((play_end_frame_ > 0) ? (int)play_end_frame_ : (int)meas_rows_.size());
  if (totalFrames <= 0) totalFrames = (int)meas_rows_.size();
  if (totalFrames > 0) curIdx = std::max(0, std::min((int)play_frame_, totalFrames - 1));

  QVector<double> disp(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> speed(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> accel(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> area(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> perim(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> major(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> minor(totalFrames, std::numeric_limits<double>::quiet_NaN());
  QVector<double> circ(totalFrames, std::numeric_limits<double>::quiet_NaN());

  for (const auto& r : meas_rows_) {
    if (r.key < 0 || r.key >= totalFrames) continue;
    int i = (int)r.key;
    disp[i]=r.disp; speed[i]=r.speed; accel[i]=r.accel; area[i]=r.area; perim[i]=r.perim; major[i]=r.major; minor[i]=r.minor; circ[i]=r.circ;
  }

  auto metricValues = [&](int idx)->QVector<double> {
    if (idx == 1) return speed;
    if (idx == 2) return accel;
    if (idx == 3) return area;
    if (idx == 4) return perim;
    if (idx == 5) return major;
    if (idx == 6) return minor;
    if (idx == 7) return circ;
    return disp;
  };
  auto metricLabel = [&](int idx)->QString {
    if (idx == 1) return QString("Speed (%1/frame)").arg(lenU);
    if (idx == 2) return QString("Acceleration (%1/frame²)").arg(lenU);
    if (idx == 3) return QString("Area (%1)").arg(areaU);
    if (idx == 4) return QString("Perimeter (%1)").arg(lenU);
    if (idx == 5) return QString("Major Axis (%1)").arg(lenU);
    if (idx == 6) return QString("Minor Axis (%1)").arg(lenU);
    if (idx == 7) return "Circularity (-)";
    return QString("Displacement (%1)").arg(lenU);
  };

  auto fill=[&](QCustomPlot* p, const QVector<double>& vals, const QString& yLabel){
    if(!p) return;
    QVector<double> x(vals.size());
    double yMin = 0.0, yMax = 1.0;
    bool has=false;
    for(int i=0;i<vals.size();++i){
      x[i]=i;
      if (std::isfinite(vals[i])) {
        if(!has){ yMin=yMax=vals[i]; has=true; }
        else { yMin=std::min(yMin, vals[i]); yMax=std::max(yMax, vals[i]); }
      }
    }
    if(!has){ yMin=0.0; yMax=1.0; }
    if(std::abs(yMax-yMin) < 1e-9){ yMin-=1.0; yMax+=1.0; }
    if(p->graphCount()<2){ while(p->graphCount()<2) p->addGraph(); p->graph(1)->setPen(QPen(QColor(239,68,68),1.6)); }
    p->graph(0)->setData(x, vals);
    p->graph(1)->setData({}, {});
    if (curIdx >= 0) {
      QVector<double> cx(2), cy(2); cx[0]=curIdx; cx[1]=curIdx; cy[0]=yMin; cy[1]=yMax; p->graph(1)->setData(cx,cy);
    }
    p->xAxis->setLabel("Frame");
    p->yAxis->setLabel(yLabel);
    p->xAxis->setRange(0, std::max(1, vals.size()-1));
    p->yAxis->setRange(yMin, yMax);
    p->replot();
  };

  fill(leftDispPlot_, metricValues(cbDispMetric_ ? cbDispMetric_->currentIndex() : 0), metricLabel(cbDispMetric_ ? cbDispMetric_->currentIndex() : 0));
  fill(leftSpeedPlot_, metricValues(cbSpeedMetric_ ? cbSpeedMetric_->currentIndex() : 1), metricLabel(cbSpeedMetric_ ? cbSpeedMetric_->currentIndex() : 1));
  fill(leftAreaPlot_, metricValues(cbAreaMetric_ ? cbAreaMetric_->currentIndex() : 3), metricLabel(cbAreaMetric_ ? cbAreaMetric_->currentIndex() : 3));
  fill(leftPerimPlot_, metricValues(cbPerimMetric_ ? cbPerimMetric_->currentIndex() : 4), metricLabel(cbPerimMetric_ ? cbPerimMetric_->currentIndex() : 4));
  fill(leftCircPlot_, metricValues(cbCircMetric_ ? cbCircMetric_->currentIndex() : 7), metricLabel(cbCircMetric_ ? cbCircMetric_->currentIndex() : 7));

  if (leftMeasureTable_) {
    leftMeasureTable_->setHorizontalHeaderLabels({
      QString("Disp (%1)").arg(lenU), QString("Speed (%1/frame)").arg(lenU), QString("Accel (%1/frame²)").arg(lenU),
      QString("Area (%1)").arg(areaU), QString("Perimeter (%1)").arg(lenU), QString("Major (%1)").arg(lenU), QString("Minor (%1)").arg(lenU),
      "Circularity (-)", "Scale (mm/px)"
    });
    leftMeasureTable_->setRowCount(totalFrames);
    auto setNum=[&](int r,int c,double v){ leftMeasureTable_->setItem(r,c,new QTableWidgetItem(std::isfinite(v)?QString::number(v,'f',4):"-")); };
    for (int i=0;i<totalFrames;++i) {
      const MeasureRow* rp = nullptr;
      for (const auto& r : meas_rows_) if (r.key == i) { rp = &r; break; }
      if (rp) {
        const double sc=(mm_per_pixel_>0.0?mm_per_pixel_:1.0);
        setNum(i,0,rp->disp); setNum(i,1,rp->speed); setNum(i,2,rp->accel); setNum(i,3,rp->area); setNum(i,4,rp->perim); setNum(i,5,rp->major); setNum(i,6,rp->minor); setNum(i,7,rp->circ); setNum(i,8,sc);
      } else {
        for (int c=0;c<8;++c) setNum(i,c,std::numeric_limits<double>::quiet_NaN());
        setNum(i,8,(mm_per_pixel_>0.0?mm_per_pixel_:1.0));
      }
    }
    leftMeasureTable_->clearSelection();
    if (curIdx >= 0 && curIdx < leftMeasureTable_->rowCount()) {
      leftMeasureTable_->selectRow(curIdx);
      if (leftMeasureTable_->item(curIdx, 0)) leftMeasureTable_->scrollToItem(leftMeasureTable_->item(curIdx,0), QAbstractItemView::PositionAtCenter);
    }
  }
}

void MainWindow::savePlotAsBmp(QCustomPlot* plot, const QString& nameHint) {
  if (!plot) return;
  QString def = QString("%1_%2.bmp").arg(nameHint, QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
  QString path = QFileDialog::getSaveFileName(this, "Save Plot BMP", def, "Bitmap (*.bmp)");
  if (path.isEmpty()) return;
  const int w = std::max(1920, plot->width() * 2);
  const int h = std::max(1080, plot->height() * 2);
  if (!plot->saveBmp(path, w, h, 1.0, -1)) {
    QMessageBox::warning(this, "Save BMP", "Failed to save plot image.");
  }
}

void MainWindow::onCaptureVisualSnapshot() {
  if (!visualDashHost_) return;
  QString def = QString("visual_window_%1.bmp").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
  QString path = QFileDialog::getSaveFileName(this, "Save Visual Snapshot", def, "Bitmap (*.bmp)");
  if (path.isEmpty()) return;
  QPixmap pm = visualDashHost_->grab();
  if (!pm.save(path, "BMP")) QMessageBox::warning(this, "Snapshot", "Failed to save snapshot.");
}

void MainWindow::onExportTableCsv() {
  if (!leftMeasureTable_) return;
  QString def = QString("measurements_%1.csv").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
  QString path = QFileDialog::getSaveFileName(this, "Export CSV", def, "CSV (*.csv)");
  if (path.isEmpty()) return;
  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::warning(this, "Export CSV", "Failed to open file.");
    return;
  }
  QTextStream ts(&f);
  QStringList headers;
  for (int c=0;c<leftMeasureTable_->columnCount();++c) {
    auto* hi = leftMeasureTable_->horizontalHeaderItem(c);
    headers << (hi ? hi->text() : QString("Col%1").arg(c));
  }
  ts << "Frame," << headers.join(',') << "\n";
  for (int r=0;r<leftMeasureTable_->rowCount();++r) {
    ts << r;
    for (int c=0;c<leftMeasureTable_->columnCount();++c) {
      auto* it = leftMeasureTable_->item(r,c);
      QString v = it ? it->text() : "";
      if (v.contains(',')) v = QString("\"") + v + "\"";
      ts << ',' << v;
    }
    ts << "\n";
  }
  f.close();
}

void MainWindow::onExportVisualMp4() {
  if (!visualDashHost_ || !progressSlider_) return;
  QString def = QString("visual_record_%1.mp4").arg(QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss"));
  QString path = QFileDialog::getSaveFileName(this, "Export MP4", def, "MP4 (*.mp4)");
  if (path.isEmpty()) return;

  const int start = progressSlider_->minimum();
  const int end = progressSlider_->maximum();
  if (end < start) return;
  int savedFrame = (int)play_frame_;

  QImage first = visualDashHost_->grab().toImage().convertToFormat(QImage::Format_RGB888);
  cv::Mat firstMat(first.height(), first.width(), CV_8UC3, (void*)first.bits(), first.bytesPerLine());
  cv::Mat firstBgr; cv::cvtColor(firstMat, firstBgr, cv::COLOR_RGB2BGR);
  cv::VideoWriter writer(path.toStdString(), cv::VideoWriter::fourcc('m','p','4','v'), std::max(1.0, play_fps_), firstBgr.size());
  if (!writer.isOpened()) {
    QMessageBox::warning(this, "Export MP4", "Failed to create mp4 writer.");
    return;
  }

  for (int i=start; i<=end; ++i) {
    play_frame_ = i;
    onTick();
    qApp->processEvents();
    QImage qi = visualDashHost_->grab().toImage().convertToFormat(QImage::Format_RGB888);
    cv::Mat rgb(qi.height(), qi.width(), CV_8UC3, (void*)qi.bits(), qi.bytesPerLine());
    cv::Mat bgr; cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    writer.write(bgr);
  }
  writer.release();
  play_frame_ = savedFrame;
  onTick();
}

void MainWindow::onPoseFromWorker(const PoseResult& r) {
  if (r.ok) {
    t_wr_ = Eigen::Vector3d(r.t[0], r.t[1], r.t[2]);
    Eigen::Vector3d aavec(r.aa[0], r.aa[1], r.aa[2]);
    double angle = aavec.norm();
    Eigen::Vector3d axis = (angle < 1e-12) ? Eigen::Vector3d(1,0,0) : (aavec/angle);
    R_wr_ = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    last_inliers_ = r.inliers;
    traj_.push_back(TrajRow{ r.solve_ts_ms, t_wr_, aavec, r.inliers });
  }
  if (lblLatency_) lblLatency_->setText(QString("Latency: %1 ms (obs=%2)")
                                       .arg(r.latency_ms,0,'f',1).arg(r.obs_count));
  if (lblPose_) {
    if (r.ok) {
      lblPose_->setText(QString("Pose t=[%1, %2, %3], aa=[%4, %5, %6], inliers=%7")
                        .arg(r.t[0],0,'f',4).arg(r.t[1],0,'f',4).arg(r.t[2],0,'f',4)
                        .arg(r.aa[0],0,'f',4).arg(r.aa[1],0,'f',4).arg(r.aa[2],0,'f',4)
                        .arg(r.inliers));
    } else {
      lblPose_->setText("Pose: no valid solution");
    }
  }
  refreshTrajectoryPlot();
}

void MainWindow::rebuildSourceDocks() {
  // Remove existing docks
  for (auto* d : camDocks_) {
    if (d) { removeDockWidget(d); d->deleteLater(); }
  }
  camDocks_.clear();
  camLabels_.clear();

  for (int i=0;i<(int)sources_.size();++i) {
    QDockWidget* d = new QDockWidget(QString("View %1").arg(i), this);
    d->setAllowedAreas(Qt::AllDockWidgetAreas);
    QLabel* lab = new QLabel(d);
    lab->setAlignment(Qt::AlignCenter);
    lab->setMinimumSize(320, 240);
    lab->setStyleSheet("background-color: #111; border: 1px solid #333;");
    d->setWidget(lab);
    addDockWidget(Qt::BottomDockWidgetArea, d);
    camDocks_.push_back(d);
    camLabels_.push_back(lab);
  }
}

void MainWindow::updateSourceDocks(const std::vector<cv::Mat>& frames) {
  if ((int)camLabels_.size() != (int)frames.size()) return;
  for (int i=0;i<(int)frames.size();++i) {
    if (!camLabels_[i]) continue;
    if (frames[i].empty()) continue;
    QImage img = matToQImage(frames[i]);
    camLabels_[i]->setPixmap(QPixmap::fromImage(img).scaled(
      camLabels_[i]->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
  }
}

void MainWindow::onSaveLayout() {
  settings_.setValue("mainWindow/geometry", saveGeometry());
  settings_.setValue("mainWindow/state", saveState());
  logLine("Layout saved to settings.");
}

void MainWindow::onRestoreLayout() {
  if (settings_.contains("mainWindow/geometry")) restoreGeometry(settings_.value("mainWindow/geometry").toByteArray());
  if (settings_.contains("mainWindow/state")) restoreState(settings_.value("mainWindow/state").toByteArray());
  logLine("Layout restored from settings.");
}

void MainWindow::onToggleDocks(bool on) {
  show_docks_ = on;
  if (show_docks_) {
    rebuildSourceDocks();
    logLine("Per-source docks: ON");
  } else {
    // remove existing docks
    for (auto* d : camDocks_) {
      if (d) { removeDockWidget(d); d->deleteLater(); }
    }
    camDocks_.clear();
    camLabels_.clear();
    logLine("Per-source docks: OFF");
  }
}


int MainWindow::videoSourceCount() const {
  int c=0;
  for (const auto& s : sources_) if (!s.is_cam && s.mode_owner==(int)mode_) c++;
  return c;
}

void MainWindow::stopCaptureBlocking() {
  if (!captureWorker_) return;
  // BlockingQueuedConnection ensures timer stops before returning
  QMetaObject::invokeMethod(captureWorker_, "stop", Qt::BlockingQueuedConnection);
}

void MainWindow::updatePlaybackParams() {
  if (!captureWorker_) return;
  int vidN = videoSourceCount();
  bool sync = sync_play_ && (vidN >= 2); // only meaningful with >=2 videos
  QMetaObject::invokeMethod(captureWorker_, "setSyncModeSlot", Qt::BlockingQueuedConnection, Q_ARG(bool, sync));

  // compute range and fps for sync video playback
  if (sync) {
    int64_t minFrames = (int64_t)9e18;
    double fps = 0.0;
    bool found=false;
    QMutexLocker srcLock(&sources_mutex_);
    for (auto& s : sources_) {
      if (s.is_cam || s.mode_owner!=(int)mode_) continue;
      if (s.is_image_seq) {
        int64_t fc = (int64_t)s.seq_files.size();
        if (fc > 0) { minFrames = std::min(minFrames, fc); found = true; }
        if (fps <= 0.0) fps = 30.0;
        continue;
      }
      if (!s.cap.isOpened()) continue;
      double fc = s.cap.get(cv::CAP_PROP_FRAME_COUNT);
      if (fc > 0) { minFrames = std::min(minFrames, (int64_t)fc); found=true; }
      if (fps <= 0.0) {
        double vf = s.cap.get(cv::CAP_PROP_FPS);
        if (vf > 0.0) fps = vf;
      }
    }
    if (!found || minFrames <= 0) {
      minFrames = 0;
    }
    play_end_frame_ = minFrames;
    play_fps_ = (fps > 0.0 ? fps : 30.0);
    QMetaObject::invokeMethod(captureWorker_, "setPlaybackRangeSlot", Qt::BlockingQueuedConnection,
                              Q_ARG(qint64, (qint64)0), Q_ARG(qint64, (qint64)play_end_frame_), Q_ARG(double, play_fps_));
  } else {
    int64_t maxFrames = 0;
    QMutexLocker srcLock(&sources_mutex_);
    for (const auto& s : sources_) {
      if (s.is_cam || s.mode_owner!=(int)mode_) continue;
      if (s.is_image_seq) {
        maxFrames = std::max<int64_t>(maxFrames, (int64_t)s.seq_files.size());
        continue;
      }
      if (!s.cap.isOpened()) continue;
      double fc = s.cap.get(cv::CAP_PROP_FRAME_COUNT);
      if (fc > 0) maxFrames = std::max<int64_t>(maxFrames, (int64_t)fc);
    }
    play_end_frame_ = maxFrames;
  }
  updateProgressUI(play_frame_, play_end_frame_);
}


void MainWindow::updateProgressUI(int64_t frame, int64_t endFrame) {
  if (!progressSlider_) return;
  int maxVal = (int)std::max<int64_t>(0, endFrame > 0 ? (endFrame - 1) : 0);
  int val = (int)std::max<int64_t>(0, std::min<int64_t>(frame, maxVal));
  progressSlider_->setRange(0, maxVal);
  progressSlider_->blockSignals(true);
  progressSlider_->setValue(val);
  progressSlider_->blockSignals(false);
  if (editCurFrame_) editCurFrame_->setText(QString::number(val));
  if (lblTotalFrame_) lblTotalFrame_->setText(QString("/ %1").arg(maxVal));
  refreshTrajectoryPlot();
}


void MainWindow::stepAllVideos(int delta) {
  updatePlaybackParams();
  stopCaptureBlocking();
  playback_running_ = false;
  bool stepped = false;
  int64_t progressFrame = play_frame_;
  int64_t desiredFrame = std::max<int64_t>(0, play_frame_ + delta);
  if (play_end_frame_ > 0) desiredFrame = std::min<int64_t>(desiredFrame, play_end_frame_-1);
  std::vector<cv::Mat> steppedFrames;

  {
    QMutexLocker srcLock(&sources_mutex_);
    steppedFrames.resize(sources_.size());
    for (int i=0;i<(int)sources_.size();++i) {
      auto& src = sources_[i];
      if (src.is_cam || src.mode_owner!=(int)mode_) continue;
      cv::Mat f;
      int64_t target = desiredFrame;
      if (src.is_image_seq) {
        if (src.seq_files.isEmpty()) continue;
        target = std::max<int64_t>(0, std::min<int64_t>(target, (int64_t)src.seq_files.size()-1));
        src.seq_idx = (int)target;
        f = imreadUnicodePath(src.seq_files[(int)target], cv::IMREAD_COLOR);
      } else {
        if (!src.cap.isOpened()) continue;
        if (play_end_frame_ <= 0) {
          double cnt = src.cap.get(cv::CAP_PROP_FRAME_COUNT);
          if (cnt > 0) target = std::min<int64_t>(target, std::max<int64_t>(0, (int64_t)std::llround(cnt)-1));
        }
        src.cap.set(cv::CAP_PROP_POS_FRAMES, (double)target);
        src.cap.read(f);
      }
      if (!f.empty()) {
        // If Detect-All generated a visualized frame for this source/frame, reuse it
        // so marker overlays remain persistent when stepping prev/next.
        auto srcIt = detect_overlay_cache_.find(i);
        if (srcIt != detect_overlay_cache_.end()) {
          auto frameIt = srcIt->second.find(target);
          if (frameIt != srcIt->second.end() && !frameIt->second.empty()) {
            f = frameIt->second;
          }
        }
        steppedFrames[i] = f;
        progressFrame = target;
        stepped = true;
      }
    }
  }

  if (stepped) {
    {
      QMutexLocker frameLock(&frames_mutex_);
      if ((int)last_frames_.size() != (int)steppedFrames.size()) last_frames_.resize(steppedFrames.size());
      for (int i=0;i<(int)steppedFrames.size();++i) {
        if (!steppedFrames[i].empty()) last_frames_[i] = steppedFrames[i];
      }
    }
    play_frame_ = progressFrame;
    updateProgressUI(play_frame_, play_end_frame_);
    onTick();
    logLine(QString("Step frame: %1").arg(delta > 0 ? "next" : "prev"));
  } else {
    logLine("Step frame ignored: no video source ready.");
  }
}

void MainWindow::onStepPrevFrame() { stepAllVideos(-1); }
void MainWindow::onStepNextFrame() { stepAllVideos(1); }

void MainWindow::onToolPan() {
  if (btnToolPan_) btnToolPan_->setChecked(true);
  if (btnToolPoint_) btnToolPoint_->setChecked(false);
  if (btnToolLine_) btnToolLine_->setChecked(false);
  for (auto* v : sourceViews_) if (v) v->setToolMode(ImageViewer::PanTool);
}

void MainWindow::onToolPoint() {
  if (btnToolPan_) btnToolPan_->setChecked(false);
  if (btnToolPoint_) btnToolPoint_->setChecked(true);
  if (btnToolLine_) btnToolLine_->setChecked(false);
  for (auto* v : sourceViews_) if (v) v->setToolMode(ImageViewer::PointTool);
}

void MainWindow::onToolLine() {
  if (btnToolPan_) btnToolPan_->setChecked(false);
  if (btnToolPoint_) btnToolPoint_->setChecked(false);
  if (btnToolLine_) btnToolLine_->setChecked(true);
  for (auto* v : sourceViews_) if (v) v->setToolMode(ImageViewer::LineTool);
}

void MainWindow::onZoomIn() { for (auto* v : sourceViews_) if (v) v->zoomIn(); }
void MainWindow::onZoomOut() { for (auto* v : sourceViews_) if (v) v->zoomOut(); }
void MainWindow::onResetView() { for (auto* v : sourceViews_) if (v) v->resetView(); }
void MainWindow::onClearAnnotations() { for (auto* v : sourceViews_) if (v) v->clearAnnotations(); }

void MainWindow::onProgressSliderReleased() {
  if (!progressSlider_) return;
  int target = progressSlider_->value();
  {
    QMutexLocker srcLock(&sources_mutex_);
    for (auto& src : sources_) {
      if (src.is_cam || src.mode_owner!=(int)mode_) continue;
      if (src.is_image_seq) {
        if (!src.seq_files.isEmpty()) src.seq_idx = std::max(0, std::min(target, (int)src.seq_files.size()-1));
        continue;
      }
      if (!src.cap.isOpened()) continue;
      src.cap.set(cv::CAP_PROP_POS_FRAMES, (double)target);
    }
  }
  play_frame_ = target;
  stepAllVideos(0);
}

void MainWindow::onFrameJumpReturnPressed() {
  if (!editCurFrame_ || !progressSlider_) return;
  bool ok = false;
  int target = editCurFrame_->text().toInt(&ok);
  if (!ok) return;
  target = std::max(progressSlider_->minimum(), std::min(progressSlider_->maximum(), target));
  progressSlider_->setValue(target);
  onProgressSliderReleased();
}

void MainWindow::onPlayAll() {
  if (playback_running_) {
    playback_running_ = false;
    stopCaptureBlocking();
    {
      QMutexLocker srcLock(&sources_mutex_);
      for (int i=0;i<(int)sources_.size();++i) {
        if (!sources_[i].is_cam && sources_[i].mode_owner==(int)mode_) source_enabled_[i] = false;
      }
    }
    if (btnPlayAll_) {
      btnPlayAll_->setChecked(false);
      btnPlayAll_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
      btnPlayAll_->setToolTip("Play");
    }
    logLine("Playback paused.");
    refreshSourceList();
    return;
  }

  // Start playback when user presses Play.
  int vidN = videoSourceCount();
  if (vidN < 1) {
    QMessageBox::information(this, "Play", "Need at least 1 video/image source in current tab.");
    return;
  }

  stopCaptureBlocking();

  {
    QMutexLocker srcLock(&sources_mutex_);
    // Enable all video sources for playback
    for (int i=0;i<(int)sources_.size();++i) if (!sources_[i].is_cam && sources_[i].mode_owner==(int)mode_) source_enabled_[i]=true;

  }
  playback_running_ = true;
  if (btnPlayAll_) {
    btnPlayAll_->setChecked(true);
    btnPlayAll_->setIcon(style()->standardIcon(QStyle::SP_MediaPause));
    btnPlayAll_->setToolTip("Pause");
  }
  updateProgressUI(play_frame_, play_end_frame_);
  //if (lblPlayState_) 
  //    lblPlayState_->setText("State: PLAY (SYNC)");

  updatePlaybackParams();
  timer_.start(33);
  QMetaObject::invokeMethod(captureWorker_, "start", Qt::QueuedConnection);
}

void MainWindow::onStopAll() {
  playback_running_ = false;
  if (btnPlayAll_) {
    btnPlayAll_->setChecked(false);
    btnPlayAll_->setIcon(style()->standardIcon(QStyle::SP_MediaPlay));
    btnPlayAll_->setToolTip("Play");
  }
  //if (lblPlayState_) lblPlayState_->setText("State: STOP");
  stopCaptureBlocking();
  updateProgressUI(play_frame_, play_end_frame_);
}
