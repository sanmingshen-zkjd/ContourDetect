#pragma once

#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QColor>
#include <QPoint>
#include <QImage>

#include <opencv2/core.hpp>

#include <functional>

class SegmentationView : public QGraphicsView {
  Q_OBJECT
public:
  enum ToolMode {
    PanTool,
    PaintTool,
    EraseTool
  };

  explicit SegmentationView(QWidget* parent = nullptr);

  void setBaseImage(const QImage& image);
  void setOverlayImage(const QImage& image);
  void setAnnotationPreview(const QImage& image);
  void clearAllLayers();

  void setToolMode(ToolMode mode);
  ToolMode toolMode() const { return toolMode_; }

  void setBrushRadius(int radius);
  int brushRadius() const { return brushRadius_; }

  void setActiveClass(int classIndex, const QColor& color);
  void setPaintEnabled(bool enabled) { paintEnabled_ = enabled; }

  void zoomIn();
  void zoomOut();
  void resetView();

  std::function<void(const QPoint& imagePos, int radius, bool erase)> onBrushStroke;
  std::function<void(const QPoint& imagePos)> onMouseHover;

protected:
  void wheelEvent(QWheelEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void resizeEvent(QResizeEvent* event) override;
  void leaveEvent(QEvent* event) override;

private:
  QPoint sceneToImage(const QPointF& scenePos) const;
  void applyZoom(double factor);
  void updateCursor();
  void updateBrushPreview(const QPoint& imagePos);
  void clearBrushPreview();

  QGraphicsScene scene_;
  QGraphicsPixmapItem* baseItem_ = nullptr;
  QGraphicsPixmapItem* overlayItem_ = nullptr;
  QGraphicsPixmapItem* annotationItem_ = nullptr;
  QGraphicsEllipseItem* brushItem_ = nullptr;

  ToolMode toolMode_ = PanTool;
  bool paintEnabled_ = true;
  bool mousePressed_ = false;
  int brushRadius_ = 12;
  QColor activeColor_ = Qt::red;
  double zoomFactor_ = 1.0;
  QSize imageSize_;
};
