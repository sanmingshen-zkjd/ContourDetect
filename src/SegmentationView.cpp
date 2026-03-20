#include "SegmentationView.h"

#include <QMouseEvent>
#include <QWheelEvent>
#include <QFrame>
#include <QPen>
#include <QBrush>
#include <cmath>

SegmentationView::SegmentationView(QWidget* parent)
    : QGraphicsView(parent) {
  setScene(&scene_);
  setRenderHint(QPainter::Antialiasing, true);
  setFrameShape(QFrame::StyledPanel);
  setBackgroundBrush(QColor(18, 18, 18));
  setDragMode(QGraphicsView::ScrollHandDrag);
  setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
  setResizeAnchor(QGraphicsView::AnchorViewCenter);

  baseItem_ = scene_.addPixmap(QPixmap());
  overlayItem_ = scene_.addPixmap(QPixmap());
  overlayItem_->setOpacity(1.0);
  annotationItem_ = scene_.addPixmap(QPixmap());
  annotationItem_->setOpacity(1.0);

  brushItem_ = scene_.addEllipse(QRectF(), QPen(QColor(255, 255, 255, 180), 1.5), Qt::NoBrush);
  brushItem_->setVisible(false);
  brushItem_->setZValue(10.0);

  pendingTraceItem_ = scene_.addPath(QPainterPath(), QPen(QColor(255, 210, 90), 2.0), Qt::NoBrush);
  pendingTraceItem_->setVisible(false);
  pendingTraceItem_->setZValue(11.0);

  activeTraceItem_ = scene_.addPath(QPainterPath(), QPen(QColor(255, 255, 255, 160), 2.0), Qt::NoBrush);
  activeTraceItem_->setVisible(false);
  activeTraceItem_->setZValue(12.0);
  updateCursor();
}

void SegmentationView::setBaseImage(const QImage& image) {
  baseItem_->setPixmap(QPixmap::fromImage(image));
  imageSize_ = image.size();
  scene_.setSceneRect(QRectF(QPointF(0, 0), QSizeF(image.size())));
  resetView();
}

void SegmentationView::setOverlayImage(const QImage& image) {
  overlayItem_->setPixmap(QPixmap::fromImage(image));
}

void SegmentationView::setAnnotationPreview(const QImage& image) {
  annotationItem_->setPixmap(QPixmap::fromImage(image));
}

void SegmentationView::clearAllLayers() {
  baseItem_->setPixmap(QPixmap());
  overlayItem_->setPixmap(QPixmap());
  annotationItem_->setPixmap(QPixmap());
  imageSize_ = QSize();
  scene_.setSceneRect(QRectF());
  clearPendingTrace();
  activeTracePolygon_.clear();
  activeTraceItem_->setPath(QPainterPath());
  activeTraceItem_->setVisible(false);
}

void SegmentationView::setToolMode(ToolMode mode) {
  toolMode_ = mode;
  setDragMode(toolMode_ == PanTool ? QGraphicsView::ScrollHandDrag : QGraphicsView::NoDrag);
  if (toolMode_ != TraceTool) {
    traceDrawing_ = false;
    activeTracePolygon_.clear();
    activeTraceItem_->setPath(QPainterPath());
    activeTraceItem_->setVisible(false);
  }
  updateCursor();
}

void SegmentationView::setBrushRadius(int radius) {
  brushRadius_ = qMax(1, radius);
  if (brushItem_->isVisible()) {
    updateBrushPreview(brushItem_->rect().center().toPoint());
  }
}

void SegmentationView::setActiveClass(int classIndex, const QColor& color) {
  Q_UNUSED(classIndex);
  activeColor_ = color;
  QPen pen(activeColor_.lighter(160), 1.5);
  brushItem_->setPen(pen);
  pendingTraceItem_->setPen(QPen(activeColor_.lighter(130), 2.0));
  activeTraceItem_->setPen(QPen(activeColor_.lighter(170), 2.0));
}

void SegmentationView::setPendingTrace(const QPolygon& trace, const QColor& color) {
  if (trace.isEmpty()) {
    clearPendingTrace();
    return;
  }
  pendingTraceItem_->setPen(QPen(color.lighter(130), 2.0));
  pendingTraceItem_->setPath(polygonToPath(trace));
  pendingTraceItem_->setVisible(true);
}

void SegmentationView::clearPendingTrace() {
  pendingTraceItem_->setPath(QPainterPath());
  pendingTraceItem_->setVisible(false);
}

void SegmentationView::zoomIn() { applyZoom(1.15); }
void SegmentationView::zoomOut() { applyZoom(1.0 / 1.15); }

void SegmentationView::resetView() {
  resetTransform();
  zoomFactor_ = 1.0;
  if (!baseItem_->pixmap().isNull()) {
    fitInView(baseItem_, Qt::KeepAspectRatio);
  }
}

void SegmentationView::wheelEvent(QWheelEvent* event) {
  if (event->modifiers() & Qt::ControlModifier) {
    if (event->angleDelta().y() > 0) zoomIn();
    else zoomOut();
    event->accept();
    return;
  }
  QGraphicsView::wheelEvent(event);
}

void SegmentationView::mousePressEvent(QMouseEvent* event) {
  const QPoint imagePos = sceneToImage(mapToScene(event->pos()));
  if (toolMode_ == TraceTool && paintEnabled_ && event->button() == Qt::LeftButton && imagePos.x() >= 0) {
    traceDrawing_ = true;
    activeTracePolygon_.clear();
    activeTracePolygon_ << imagePos;
    activeTraceItem_->setPath(polygonToPath(activeTracePolygon_));
    activeTraceItem_->setVisible(true);
    return;
  }

  if ((toolMode_ == PaintTool || toolMode_ == EraseTool) && paintEnabled_ && event->button() == Qt::LeftButton) {
    if (imagePos.x() >= 0 && onBrushStroke) {
      mousePressed_ = true;
      onBrushStroke(imagePos, brushRadius_, toolMode_ == EraseTool);
      updateBrushPreview(imagePos);
      return;
    }
  }
  QGraphicsView::mousePressEvent(event);
}

void SegmentationView::mouseMoveEvent(QMouseEvent* event) {
  const QPoint imagePos = sceneToImage(mapToScene(event->pos()));
  if (imagePos.x() >= 0) {
    if (toolMode_ == TraceTool && traceDrawing_) {
      if (activeTracePolygon_.isEmpty() || activeTracePolygon_.last() != imagePos) {
        activeTracePolygon_ << imagePos;
        activeTraceItem_->setPath(polygonToPath(activeTracePolygon_));
      }
      return;
    }

    updateBrushPreview(imagePos);
    if (onMouseHover) {
      onMouseHover(imagePos);
    }
    if (mousePressed_ && paintEnabled_ && (toolMode_ == PaintTool || toolMode_ == EraseTool) && onBrushStroke) {
      onBrushStroke(imagePos, brushRadius_, toolMode_ == EraseTool);
      return;
    }
  } else {
    clearBrushPreview();
  }
  QGraphicsView::mouseMoveEvent(event);
}

void SegmentationView::mouseReleaseEvent(QMouseEvent* event) {
  if (toolMode_ == TraceTool && traceDrawing_ && event->button() == Qt::LeftButton) {
    traceDrawing_ = false;
    activeTraceItem_->setVisible(false);
    activeTraceItem_->setPath(QPainterPath());
    if (activeTracePolygon_.size() > 2 && onTraceFinished) {
      onTraceFinished(activeTracePolygon_);
    }
    activeTracePolygon_.clear();
    return;
  }

  mousePressed_ = false;
  QGraphicsView::mouseReleaseEvent(event);
}

void SegmentationView::resizeEvent(QResizeEvent* event) {
  QGraphicsView::resizeEvent(event);
  if (zoomFactor_ == 1.0 && !baseItem_->pixmap().isNull()) {
    fitInView(baseItem_, Qt::KeepAspectRatio);
  }
}

void SegmentationView::leaveEvent(QEvent* event) {
  clearBrushPreview();
  QGraphicsView::leaveEvent(event);
}

QPoint SegmentationView::sceneToImage(const QPointF& scenePos) const {
  if (imageSize_.isEmpty()) {
    return QPoint(-1, -1);
  }
  const int x = static_cast<int>(std::round(scenePos.x()));
  const int y = static_cast<int>(std::round(scenePos.y()));
  if (x < 0 || y < 0 || x >= imageSize_.width() || y >= imageSize_.height()) {
    return QPoint(-1, -1);
  }
  return QPoint(x, y);
}

void SegmentationView::applyZoom(double factor) {
  zoomFactor_ *= factor;
  zoomFactor_ = qBound(0.1, zoomFactor_, 20.0);
  scale(factor, factor);
}

void SegmentationView::updateCursor() {
  if (toolMode_ == PanTool) {
    setCursor(Qt::OpenHandCursor);
  } else if (toolMode_ == TraceTool) {
    setCursor(Qt::PointingHandCursor);
  } else {
    setCursor(Qt::CrossCursor);
  }
}

void SegmentationView::updateBrushPreview(const QPoint& imagePos) {
  if (toolMode_ == PanTool || toolMode_ == TraceTool || imagePos.x() < 0) {
    clearBrushPreview();
    return;
  }
  brushItem_->setVisible(true);
  brushItem_->setRect(imagePos.x() - brushRadius_, imagePos.y() - brushRadius_, brushRadius_ * 2, brushRadius_ * 2);
}

void SegmentationView::clearBrushPreview() {
  brushItem_->setVisible(false);
}

QPainterPath SegmentationView::polygonToPath(const QPolygon& polygon) const {
  QPainterPath path;
  if (polygon.isEmpty()) {
    return path;
  }
  path.moveTo(polygon.first());
  for (int i = 1; i < polygon.size(); ++i) {
    path.lineTo(polygon[i]);
  }
  path.closeSubpath();
  return path;
}
