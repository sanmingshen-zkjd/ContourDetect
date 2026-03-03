// mainwindow.h
#ifndef GPAPHICSVIEW_H
#define GPAPHICSVIEW_H

#include <QMainWindow>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include "lineitem.h"
#include "rectitem.h"
#include "polygonitem.h"

class GraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    enum DrawMode {
        SelectMode,
        PanMode,
        DrawLine,
        DrawRect,
        DrawPolygon
    };

    explicit GraphicsView(QWidget *parent = nullptr);

    void setDrawMode(DrawMode mode);
    void zoomIn();
    void zoomOut();
    void resetView();
    void setImage(const QImage &image);
    void clearDrawings();
    void deleteSelectedItems();

    // 序列化
    QByteArray saveDrawings() const;
    void loadDrawings(const QByteArray &data);

    //增加图像自适应部分代码
    void setAutoFitImage(bool autoFit);
    bool autoFitImage() const { return m_autoFitImage; }

protected:
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

    //增加图像自适应部分代码
    void resizeEvent(QResizeEvent *event) override;

private:
    void updateCursor();
    void zoom(double factor, const QPointF &centerPos);
    void clearSelection();

    QGraphicsScene *m_scene;
    QGraphicsPixmapItem *m_imageItem;
    DrawMode m_drawMode;
    bool m_isDrawing;
    QPointF m_startPoint;

    // 临时图形项
    LineItem *m_tempLine;
    RectItem *m_tempRect;
    PolygonItem *m_tempPolygon;

    // 绘制的图形项
    QList<EditableItem*> m_drawnItems;
    QPolygonF m_polygonPoints;

    // 缩放相关
    double m_zoomFactor;
    double m_minZoom;
    double m_maxZoom;

    bool m_autoFitImage;  // 添加这个
    bool m_userZoomed;
};
#endif // MAINWINDOW_H
