// mainwindow.cpp
#include "graphicsview.h"
#include <QToolBar>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QPen>
#include <QBrush>
#include <QCursor>
#include <QWheelEvent>
#include <QMenuBar>
#include <QStatusBar>
#include <QLabel>
#include <QDataStream>
#include <QApplication>
#include <QKeyEvent>
#include <QScrollBar>
#include <QDebug>

GraphicsView::GraphicsView(QWidget *parent)
    : QGraphicsView(parent)
    , m_scene(new QGraphicsScene(this))
    , m_imageItem(nullptr)
    , m_drawMode(SelectMode)
    , m_isDrawing(false)
    , m_tempLine(nullptr)
    , m_tempRect(nullptr)
    , m_tempPolygon(nullptr)
    , m_zoomFactor(1.0)
    , m_minZoom(0.1)
    , m_maxZoom(40.0)
    , m_autoFitImage(true)  // 默认启用自适应
    , m_userZoomed(false)  // 初始化
{
    setScene(m_scene);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setRenderHint(QPainter::Antialiasing, false); // 拖动时关闭抗锯齿
    setOptimizationFlag(QGraphicsView::DontAdjustForAntialiasing, true);
    setOptimizationFlag(QGraphicsView::DontSavePainterState, true);
    setRenderHint(QPainter::Antialiasing);
    setDragMode(QGraphicsView::RubberBandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    // 设置背景
    setBackgroundBrush(QBrush(QColor(240, 240, 240)));
    // 允许鼠标追踪
    setMouseTracking(true);
    // 初始光标
    updateCursor();
}

void GraphicsView::setDrawMode(DrawMode mode)
{
    m_drawMode = mode;
    m_isDrawing = false;
    // 清除所有选中状态
    for (EditableItem *item : m_drawnItems) {
        item->setSelected(false);
    }
    switch(mode) {
    case SelectMode:
        setDragMode(QGraphicsView::RubberBandDrag);
        setInteractive(true);
        break;
    case PanMode:
        setDragMode(QGraphicsView::ScrollHandDrag);
        setInteractive(false);
        break;
    default:
        setDragMode(QGraphicsView::NoDrag);
        setInteractive(false);
        break;
    }
    updateCursor();
    // 清除临时图形项
    if (m_tempLine) {
        m_scene->removeItem(m_tempLine);
        delete m_tempLine;
        m_tempLine = nullptr;
    }
    if (m_tempRect) {
        m_scene->removeItem(m_tempRect);
        delete m_tempRect;
        m_tempRect = nullptr;
    }
    if (m_tempPolygon) {
        m_scene->removeItem(m_tempPolygon);
        delete m_tempPolygon;
        m_tempPolygon = nullptr;
    }
    m_polygonPoints.clear();
}

void GraphicsView::updateCursor()
{
    switch(m_drawMode) {
    case DrawLine:
    case DrawRect:
    case DrawPolygon:
        setCursor(Qt::CrossCursor);
        break;
    case PanMode:
        setCursor(Qt::OpenHandCursor);
        break;
    case SelectMode:
        setCursor(Qt::ArrowCursor);
        break;
    }
}

void GraphicsView::zoom(double factor, const QPointF &centerPos)
{
    // 标记用户进行了缩放操作
    m_userZoomed = true;
    // 如果用户开始缩放，自动禁用自适应模式
    if (m_autoFitImage) {
        m_autoFitImage = false;
    }
    double newZoomFactor = m_zoomFactor * factor;
    if (newZoomFactor < m_minZoom || newZoomFactor > m_maxZoom) {
        return;
    }
    QPointF oldScenePos = mapToScene(centerPos.toPoint());
    scale(factor, factor);
    m_zoomFactor = newZoomFactor;
    QPointF newScenePos = mapToScene(centerPos.toPoint());
    QPointF delta = newScenePos - oldScenePos;
    centerOn(mapToScene(viewport()->rect().center()) - delta);
}

void GraphicsView::zoomIn()
{
    QPoint center = viewport()->rect().center();
    zoom(1.2, center);
}

void GraphicsView::zoomOut()
{
    QPoint center = viewport()->rect().center();
    zoom(1/1.2, center);
}

void GraphicsView::resetView()
{
    resetTransform();
    m_zoomFactor = 1.0;
    m_userZoomed = false;  // 重置用户缩放标记 && m_autoFitImage
    if(m_imageItem)
    {
        fitInView(m_imageItem, Qt::KeepAspectRatio);
        QRectF imageRect = m_imageItem->boundingRect();
        QRectF viewRect = viewport()->rect();
        qreal scaleX = viewRect.width() / imageRect.width();
        qreal scaleY = viewRect.height() / imageRect.height();
        m_zoomFactor = qMin(scaleX, scaleY);
    }
}

void GraphicsView::setImage(const QImage &image)
{
    if(m_imageItem)
    {
        m_scene->removeItem(m_imageItem);
        delete m_imageItem;
    }
    QPixmap pixmap = QPixmap::fromImage(image);
    m_imageItem = m_scene->addPixmap(pixmap);
    m_scene->setSceneRect(m_imageItem->boundingRect());
    resetView();
}

void GraphicsView::clearDrawings()
{
    for(EditableItem *item : m_drawnItems)
    {
        m_scene->removeItem(item);
        delete item;
    }
    m_drawnItems.clear();
    if(m_tempLine)
    {
        m_scene->removeItem(m_tempLine);
        delete m_tempLine;
        m_tempLine = nullptr;
    }
    if(m_tempRect)
    {
        m_scene->removeItem(m_tempRect);
        delete m_tempRect;
        m_tempRect = nullptr;
    }
    if(m_tempPolygon)
    {
        m_scene->removeItem(m_tempPolygon);
        delete m_tempPolygon;
        m_tempPolygon = nullptr;
    }
    m_polygonPoints.clear();
    m_isDrawing = false;
}

void GraphicsView::deleteSelectedItems()
{
    // 先收集要删除的项
    QList<EditableItem*> itemsToDelete;
    for (EditableItem *item : m_drawnItems)
    {
        if (item->isSelected())
        {
            itemsToDelete.append(item);
        }
    }
    // 删除项
    for (EditableItem *item : itemsToDelete)
    {
        m_scene->removeItem(item);
        m_drawnItems.removeAll(item);
        delete item;
    }
}

QByteArray GraphicsView::saveDrawings() const
{
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);

    stream << static_cast<quint32>(m_drawnItems.size());

    for (EditableItem *item : m_drawnItems)
    {
        // 保存类型标识
        switch (item->itemType())
        {
        case EditableItem::LineItem:
            stream << QString("line");
            break;
        case EditableItem::RectItem:
            stream << QString("rect");
            break;
        case EditableItem::PolygonItem:
            stream << QString("polygon");
            break;
        }
        // 保存数据
        stream << item->saveData();
    }
    return data;
}

void GraphicsView::loadDrawings(const QByteArray &data)
{
    clearDrawings();

    QDataStream stream(data);
    quint32 count;
    stream >> count;

    for (quint32 i = 0; i < count; ++i)
    {
        QString type;
        QByteArray itemData;
        stream >> type >> itemData;
        EditableItem *item = nullptr;
        if (type == "line")
        {
            item = new LineItem();
        }
        else if (type == "rect")
        {
            item = new RectItem();
        }
        else if (type == "polygon")
        {
            item = new PolygonItem();
        }
        if (item)
        {
            item->loadData(itemData);
            m_scene->addItem(item);
            m_drawnItems.append(item);
        }
    }
}

void GraphicsView::wheelEvent(QWheelEvent *event)
{
    // 检查是否有Ctrl键按下
    if(event->modifiers() & Qt::ControlModifier)
    {
        // Ctrl+滚轮：以鼠标位置为中心缩放
        double scaleFactor = 1.15;
        QPoint mousePos = event->position().toPoint();
        // 标记用户进行了缩放
        m_userZoomed = true;
        // 用户缩放时禁用自适应
        if (m_autoFitImage)
        {
            m_autoFitImage = false;
        }
        if(event->angleDelta().y() > 0)
        {
            // 向上滚动：放大
            zoom(scaleFactor, mousePos);
        }
        else
        {
            // 向下滚动：缩小
            zoom(1/scaleFactor, mousePos);
        }
        event->accept();
    }
    else if (event->modifiers() & Qt::ShiftModifier)
    {
        // Shift+滚轮：水平滚动
        QScrollBar *hBar = horizontalScrollBar();
        if (hBar)
        {
            int delta = event->angleDelta().y();
            hBar->setValue(hBar->value() - delta);
        }
        event->accept();
    }
    else
    {
        // 普通滚轮：垂直滚动
        QGraphicsView::wheelEvent(event);
    }
}

void GraphicsView::mousePressEvent(QMouseEvent *event)
{
    QPointF scenePos = mapToScene(event->pos());
    // 获取点击的图形项
    QGraphicsItem *clickedItem = m_scene->itemAt(scenePos, transform());
    EditableItem *editableItem = dynamic_cast<EditableItem*>(clickedItem);
    // 如果点击的是自定义图形项
    if (editableItem)
    {
        editableItem->setContextMenuEnabled(true);  // 启用右键菜单??????????????????????????????????????????????????
        // 先清除其他所有项的选择状态
        for (EditableItem *item : m_drawnItems)
        {
            if (item != editableItem)
            {
                item->setSelected(false);
            }
        }
    }
    else if (!clickedItem || clickedItem == m_imageItem)
    {
        // 点击了空白处，清除所有选中
        for (EditableItem *item : m_drawnItems)
        {
            item->setSelected(false);
        }
    }
    // 在平移模式下，按下鼠标时改为闭合手形
    if(m_drawMode == PanMode && event->button() == Qt::LeftButton)
    {
        setCursor(Qt::ClosedHandCursor);
    }
    // 中键按下：临时切换到平移模式
    if(event->button() == Qt::MiddleButton)
    {
        setDragMode(QGraphicsView::ScrollHandDrag);
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    switch(m_drawMode)
    {
    case DrawLine:
        if(event->button() == Qt::LeftButton)
        {
            m_startPoint = scenePos;
            m_isDrawing = true;
            m_tempLine = new LineItem(QLineF(scenePos, scenePos));
            m_scene->addItem(m_tempLine);
        }
        break;
    case DrawRect:
        if(event->button() == Qt::LeftButton)
        {
            m_startPoint = scenePos;
            m_isDrawing = true;
            m_tempRect = new RectItem(QRectF(scenePos, scenePos));
            m_scene->addItem(m_tempRect);
        }
        break;
    case DrawPolygon:
        if(event->button() == Qt::LeftButton)
        {
            if(!m_isDrawing)
            {
                // 开始新的多边形 - 禁用临时多边形的右键菜单
                m_polygonPoints.clear();
                m_polygonPoints.append(scenePos);
                m_isDrawing = true;
                m_tempPolygon = new PolygonItem(m_polygonPoints);
                m_scene->addItem(m_tempPolygon);
            }
            else
            {
                // 添加新点
                m_polygonPoints.append(scenePos);
                if (m_tempPolygon && m_polygonPoints.size() >= 3)
                {
                    m_tempPolygon->setPolygon(m_polygonPoints);
                }
            }
        }
        else if(event->button() == Qt::RightButton && m_isDrawing)
        {
            // 右键结束多边形绘制
            if(m_polygonPoints.size() >= 3)
            {
                PolygonItem *polygonItem = new PolygonItem(m_polygonPoints);
                polygonItem->setContextMenuEnabled(false);  // 禁用右键菜单
                m_scene->addItem(polygonItem);
                m_drawnItems.append(polygonItem);
                polygonItem->setSelected(true);
            }
            // 清理临时多边形
            if (m_tempPolygon)
            {
                m_scene->removeItem(m_tempPolygon);
                delete m_tempPolygon;
                m_tempPolygon = nullptr;
            }
            m_polygonPoints.clear();
            m_isDrawing = false;
            // 自动切换到选择模式
            setDrawMode(SelectMode);
        }
        break;
    default:
        QGraphicsView::mousePressEvent(event);
        break;
    }
}

void GraphicsView::mouseMoveEvent(QMouseEvent *event)
{
    QPointF scenePos = mapToScene(event->pos());

    switch(m_drawMode) {
    case DrawLine:
        if(m_isDrawing && m_tempLine) {
            m_tempLine->setLine(QLineF(m_startPoint, scenePos));
        }
        break;

    case DrawRect:
        if(m_isDrawing && m_tempRect) {
            QRectF rect(m_startPoint, scenePos);
            m_tempRect->setRect(rect.normalized());
        }
        break;

    case DrawPolygon:
        if(m_isDrawing && m_tempPolygon && !m_polygonPoints.isEmpty()) {
            QPolygonF tempPolygon = m_polygonPoints;
            tempPolygon.append(scenePos);
            m_tempPolygon->setPolygon(tempPolygon);
        }
        break;

    default:
        QGraphicsView::mouseMoveEvent(event);
        break;
    }
}

void GraphicsView::mouseReleaseEvent(QMouseEvent *event)
{
    if(m_drawMode == PanMode && event->button() == Qt::LeftButton) {
        setCursor(Qt::OpenHandCursor);
    }

    if(event->button() == Qt::MiddleButton) {
        if (m_drawMode == PanMode) {
            setDragMode(QGraphicsView::ScrollHandDrag);
            setCursor(Qt::OpenHandCursor);
        } else {
            setDragMode(QGraphicsView::NoDrag);
            updateCursor();
        }
        event->accept();
        return;
    }

    switch(m_drawMode) {
    case DrawLine:
        if(event->button() == Qt::LeftButton && m_isDrawing && m_tempLine) {
            LineItem *lineItem = new LineItem(m_tempLine->line());
            m_scene->addItem(lineItem);
            m_drawnItems.append(lineItem);
            // 选中新创建的图形
            lineItem->setSelected(true);
            m_scene->removeItem(m_tempLine);
            delete m_tempLine;
            m_tempLine = nullptr;
            m_isDrawing = false;
            // 自动切换到选择模式
            setDrawMode(SelectMode);
        }
        break;
    case DrawRect:
        if(event->button() == Qt::LeftButton && m_isDrawing && m_tempRect) {
            RectItem *rectItem = new RectItem(m_tempRect->rect());
            m_scene->addItem(rectItem);
            m_drawnItems.append(rectItem);
            // 选中新创建的图形
            rectItem->setSelected(true);
            m_scene->removeItem(m_tempRect);
            delete m_tempRect;
            m_tempRect = nullptr;
            m_isDrawing = false;
            // 自动切换到选择模式
            setDrawMode(SelectMode);
        }
        break;

    default:
        QGraphicsView::mouseReleaseEvent(event);
        break;
    }
}

void GraphicsView::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) {
        deleteSelectedItems();
        event->accept();
    } else {
        QGraphicsView::keyPressEvent(event);
    }
}

void GraphicsView::clearSelection()
{
    for (EditableItem *item : m_drawnItems) {
        item->setSelected(false);
    }
}


void GraphicsView::resizeEvent(QResizeEvent *event)
{
    QGraphicsView::resizeEvent(event);

    // 如果启用自适应且存在图像，且用户没有手动缩放
    if (m_autoFitImage && m_imageItem && !m_userZoomed) {
        fitInView(m_imageItem, Qt::KeepAspectRatio);

        // 重新计算缩放因子
        QRectF viewRect = viewport()->rect();
        QRectF imageRect = m_imageItem->boundingRect();

        if (!imageRect.isEmpty()) {
            qreal scaleX = viewRect.width() / imageRect.width();
            qreal scaleY = viewRect.height() / imageRect.height();
            m_zoomFactor = qMin(scaleX, scaleY);
        }
    }

    update();
}


void GraphicsView::setAutoFitImage(bool autoFit)
{
    m_autoFitImage = autoFit;
    if (autoFit && m_imageItem) {
        fitInView(m_imageItem, Qt::KeepAspectRatio);
        // 计算并设置当前缩放因子
        QRectF viewRect = viewport()->rect();
        QRectF imageRect = m_imageItem->boundingRect();
        if (!imageRect.isEmpty()) {
            qreal scaleX = viewRect.width() / imageRect.width();
            qreal scaleY = viewRect.height() / imageRect.height();
            m_zoomFactor = qMin(scaleX, scaleY);
        }
        m_userZoomed = false;  // 重置用户缩放标记
        update();
    }
}
