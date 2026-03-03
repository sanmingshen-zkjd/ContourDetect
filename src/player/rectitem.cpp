// rectitem.cpp
#include "rectitem.h"
#include <QDataStream>
#include <QPen>
#include <QBrush>

RectItem::RectItem(const QRectF &rect, QGraphicsItem *parent)
    : EditableItem(EditableItem::RectItem, parent)
    , m_rect(rect)
{
    setLineColor(Qt::blue);
    setFillColor(QColor(0, 0, 255, 30));
    setLineWidth(2);
    m_hasFillColor = true;
    updateShape();
}

void RectItem::setRect(const QRectF &rect)
{
    prepareGeometryChange();
    m_rect = rect;
    updateShape();
    update();
}

QRectF RectItem::rect() const
{
    return m_rect;
}

QRectF RectItem::boundingRect() const
{
    qreal extra = m_lineWidth / 2.0 + HANDLE_SIZE;
    return m_rect.adjusted(-extra, -extra, extra, extra);
}

void RectItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setRenderHint(QPainter::Antialiasing);
    QPen pen(m_lineColor, m_lineWidth);
    painter->setPen(pen);

    if (m_hasFillColor && m_fillColor.alpha() > 0) {
        painter->setBrush(m_fillColor);
    } else {
        painter->setBrush(Qt::NoBrush);
    }

    painter->drawRect(m_rect);

    // 绘制标签
    if (!m_label.isEmpty()) {
        painter->setPen(Qt::black);
        QPointF center = m_rect.center();
        painter->drawText(center, m_label);
    }

    // 绘制控制点
    EditableItem::paint(painter, option, widget);
}

QByteArray RectItem::saveData() const
{
    QByteArray data = EditableItem::saveData();
    QDataStream stream(&data, QIODevice::Append);
    stream << m_rect;
    return data;
}

void RectItem::loadData(const QByteArray &data)
{
    QDataStream stream(data);

    // 读取基类数据
    int type;
    stream >> type
        >> m_label
        >> m_lineColor
        >> m_fillColor
        >> m_lineWidth
        >> m_hasFillColor;

    QPointF position;
    qreal rotation, scaleValue;
    stream >> position >> rotation >> scaleValue;

    setPos(position);
    setRotation(rotation);
    setScale(scaleValue);

    // 读取子类数据
    stream >> m_rect;

    m_type = static_cast<ItemType>(type);
    updateShape();
    update();
}

void RectItem::resizeItem(int vertexIndex, const QPointF &delta)
{
    if (vertexIndex >= 0 && vertexIndex < 4) {
        QPointF newPos = getVertexPosition(vertexIndex) + delta;
        setVertexPosition(vertexIndex, newPos);
    }
}

void RectItem::updateShape()
{
    prepareGeometryChange();
}


// rectitem.cpp 中添加实现：
int RectItem::getVertexCount() const
{
    return 4;  // 矩形有4个顶点
}

QPointF RectItem::getVertexPosition(int index) const
{
    switch (index) {
    case 0: return m_rect.topLeft();      // 左上
    case 1: return m_rect.topRight();     // 右上
    case 2: return m_rect.bottomRight();  // 右下
    case 3: return m_rect.bottomLeft();   // 左下
    default: return QPointF();
    }
}

void RectItem::setVertexPosition(int index, const QPointF &pos)
{
    QRectF newRect = m_rect;

    switch (index) {
    case 0: newRect.setTopLeft(pos); break;
    case 1: newRect.setTopRight(pos); break;
    case 2: newRect.setBottomRight(pos); break;
    case 3: newRect.setBottomLeft(pos); break;
    }

    // 确保矩形有效
    if (newRect.width() < 10) newRect.setWidth(10);
    if (newRect.height() < 10) newRect.setHeight(10);

    m_rect = newRect;
    updateShape();
    update();
}
