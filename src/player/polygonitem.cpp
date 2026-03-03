// polygonitem.cpp - 完整实现
#include "polygonitem.h"
#include <QDataStream>
#include <QPen>
#include <QBrush>
#include <QPainter>
#include <QDebug>

PolygonItem::PolygonItem(const QPolygonF &polygon, QGraphicsItem *parent)
    : EditableItem(EditableItem::PolygonItem, parent)
    , m_polygon(polygon)
{
    // 确保多边形至少有一个点
    if (m_polygon.isEmpty())
    {
        // 添加一个默认点避免空多边形
        m_polygon << QPointF(0, 0) << QPointF(50, 0) << QPointF(25, 50);
    }
    // 设置填充颜色（确保alpha值不为0）
    setFillColor(QColor(0, 255, 0, 50));  // 增加alpha值，确保可见
    setLineColor(Qt::green);
    setLineWidth(2);
    // 确保多边形有填充颜色
    m_hasFillColor = true;
    updateShape();
}

void PolygonItem::setPolygon(const QPolygonF &polygon)
{
    prepareGeometryChange();
    m_polygon = polygon;
    updateShape();
    update();
}

QPolygonF PolygonItem::polygon() const
{
    return m_polygon;
}

QRectF PolygonItem::boundingRect() const
{
    if (m_polygon.isEmpty())
    {
        return QRectF();
    }
    qreal extra = m_lineWidth / 2.0 + HANDLE_SIZE;
    QRectF rect = m_polygon.boundingRect();
    return rect.adjusted(-extra, -extra, extra, extra);
}

void PolygonItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    if (m_polygon.isEmpty() || m_polygon.size() < 3)
    {
        return;
    }
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    // 绘制多边形填充
    if (m_hasFillColor && m_fillColor.alpha() > 0)
    {
        painter->setBrush(m_fillColor);
    }
    else
    {
        painter->setBrush(Qt::NoBrush);
    }
    // 绘制多边形边框
    QPen pen(m_lineColor, m_lineWidth);
    pen.setJoinStyle(Qt::MiterJoin);
    painter->setPen(pen);
    painter->drawPolygon(m_polygon);
    // 绘制标签
    if (!m_label.isEmpty())
    {
        painter->setPen(Qt::black);
        QPointF center = m_polygon.boundingRect().center();
        painter->drawText(center, m_label);
    }
    painter->restore();
    // 绘制控制点（基类实现）
    EditableItem::paint(painter, option, widget);
}

QByteArray PolygonItem::saveData() const
{
    QByteArray data = EditableItem::saveData();
    QDataStream stream(&data, QIODevice::Append);
    stream << m_polygon;
    return data;
}

void PolygonItem::loadData(const QByteArray &data)
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
    stream >> m_polygon;

    // 验证多边形数据
    if (m_polygon.size() < 3) {
        qWarning() << "Loaded polygon has less than 3 points, creating default triangle";
        m_polygon.clear();
        m_polygon << QPointF(0, 0) << QPointF(50, 0) << QPointF(25, 50);
    }

    m_type = static_cast<ItemType>(type);
    updateShape();
    update();
}

// 顶点控制接口实现
int PolygonItem::getVertexCount() const
{
    return m_polygon.size();
}

QPointF PolygonItem::getVertexPosition(int index) const
{
    if (index >= 0 && index < m_polygon.size())
    {
        return m_polygon[index];
    }
    qWarning() << "Invalid vertex index:" << index;
    return QPointF();
}

void PolygonItem::setVertexPosition(int index, const QPointF &pos)
{
    if (index >= 0 && index < m_polygon.size())
    {
        prepareGeometryChange();
        m_polygon[index] = pos;
        updateShape();
        update();
    }
    else
    {
        qWarning() << "Invalid vertex index for setVertexPosition:" << index;
    }
}

void PolygonItem::resizeItem(int vertexIndex, const QPointF &delta)
{
    if (vertexIndex >= 0 && vertexIndex < m_polygon.size())
    {
        prepareGeometryChange();
        m_polygon[vertexIndex] += delta;
        updateShape();
        update();
    }
    else
    {
        qWarning() << "Invalid vertex index for resizeItem:" << vertexIndex;
    }
}

void PolygonItem::updateShape()
{
    prepareGeometryChange();
    // 这里可以添加额外的形状更新逻辑
}
