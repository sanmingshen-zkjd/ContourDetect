// lineitem.cpp
#include "lineitem.h"
#include <QDataStream>
#include <QPen>
#include <QBrush>

LineItem::LineItem(const QLineF &line, QGraphicsItem *parent)
    : EditableItem(EditableItem::LineItem, parent)  // 调用基类构造函数
    , m_line(line)
{
    setLineColor(Qt::red);
    setLineWidth(2);
    m_hasFillColor = false;
    updateShape();
}

void LineItem::setLine(const QLineF &line)
{
    prepareGeometryChange();
    m_line = line;
    updateShape();
    update();
}

QLineF LineItem::line() const
{
    return m_line;
}

QRectF LineItem::boundingRect() const
{
    // 使用shape的boundingRect作为基础，但稍微扩大以包含控制点
    QRectF shapeRect = shape().boundingRect();

    if (shapeRect.isEmpty()) {
        qreal extra = m_lineWidth / 2.0 + HANDLE_SIZE;
        return QRectF(m_line.p1(), m_line.p1())
            .normalized()
            .adjusted(-extra, -extra, extra, extra);
    }

    // 添加控制点的空间
    qreal handleExtra = HANDLE_SIZE;
    QRectF handlesRect = QRectF(m_line.p1(), m_line.p2())
                             .normalized()
                             .adjusted(-handleExtra, -handleExtra, handleExtra, handleExtra);

    return shapeRect.united(handlesRect);
}

void LineItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setRenderHint(QPainter::Antialiasing);
    QPen pen(m_lineColor, m_lineWidth);
    pen.setCapStyle(Qt::RoundCap);
    painter->setPen(pen);
    painter->drawLine(m_line);
    // 绘制标签
    if (!m_label.isEmpty()) {
        painter->setPen(Qt::black);
        QPointF center = (m_line.p1() + m_line.p2()) / 2.0;
        painter->drawText(center, m_label);
    }
    // 绘制控制点
    EditableItem::paint(painter, option, widget);
}

QByteArray LineItem::saveData() const
{
    QByteArray data = EditableItem::saveData();
    QDataStream stream(&data, QIODevice::Append);
    stream << m_line;
    return data;
}

void LineItem::loadData(const QByteArray &data)
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
    stream >> m_line;
    m_type = static_cast<ItemType>(type);
    updateShape();
    update();
}

void LineItem::resizeItem(int vertexIndex, const QPointF &delta)
{
    if (vertexIndex >= 0 && vertexIndex < 2) {
        QPointF newPos = getVertexPosition(vertexIndex) + delta;
        setVertexPosition(vertexIndex, newPos);
    }
}

void LineItem::updateShape()
{
    // 更新边界矩形
    prepareGeometryChange();
}


// lineitem.cpp 中添加实现：
int LineItem::getVertexCount() const
{
    return 2;  // 直线有两个端点
}

QPointF LineItem::getVertexPosition(int index) const
{
    if (index == 0) return m_line.p1();
    if (index == 1) return m_line.p2();
    return QPointF();
}

void LineItem::setVertexPosition(int index, const QPointF &pos)
{
    if (index == 0) {
        m_line.setP1(pos);
    } else if (index == 1) {
        m_line.setP2(pos);
    }
    updateShape();
    update();
}


QRectF LineItem::calculateRotatedBoundingRect() const
{
    if (m_line.isNull())
        return QRectF();
    QPointF p1 = m_line.p1();
    QPointF p2 = m_line.p2();
    // 计算直线长度和角度
    qreal length = m_line.length();
    if (length == 0) {
        // 如果直线长度为0，返回一个点的小矩形
        qreal extra = m_lineWidth / 2.0 + HANDLE_SIZE;
        return QRectF(p1.x() - extra, p1.y() - extra,
                      2 * extra, 2 * extra);
    }
    // 计算方向向量
    QPointF dir = (p2 - p1) / length;
    // 计算垂直方向（法向量）
    QPointF normal(-dir.y(), dir.x());
    // 计算线的半宽（包括画笔宽度）
    qreal halfWidth = m_lineWidth /2.0;
    // 计算四个顶点
    QPointF points[4] = {
        p1 + normal * halfWidth,  // 左上
        p1 - normal * halfWidth,  // 左下
        p2 - normal * halfWidth,  // 右下
        p2 + normal * halfWidth   // 右上
    };
    // 找到最小和最大的x、y值
    qreal minX = points[0].x();
    qreal maxX = points[0].x();
    qreal minY = points[0].y();
    qreal maxY = points[0].y();
    for (int i = 1; i < 4; ++i) {
        if (points[i].x() < minX) minX = points[i].x();
        if (points[i].x() > maxX) maxX = points[i].x();
        if (points[i].y() < minY) minY = points[i].y();
        if (points[i].y() > maxY) maxY = points[i].y();
    }
    // 考虑控制点的大小
    qreal handleExtra = HANDLE_SIZE;
    return QRectF(minX - handleExtra, minY - handleExtra,
                  maxX - minX + 2 * handleExtra,
                  maxY - minY + 2 * handleExtra);
}

// lineitem.cpp 中的实现
QPainterPath LineItem::shape() const
{
    // 创建一个函数，返回直线的多边形表示，增加选择容差
    auto createLinePolygon = [](const QLineF& line, qreal width, qreal selectionTolerance = 10.0) -> QPolygonF {
        if (line.isNull())
            return QPolygonF();

        QPointF p1 = line.p1();
        QPointF p2 = line.p2();
        qreal length = line.length();

        if (length < 0.001) {
            // 点的情况，返回一个扩大的矩形
            qreal radius = width/2 + selectionTolerance;
            return QRectF(p1.x() - radius, p1.y() - radius, 2 * radius, 2 * radius);
        }

        // 计算方向向量和法向量
        QPointF dir = (p2 - p1) / length;
        QPointF normal(-dir.y(), dir.x());

        // 增加选择容差
        qreal totalWidth = width + 2 * selectionTolerance;
        QPointF offset = normal * (totalWidth / 2);

        // 考虑圆头，在端点处延伸
        QPointF extend = dir * (totalWidth / 4);

        // 创建多边形
        QPolygonF polygon;
        polygon << p1 - extend + offset  // 左上
                << p1 - extend - offset  // 左下
                << p2 + extend - offset  // 右下
                << p2 + extend + offset; // 右上

        return polygon;
    };

    // 将多边形转换为QPainterPath，使用10像素的选择容差
    QPolygonF polygon = createLinePolygon(m_line, m_lineWidth, 10.0);
    QPainterPath path;
    path.addPolygon(polygon);
    path.closeSubpath();  // 封闭多边形

    return path;
}
