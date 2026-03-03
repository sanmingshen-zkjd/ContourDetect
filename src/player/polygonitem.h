#ifndef POLYGONITEM_H
#define POLYGONITEM_H

#include "editableitem.h"
#include <QPolygonF>

class PolygonItem : public EditableItem
{
public:
    PolygonItem(const QPolygonF &polygon = QPolygonF(), QGraphicsItem *parent = nullptr);

    void setPolygon(const QPolygonF &polygon);
    QPolygonF polygon() const;

    // QGraphicsItem 接口
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

    // 序列化
    QByteArray saveData() const override;
    void loadData(const QByteArray &data) override;

    // 顶点控制接口
    int getVertexCount() const override;
    QPointF getVertexPosition(int index) const override;
    void setVertexPosition(int index, const QPointF &pos) override;

protected:
    // 控制点操作
    void resizeItem(int vertexIndex, const QPointF &delta) override;
    void updateShape() override;

private:
    QPolygonF m_polygon;
};

#endif // POLYGONITEM_H
