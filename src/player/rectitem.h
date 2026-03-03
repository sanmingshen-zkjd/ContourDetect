// rectitem.h
#ifndef RECTITEM_H
#define RECTITEM_H

#include "editableitem.h"

class RectItem : public EditableItem
{
public:
    RectItem(const QRectF &rect = QRectF(), QGraphicsItem *parent = nullptr);

    void setRect(const QRectF &rect);
    QRectF rect() const;

    // QGraphicsItem 接口
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

    // 序列化
    QByteArray saveData() const override;
    void loadData(const QByteArray &data) override;

    int getVertexCount() const override;
    QPointF getVertexPosition(int index) const override;
    void setVertexPosition(int index, const QPointF &pos) override;

protected:
    // 控制点操作
    void resizeItem(int vertexIndex, const QPointF &delta) override;
    void updateShape() override;

private:
    QRectF m_rect;
};

#endif // RECTITEM_H
