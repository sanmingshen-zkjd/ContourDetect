// lineitem.h
#ifndef LINEITEM_H
#define LINEITEM_H

#include "editableitem.h"
#include <QLineF>

// 也要修改为继承自 EditableItem
class LineItem : public EditableItem
{
public:
    LineItem(const QLineF &line = QLineF(), QGraphicsItem *parent = nullptr);

    void setLine(const QLineF &line);
    QLineF line() const;

    // QGraphicsItem 接口
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

    // 序列化
    QByteArray saveData() const override;
    void loadData(const QByteArray &data) override;

    int getVertexCount() const override;
    QPointF getVertexPosition(int index) const override;
    void setVertexPosition(int index, const QPointF &pos) override;


private:
    // 辅助函数：计算旋转的包围矩形
    QRectF calculateRotatedBoundingRect() const;

protected:
    // 控制点操作
    void resizeItem(int vertexIndex, const QPointF &delta) override;
    void updateShape() override;
    QPainterPath shape() const override;

private:
    QLineF m_line;
};

#endif // LINEITEM_H
