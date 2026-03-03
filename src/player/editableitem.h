// editableitem.h
#ifndef EDITABLEITEM_H
#define EDITABLEITEM_H

#include <QGraphicsObject>  // 改为继承自 QGraphicsObject，这样就有了 QObject 的功能
#include <QPainter>
#include <QMenu>
#include <QAction>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneContextMenuEvent>
#include <QColorDialog>
#include <QInputDialog>

// enum HandleType {
//     NoHandle,
//     TopLeftHandle,
//     TopMiddleHandle,
//     TopRightHandle,
//     MiddleLeftHandle,
//     MiddleRightHandle,
//     BottomLeftHandle,
//     BottomMiddleHandle,
//     BottomRightHandle
// };

// 改为继承自 QGraphicsObject
class EditableItem : public QGraphicsObject
{
    Q_OBJECT

public:
    enum ItemType {
        LineItem,
        RectItem,
        PolygonItem
    };

    EditableItem(ItemType type, QGraphicsItem *parent = nullptr);

    // QGraphicsItem 接口
    QRectF boundingRect() const override = 0;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;

    // 属性设置
    virtual void setLabel(const QString &label);
    QString label() const { return m_label; }

    virtual void setLineColor(const QColor &color);
    QColor lineColor() const { return m_lineColor; }

    virtual void setFillColor(const QColor &color);
    QColor fillColor() const { return m_fillColor; }

    virtual void setLineWidth(int width);
    int lineWidth() const { return m_lineWidth; }

    void setSelected(bool selected);
    bool isSelected() const { return m_selected; }

    ItemType itemType() const { return m_type; }

    // 控制点相关
    void setHandlesVisible(bool visible);
    bool handlesVisible() const { return m_showHandles; }

    // 序列化
    virtual QByteArray saveData() const;
    virtual void loadData(const QByteArray &data);


    // 在 EditableItem 类的 public 部分添加：
    virtual int getVertexCount() const = 0;
    virtual QPointF getVertexPosition(int index) const = 0;
    virtual void setVertexPosition(int index, const QPointF &pos) = 0;


    void setContextMenuEnabled(bool enabled)
    {
        m_contextMenuEnabled = enabled;
    }

protected:
    // 鼠标事件
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

    // 上下文菜单
    void contextMenuEvent(QGraphicsSceneContextMenuEvent *event) override;

    // 悬停事件
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

    // 绘制控制点
    void drawHandles(QPainter *painter);

    // 控制点操作
    virtual int getHandleAt(const QPointF &pos) const;
    virtual QRectF getHandleRect(int handle) const;
    virtual void updateCursor(int handle);
    virtual void resizeItem(int vertexIndex, const QPointF &delta) = 0;

    // 纯虚函数 - 子类实现
    virtual void updateShape() = 0;

protected slots:
    // 上下文菜单槽函数
    void showPropertiesDialog();
    void changeLineColor();
    void changeFillColor();
    void changeLineWidth();
    void changeLabel();
    void deleteItem();

protected:
    ItemType m_type;
    QString m_label;
    QColor m_lineColor;
    QColor m_fillColor;
    int m_lineWidth;
    bool m_hasFillColor;

    // 选择状态
    bool m_selected;
    bool m_showHandles;

    // 控制点相关
    bool m_isResizing;
    bool m_isMoving;  // 添加这个
    int m_activeHandle;
    QPointF m_lastMousePos;
    QRectF m_originalRect;

    //用来解决右键冲突问题：多边形是右键结束，同时，右键又会弹出菜单
    bool m_contextMenuEnabled;  // 添加这个

    // 控制点大小
    static constexpr qreal HANDLE_SIZE = 10.0;

private:
    void createContextMenu();
    QMenu *m_contextMenu;
};

#endif // EDITABLEITEM_H
