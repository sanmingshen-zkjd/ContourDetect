// editableitem.cpp
#include "editableitem.h"
#include <QApplication>
#include <QGraphicsScene>
#include <QMessageBox>
#include <QDataStream>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QSpinBox>
#include <QDialogButtonBox>
#include <QPushButton>

EditableItem::EditableItem(ItemType type, QGraphicsItem *parent)
    : QGraphicsObject(parent)  // 使用 QGraphicsObject
    , m_type(type)
    , m_label("")
    , m_lineColor(Qt::red)
    , m_fillColor(Qt::transparent)
    , m_lineWidth(2)
    , m_hasFillColor(type != LineItem)
    , m_selected(false)
    , m_showHandles(false)
    , m_isResizing(false)
    , m_isMoving(false)
    , m_activeHandle(-1)
    , m_contextMenuEnabled(true)
    , m_contextMenu(nullptr)
{
    setFlag(QGraphicsItem::ItemIsSelectable, true);
    setFlag(QGraphicsItem::ItemIsMovable, true);
    setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    setFlag(QGraphicsItem::ItemSendsScenePositionChanges, true);
    setAcceptHoverEvents(true);

    createContextMenu();
}

void EditableItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    if (m_selected) {
        // 绘制控制点
        if (m_showHandles) {
            drawHandles(painter);
        }
    }
}

void EditableItem::setLabel(const QString &label)
{
    m_label = label;
    update();
}

void EditableItem::setLineColor(const QColor &color)
{
    m_lineColor = color;
    update();
}

void EditableItem::setFillColor(const QColor &color)
{
    m_fillColor = color;
    update();
}

void EditableItem::setLineWidth(int width)
{
    m_lineWidth = width;
    update();
}

void EditableItem::setSelected(bool selected)
{
    m_selected = selected;
    m_showHandles = selected;
    update();
}

void EditableItem::setHandlesVisible(bool visible)
{
    m_showHandles = visible;
    update();
}

QByteArray EditableItem::saveData() const
{
    QByteArray data;
    QDataStream stream(&data, QIODevice::WriteOnly);
    stream << static_cast<int>(m_type)
           << m_label
           << m_lineColor
           << m_fillColor
           << m_lineWidth
           << m_hasFillColor
           << pos()
           << rotation()
           << scale();
    return data;
}

void EditableItem::loadData(const QByteArray &data)
{
    QDataStream stream(data);
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

    m_type = static_cast<ItemType>(type);
    update();
}

void EditableItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    // 首先选中图形（如果还未选中）
    if (!m_selected) {
        setSelected(true);
        m_isMoving = true;
        QGraphicsObject::mousePressEvent(event);
        return;
    }
    // 图形已选中，检查是否点击了控制点
    m_activeHandle = getHandleAt(event->pos());
    if (m_activeHandle != -1) {
        // 点击了控制点，开始调整大小
        m_isResizing = true;
        m_lastMousePos = event->scenePos();
        m_originalRect = boundingRect();
        event->accept();
        return;
    }
    // 点击了图形但不在控制点上，可以移动整个图形
    m_isMoving = true;
    QGraphicsObject::mousePressEvent(event);
}

void EditableItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if (m_isResizing && m_activeHandle != -1) {
        QPointF delta = event->scenePos() - m_lastMousePos;
        resizeItem(m_activeHandle, delta);
        m_lastMousePos = event->scenePos();
        // 强制更新，防止拖影
        update();
        if (scene()) {
            scene()->update(); // 更新整个场景
        }
        event->accept();
    } else if (m_isMoving) {
        // 正常移动时也强制更新
        QGraphicsObject::mouseMoveEvent(event);
        update();
        if (scene()) {
            scene()->update();
        }
    } else {
        QGraphicsObject::mouseMoveEvent(event);
    }
}

void EditableItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    if (m_isResizing) {
        m_isResizing = false;
        m_activeHandle = -1;
        // 释放时再次更新，清除残留
        update();
        if (scene()) {
            scene()->update();
        }
        event->accept();
    } else {
        m_isMoving = false;
        QGraphicsObject::mouseReleaseEvent(event);
    }
}

void EditableItem::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    // 如果右键菜单被禁用，则不显示
    if (!m_contextMenuEnabled) {
        event->ignore();
        return;
    }
    // 显示右键菜单
    if (m_contextMenu) {
        m_contextMenu->exec(event->screenPos());
        event->accept();  // 标记事件已处理
    } else {
        event->ignore();
    }
}

void EditableItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    if (m_selected && m_showHandles) {
        int handle = getHandleAt(event->pos());
        updateCursor(handle);
    } else {
        setCursor(Qt::ArrowCursor);
    }
    QGraphicsObject::hoverEnterEvent(event);
}

void EditableItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    if (m_selected && m_showHandles) {
        int handle = getHandleAt(event->pos());
        updateCursor(handle);
    } else {
        setCursor(Qt::ArrowCursor);
    }
    QGraphicsObject::hoverMoveEvent(event);
}

void EditableItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    setCursor(Qt::ArrowCursor);
    QGraphicsObject::hoverLeaveEvent(event);
}

void EditableItem::drawHandles(QPainter *painter)
{
    painter->save();
    QBrush handleBrush(Qt::white);
    QPen handlePen(Qt::blue, 2);
    painter->setBrush(handleBrush);
    painter->setPen(handlePen);
    int vertexCount = getVertexCount();
    for (int i = 0; i < vertexCount; ++i) {
        QRectF rect = getHandleRect(i);
        if (!rect.isNull()) {
            painter->drawRect(rect);
        }
    }
    painter->restore();
}

int EditableItem::getHandleAt(const QPointF &pos) const
{
    int vertexCount = getVertexCount();
    for (int i = 0; i < vertexCount; ++i) {
        QRectF rect = getHandleRect(i);
        if (rect.contains(pos)) {
            return i;  // 返回顶点索引
        }
    }
    return -1;  // 没有找到控制点
}

QRectF EditableItem::getHandleRect(int vertexIndex) const
{
    if (vertexIndex >= 0 && vertexIndex < getVertexCount()) {
        QPointF vertexPos = getVertexPosition(vertexIndex);
        qreal halfHandle = HANDLE_SIZE / 2.0;
        return QRectF(vertexPos - QPointF(halfHandle, halfHandle),
                      QSizeF(HANDLE_SIZE, HANDLE_SIZE));
    }
    return QRectF();
}

void EditableItem::updateCursor(int vertexIndex)
{
    // 所有控制点都显示为十字光标或移动光标
    if (vertexIndex >= 0) {
        setCursor(Qt::SizeAllCursor);  // 或者 Qt::CrossCursor
    } else {
        setCursor(Qt::ArrowCursor);
    }
}

void EditableItem::createContextMenu()
{
    m_contextMenu = new QMenu();

    QAction *propertiesAction = m_contextMenu->addAction("property...");
    connect(propertiesAction, &QAction::triggered, this, &EditableItem::showPropertiesDialog);

    QAction *lineColorAction = m_contextMenu->addAction("modify color...");
    connect(lineColorAction, &QAction::triggered, this, &EditableItem::changeLineColor);

    if (m_hasFillColor) {
        QAction *fillColorAction = m_contextMenu->addAction("modify fill color...");
        connect(fillColorAction, &QAction::triggered, this, &EditableItem::changeFillColor);
    }

    QAction *lineWidthAction = m_contextMenu->addAction("modify line width...");
    connect(lineWidthAction, &QAction::triggered, this, &EditableItem::changeLineWidth);

    QAction *labelAction = m_contextMenu->addAction("modify label...");
    connect(labelAction, &QAction::triggered, this, &EditableItem::changeLabel);

    m_contextMenu->addSeparator();

    QAction *deleteAction = m_contextMenu->addAction("delete");
    connect(deleteAction, &QAction::triggered, this, &EditableItem::deleteItem);
}

void EditableItem::showPropertiesDialog()
{
    QDialog dialog;
    dialog.setWindowTitle("image property");
    dialog.setMinimumWidth(300);

    QVBoxLayout *layout = new QVBoxLayout(&dialog);

    QFormLayout *formLayout = new QFormLayout();

    // 标签
    QLineEdit *labelEdit = new QLineEdit(m_label, &dialog);
    formLayout->addRow("标签:", labelEdit);

    // 线宽
    QSpinBox *lineWidthSpin = new QSpinBox(&dialog);
    lineWidthSpin->setRange(1, 20);
    lineWidthSpin->setValue(m_lineWidth);
    formLayout->addRow("线宽:", lineWidthSpin);

    // 线条颜色
    QPushButton *lineColorBtn = new QPushButton("选择...", &dialog);
    lineColorBtn->setStyleSheet(QString("background-color: %1;").arg(m_lineColor.name()));
    connect(lineColorBtn, &QPushButton::clicked, [&]() {
        QColor color = QColorDialog::getColor(m_lineColor, &dialog, "选择线条颜色");
        if (color.isValid()) {
            setLineColor(color);
            lineColorBtn->setStyleSheet(QString("background-color: %1;").arg(color.name()));
        }
    });
    formLayout->addRow("线条颜色:", lineColorBtn);

    // 填充颜色（如果适用）
    if (m_hasFillColor) {
        QPushButton *fillColorBtn = new QPushButton("choose...", &dialog);
        QString style = QString("background-color: %1;").arg(m_fillColor.name());
        if (m_fillColor.alpha() == 0) {
            style = "background-color: white; border: 1px solid gray;";
            fillColorBtn->setText("no fill");
        }
        fillColorBtn->setStyleSheet(style);

        connect(fillColorBtn, &QPushButton::clicked, [&]() {
            QColor color = QColorDialog::getColor(m_fillColor, &dialog, "select fill color",
                                                  QColorDialog::ShowAlphaChannel);
            if (color.isValid()) {
                setFillColor(color);
                QString style = QString("background-color: %1;").arg(color.name());
                if (color.alpha() == 0) {
                    style = "background-color: white; border: 1px solid gray;";
                    fillColorBtn->setText("no fill");
                } else {
                    fillColorBtn->setText("");
                }
                fillColorBtn->setStyleSheet(style);
            }
        });
        formLayout->addRow("fill color:", fillColorBtn);
    }

    layout->addLayout(formLayout);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
                                                       Qt::Horizontal, &dialog);
    layout->addWidget(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    if (dialog.exec() == QDialog::Accepted) {
        setLabel(labelEdit->text());
        setLineWidth(lineWidthSpin->value());
    }
}

void EditableItem::changeLineColor()
{
    QColor color = QColorDialog::getColor(m_lineColor, nullptr, "选择线条颜色");
    if (color.isValid()) {
        setLineColor(color);
    }
}

void EditableItem::changeFillColor()
{
    if (!m_hasFillColor) return;

    QColor color = QColorDialog::getColor(m_fillColor, nullptr, "选择填充颜色",
                                          QColorDialog::ShowAlphaChannel);
    if (color.isValid()) {
        setFillColor(color);
    }
}

void EditableItem::changeLineWidth()
{
    bool ok;
    int width = QInputDialog::getInt(nullptr, "修改线宽", "请输入线宽(1-20):",
                                     m_lineWidth, 1, 20, 1, &ok);
    if (ok) {
        setLineWidth(width);
    }
}

void EditableItem::changeLabel()
{
    bool ok;
    QString label = QInputDialog::getText(nullptr, "modify label", "please input label:",QLineEdit::Normal, m_label, &ok);
    if (ok && !label.isEmpty()) {
        setLabel(label);
    }
}

void EditableItem::deleteItem()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(nullptr,"delete","are u sure to delete?",QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::Yes) 
    {
        this->deleteLater();
    }
}
