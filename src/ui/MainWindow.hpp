#pragma once

#include "app/MonocularContourApp.hpp"

#include <QMainWindow>

class QLabel;
class QPushButton;
class QTimer;

namespace contour {

class MainWindow final : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onPlayPause();
    void onStepNext();
    void onTick();

private:
    void refreshStatus();

    MonocularContourApp m_app;
    QTimer* m_timer {nullptr};
    QLabel* m_frameLabel {nullptr};
    QLabel* m_measureLabel {nullptr};
    QPushButton* m_playButton {nullptr};
};

} // namespace contour
