#include "ui/MainWindow.hpp"

#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

namespace contour {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    auto* central = new QWidget(this);
    auto* layout = new QVBoxLayout(central);

    m_frameLabel = new QLabel("Frame: -", central);
    m_measureLabel = new QLabel("Measurement: -", central);
    m_playButton = new QPushButton("Play", central);
    auto* stepButton = new QPushButton("Next Frame", central);

    layout->addWidget(m_frameLabel);
    layout->addWidget(m_measureLabel);
    layout->addWidget(m_playButton);
    layout->addWidget(stepButton);
    setCentralWidget(central);

    m_app.loadFrames({"frame-0", "frame-1", "frame-2"});

    m_timer = new QTimer(this);
    m_timer->setInterval(33);

    connect(m_playButton, &QPushButton::clicked, this, &MainWindow::onPlayPause);
    connect(stepButton, &QPushButton::clicked, this, &MainWindow::onStepNext);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::onTick);

    connect(&m_app.player(), &PlayerController::frameChanged, this, [this](const FrameData& frame) {
        m_frameLabel->setText(QString("Frame: %1 (%2)").arg(frame.index).arg(frame.payload));
        const auto measure = m_app.processCurrent();
        m_measureLabel->setText(QString("Measurement length: %1").arg(measure.contourLength));
    });

    refreshStatus();
}

void MainWindow::onPlayPause() {
    if (m_app.player().state().isPlaying) {
        m_app.player().pause();
        m_timer->stop();
    } else {
        m_app.player().play();
        m_timer->start();
    }
    refreshStatus();
}

void MainWindow::onStepNext() {
    m_app.player().step();
}

void MainWindow::onTick() {
    m_app.player().tick();
    refreshStatus();
}

void MainWindow::refreshStatus() {
    m_playButton->setText(m_app.player().state().isPlaying ? "Pause" : "Play");
}

} // namespace contour
