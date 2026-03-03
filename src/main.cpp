#include "ui/MainWindow.hpp"

#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    contour::MainWindow window;
    window.resize(480, 240);
    window.show();
    return app.exec();
}
