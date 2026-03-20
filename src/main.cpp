#include <QApplication>
#include <QFont>
#include <QGuiApplication>
#include <QScreen>

#include "MainWindow.h"

int main(int argc, char** argv) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
  QGuiApplication::setHighDpiScaleFactorRoundingPolicy(
      Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);
#endif

  QApplication app(argc, argv);
  app.setApplicationName("QtTrainableSegmentation");
  app.setOrganizationName("OpenAI");

  QFont font = app.font();
  font.setPointSizeF(qMax(11.0, font.pointSizeF() > 0 ? font.pointSizeF() + 1.0 : 11.0));
  app.setFont(font);

  app.setStyleSheet(
      "QMainWindow,QWidget{background:#20242b;color:#edf2f7;}"
      "QGroupBox{border:1px solid #40495b;border-radius:8px;margin-top:10px;padding-top:8px;font-weight:600;}"
      "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}"
      "QPushButton,QToolButton,QComboBox,QSpinBox{min-height:32px;padding:4px 8px;background:#2d3440;border:1px solid #4b5566;border-radius:6px;}"
      "QPushButton:hover,QToolButton:hover{background:#394356;}"
      "QPushButton:pressed,QToolButton:pressed{background:#1c222c;}"
      "QListWidget,QTextEdit{background:#181c23;border:1px solid #394150;border-radius:6px;}"
      "QLabel{color:#edf2f7;}"
      "QMenuBar,QToolBar{background:#262c36;}"
      "QScrollBar:vertical,QScrollBar:horizontal{background:#20242b;}"
  );

  MainWindow window;
  window.show();
  return app.exec();
}
