#include <QApplication>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QFont>
#include <QGuiApplication>
#include <QTextStream>

#include "InferencePipeline.h"
#include "MainWindow.h"
#include "ServiceServer.h"

namespace {
int runHeadless(int argc, char** argv) {
  QCoreApplication app(argc, argv);
  app.setApplicationName("QtTrainableSegmentation");
  QCommandLineParser parser;
  parser.setApplicationDescription("Qt Trainable Segmentation headless/service mode");
  parser.addHelpOption();
  parser.addOption({"headless", "Run headless inference mode."});
  parser.addOption({"server", "Run HTTP inference service."});
  parser.addOption({"model", "Classical model YAML path.", "path"});
  parser.addOption({"onnx-model", "ONNX segmentation model path.", "path"});
  parser.addOption({"input", "Input image file or directory.", "path"});
  parser.addOption({"output", "Output file or directory.", "path"});
  parser.addOption({"probability-class", "Optional probability class index.", "index", "0"});
  parser.addOption({"backend", "Inference backend: classical or onnx.", "name", "classical"});
  parser.addOption({"port", "HTTP service port.", "port", "8080"});
  parser.process(app);

  if (parser.isSet("server")) {
    ServiceServer server;
    server.setClassicalModelPath(parser.value("model"));
    server.setOnnxModelPath(parser.value("onnx-model"));
    QString error;
    if (!server.start(static_cast<quint16>(parser.value("port").toUShort()), &error)) {
      QTextStream(stderr) << "Failed to start service: " << error << "\n";
      return 2;
    }
    QTextStream(stdout) << "Service listening on port " << parser.value("port") << "\n";
    return app.exec();
  }

  if (!parser.isSet("headless")) {
    return -1;
  }
  if (!parser.isSet("input") || !parser.isSet("output")) {
    QTextStream(stderr) << "--headless requires --input and --output\n";
    return 3;
  }

  QString error;
  const QString backend = parser.value("backend");
  bool ok = false;
  if (backend == QStringLiteral("onnx")) {
    if (!parser.isSet("onnx-model")) {
      QTextStream(stderr) << "--backend onnx requires --onnx-model\n";
      return 4;
    }
    ok = InferencePipeline::runOnnxInference({parser.value("onnx-model"), parser.value("input"), parser.value("output")}, &error);
  } else {
    if (!parser.isSet("model")) {
      QTextStream(stderr) << "Classical backend requires --model\n";
      return 5;
    }
    ok = InferencePipeline::runClassicalInference({parser.value("model"), parser.value("input"), parser.value("output"), parser.value("probability-class").toInt()}, &error);
  }
  if (!ok) {
    QTextStream(stderr) << error << "\n";
    return 6;
  }
  return 0;
}
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const QString arg = QString::fromLocal8Bit(argv[i]);
    if (arg == QStringLiteral("--headless") || arg == QStringLiteral("--server")) {
      return runHeadless(argc, argv);
    }
  }

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
