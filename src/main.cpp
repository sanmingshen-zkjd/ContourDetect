#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QTextStream>
#include <QFont>
#include <QGuiApplication>

#include <opencv2/imgcodecs.hpp>

#include "ImageIOUtils.h"
#include "MainWindow.h"
#include "SegmentationEngine.h"

namespace {
int runCli(int argc, char** argv) {
  QCoreApplication app(argc, argv);
  app.setApplicationName("QtTrainableSegmentation");
  QCommandLineParser parser;
  parser.setApplicationDescription("Qt Trainable Segmentation CLI inference mode");
  parser.addHelpOption();
  parser.addOption({"cli", "Run CLI inference mode."});
  parser.addOption({"model", "Classifier YAML path.", "path"});
  parser.addOption({"input", "Input image file or directory.", "path"});
  parser.addOption({"output", "Output file or directory.", "path"});
  parser.addOption({"probability-class", "Optional probability class index.", "index", "0"});
  parser.process(app);

  if (!parser.isSet("cli")) {
    return -1;
  }
  if (!parser.isSet("model") || !parser.isSet("input") || !parser.isSet("output")) {
    QTextStream(stderr) << "--cli requires --model, --input, and --output\n";
    return 2;
  }

  SegmentationClassifier model;
  std::vector<SegmentationClassInfo> classes;
  SegmentationFeatureSettings featureSettings;
  SegmentationClassifierSettings classifierSettings;
  SegmentationTrainingStats stats;
  if (!model.load(parser.value("model"), &classes, &featureSettings, &classifierSettings, &stats)) {
    QTextStream(stderr) << "Failed to load model: " << parser.value("model") << "\n";
    return 3;
  }

  const QString inputPath = parser.value("input");
  const QString outputPath = parser.value("output");
  const int probabilityClass = parser.value("probability-class").toInt();

  QStringList inputs;
  QFileInfo inputInfo(inputPath);
  if (inputInfo.isDir()) {
    QDirIterator it(inputPath, QStringList() << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp" << "*.tif" << "*.tiff", QDir::Files);
    while (it.hasNext()) inputs << it.next();
  } else {
    inputs << inputPath;
  }
  if (inputs.isEmpty()) {
    QTextStream(stderr) << "No input images found.\n";
    return 4;
  }
  if (inputs.size() > 1) {
    QDir().mkpath(outputPath);
  }

  for (const QString& imagePath : inputs) {
    std::vector<cv::Mat> slices;
    QString error;
    if (!ImageIOUtils::loadImageVolume(imagePath, &slices, &error) || slices.empty()) {
      QTextStream(stderr) << error << "\n";
      return 5;
    }
    const QString baseName = QFileInfo(imagePath).completeBaseName();
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(slices.size()); ++sliceIndex) {
      const cv::Mat featureStack = SegmentationEngine::computeFeatureStack(slices[sliceIndex], featureSettings);
      const cv::Mat labels = SegmentationEngine::applyModelLabels(model, featureStack, slices[sliceIndex].rows, slices[sliceIndex].cols);
      if (labels.empty()) {
        QTextStream(stderr) << "Failed to infer: " << imagePath << "\n";
        return 6;
      }
      cv::Mat exportMask;
      labels.convertTo(exportMask, CV_8U);
      QString targetPath;
      if (inputs.size() == 1 && slices.size() == 1) {
        targetPath = outputPath;
      } else {
        targetPath = QDir(outputPath).filePath(QString("%1_slice_%2.png").arg(baseName).arg(sliceIndex, 3, 10, QLatin1Char('0')));
      }
      if (!cv::imwrite(targetPath.toStdString(), exportMask)) {
        QTextStream(stderr) << "Failed to write: " << targetPath << "\n";
        return 7;
      }
      if (model.supportsProbability() && probabilityClass >= 0 && probabilityClass < static_cast<int>(classes.size())) {
        const cv::Mat probs = SegmentationEngine::applyModelProbabilities(model, featureStack, slices[sliceIndex].rows, slices[sliceIndex].cols, probabilityClass);
        if (!probs.empty()) {
          cv::Mat prob8;
          probs.convertTo(prob8, CV_8U, 255.0);
          const QString probPath = targetPath.left(targetPath.lastIndexOf('.')) + QString("_prob_class_%1.png").arg(probabilityClass);
          cv::imwrite(probPath.toStdString(), prob8);
        }
      }
    }
  }
  return 0;
}
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (QString::fromLocal8Bit(argv[i]) == QStringLiteral("--cli")) {
      return runCli(argc, argv);
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
