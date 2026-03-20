#pragma once

#include <QObject>
#include <QTcpServer>

class QTcpSocket;

class ServiceServer : public QObject {
  Q_OBJECT
public:
  explicit ServiceServer(QObject* parent = nullptr);
  bool start(quint16 port, QString* error = nullptr);

  void setClassicalModelPath(const QString& path) { classicalModelPath_ = path; }
  void setOnnxModelPath(const QString& path) { onnxModelPath_ = path; }

private slots:
  void onNewConnection();

private:
  void handleRequest(QTcpSocket* socket);
  QByteArray jsonResponse(int statusCode, const QByteArray& body) const;

  QTcpServer server_;
  QString classicalModelPath_;
  QString onnxModelPath_;
};
