#include "ServiceServer.h"

#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTcpSocket>

#include "InferencePipeline.h"

ServiceServer::ServiceServer(QObject* parent)
    : QObject(parent) {
  connect(&server_, &QTcpServer::newConnection, this, &ServiceServer::onNewConnection);
}

bool ServiceServer::start(quint16 port, QString* error) {
  if (!server_.listen(QHostAddress::Any, port)) {
    if (error) *error = server_.errorString();
    return false;
  }
  return true;
}

void ServiceServer::onNewConnection() {
  while (server_.hasPendingConnections()) {
    QTcpSocket* socket = server_.nextPendingConnection();
    connect(socket, &QTcpSocket::readyRead, this, [this, socket]() { handleRequest(socket); });
    connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
  }
}

QByteArray ServiceServer::jsonResponse(int statusCode, const QByteArray& body) const {
  QByteArray response = "HTTP/1.1 " + QByteArray::number(statusCode) + (statusCode == 200 ? " OK\r\n" : " ERROR\r\n");
  response += "Content-Type: application/json\r\n";
  response += "Content-Length: " + QByteArray::number(body.size()) + "\r\n\r\n";
  response += body;
  return response;
}

void ServiceServer::handleRequest(QTcpSocket* socket) {
  const QByteArray request = socket->readAll();
  const QList<QByteArray> parts = request.split('\n');
  if (parts.isEmpty()) return;
  const QByteArray requestLine = parts.first().trimmed();
  if (requestLine.startsWith("GET /health")) {
    const QByteArray body = QJsonDocument(QJsonObject{{"status", "ok"}}).toJson(QJsonDocument::Compact);
    socket->write(jsonResponse(200, body));
    socket->disconnectFromHost();
    return;
  }

  if (!requestLine.startsWith("POST /infer")) {
    const QByteArray body = QJsonDocument(QJsonObject{{"error", "unsupported route"}}).toJson(QJsonDocument::Compact);
    socket->write(jsonResponse(404, body));
    socket->disconnectFromHost();
    return;
  }

  const int bodyStart = request.indexOf("\r\n\r\n");
  const QByteArray payload = bodyStart >= 0 ? request.mid(bodyStart + 4) : QByteArray();
  const QJsonDocument doc = QJsonDocument::fromJson(payload);
  if (!doc.isObject()) {
    const QByteArray body = QJsonDocument(QJsonObject{{"error", "invalid JSON payload"}}).toJson(QJsonDocument::Compact);
    socket->write(jsonResponse(400, body));
    socket->disconnectFromHost();
    return;
  }

  const QJsonObject obj = doc.object();
  const QString input = obj.value("input").toString();
  const QString output = obj.value("output").toString();
  const QString backend = obj.value("backend").toString("classical");
  QString error;
  bool ok = false;
  if (backend == QStringLiteral("onnx")) {
    ok = InferencePipeline::runOnnxInference({obj.value("model").toString(onnxModelPath_), input, output}, &error);
  } else {
    ok = InferencePipeline::runClassicalInference({obj.value("model").toString(classicalModelPath_), input, output, obj.value("probabilityClass").toInt(0)}, &error);
  }

  const QByteArray body = QJsonDocument(ok
      ? QJsonObject{{"status", "ok"}, {"output", output}}
      : QJsonObject{{"status", "error"}, {"error", error}}).toJson(QJsonDocument::Compact);
  socket->write(jsonResponse(ok ? 200 : 500, body));
  socket->disconnectFromHost();
}
