cmake_minimum_required(VERSION 3.16)
project(QtTrainableSegmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt6 Qt5 COMPONENTS Widgets PrintSupport REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets PrintSupport REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(QtTrainableSegmentation
  src/main.cpp
  src/MainWindow.cpp
  src/MainWindow.h
  src/SegmentationEngine.cpp
  src/SegmentationEngine.h
  src/SegmentationView.cpp
  src/SegmentationView.h
  src/QCustomPlot.cpp
  src/QCustomPlot.h
  assets/resourcefile.qrc
)

target_include_directories(QtTrainableSegmentation PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(QtTrainableSegmentation PRIVATE
  Qt${QT_VERSION_MAJOR}::Widgets
  Qt${QT_VERSION_MAJOR}::PrintSupport
  ${OpenCV_LIBS}
)
