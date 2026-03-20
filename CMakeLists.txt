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

set(OPENCV_LINK_LIBS)

# Prefer imported OpenCV targets when available because they preserve
# Debug/Release-specific library selection on multi-config generators
# such as Visual Studio.
if(TARGET opencv_world)
  list(APPEND OPENCV_LINK_LIBS opencv_world)
elseif(TARGET opencv_core)
  foreach(module IN ITEMS opencv_core opencv_imgproc opencv_imgcodecs)
    if(TARGET ${module})
      list(APPEND OPENCV_LINK_LIBS ${module})
    endif()
  endforeach()
endif()

# Fall back to package variables exported by older OpenCV CMake packages.
if(NOT OPENCV_LINK_LIBS)
  if(OpenCV_LIBS)
    list(APPEND OPENCV_LINK_LIBS ${OpenCV_LIBS})
  endif()
  if(OpenCV_LIBRARIES)
    list(APPEND OPENCV_LINK_LIBS ${OpenCV_LIBRARIES})
  endif()
endif()

# Some Windows OpenCV package layouts expose the module names but not as
# imported targets. Linking them explicitly avoids missing symbols such as
# GaussianBlur/cvtColor in Debug builds.
if(MSVC AND NOT OPENCV_LINK_LIBS)
  list(APPEND OPENCV_LINK_LIBS opencv_core opencv_imgproc opencv_imgcodecs)
endif()

list(REMOVE_DUPLICATES OPENCV_LINK_LIBS)

if(NOT OPENCV_LINK_LIBS)
  message(FATAL_ERROR "Unable to determine which OpenCV libraries to link. Check your OpenCV_DIR/CMAKE_PREFIX_PATH.")
endif()

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV link libs: ${OPENCV_LINK_LIBS}")

target_link_libraries(QtTrainableSegmentation PRIVATE
  Qt${QT_VERSION_MAJOR}::Widgets
  Qt${QT_VERSION_MAJOR}::PrintSupport
  ${OPENCV_LINK_LIBS}
)
