cmake_minimum_required(VERSION 3.16)
project(multicam_rig_toolkit_qt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows target: Qt 5.15 + Visual Studio 2019 (MSVC)
# Use Qt5 Widgets only (no Qt6 fallback) per project requirement.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 查找依赖包
find_package(Qt5 5.15 COMPONENTS Widgets PrintSupport REQUIRED)
find_package(OpenCV 4.8 REQUIRED) # 明确指定OpenCV版本，更精准
find_package(Eigen3 REQUIRED)

# 定义第三方库的路径（建议改成相对路径或通过环境变量传递，避免硬编码）
set(THIRD_PARTY_ROOT "D:/thirdparty")
set(CERES_ROOT "D:/MyWork/FFTDemo/FFTDemo/packages/ceres.2.0.0/build/native")

# 包含目录（使用target_include_directories替代全局include_directories）
set(EXTRA_INCLUDE_DIRS
    ${THIRD_PARTY_ROOT}/opencv/opencv/opencv-4.8.0/build/install/include
    ${CERES_ROOT}/include
    ${CERES_ROOT}/include/ceres/internal/miniglog
)

# 按编译类型定义库路径和库文件
set(OPENCV_DEBUG_LIB_DIR "${THIRD_PARTY_ROOT}/opencv/opencv/opencv-4.8.0/build/lib/Debug")
set(OPENCV_RELEASE_LIB_DIR "${THIRD_PARTY_ROOT}/opencv/opencv/opencv-4.8.0/build/lib/Release")
set(CERES_DEBUG_LIB_DIR "${CERES_ROOT}/lib/x64/Debug")
set(CERES_RELEASE_LIB_DIR "${CERES_ROOT}/lib/x64/Release")

# 定义需要链接的库文件（按Debug/Release区分）
set(DEBUG_LIBS
    ${OPENCV_DEBUG_LIB_DIR}/opencv_highgui480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_video480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_aruco480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_core480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_videoio480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_videostab480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_calib3d480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_ccalib480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_imgproc480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_imgcodecs480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_objdetect480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_tracking480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_ximgproc480d.lib
    ${OPENCV_DEBUG_LIB_DIR}/opencv_xobjdetect480d.lib
    ${CERES_DEBUG_LIB_DIR}/ceres-debug.lib
)

set(RELEASE_LIBS
    ${OPENCV_RELEASE_LIB_DIR}/opencv_highgui480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_video480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_aruco480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_core480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_videoio480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_videostab480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_calib3d480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_ccalib480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_imgproc480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_imgcodecs480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_objdetect480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_tracking480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_ximgproc480.lib
    ${OPENCV_RELEASE_LIB_DIR}/opencv_xobjdetect480.lib
    ${CERES_RELEASE_LIB_DIR}/ceres.lib
)

# 资源文件
set(RESOURCE_FILES ./assets/resourcefile.qrc)

# 添加可执行文件
add_executable(multicam_rig_toolkit_qt
  src/main.cpp
  src/MainWindow.h
  src/MainWindow.cpp
  src/Core.h
  src/Core.cpp
  src/CaptureWorker.h
  src/SolveWorker.h
  src/player/graphicsview.cpp
  src/player/graphicsview.h
  src/player/editableitem.cpp
  src/player/editableitem.h
  src/player/lineitem.cpp
  src/player/lineitem.h
  src/player/polygonitem.cpp
  src/player/polygonitem.h
  src/player/rectitem.cpp
  src/player/rectitem.h
  src/QCustomPlot.cpp
  src/QCustomPlot.h
  ${RESOURCE_FILES} # 修复：原代码用了${RESOURCE_OBJECTS}，应该是${RESOURCE_FILES}
)

# 设置包含目录（仅作用于当前目标，更规范）
target_include_directories(multicam_rig_toolkit_qt
  PRIVATE
    ${EXTRA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src # 添加源码根目录，方便头文件包含
)

# 按编译类型链接对应的库文件
target_link_libraries(multicam_rig_toolkit_qt
  PRIVATE
    Qt5::Widgets
    Qt5::PrintSupport
    Eigen3::Eigen
    $<$<CONFIG:Debug>:${DEBUG_LIBS}>
    $<$<CONFIG:Release>:${RELEASE_LIBS}>
)

# 可选：强制设置VS工程的输出目录，方便管理
set_target_properties(multicam_rig_toolkit_qt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)