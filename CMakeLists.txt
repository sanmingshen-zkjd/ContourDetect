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

function(_append_opencv_ms_target module version_suffix)
  if(NOT MSVC)
    return()
  endif()

  set(_roots)
  foreach(candidate
      "${OpenCV_DIR}"
      "${OpenCV_CONFIG_PATH}"
      "${OpenCV_INSTALL_PATH}"
      "$ENV{OpenCV_DIR}")
    if(candidate)
      list(APPEND _roots "${candidate}")
    endif()
  endforeach()

  set(_search_paths)
  foreach(root IN LISTS _roots)
    list(APPEND _search_paths
      "${root}"
      "${root}/lib"
      "${root}/x64/vc16/lib"
      "${root}/x64/vc17/lib"
      "${root}/x64/vc15/lib"
      "${root}/build/lib"
      "${root}/build/x64/vc16/lib"
      "${root}/build/x64/vc17/lib"
      "${root}/build/x64/vc15/lib")
  endforeach()
  list(REMOVE_DUPLICATES _search_paths)

  find_library(OPENCV_${module}_DEBUG_LIB
    NAMES
      opencv_${module}${version_suffix}d
      opencv_${module}d
    PATHS ${_search_paths}
    NO_DEFAULT_PATH)

  find_library(OPENCV_${module}_RELEASE_LIB
    NAMES
      opencv_${module}${version_suffix}
      opencv_${module}
    PATHS ${_search_paths}
    NO_DEFAULT_PATH)

  if(OPENCV_${module}_DEBUG_LIB OR OPENCV_${module}_RELEASE_LIB)
    if(OPENCV_${module}_DEBUG_LIB)
      list(APPEND OPENCV_LINK_LIBS debug ${OPENCV_${module}_DEBUG_LIB})
    endif()
    if(OPENCV_${module}_RELEASE_LIB)
      list(APPEND OPENCV_LINK_LIBS optimized ${OPENCV_${module}_RELEASE_LIB})
    endif()
    set(OPENCV_LINK_LIBS "${OPENCV_LINK_LIBS}" PARENT_SCOPE)
  endif()
endfunction()

# On MSVC, prefer resolving the actual .lib files first.
# This is the most reliable option for OpenCV 4.8.0 package layouts that use
# names such as opencv_imgproc480d.lib / opencv_imgproc480.lib.
if(MSVC)
  set(_opencv_version_suffix "")
  if(OpenCV_VERSION)
    string(REPLACE "." ";" _opencv_version_parts "${OpenCV_VERSION}")
    list(LENGTH _opencv_version_parts _opencv_version_len)
    if(_opencv_version_len GREATER 0)
      list(GET _opencv_version_parts 0 _opencv_ver_major)
    else()
      set(_opencv_ver_major 0)
    endif()
    if(_opencv_version_len GREATER 1)
      list(GET _opencv_version_parts 1 _opencv_ver_minor)
    else()
      set(_opencv_ver_minor 0)
    endif()
    if(_opencv_version_len GREATER 2)
      list(GET _opencv_version_parts 2 _opencv_ver_patch)
    else()
      set(_opencv_ver_patch 0)
    endif()
    set(_opencv_version_suffix "${_opencv_ver_major}${_opencv_ver_minor}${_opencv_ver_patch}")
  endif()

  foreach(module IN ITEMS core imgproc imgcodecs)
    _append_opencv_ms_target(${module} "${_opencv_version_suffix}")
  endforeach()
endif()

# If explicit MSVC library resolution did not succeed, prefer imported OpenCV
# targets because they preserve Debug/Release-specific selection on multi-config
# generators such as Visual Studio.
if(NOT OPENCV_LINK_LIBS)
  if(TARGET opencv_world)
    list(APPEND OPENCV_LINK_LIBS opencv_world)
  elseif(TARGET opencv_core)
    foreach(module IN ITEMS opencv_core opencv_imgproc opencv_imgcodecs)
      if(TARGET ${module})
        list(APPEND OPENCV_LINK_LIBS ${module})
      endif()
    endforeach()
  endif()
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

# Final MSVC fallback: try plain module names in case a toolchain file already
# added the corresponding library directories globally.
if(MSVC AND NOT OPENCV_LINK_LIBS)
  list(APPEND OPENCV_LINK_LIBS opencv_core opencv_imgproc opencv_imgcodecs)
endif()

list(REMOVE_DUPLICATES OPENCV_LINK_LIBS)

if(NOT OPENCV_LINK_LIBS)
  message(FATAL_ERROR "Unable to determine which OpenCV libraries to link. Check your OpenCV_DIR/CMAKE_PREFIX_PATH.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV link libs: ${OPENCV_LINK_LIBS}")

target_link_libraries(QtTrainableSegmentation PRIVATE
  Qt${QT_VERSION_MAJOR}::Widgets
  Qt${QT_VERSION_MAJOR}::PrintSupport
  ${OPENCV_LINK_LIBS}
)
