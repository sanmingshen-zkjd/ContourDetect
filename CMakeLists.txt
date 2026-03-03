cmake_minimum_required(VERSION 3.21)
project(ContourDetectQt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt5 5.15 REQUIRED COMPONENTS Widgets)

add_executable(ContourDetectQt
    src/main.cpp
    src/app/MonocularContourApp.cpp
    src/player/PlayerController.cpp
    src/preprocess/PreProcessPipeline.cpp
    src/measurement/MeasurementEngine.cpp
    src/ui/MainWindow.cpp
)

target_include_directories(ContourDetectQt PRIVATE src)

target_link_libraries(ContourDetectQt PRIVATE Qt5::Widgets)
