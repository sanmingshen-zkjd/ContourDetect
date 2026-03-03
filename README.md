# ContourDetect（Qt/C++ 单目版）

按你的要求，项目保持 **Qt + C++** 开发框架，并参考 Multi-Cam6DPoseTracker 的播放器架构改为单目处理：

- Player（保留播放器核心能力）
- PreProcess（替代原 Calibration）
- Measurement（测量逻辑接口保持不变）

## 架构

```text
MainWindow (Qt Widgets)
   -> MonocularContourApp
      -> PlayerController
      -> PreProcessPipeline
      -> MeasurementEngine
```

## 功能对应

- 播放器：播放/暂停、逐帧、seek、循环模式、变速。
- 单目预处理：`PreProcessPipeline` 统一处理入口。
- 测量：继续以预处理结果为输入，输出 `MeasurementResult`。

## 构建

```bash
cmake -S . -B build
cmake --build build
./build/ContourDetectQt
```

> 依赖：Qt5.15 Widgets、CMake 3.21+、C++17。
