# ContourDetect（单目版架构）

本仓库按照 **Multi-Cam6DPoseTracker 的播放器设计思路**重构为单目流程，核心链路为：

1. **Player**（完整保留播放器能力）
2. **PreProcess**（替代原始标定 Calibration）
3. **Measurement**（测量部分保持不变接口）

## 架构映射

- Multi-Cam 的多路相机播放 -> 单路 `PlayerController`
- Calibration -> `PreProcessPipeline`
- Measurement -> `MeasurementEngine`

## 目录

- `src/contour_detect/player.py`：播放器控制（播放/暂停/跳转/逐帧/循环/变速）
- `src/contour_detect/preprocess.py`：单目预处理配置与流程
- `src/contour_detect/measurement.py`：测量引擎（接口保持稳定）
- `src/contour_detect/app.py`：应用编排（Player -> PreProcess -> Measurement）

## 快速使用

```python
from contour_detect import MonocularContourApp

app = MonocularContourApp()
app.load_frames(["f0", "f1", "f2"])
result = app.process_current()
print(result)
```
