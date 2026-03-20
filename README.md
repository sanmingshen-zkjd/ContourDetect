# Qt Trainable Segmentation

一个使用 **Qt Widgets + C++ + OpenCV** 实现的交互式可训练图像分割工具，工作流参考 Fiji / ImageJ 的 **Plugins › Segmentation › Trainable Weka Segmentation**。

## 当前实现的核心能力

- 打开单张图像并进行交互式像素级标注。
- 支持多类别训练样本管理，默认提供 2 个类别，并可动态新增类别；其中 Class 1 预设为前景目标，Class 2 预设为背景对象。
- 支持画笔标注、擦除、平移、缩放、ROI trace 绘制、trace 提交与结果叠加显示。
- 自动提取多种像素特征：
  - 原始灰度
  - 多尺度高斯平滑
  - 梯度幅值
  - Laplacian
  - 局部均值 / 局部标准差
  - 归一化 X/Y 空间位置
- 支持多种可切换分类器：**Gaussian Naive Bayes / Random Forest / SVM (RBF)**。
- 生成整图分割结果与按类别概率图。
- 支持结果统计绘图。
- 支持保存 / 加载：
  - 训练数据（JSON + 每类 mask PNG）
  - 分类器（YAML）
- 支持导出最终标签图。
- 训练好的分类器可直接应用到当前图像或另选图像文件。

## 界面对应关系

左侧面板对应 Trainable Weka Segmentation 常见操作：

- **Train classifier**：根据用户标注训练分类器
- **Toggle overlay**：开关分割叠加层
- **Create result**：弹出结果查看器
- **Get probability**：显示类别概率图（当前支持 GaussianNB / Random Forest）
- **Plot result**：统计各类别像素数量
- **Apply classifier**：对当前图像或另选图像重新执行整图分类
- **Load / Save classifier**：加载/保存训练好的模型
- **Load / Save data**：加载/保存标注数据
- **Create new class**：新增类别
- **Settings**：配置特征提取项
- **Classifier**：选择当前训练器（GaussianNB / Random Forest / SVM）
- **Trace ROI**：先画出一个自由曲线 ROI
- **Add ROI to selected class**：把当前 ROI 提交到当前选中的类别

## 依赖

- CMake >= 3.16
- Qt 5 或 Qt 6（Widgets + PrintSupport）
- OpenCV（core / imgproc / imgcodecs / ml）

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

> 如果 CMake 找不到 Qt，请显式设置 `CMAKE_PREFIX_PATH` 指向 Qt 安装目录。

例如：

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x/gcc_64
```

### Windows / Visual Studio 说明

如果你在 **MSVC Debug** 下遇到 `GaussianBlur`、`cvtColor` 等 `LNK2019`，通常是 OpenCV 的 `imgproc` / `imgcodecs` 没有按 Debug 配置正确参与链接。当前 `CMakeLists.txt` 已优先使用 OpenCV 的 imported targets（如 `opencv_world` 或 `opencv_core`/`opencv_imgproc`/`opencv_imgcodecs`），并在旧版 OpenCV 包配置下回退到显式模块链接，以尽量避免这一类问题。

如果仍有问题，请检查：

1. `OpenCV_DIR` 是否指向与你当前编译器/架构匹配的 OpenCV CMake 配置目录。
2. Debug 构建是否对应 Debug OpenCV 库，Release 构建是否对应 Release OpenCV 库。
3. `cmake` 配置输出中的 `OpenCV link libs:` 是否包含 `opencv_imgproc` / `opencv_imgcodecs` 或 `opencv_world`。
4. 如果你使用的是 OpenCV 4.8.0 官方/自编译 Windows 包，确认其 `lib` 目录里确实存在类似 `opencv_core480d.lib`、`opencv_imgproc480d.lib`、`opencv_imgcodecs480d.lib`（Debug）以及对应的不带 `d` 的 Release 库。

对于这种目录结构，现在的 CMake 会优先尝试直接解析这些具体 `.lib` 文件。

一个典型的 VS2019 + Qt5.15 + OpenCV 4.8.0 配置示例是：

```bash
cmake -S . -B build -G "Visual Studio 16 2019" -A x64 ^
  -DQt5_DIR="C:/Qt/5.15.2/msvc2019_64/lib/cmake/Qt5" ^
  -DOpenCV_DIR="C:/opencv/build"
```

## 使用流程

1. 打开图像。
2. 从右侧类别列表选择当前要标注的类别。
3. 先使用工具栏里的 **Trace ROI** 画出一个自由曲线 ROI，并点击 **Add ROI to selected class** 提交。
4. 也可以继续使用 `Paint` 工具在图像上补充像素级样本。
5. 如果你创建了更多类别，可以随时切换当前类别并继续标注；trace 列表会跟随当前选中类别刷新。
6. 选择需要的 **Classifier**。
7. 点击 **Train classifier**。
8. 通过 **Apply classifier** 将模型应用到当前图像，或选择另一个图像文件进行推理。
9. 查看叠加结果、概率图与结果统计。
10. 根据需要继续补充标注并重新训练。
11. 保存训练数据或导出分类器。

## 训练数据格式

保存训练数据时会生成：

- 一个 JSON 清单文件
- 每个类别对应的 mask PNG 文件

这样便于后续再次打开项目继续标注和训练。

## 说明

- 当前实现面向 **2D 单张图像** 的交互式训练分割。
- 为保证项目可直接在 C++/Qt/OpenCV 中落地，分类器采用轻量级的高斯朴素贝叶斯实现，而不是直接依赖 Java/Weka 运行时。
- 如果需要进一步逼近 Fiji 版本，还可以继续扩展：
  - 更多滤波器特征（Hessian、Gabor、膜结构、纹理特征）
  - 超像素
  - 3D stack / 多通道图像
  - 批处理推理
  - 模型对比与交叉验证


> 注：当前“概率图”支持 Gaussian Naive Bayes 与 Random Forest；SVM 已支持训练/预测/保存加载，但暂未提供严格概率输出。
