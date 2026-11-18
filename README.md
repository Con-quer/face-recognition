# 基于 OpenCV 与 YOLO11 的实时人脸识别

用 OpenCV 的 Haar 级联做**人脸检测**、用 YOLO11 分类模型做**人脸识别**的小型完整项目。
代码按机器学习的三个环节组织：**准备数据 → 训练模型 → 开始识别**。

原始教程见 [基于OpenCV和YOLO11的实时人脸识别.md](基于OpenCV和YOLO11的实时人脸识别.md)，
本项目在保持同样思路的前提下，把它整理成了一个可直接运行、可测试的工程。

| 教程脚本 | 本项目对应入口 | 实际实现 |
| --- | --- | --- |
| `get_picture.py` | `python scripts/get_picture.py` | `src/facerec/collect.py` |
| `train.py` | `python scripts/train.py` | `src/facerec/trainer.py` |
| `start.py` | `python scripts/start.py` | `src/facerec/recognizer.py` |

> 想先搞懂"一个完整项目由什么组成、怎么部署、谁来维护"这类通用问题？看
> [Python 项目开发与部署入门](docs/Python项目开发与部署入门.md)（与具体项目无关）。
> 想了解本项目每个文件的作用、数据怎么流动、出问题先查哪里，看
> [项目使用与维护指南](docs/项目使用与维护指南.md)；做课程设计或答辩可参考 [项目计划书](docs/项目计划书.md)。

## 功能特性

- **采集**：摄像头预览中按 `S` 保存人脸，自动裁剪、自动过滤模糊样本、断点续编号。
- **划分**：采集完成后按比例随机划分训练集 / 验证集，重复运行不会破坏已有划分。
- **训练**：基于 `yolo11n-cls.pt` 训练分类模型，输出到 `runs/classify/face_recognize/weights/`。
- **识别**：实时画面叠加姓名、置信度、FPS 与 Top-K 结果，支持镜像切换。
- **工程化**：统一的 `main.py` 命令行入口、中文日志、环境自检、Pillow 中文渲染、71 个单元测试。

## 目录结构

```text
.
├── main.py                     # 统一命令行入口
├── scripts/                    # 与教程同名的三个脚本（薄封装）
│   ├── get_picture.py
│   ├── train.py
│   └── start.py
├── src/facerec/                # 项目主体代码
│   ├── config.py               # 路径与超参数
│   ├── dataset.py              # 数据集目录、姓名校验、训练/验证集划分
│   ├── camera.py               # 摄像头封装
│   ├── detector.py             # Haar 级联人脸检测与裁剪
│   ├── drawing.py              # 中英文文字绘制
│   ├── gui.py                  # OpenCV 窗口相关工具
│   ├── collect.py              # 采集阶段
│   ├── trainer.py              # 训练阶段
│   ├── recognizer.py           # 识别阶段
│   ├── diagnostics.py          # 环境自检
│   ├── logging_utils.py        # 日志
│   └── cli.py                  # 命令行参数解析
├── assets/                     # 随项目携带的 Haar 级联文件
├── dataset/                    # 数据集（train/ 与 val/）
├── runs/                       # 训练产物（训练后生成）
├── docs/                       # 通用开发部署入门、项目使用与维护指南、项目计划书
└── tests/                      # 单元测试
```

## 环境准备

需要 Python 3.9 及以上。推荐先建虚拟环境，避免和系统里的其他包互相污染：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

安装完成后自检一次环境：

```powershell
python main.py doctor
```

正常输出示例：

```text
[ OK ] Python 版本：3.13.9（满足 >=3.9）
[ OK ] OpenCV：4.14.0（支持窗口显示）
[ OK ] Haar 人脸检测器：...\assets\haarcascade_frontalface_default.xml
[WARN] Ultralytics：未安装，训练与识别不可用（pip install -r requirements.txt）
[ OK ] 中文字体：C:/Windows/Fonts/msyh.ttc
[WARN] 数据集：...\dataset 中还没有样本，请先运行采集
```

## 快速开始

### 第一步：采集人脸样本

```powershell
python main.py collect --name 张三 --target 40
```

弹出的摄像头窗口里：

| 按键 | 作用 |
| --- | --- |
| `S` | 保存当前画面中最大的人脸（`--all-faces` 可保存全部） |
| `ESC` / `Q` | 结束采集 |

采集建议：正对镜头，缓慢左右转动头部，光线均匀；**同一个人至少 20 张**，视角和距离尽量多样。
采集满 `--target` 张会自动结束，并按 0.2 的比例把一部分样本划到验证集。

### 第二步：训练模型

```powershell
python main.py train --epochs 30 --device cpu
```

首次运行会自动下载 `yolo11n-cls.pt` 预训练权重（需要联网）。
训练完成后权重位于：

```text
runs/classify/face_recognize/weights/best.pt
```

有 NVIDIA 显卡时可以改成 `--device 0`，速度会快很多。

### 第三步：实时识别

```powershell
python main.py recognize
```

窗口左上角会显示识别结果与置信度，下方显示 FPS；
按 `ESC` / `Q` 退出，`M` 切换镜像，`H` 隐藏/显示信息。

也可以一条命令走完采集与训练：

```powershell
python main.py pipeline --name 张三 --target 40 --epochs 30
```

## 命令行参考

```powershell
python main.py collect     # 采集样本
python main.py train       # 训练模型
python main.py recognize   # 实时识别
python main.py pipeline    # 采集 + 训练
python main.py doctor      # 环境自检
python main.py info        # 查看数据集统计
```

每个子命令都支持 `--help`，例如 `python main.py collect --help`。
常用的几个参数：

| 参数 | 说明 |
| --- | --- |
| `--camera 1` | 摄像头索引，默认 0；笔记本外接摄像头常用 1 |
| `--target 40` | 采集够多少张后自动结束 |
| `--blur-threshold 40` | 清晰度阈值（拉普拉斯方差），低于该值不保存；`0` 表示不过滤 |
| `--min-face-size 60` | 小于该边长的人脸框直接忽略 |
| `--val-ratio 0.2` | 验证集比例 |
| `--epochs 30` | 训练轮数 |
| `--device cpu` | `cpu` / `0` / `0,1` / `auto` |
| `--conf 0.5` | 识别置信度阈值，低于该值显示 `unknown` |

## 数据集结构

Ultralytics 的分类任务要求「一个类别一个文件夹」：

```text
dataset/
├── train/
│   ├── 张三/
│   │   ├── face_0000.jpg
│   │   └── ...
│   └── 李四/
└── val/
    ├── 张三/
    └── 李四/
```

类别名就是文件夹名（也就是识别结果里显示的姓名）。
采集脚本会自动建目录、接着已有编号继续保存，不需要手工整理。

## 常见问题

**1. `ModuleNotFoundError: No module named 'numpy.core.multiarray'`**

numpy 安装不完整或版本与 OpenCV 不匹配，重装即可：

```powershell
pip install -U --force-reinstall numpy opencv-python
```

**2. `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`**

OpenCV 5.x 的部分发行版不再包含 `objdetect` 模块（Haar 级联），本项目的采集阶段依赖它，
安装 4.x 版本即可：

```powershell
pip install "opencv-python>=4.8,<5"
```

**3. 项目路径里有中文会不会有问题？**

Windows 上的 OpenCV 用窄字符接口打开文件，路径含中文时 `cv2.imread` / `cv2.imwrite` /
`cv2.CascadeClassifier` 会直接失败。本项目已经处理了两处最容易踩坑的地方：

- 级联文件在需要时自动转存到纯英文的临时目录再加载；
- 人脸样本改用 `cv2.imencode` 写盘，中文姓名目录也能正常保存。

不过**训练阶段由 Ultralytics 负责读写文件，仍建议把项目放在纯英文路径下**
（例如 `D:\projects\facerec`）再开始训练，可以省掉一类难排查的问题。

**4. 摄像头打不开**

先确认没有被相机、会议软件占用，再检查系统隐私设置里是否允许了摄像头权限，
最后可以换索引试试：`--camera 1`。

**5. 窗口里的中文变成问号**

说明系统没有找到中文字体。`python main.py doctor` 会给出提示；
也可以在 `src/facerec/drawing.py` 的 `FONT_CANDIDATES` 里补上你自己的字体路径。

**6. 训练报错说验证集为空 / 样本太少**

每个人至少准备 20 张、最好 2 个人以上。训练前会自动把验证集补出来，
也可以先看一眼 `python main.py info` 的统计结果。

## 相对教程的改进

原教程的代码思路完全保留，主要补了工程上容易出问题的地方：

1. **划分逻辑**：随机抽样代替 `listdir` 的任意顺序；验证集非空时跳过，重复运行不会把关联网越用越小；
   至少给训练集留 1 张，避免样本太少直接报错。
2. **数据质量**：新增拉普拉斯方差模糊过滤、最小人脸尺寸过滤，减少脏数据。
3. **中文支持**：`cv2.putText` 不支持中文，这里改用 Pillow 渲染，姓名能正确显示。
4. **中文路径**：级联文件自动转存、图片用 `imencode` 写盘，绕开 OpenCV 的窄字符路径限制。
5. **可复现**：固定随机种子，重复采集会接着旧编号保存，不会覆盖已有样本。
6. **可维护**：拆分为独立模块，统一的命令行入口与日志，`--help` 自解释。

## 测试

```powershell
pip install pytest
python -m pytest
```

测试不依赖摄像头和真实硬件：采集流程用假摄像头驱动，覆盖编号续写、模糊过滤、
验证集划分、中文路径写盘、命令行行为等 71 个用例。

## 文档

| 文档 | 适合谁 | 内容 |
| --- | --- | --- |
| [README.md](README.md) | 所有人 | 安装、命令、参数、常见问题 |
| [docs/Python项目开发与部署入门.md](docs/Python项目开发与部署入门.md) | 没做过完整项目/部署的人 | Python 项目通用知识：环境、依赖、配置、测试、git、交付形态、部署流程、CI、角色与维护 |
| [docs/项目使用与维护指南.md](docs/项目使用与维护指南.md) | 拿到本项目的人 | 每个文件的作用、数据流、角色分工、维护清单、术语表 |
| [docs/项目计划书.md](docs/项目计划书.md) | 课程设计/答辩 | 目标、范围、里程碑、分工、风险、验收标准、迭代路线 |

## 安装为命令（可选）

```powershell
pip install -e .
facerec doctor
facerec collect --name 张三
```
