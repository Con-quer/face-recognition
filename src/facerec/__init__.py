"""基于 OpenCV 与 YOLO11 的实时人脸识别。

模块划分（与原教程的三个脚本一一对应）：

============  ===============================  ===============
模块          职责                             对应教程脚本
============  ===============================  ===============
config        路径与超参数
dataset       数据集目录、姓名校验、划分
camera        摄像头封装
detector      OpenCV Haar 级联人脸检测与裁剪
drawing       中英文文字绘制
collect       采集阶段                          get_picture.py
trainer       训练阶段                          train.py
recognizer    识别阶段                          start.py
diagnostics   环境自检
cli           命令行入口
============  ===============================  ===============
"""

from __future__ import annotations

from .config import (
    ASSETS_DIR,
    CLASSIFY_DIR,
    DATASET_DIR,
    DEFAULT_CASCADE,
    DEFAULT_RUN_NAME,
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    RUNS_DIR,
    TRAIN_DIR,
    VAL_DIR,
    CollectConfig,
    RecognizeConfig,
    TrainConfig,
)

__version__ = "1.0.0"

__all__ = [
    "ASSETS_DIR",
    "CLASSIFY_DIR",
    "DATASET_DIR",
    "DEFAULT_CASCADE",
    "DEFAULT_RUN_NAME",
    "DEFAULT_WEIGHTS",
    "PROJECT_ROOT",
    "RUNS_DIR",
    "TRAIN_DIR",
    "VAL_DIR",
    "CollectConfig",
    "RecognizeConfig",
    "TrainConfig",
    "__version__",
]
