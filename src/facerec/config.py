"""项目配置：集中管理目录结构与各阶段超参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: 项目根目录。本文件位于 ``<repo>/src/facerec/config.py``，向上三层即仓库根目录。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: 数据集根目录。Ultralytics 分类任务要求 ``dataset/train/<类别>/`` 与 ``dataset/val/<类别>/``。
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
TRAIN_DIR: Path = DATASET_DIR / "train"
VAL_DIR: Path = DATASET_DIR / "val"

#: 训练输出目录（与 Ultralytics 默认的 ``runs/classify`` 保持一致）。
RUNS_DIR: Path = PROJECT_ROOT / "runs"
CLASSIFY_DIR: Path = RUNS_DIR / "classify"

#: 随项目携带的静态资源目录。
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

DEFAULT_RUN_NAME: str = "face_recognize"

#: 训练完成后权重所在位置，识别阶段默认从这里加载。
DEFAULT_WEIGHTS: Path = CLASSIFY_DIR / DEFAULT_RUN_NAME / "weights" / "best.pt"

#: 随项目携带的 Haar 级联分类器，避免部分 OpenCV 发行版不带 data 目录时无法运行。
DEFAULT_CASCADE: Path = ASSETS_DIR / "haarcascade_frontalface_default.xml"


@dataclass
class CollectConfig:
    """采集阶段（对应 ``get_picture.py``）的参数。"""

    #: 数据集根目录，默认就是项目下的 ``dataset/``。
    data_root: Path = DATASET_DIR
    #: Haar 级联文件路径，None 表示自动查找（项目 assets -> OpenCV 自带）。
    cascade_path: Path | None = None

    #: 摄像头索引，笔记本内置摄像头通常是 0，外接摄像头可以试试 1。
    camera_index: int = 0
    width: int = 1280
    height: int = 720

    #: 人脸框边长小于该值（像素）时丢弃，用来过滤远处目标与误检。
    min_face_size: int = 60
    #: 拉普拉斯方差阈值，低于该值视为模糊样本不予保存。设为 0 可关闭该过滤。
    blur_threshold: float = 40.0

    #: 保存图片的格式与 JPEG 压缩质量。
    image_format: str = "jpg"
    jpeg_quality: int = 95
    #: 保存前把人脸缩放到该边长，0 表示保持原始尺寸。
    face_size: int = 0

    #: 一帧中检测到多张人脸时是否全部保存（默认只保存面积最大的一张）。
    save_all_faces: bool = False
    #: 采集够多少张后自动结束，None 表示由用户按 ESC 结束。
    target_count: int | None = None

    #: 采集完成后自动按该比例划分验证集。
    val_ratio: float = 0.2
    #: 划分时的随机种子，固定种子保证结果可复现。
    seed: int = 42
    auto_split: bool = True

    #: 是否使用 tkinter 弹窗询问姓名（无图形界面时自动回退到命令行输入）。
    use_gui_prompt: bool = True
    window_name: str = "采集人脸 - 按 S 保存 / ESC 退出"


@dataclass
class TrainConfig:
    """训练阶段（对应 ``train.py``）的参数。"""

    #: 预训练权重。人脸分类任务沿用 Ultralytics 的 ``-cls`` 分类模型。
    model: str = "yolo11n-cls.pt"
    data: Path = DATASET_DIR
    epochs: int = 30
    imgsz: int = 224
    batch: int = 4
    #: ``cpu`` / ``0`` / ``0,1``，也可以用 ``auto`` 交给 Ultralytics 自行选择。
    device: str = "cpu"
    workers: int = 2
    patience: int = 10
    seed: int = 42

    name: str = DEFAULT_RUN_NAME
    project: Path = CLASSIFY_DIR

    #: 训练前若验证集为空，自动按比例划分一份出来。
    auto_split: bool = True
    val_ratio: float = 0.2

    @property
    def run_dir(self) -> Path:
        """本次训练的输出目录。"""
        return Path(self.project) / self.name

    @property
    def weights_path(self) -> Path:
        """训练产出的最优权重路径（若存在 ``best.pt`` 则优先取它）。"""
        weights = self.run_dir / "weights"
        best = weights / "best.pt"
        return best if best.exists() else weights / "last.pt"


@dataclass
class RecognizeConfig:
    """识别阶段（对应 ``start.py``）的参数。"""

    weights: Path = DEFAULT_WEIGHTS
    camera_index: int = 0
    width: int = 1280
    height: int = 720

    #: 默认水平镜像画面，自拍时更符合直觉。
    mirror: bool = True
    #: 低于该置信度时显示为 unknown。
    conf_threshold: float = 0.5
    #: 是否叠加 Ultralytics 的 ``plot()`` 结果图。
    use_plot: bool = True
    #: 是否显示左上角的 FPS 与提示信息。
    show_hud: bool = True
    window_name: str = "人脸识别"


def describe(config: CollectConfig | TrainConfig | RecognizeConfig) -> dict[str, object]:
    """把配置对象转成便于打印的普通字典（``Path`` 转为字符串）。"""
    result: dict[str, object] = {}
    for key, value in config.__dict__.items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result
