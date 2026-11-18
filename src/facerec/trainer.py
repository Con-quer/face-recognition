"""训练阶段：用 YOLO11 分类模型训练人脸识别器（对应教程 ``train.py``）。"""

from __future__ import annotations

from pathlib import Path

from .config import TrainConfig
from .dataset import SplitResult, split_all, validate_for_training
from .logging_utils import get_logger

logger = get_logger(__name__)

_ULTRALYTICS_HINT = (
    "未安装 Ultralytics，无法训练或推理。请先安装依赖：\n"
    "  pip install -r requirements.txt\n"
    "国内网络较慢时可使用镜像：\n"
    "  pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple"
)


def require_ultralytics():
    """导入并返回 ``YOLO`` 类，未安装时给出中文安装提示。"""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(_ULTRALYTICS_HINT) from exc
    return YOLO


def prepare_dataset(config: TrainConfig) -> list[SplitResult]:
    """训练前检查数据集，返回自动划分的结果。"""
    splits: list[SplitResult] = []
    if config.auto_split:
        splits = split_all(config.data, ratio=config.val_ratio, seed=config.seed)
        for item in splits:
            if not item.skipped:
                logger.info(
                    "自动划分 %s：验证集 %s 张，训练集 %s 张",
                    item.name,
                    item.val_count,
                    item.train_count,
                )

    errors, warnings = validate_for_training(config.data)
    for message in warnings:
        logger.warning(message)
    if errors:
        raise RuntimeError("数据集不满足训练要求：\n  - " + "\n  - ".join(errors))
    return splits


def train_model(config: TrainConfig | None = None) -> Path:
    """按配置训练模型，返回最优权重 ``best.pt`` 的路径。"""
    config = config or TrainConfig()
    logger.info(
        "训练配置：模型=%s，轮数=%s，尺寸=%s，批大小=%s，设备=%s",
        config.model,
        config.epochs,
        config.imgsz,
        config.batch,
        config.device,
    )
    logger.info("数据集：%s", config.data)
    logger.info("输出目录：%s", config.run_dir)

    YOLO = require_ultralytics()
    prepare_dataset(config)

    # 第一次运行会自动下载 yolo11n-cls.pt 预训练权重。
    model = YOLO(config.model)
    model.train(
        data=str(config.data),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        workers=config.workers,
        patience=config.patience,
        seed=config.seed,
        project=str(config.project),
        name=config.name,
        val=True,
    )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", config.run_dir))
    for candidate in (
        save_dir / "weights" / "best.pt",
        save_dir / "weights" / "last.pt",
        config.weights_path,
    ):
        if candidate.is_file():
            logger.info("训练完成，权重文件：%s", candidate)
            return candidate

    raise FileNotFoundError(f"训练结束但找不到权重文件，请检查目录：{save_dir / 'weights'}")
