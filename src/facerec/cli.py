"""命令行入口：采集 / 训练 / 识别 / 自检 / 查看数据集。"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import (
    CLASSIFY_DIR,
    DATASET_DIR,
    DEFAULT_RUN_NAME,
    DEFAULT_WEIGHTS,
    CollectConfig,
    RecognizeConfig,
    TrainConfig,
    describe,
)
from .dataset import summarize
from .diagnostics import format_report, run_diagnostics
from .logging_utils import get_logger

logger = get_logger(__name__)


def _display_width(text: str) -> int:
    """估算终端显示宽度，中日韩文字按 2 列计算。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facerec",
        description="基于 OpenCV 与 YOLO11 的实时人脸识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py collect --name 张三 --target 40   # 采集 40 张人脸样本\n"
            "  python main.py train --epochs 50 --device cpu    # 训练分类模型\n"
            "  python main.py recognize                         # 打开摄像头实时识别\n"
            "  python main.py doctor --camera                    # 检查运行环境\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"facerec {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="采集人脸样本并划分训练/验证集")
    collect.add_argument("--name", help="姓名；不填则弹窗询问")
    collect.add_argument("--camera", type=int, default=0, help="摄像头索引（默认 0）")
    collect.add_argument("--data", type=Path, default=DATASET_DIR, help="数据集根目录")
    collect.add_argument("--target", type=int, default=None, help="采集够多少张后自动结束")
    collect.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例（默认 0.2）")
    collect.add_argument("--no-split", action="store_true", help="采集后不划分验证集")
    collect.add_argument("--all-faces", action="store_true", help="一帧中的多张人脸全部保存")
    collect.add_argument("--min-face-size", type=int, default=60, help="最小人脸边长（默认 60）")
    collect.add_argument(
        "--blur-threshold",
        type=float,
        default=40.0,
        help="清晰度阈值，低于该值不保存；0 表示不过滤",
    )
    collect.add_argument("--image-format", default="jpg", choices=["jpg", "png"], help="保存格式")
    collect.add_argument(
        "--face-size", type=int, default=0, help="保存前缩放人脸边长，0 表示不缩放"
    )
    collect.add_argument("--seed", type=int, default=42, help="划分随机种子")
    collect.add_argument(
        "--no-gui-prompt", action="store_true", help="不弹窗，直接在命令行输入姓名"
    )
    collect.set_defaults(handler=_run_collect)

    train = subparsers.add_parser("train", help="训练 YOLO11 分类模型")
    train.add_argument("--data", type=Path, default=DATASET_DIR, help="数据集根目录")
    train.add_argument("--model", default="yolo11n-cls.pt", help="预训练权重")
    train.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    train.add_argument("--imgsz", type=int, default=224, help="输入尺寸（默认 224）")
    train.add_argument("--batch", type=int, default=4, help="批大小（默认 4）")
    train.add_argument("--device", default="cpu", help="cpu / 0 / 0,1 / auto（默认 cpu）")
    train.add_argument("--workers", type=int, default=2, help="数据加载线程数")
    train.add_argument("--patience", type=int, default=10, help="早停耐心值")
    train.add_argument("--seed", type=int, default=42, help="随机种子")
    train.add_argument("--name", default=DEFAULT_RUN_NAME, help="本次运行的名称")
    train.add_argument("--project", type=Path, default=CLASSIFY_DIR, help="训练输出根目录")
    train.add_argument("--no-auto-split", action="store_true", help="不自动划分验证集")
    train.set_defaults(handler=_run_train)

    recognize = subparsers.add_parser("recognize", help="打开摄像头实时识别")
    recognize.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="模型权重路径")
    recognize.add_argument("--camera", type=int, default=0, help="摄像头索引（默认 0）")
    recognize.add_argument("--width", type=int, default=1280, help="采集宽度")
    recognize.add_argument("--height", type=int, default=720, help="采集高度")
    recognize.add_argument("--conf", type=float, default=0.5, help="置信度阈值（默认 0.5）")
    recognize.add_argument("--no-mirror", action="store_true", help="不镜像画面")
    recognize.add_argument("--no-plot", action="store_true", help="不叠加 Ultralytics 的结果图")
    recognize.add_argument("--no-hud", action="store_true", help="不显示 FPS 与提示信息")
    recognize.set_defaults(handler=_run_recognize)

    pipeline = subparsers.add_parser("pipeline", help="先采集再训练（等价于 collect + train）")
    pipeline.add_argument("--name", required=True, help="姓名")
    pipeline.add_argument("--target", type=int, default=40, help="采集张数（默认 40）")
    pipeline.add_argument("--epochs", type=int, default=30, help="训练轮数")
    pipeline.add_argument("--device", default="cpu", help="cpu / 0 / auto")
    pipeline.add_argument("--data", type=Path, default=DATASET_DIR, help="数据集根目录")
    pipeline.add_argument("--camera", type=int, default=0, help="摄像头索引")
    pipeline.set_defaults(handler=_run_pipeline)

    doctor = subparsers.add_parser("doctor", help="检查运行环境")
    doctor.add_argument("--data", type=Path, default=DATASET_DIR, help="数据集根目录")
    doctor.add_argument("--camera", type=int, default=None, help="附加测试该索引的摄像头")
    doctor.set_defaults(handler=_run_doctor)

    info = subparsers.add_parser("info", help="查看数据集统计信息")
    info.add_argument("--data", type=Path, default=DATASET_DIR, help="数据集根目录")
    info.set_defaults(handler=_run_info)

    return parser


def _run_collect(args: argparse.Namespace) -> int:
    from .collect import CollectionCancelled, collect

    config = CollectConfig(
        data_root=args.data,
        camera_index=args.camera,
        min_face_size=args.min_face_size,
        blur_threshold=args.blur_threshold,
        image_format=args.image_format,
        face_size=args.face_size,
        save_all_faces=args.all_faces,
        target_count=args.target,
        val_ratio=args.val_ratio,
        seed=args.seed,
        auto_split=not args.no_split,
        use_gui_prompt=not args.no_gui_prompt,
    )
    logger.debug("采集配置：%s", describe(config))
    try:
        result = collect(args.name, config)
    except CollectionCancelled as exc:
        logger.warning("%s", exc)
        return 1
    logger.info(
        "采集结束：%s 共 %s 张样本（本次 %s 张，跳过模糊 %s 张）",
        result.name,
        result.total,
        result.saved,
        result.skipped_blurry,
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from .trainer import train_model

    config = TrainConfig(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        name=args.name,
        project=args.project,
        auto_split=not args.no_auto_split,
    )
    weights = train_model(config)
    logger.info("可以开始识别了：python main.py recognize --weights %s", weights)
    return 0


def _run_recognize(args: argparse.Namespace) -> int:
    from .recognizer import run

    config = RecognizeConfig(
        weights=args.weights,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        conf_threshold=args.conf,
        use_plot=not args.no_plot,
        show_hud=not args.no_hud,
    )
    run(config)
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    collect_code = _run_collect(
        argparse.Namespace(
            name=args.name,
            camera=args.camera,
            data=args.data,
            target=args.target,
            val_ratio=0.2,
            no_split=False,
            all_faces=False,
            min_face_size=60,
            blur_threshold=40.0,
            image_format="jpg",
            face_size=0,
            seed=42,
            no_gui_prompt=False,
        )
    )
    if collect_code != 0:
        return collect_code
    return _run_train(
        argparse.Namespace(
            data=args.data,
            model="yolo11n-cls.pt",
            epochs=args.epochs,
            imgsz=224,
            batch=4,
            device=args.device,
            workers=2,
            patience=10,
            seed=42,
            name=DEFAULT_RUN_NAME,
            project=CLASSIFY_DIR,
            no_auto_split=False,
        )
    )


def _run_doctor(args: argparse.Namespace) -> int:
    checks, passed = run_diagnostics(camera_index=args.camera, root=args.data)
    print(format_report(checks))
    print()
    print(
        "结论：环境正常，可以开始使用。"
        if passed
        else "结论：存在必须处理的问题，请按上面的提示修复。"
    )
    return 0 if passed else 1


def _run_info(args: argparse.Namespace) -> int:
    stats = summarize(args.data)
    print(f"数据集目录：{Path(args.data).resolve()}")
    if not stats:
        print("还没有任何样本，先运行：python main.py collect")
        return 0

    name_width = max([_display_width("类别"), *(_display_width(name) for name in stats)]) + 2
    print()
    print(f"{_pad('类别', name_width)}训练集   验证集   合计")
    for name, counts in stats.items():
        total = counts["train"] + counts["val"]
        print(f"{_pad(name, name_width)}{counts['train']:<8}{counts['val']:<9}{total}")
    grand_total = sum(item["train"] + item["val"] for item in stats.values())
    print(f"\n共 {len(stats)} 个类别，{grand_total} 张图片")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    get_logger(level=getattr(logging, args.log_level))

    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        logger.warning("已中断")
        return 130
    except ImportError as exc:
        logger.error(
            "导入依赖失败：%s\n"
            "请先安装或修复依赖：pip install -r requirements.txt\n"
            "如果提示 numpy.core.multiarray 之类的错误，说明环境里的 numpy 不完整，可以执行：\n"
            "  pip install -U --force-reinstall numpy opencv-python",
            exc,
        )
        return 1
    except (RuntimeError, ValueError, FileNotFoundError, PermissionError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
