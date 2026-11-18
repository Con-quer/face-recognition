"""环境自检：依赖、字体、摄像头与数据集状态一次看清。"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import DATASET_DIR
from .dataset import summarize
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CheckResult:
    """单项检查结果。"""

    name: str
    ok: bool
    detail: str
    #: 标记为关键的检查失败时，整体自检返回失败状态。
    fatal: bool = False


def check_python() -> CheckResult:
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= (3, 9)
    return CheckResult(
        "Python 版本",
        ok,
        f"{version}（{'满足 >=3.9' if ok else '过低，请升级到 3.9 及以上'}）",
        fatal=True,
    )


def check_opencv() -> CheckResult:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001 - 需要把任何导入错误都报告出来
        return CheckResult("OpenCV", False, f"导入失败：{exc}", fatal=True)

    version = getattr(cv2, "__version__", "unknown")
    gui = hasattr(cv2, "imshow") and hasattr(cv2, "namedWindow")
    detail = f"{version}（{'支持窗口显示' if gui else '缺少 GUI 支持，无法实时预览'}）"
    return CheckResult("OpenCV", True, detail, fatal=True)


def check_cascade() -> CheckResult:
    try:
        from .detector import resolve_cascade_path

        path = resolve_cascade_path()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Haar 人脸检测器", False, str(exc), fatal=True)
    return CheckResult("Haar 人脸检测器", True, str(path), fatal=True)


def check_ultralytics() -> CheckResult:
    try:
        import ultralytics
    except ImportError:
        return CheckResult(
            "Ultralytics",
            False,
            "未安装，训练与识别不可用（pip install -r requirements.txt）",
        )

    version = getattr(ultralytics, "__version__", "unknown")
    detail = version
    try:
        import torch

        detail += f"；torch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}"
    except ImportError:
        detail += "；缺少 torch"
    return CheckResult("Ultralytics", True, detail)


def check_font() -> CheckResult:
    try:
        from .drawing import find_font_path

        path = find_font_path()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("中文字体", False, f"检测失败：{exc}")

    if path is None:
        return CheckResult("中文字体", False, "未找到中文字体，画面上的中文姓名会显示异常")
    return CheckResult("中文字体", True, path)


def check_dataset(root: str | Path = DATASET_DIR) -> CheckResult:
    stats = summarize(root)
    if not stats:
        return CheckResult("数据集", False, f"{root} 中还没有样本，请先运行采集")

    total = sum(item["train"] + item["val"] for item in stats.values())
    detail = f"{len(stats)} 个类别，共 {total} 张图片（"
    detail += "，".join(
        f"{name} 训练 {item['train']}/验证 {item['val']}" for name, item in stats.items()
    )
    return CheckResult("数据集", True, detail + "）")


def check_camera(index: int = 0) -> CheckResult:
    try:
        from .camera import Camera, CameraError
    except Exception as exc:  # noqa: BLE001
        return CheckResult("摄像头", False, f"OpenCV 不可用：{exc}")

    camera = Camera(index)
    try:
        camera.open()
        frame = camera.read()
    except CameraError as exc:
        return CheckResult("摄像头", False, str(exc).splitlines()[0])
    finally:
        camera.close()

    return CheckResult(
        "摄像头", True, f"索引 {index} 可读取，画面 {frame.shape[1]}x{frame.shape[0]}"
    )


def run_diagnostics(
    *,
    camera_index: int | None = None,
    root: str | Path = DATASET_DIR,
) -> tuple[list[CheckResult], bool]:
    """执行全部检查，返回 ``(结果列表, 是否全部关键项通过)``。"""
    logger.info("运行环境：%s %s", platform.system(), platform.release())

    checks = [
        check_python(),
        check_opencv(),
        check_cascade(),
        check_ultralytics(),
        check_font(),
        check_dataset(root),
    ]
    if camera_index is not None:
        checks.append(check_camera(camera_index))

    passed = all(item.ok or not item.fatal for item in checks)
    return checks, passed


def format_report(checks: list[CheckResult]) -> str:
    """把检查结果排版成可读的文本表格。"""
    lines = []
    for item in checks:
        if item.ok:
            mark = "[ OK ]"
        elif item.fatal:
            mark = "[FAIL]"
        else:
            mark = "[WARN]"
        lines.append(f"{mark} {item.name}：{item.detail}")
    return "\n".join(lines)
