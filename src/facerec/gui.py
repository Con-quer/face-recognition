"""OpenCV 窗口显示相关的小工具。"""

from __future__ import annotations

import cv2

from .logging_utils import get_logger

logger = get_logger(__name__)

_HEADLESS_HINT = (
    "当前环境无法创建显示窗口，通常是安装了 headless 版 OpenCV 或没有图形界面。\n"
    "请改装带 GUI 的版本：pip install -U --force-reinstall opencv-python"
)


def ensure_display(window_name: str, width: int | None = None, height: int | None = None) -> None:
    """提前创建一个窗口，尽早暴露 headless 环境问题。"""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        if width and height:
            cv2.resizeWindow(window_name, int(width), int(height))
    except cv2.error as exc:
        raise RuntimeError(_HEADLESS_HINT) from exc


def show(window_name: str, image) -> None:
    """显示一帧画面，headless 环境下给出可读的错误提示。"""
    try:
        cv2.imshow(window_name, image)
    except cv2.error as exc:
        raise RuntimeError(_HEADLESS_HINT) from exc


def destroy_all() -> None:
    """关闭所有 OpenCV 窗口。"""
    try:
        cv2.destroyAllWindows()
    except cv2.error:  # pragma: no cover - headless 环境
        pass


def wait_key(delay: int = 1) -> int:
    """读取按键，统一屏蔽 ``waitKey`` 的平台差异。"""
    key = cv2.waitKey(delay)
    return -1 if key == -1 else key & 0xFF
