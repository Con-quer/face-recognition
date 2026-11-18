"""摄像头封装：打开、读取、释放，并在失败时给出明确错误。"""

from __future__ import annotations

import cv2

from .logging_utils import get_logger

logger = get_logger(__name__)


class CameraError(RuntimeError):
    """摄像头打开或读取失败。"""


class Camera:
    """对 ``cv2.VideoCapture`` 的轻量封装，支持 ``with`` 语句。"""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> Camera:
        """打开摄像头，失败时抛出 :class:`CameraError`。"""
        capture = cv2.VideoCapture(self.index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"无法打开摄像头（索引 {self.index}）。请检查：\n"
                "  1. 摄像头是否被其他程序占用（相机、会议软件等）；\n"
                "  2. 是否允许了本程序的摄像头权限；\n"
                "  3. 换一个索引试试，例如 --camera 1。"
            )
        self._capture = capture
        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("摄像头 %s 已打开，分辨率 %sx%s", self.index, actual_w, actual_h)
        return self

    @property
    def opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def read(self, attempts: int = 3):
        """读取一帧；连续失败 ``attempts`` 次后抛出 :class:`CameraError`。"""
        if self._capture is None:
            raise CameraError("摄像头尚未打开，请先调用 open()")
        for _ in range(max(1, attempts)):
            ok, frame = self._capture.read()
            if ok and frame is not None:
                return frame
        raise CameraError("读取摄像头画面失败，请确认设备是否被拔出或被占用")

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        if not self.opened:
            self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "opened" if self.opened else "closed"
        return f"Camera(index={self.index}, {self.width}x{self.height}, {state})"
