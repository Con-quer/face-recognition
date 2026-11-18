"""基于 OpenCV Haar 级联分类器的人脸检测与裁剪。"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_CASCADE
from .logging_utils import get_logger

logger = get_logger(__name__)

_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


@lru_cache(maxsize=8)
def ascii_safe_path(path: str | Path) -> Path:
    """把非 ASCII 路径上的文件复制到临时目录，返回 OpenCV 能打开的路径。

    OpenCV 在 Windows 上通过窄字符接口打开文件，路径含中文时
    ``cv2.CascadeClassifier`` / ``cv2.imread`` 会直接失败。项目目录名、
    样本文件名都可能是中文，因此这里统一做一次转存。
    """
    source = Path(path)
    if _is_ascii(str(source)):
        return source

    try:
        temp_root = Path(tempfile.gettempdir())
    except (OSError, ValueError):  # pragma: no cover - 极端环境下拿不到临时目录
        return source
    if not _is_ascii(str(temp_root)):
        # 临时目录也含非 ASCII 字符，退而使用原路径，由调用方给出提示。
        return source

    staging = temp_root / "facerec-cache"
    staging.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:8]
    target = staging / f"{digest}-{_ascii_slug(source.stem)}{_ascii_slug(source.suffix)}"
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, target)
    return target


def _ascii_slug(text: str) -> str:
    """把文件名片段压缩成 ASCII 安全形式。"""
    cleaned = "".join(
        char if char.isascii() and (char.isalnum() or char in "-_.") else "_" for char in text
    )
    return cleaned[:48]


@dataclass(frozen=True)
class Face:
    """一个人脸框，坐标以画面左上角为原点。"""

    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)``，与 ``cv2.rectangle`` 的用法兼容。"""
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def is_square_enough(self) -> bool:
        return self.w > 0 and self.h > 0


def resolve_cascade_path(explicit: str | Path | None = None) -> Path:
    """定位 Haar 级联文件：命令行指定 > 项目 ``assets`` > OpenCV 自带。

    显式指定的路径不存在时直接报错，避免"以为换了个模型，其实还在用默认值"。
    """
    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"指定的 Haar 级联文件不存在：{explicit_path}")
        return explicit_path

    candidates: list[Path] = [DEFAULT_CASCADE]
    try:
        candidates.append(Path(cv2.data.haarcascades) / _CASCADE_FILENAME)
    except AttributeError:  # pragma: no cover - 极少数裁剪版 OpenCV 没有 cv2.data
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  ".join(str(item) for item in candidates)
    raise FileNotFoundError(
        "找不到 Haar 级联文件 haarcascade_frontalface_default.xml，已尝试：\n  "
        f"{searched}\n"
        "可以重新安装带 data 目录的 OpenCV（pip install -U opencv-python），"
        "或把该文件放到项目的 assets 目录下。"
    )


class FaceDetector:
    """包装 ``cv2.CascadeClassifier``，返回结构化的 :class:`Face` 列表。"""

    def __init__(
        self,
        cascade_path: str | Path | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 60,
        equalize: bool = True,
    ) -> None:
        # OpenCV 5.x 的部分发行版不再包含 objdetect 模块，这里提前给出明确提示。
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "当前 OpenCV 版本不包含 Haar 级联检测器（cv2.CascadeClassifier）。\n"
                '请改用 4.x 版本：pip install "opencv-python>=4.8,<5"'
            )

        self.cascade_path = resolve_cascade_path(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = max(1, int(min_size))
        self.equalize = equalize

        self._cascade = cv2.CascadeClassifier(str(ascii_safe_path(self.cascade_path)))
        is_empty = getattr(self._cascade, "empty", None)
        if is_empty is None or is_empty():
            raise RuntimeError(
                f"加载 Haar 级联文件失败：{self.cascade_path}\n"
                "如果项目路径或临时目录中包含中文，建议把项目移动到纯英文路径后重试。"
            )

    def detect(self, frame: np.ndarray) -> list[Face]:
        """检测画面中的所有人脸，按面积从大到小排序。"""
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.equalize:
            # 直方图均衡化可以明显改善背光、暗光环境下的检出率。
            gray = cv2.equalizeHist(gray)

        detected = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        faces = [Face(int(x), int(y), int(w), int(h)) for x, y, w, h in detected]
        return sorted(faces, key=lambda face: face.area, reverse=True)

    def detect_largest(self, frame: np.ndarray) -> Face | None:
        """只返回面积最大的人脸，适合单人采集场景。"""
        faces = self.detect(frame)
        return faces[0] if faces else None


def crop_face(
    frame: np.ndarray,
    face: Face,
    *,
    padding: float = 0.0,
    size: int = 0,
) -> np.ndarray:
    """按人脸框裁剪画面，可选外扩与缩放。"""
    if frame is None or frame.size == 0:
        raise ValueError("待裁剪的画面为空")

    height, width = frame.shape[:2]
    pad_x = int(face.w * padding)
    pad_y = int(face.h * padding)

    x1 = max(0, face.x - pad_x)
    y1 = max(0, face.y - pad_y)
    x2 = min(width, face.x + face.w + pad_x)
    y2 = min(height, face.y + face.h + pad_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"人脸框超出画面范围：{face.box}")

    crop = frame[y1:y2, x1:x2].copy()
    if size > 0 and (crop.shape[0] != size or crop.shape[1] != size):
        interpolation = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (size, size), interpolation=interpolation)
    return crop


def blur_score(image: np.ndarray) -> float:
    """用拉普拉斯算子方差衡量清晰度，数值越小越模糊。"""
    if image is None or image.size == 0:
        return 0.0
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float) -> bool:
    """清晰度低于阈值即认为模糊；``threshold <= 0`` 表示不做过滤。"""
    if threshold <= 0:
        return False
    return blur_score(image) < threshold


@lru_cache(maxsize=4)
def default_detector(min_size: int = 60) -> FaceDetector:
    """缓存一个默认检测器，避免重复加载级联文件。"""
    return FaceDetector(min_size=min_size)
