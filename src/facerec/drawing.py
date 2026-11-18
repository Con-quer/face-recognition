"""画面文字绘制。

OpenCV 的 ``cv2.putText`` 只支持 ASCII（Hershey）字体，姓名如果是中文会显示成
一连串问号。这里在检测到非 ASCII 文本时自动改用 Pillow 渲染，并优先挑选系统中
已安装的中文字体。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)

#: 常见系统中文字体，按优先级排列。
FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


@lru_cache(maxsize=8)
def find_font_path() -> str | None:
    """返回第一个可用的中文字体路径，找不到则返回 ``None``。"""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    logger.debug("未找到可用的中文字体，中文将回退为问号显示")
    return None


@lru_cache(maxsize=32)
def _load_font(font_size: int):
    """加载可显示中文的字体，Pillow 缺失或字体不可用时返回 ``None``。"""
    try:
        from PIL import ImageFont
    except ImportError:
        # 没装 Pillow 时退化为 OpenCV 自带字体（中文会显示为问号）。
        logger.debug("未安装 Pillow，中文将无法正确绘制")
        return None

    font_path = find_font_path()
    if font_path is None:
        return None
    try:
        return ImageFont.truetype(font_path, font_size)
    except (OSError, ValueError):  # pragma: no cover - 字体文件损坏
        return None


def _is_ascii(text: str) -> bool:
    return all(ord(char) < 128 for char in text)


def text_size(text: str, font_size: int) -> tuple[int, int]:
    """估算文本占用的 ``(宽, 高)``。"""
    if _is_ascii(text):
        scale = font_size / 32.0
        (width, height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, font_size // 16)
        )
        return width, max(height, font_size)

    font = _load_font(font_size)
    if font is None:
        scale = font_size / 32.0
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        return width, max(height, font_size)

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1, 1))
    left, top, right, bottom = ImageDraw.Draw(image).textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def put_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    font_size: int = 24,
    background: tuple[int, int, int] | None = None,
    padding: int = 6,
) -> np.ndarray:
    """在图像上绘制文本并原图返回（就地修改）。

    ``org`` 为文本左上角坐标；``background`` 不为空时先铺一层底色，
    保证浅色背景下文字依旧清晰。
    """
    if not text:
        return image

    x, y = int(org[0]), int(org[1])
    width, height = text_size(text, font_size)
    height = max(height, 12)

    if background is not None:
        cv2.rectangle(
            image,
            (max(0, x - padding), max(0, y - padding)),
            (
                min(image.shape[1] - 1, x + width + padding),
                min(image.shape[0] - 1, y + height + padding),
            ),
            background,
            thickness=-1,
        )

    if _is_ascii(text):
        scale = font_size / 32.0
        thickness = max(1, font_size // 16)
        baseline = int(height * 0.8)
        cv2.putText(
            image,
            text,
            (x, y + baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return image

    font = _load_font(font_size)
    if font is None:
        # 找不到中文字体时退化为 ASCII 绘制，至少能看出标签位置。
        cv2.putText(
            image,
            text.encode("ascii", "replace").decode("ascii"),
            (x, y + int(height * 0.8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size / 32.0,
            color,
            max(1, font_size // 16),
            lineType=cv2.LINE_AA,
        )
        return image

    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    image[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    return image
