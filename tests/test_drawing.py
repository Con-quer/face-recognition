"""画面文字绘制（含中文回退）。"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="需要可用的 OpenCV")

from facerec.drawing import find_font_path, put_text, text_size  # noqa: E402


def test_ascii_text_is_drawn():
    image = np.zeros((80, 300, 3), dtype=np.uint8)
    put_text(image, "Hello 2026", (10, 10), color=(0, 255, 0), font_size=24)
    assert image.any()
    assert image[..., 1].max() > 0


def test_background_is_filled():
    image = np.zeros((80, 300, 3), dtype=np.uint8)
    put_text(image, "AB", (20, 20), background=(40, 40, 40), font_size=24)
    assert image[15, 15].tolist() == [40, 40, 40]


def test_empty_text_does_not_touch_image():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    put_text(image, "", (0, 0))
    assert not image.any()


def test_text_size_is_positive():
    width, height = text_size("Hello", 24)
    assert width > 0
    assert height > 0


@pytest.mark.skipif(find_font_path() is None, reason="系统未安装中文字体")
def test_chinese_text_is_drawn():
    image = np.zeros((120, 400, 3), dtype=np.uint8)
    put_text(image, "张三 0.93", (10, 10), font_size=32)
    assert image.any()
    assert text_size("张三", 32)[0] > 32


def test_put_text_returns_same_buffer():
    image = np.zeros((40, 200, 3), dtype=np.uint8)
    assert put_text(image, "x", (5, 5)) is image
