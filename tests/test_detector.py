"""人脸检测、裁剪与清晰度判定。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="需要可用的 OpenCV")

from facerec.detector import (  # noqa: E402
    Face,
    FaceDetector,
    ascii_safe_path,
    blur_score,
    crop_face,
    default_detector,
    is_blurry,
    resolve_cascade_path,
)


def test_bundled_cascade_is_available():
    path = resolve_cascade_path()
    assert path.is_file()
    assert path.name == "haarcascade_frontalface_default.xml"


def test_resolve_cascade_prefers_explicit_path(tmp_path):
    custom = tmp_path / "custom.xml"
    custom.write_text("<opencv_storage/>", encoding="utf-8")
    assert resolve_cascade_path(custom) == custom


def test_missing_cascade_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_cascade_path(tmp_path / "not-exists.xml")


@pytest.mark.skipif(
    not str(Path(tempfile.gettempdir())).isascii(),
    reason="系统临时目录包含非 ASCII 字符",
)
def test_ascii_safe_path_keeps_ascii_paths_untouched():
    source = Path(tempfile.gettempdir()) / "facerec-ascii-test.xml"
    source.write_text("<opencv_storage/>", encoding="utf-8")

    assert ascii_safe_path(source) == source


@pytest.mark.skipif(
    not str(Path(tempfile.gettempdir())).isascii(),
    reason="系统临时目录包含非 ASCII 字符",
)
def test_ascii_safe_path_stages_non_ascii_files(tmp_path):
    """OpenCV 打不开中文路径，这里验证会自动转存到临时目录。"""
    source = tmp_path / "级联模型.xml"
    source.write_text("<opencv_storage/>", encoding="utf-8")

    staged = ascii_safe_path(source)

    assert staged != source
    assert staged.is_file()
    assert str(staged).isascii()
    assert staged.stat().st_size == source.stat().st_size


def test_detector_loads_and_returns_empty_on_blank_frame():
    # 项目目录名本身含中文，这里同时验证检测器能在中文路径下正常加载
    detector = FaceDetector()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detector.detect(frame) == []
    assert detector.detect_largest(frame) is None
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_faces_are_sorted_by_area():
    class FakeCascade:
        def detectMultiScale(self, *args, **kwargs):
            return [(0, 0, 10, 10), (0, 0, 40, 50), (5, 5, 20, 20)]

    detector = FaceDetector()
    detector._cascade = FakeCascade()
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    faces = detector.detect(blank)
    assert [face.area for face in faces] == [2000, 400, 100]
    assert detector.detect_largest(blank).box == (0, 0, 40, 50)


def test_crop_face_shape_and_clipping():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert crop_face(frame, Face(10, 10, 30, 30)).shape == (30, 30, 3)
    # 超出边界的框会被裁剪到画面以内
    assert crop_face(frame, Face(80, 80, 50, 50)).shape == (20, 20, 3)
    # 外扩参数同样不会越界
    padded = crop_face(frame, Face(10, 10, 30, 30), padding=0.5)
    assert padded.shape[0] <= 100
    assert padded.shape[1] <= 100


def test_crop_face_resizes_when_requested():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert crop_face(frame, Face(0, 0, 100, 100), size=64).shape == (64, 64, 3)


def test_crop_face_rejects_invalid_input():
    with pytest.raises(ValueError):
        crop_face(np.zeros((0, 0, 3), dtype=np.uint8), Face(0, 0, 1, 1))
    with pytest.raises(ValueError):
        crop_face(np.zeros((40, 40, 3), dtype=np.uint8), Face(100, 100, 10, 10))


def test_blur_score_separates_sharp_and_blurry_images():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
    flat = np.full((120, 120, 3), 127, dtype=np.uint8)

    assert blur_score(noise) > blur_score(flat)
    assert blur_score(flat) == pytest.approx(0.0, abs=1e-6)
    assert is_blurry(flat, 40.0) is True
    assert is_blurry(noise, 40.0) is False
    # 阈值为 0 时关闭过滤
    assert is_blurry(flat, 0.0) is False
    assert blur_score(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_default_detector_is_cached():
    assert default_detector(60) is default_detector(60)
