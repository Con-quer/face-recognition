"""采集流程的离线测试：用假摄像头替代真实硬件。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="需要可用的 OpenCV")

import facerec.collect as collect_module  # noqa: E402
from facerec.collect import CollectResult, collect, resolve_user_name  # noqa: E402
from facerec.config import CollectConfig  # noqa: E402
from facerec.dataset import count_images, list_images  # noqa: E402
from facerec.detector import Face  # noqa: E402

ESC = 27
SAVE = ord("s")


class FakeDetector:
    """固定返回给定人脸框，避免依赖真实图像。"""

    def __init__(self, faces: list[Face]):
        self.faces = faces
        self.cascade_path = Path("fake-cascade.xml")

    def detect(self, frame) -> list[Face]:
        return list(self.faces)


class FakeCamera:
    """离线摄像头，``read`` 始终返回同一帧。"""

    frame: np.ndarray | None = None

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.index = index

    def __enter__(self) -> FakeCamera:
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def read(self, attempts: int = 3) -> np.ndarray:
        assert FakeCamera.frame is not None
        return FakeCamera.frame.copy()

    def close(self) -> None:
        return None


class KeyFeeder:
    """按顺序吐出按键，用完后返回 ESC。"""

    def __init__(self, keys):
        self._keys = list(keys)

    def __call__(self, delay: int = 1) -> int:
        return self._keys.pop(0) if self._keys else ESC


@pytest.fixture()
def offline_ui(monkeypatch):
    """把摄像头与窗口替换为离线实现。"""

    def _install(frame: np.ndarray, keys) -> None:
        FakeCamera.frame = frame
        monkeypatch.setattr(collect_module, "Camera", FakeCamera)
        monkeypatch.setattr(collect_module, "ensure_display", lambda *a, **k: None)
        monkeypatch.setattr(collect_module, "show", lambda *a, **k: None)
        monkeypatch.setattr(collect_module, "destroy_all", lambda: None)
        monkeypatch.setattr(collect_module, "wait_key", KeyFeeder(keys))

    return _install


@pytest.fixture()
def noisy_frame():
    """纹理丰富的画面，清晰度足够高，不会被模糊过滤掉。"""
    return np.random.default_rng(0).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)


def test_collect_saves_samples_and_splits(dataset_root, offline_ui, noisy_frame):
    offline_ui(noisy_frame, [SAVE, SAVE, SAVE])
    config = CollectConfig(
        data_root=dataset_root,
        use_gui_prompt=False,
        target_count=3,
        blur_threshold=40.0,
        val_ratio=0.2,
        seed=1,
    )

    result = collect("张三", config, detector=FakeDetector([Face(60, 60, 100, 100)]))

    assert isinstance(result, CollectResult)
    assert result.saved == 3
    assert result.total == 3
    assert result.save_dir == dataset_root / "train" / "张三"
    # 3 张样本按 0.2 的比例划分出 1 张做验证集
    assert count_images(dataset_root / "train" / "张三") == 2
    assert count_images(dataset_root / "val" / "张三") == 1

    # 姓名是中文，确认写出的确实是可解码的图片（cv2.imwrite 在这里会失败）
    saved_image = list_images(dataset_root / "train" / "张三")[0]
    decoded = cv2.imdecode(np.fromfile(str(saved_image), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (100, 100, 3)


def test_collect_skips_blurry_frames(dataset_root, offline_ui):
    offline_ui(np.full((240, 320, 3), 127, dtype=np.uint8), [SAVE, SAVE])
    config = CollectConfig(
        data_root=dataset_root,
        use_gui_prompt=False,
        blur_threshold=40.0,
        auto_split=False,
    )

    result = collect("李四", config, detector=FakeDetector([Face(50, 50, 90, 90)]))

    assert result.saved == 0
    assert result.skipped_blurry == 2
    assert result.total == 0


def test_collect_resumes_file_numbering(dataset_root, offline_ui, noisy_frame, make_files):
    train_class = dataset_root / "train" / "王五"
    make_files(train_class, 2)
    offline_ui(noisy_frame, [SAVE])

    config = CollectConfig(
        data_root=dataset_root,
        use_gui_prompt=False,
        target_count=1,
        auto_split=False,
    )
    collect("王五", config, detector=FakeDetector([Face(10, 10, 80, 80)]))

    names = {path.name for path in list_images(train_class)}
    assert "face_0002.jpg" in names
    assert count_images(train_class) == 3


def test_collect_can_save_every_detected_face(dataset_root, offline_ui, noisy_frame):
    offline_ui(noisy_frame, [SAVE])
    config = CollectConfig(
        data_root=dataset_root,
        use_gui_prompt=False,
        save_all_faces=True,
        auto_split=False,
    )

    result = collect(
        "赵六",
        config,
        detector=FakeDetector([Face(10, 10, 80, 80), Face(150, 40, 70, 70)]),
    )

    assert result.saved == 2
    assert count_images(dataset_root / "train" / "赵六") == 2


def test_collect_respects_target_count(dataset_root, offline_ui, noisy_frame):
    offline_ui(noisy_frame, [SAVE] * 10)
    config = CollectConfig(
        data_root=dataset_root,
        use_gui_prompt=False,
        target_count=2,
        auto_split=False,
    )

    result = collect("钱七", config, detector=FakeDetector([Face(0, 0, 120, 120)]))
    assert result.saved == 2


def test_resolve_user_name_validates_input():
    assert resolve_user_name("  张三  ") == "张三"
    with pytest.raises(ValueError):
        resolve_user_name("../../etc")
