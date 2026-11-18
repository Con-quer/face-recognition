"""配置默认值与派生路径。"""

from __future__ import annotations

from pathlib import Path

from facerec.config import (
    CLASSIFY_DIR,
    DEFAULT_RUN_NAME,
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    CollectConfig,
    RecognizeConfig,
    TrainConfig,
    describe,
)


def test_project_root_points_to_repository():
    assert (PROJECT_ROOT / "src" / "facerec" / "config.py").is_file()
    assert (PROJECT_ROOT / "main.py").is_file()


def test_default_weights_match_tutorial_layout():
    expected = CLASSIFY_DIR / DEFAULT_RUN_NAME / "weights" / "best.pt"
    assert DEFAULT_WEIGHTS == expected
    assert DEFAULT_WEIGHTS.as_posix().endswith("runs/classify/face_recognize/weights/best.pt")


def test_train_config_run_dir_and_weights(tmp_path: Path):
    config = TrainConfig(project=tmp_path, name="demo")
    assert config.run_dir == tmp_path / "demo"
    # 还没有任何权重时回退到 last.pt
    assert config.weights_path.name == "last.pt"

    weights = config.run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"weights")
    assert config.weights_path.name == "best.pt"


def test_describe_serialises_paths():
    payload = describe(RecognizeConfig(weights=Path("model.pt")))
    assert payload["weights"] == "model.pt"
    assert payload["mirror"] is True


def test_collect_defaults_follow_tutorial():
    config = CollectConfig()
    assert (config.width, config.height) == (1280, 720)
    assert config.val_ratio == 0.2
    assert config.camera_index == 0
