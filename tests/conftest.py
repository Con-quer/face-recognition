"""测试公共配置与夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """空的 ``dataset/train``、``dataset/val`` 目录结构。"""
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir(parents=True)
    return root


@pytest.fixture()
def make_files():
    """生成占位文件，用于验证目录扫描与划分逻辑（不涉及图片解码）。"""

    def _make(
        directory: Path,
        count: int,
        *,
        start: int = 0,
        suffix: str = ".jpg",
        prefix: str = "face_",
    ) -> list[Path]:
        directory.mkdir(parents=True, exist_ok=True)
        created = []
        for index in range(start, start + count):
            path = directory / f"{prefix}{index:04d}{suffix}"
            path.write_bytes(b"placeholder")
            created.append(path)
        return created

    return _make
