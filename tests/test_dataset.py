"""数据集目录、命名校验与划分逻辑。"""

from __future__ import annotations

from pathlib import Path

import pytest

from facerec import dataset
from facerec.dataset import (
    count_images,
    list_images,
    next_face_index,
    sanitize_name,
    split_all,
    split_train_val,
    summarize,
    validate_for_training,
)


@pytest.mark.parametrize("raw", ["张三", "Li Hua", "小明-2", "a.b", "汤姆 猫"])
def test_sanitize_name_accepts_normal_names(raw):
    assert sanitize_name(raw) == raw


def test_sanitize_name_trims_whitespace():
    assert sanitize_name("  张三  ") == "张三"
    # 首尾空白会被去掉，这不是错误输入
    assert sanitize_name("name ") == "name"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "..",
        "../dataset",
        "a/b",
        "a\\b",
        "张:三",
        "张*三",
        "CON",
        "aux",
        "name.",
        "x" * 33,
    ],
)
def test_sanitize_name_rejects_unsafe_input(raw):
    with pytest.raises(ValueError):
        sanitize_name(raw)


def test_list_images_requires_real_suffix(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "c.notpng").write_bytes(b"x")
    (tmp_path / "d.txt").write_bytes(b"x")
    names = [path.name for path in list_images(tmp_path)]
    assert names == ["a.jpg", "b.PNG"]
    assert count_images(tmp_path) == 2
    assert count_images(tmp_path / "missing") == 0


def test_next_face_index_resumes_numbering(tmp_path: Path, make_files):
    make_files(tmp_path, 3, start=0)
    make_files(tmp_path, 1, start=7)
    assert next_face_index(tmp_path) == 8
    assert next_face_index(tmp_path / "empty") == 0


def test_split_moves_expected_amount(dataset_root: Path, make_files):
    train_class = dataset_root / "train" / "张三"
    make_files(train_class, 10)

    result = split_train_val(
        dataset_root / "train", dataset_root / "val", "张三", ratio=0.2, seed=1
    )

    assert result.moved_count == 2
    assert result.train_count == 8
    assert result.val_count == 2
    assert not result.skipped
    assert count_images(train_class) == 8
    assert count_images(dataset_root / "val" / "张三") == 2


def test_split_is_idempotent(dataset_root: Path, make_files):
    make_files(dataset_root / "train" / "张三", 10)
    split_train_val(dataset_root / "train", dataset_root / "val", "张三")

    second = split_train_val(dataset_root / "train", dataset_root / "val", "张三")
    assert second.skipped is not None
    assert second.moved_count == 0
    assert count_images(dataset_root / "train" / "张三") == 8


def test_split_keeps_at_least_one_training_image(dataset_root: Path, make_files):
    make_files(dataset_root / "train" / "李四", 1)
    result = split_train_val(dataset_root / "train", dataset_root / "val", "李四")
    assert result.moved_count == 0
    assert result.skipped is not None
    assert count_images(dataset_root / "train" / "李四") == 1


def test_split_is_reproducible_with_seed(dataset_root: Path, make_files):
    make_files(dataset_root / "train" / "王五", 20)
    first = split_train_val(dataset_root / "train", dataset_root / "val", "王五", seed=7)

    # 换一个根目录，用同样的种子应当选出同样的文件
    other = dataset_root.parent / "other"
    make_files(other / "train" / "王五", 20)
    second = split_train_val(other / "train", other / "val", "王五", seed=7)

    assert first.moved == second.moved


def test_split_rejects_invalid_ratio(dataset_root: Path):
    with pytest.raises(ValueError):
        split_train_val(dataset_root / "train", dataset_root / "val", "张三", ratio=1.0)


def test_split_all_covers_every_class(dataset_root: Path, make_files):
    for name in ("张三", "李四"):
        make_files(dataset_root / "train" / name, 10)

    results = split_all(dataset_root, ratio=0.2, seed=3)

    assert {item.name for item in results} == {"张三", "李四"}
    stats = summarize(dataset_root)
    assert stats["张三"] == {"train": 8, "val": 2}
    assert stats["李四"] == {"train": 8, "val": 2}


def test_validate_reports_empty_dataset(dataset_root: Path):
    errors, warnings = validate_for_training(dataset_root)
    assert errors and "为空" in errors[0]
    assert warnings == []


def test_validate_warns_about_few_samples(dataset_root: Path, make_files):
    make_files(dataset_root / "train" / "张三", 3)
    errors, warnings = validate_for_training(dataset_root)
    assert errors == []
    assert any("张三" in message for message in warnings)
    assert any("只有 1 个类别" in message for message in warnings)


def test_dataset_module_uses_project_defaults():
    assert dataset.DATASET_DIR.name == "dataset"
