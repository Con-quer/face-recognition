"""数据集管理：目录结构、类别命名校验、训练/验证集划分。

目录约定（Ultralytics 分类任务的标准格式）::

    dataset/
        train/
            张三/face_0.jpg ...
            李四/face_0.jpg ...
        val/
            张三/face_0.jpg ...
            李四/face_0.jpg ...
"""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DATASET_DIR

#: 认可的图片后缀，注意必须带点号，否则 ``.notpng`` 之类的文件也会被误判。
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

_INVALID_NAME_CHARS = '<>:"/\\|?*'
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_FACE_FILE_RE = re.compile(r"^face_(\d+)\.(?:png|jpe?g|bmp|webp)$", re.IGNORECASE)


def sanitize_name(raw: object, *, max_length: int = 32) -> str:
    """校验并规范化类别名（姓名），非法时抛出 ``ValueError``。

    该名称会直接作为文件夹名使用，因此必须挡掉路径分隔符、非法字符以及
    Windows 保留设备名，避免出现 ``dataset/train/../..`` 这类越界路径。
    """
    if raw is None:
        raise ValueError("姓名不能为空")

    name = str(raw).strip()
    if not name:
        raise ValueError("姓名不能为空")

    if name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"姓名中不能包含路径分隔符：{raw!r}")

    bad_chars = sorted({char for char in name if char in _INVALID_NAME_CHARS or ord(char) < 32})
    if bad_chars:
        raise ValueError(f"姓名中包含非法字符 {''.join(bad_chars)}：{raw!r}")

    if name != name.rstrip(" ."):
        raise ValueError("姓名不能以空格或点号结尾")

    if name.split(".")[0].upper() in _RESERVED_NAMES:
        raise ValueError(f"“{name}”是 Windows 保留名称，请换一个")

    if len(name) > max_length:
        raise ValueError(f"姓名过长（最多 {max_length} 个字符，当前 {len(name)} 个）")

    return name


def ensure_layout(root: str | Path = DATASET_DIR) -> tuple[Path, Path]:
    """创建 ``train`` / ``val`` 目录并返回它们。"""
    train_dir = Path(root) / "train"
    val_dir = Path(root) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir


def list_images(directory: str | Path) -> list[Path]:
    """按文件名排序列出目录下的图片，目录不存在时返回空列表。"""
    target = Path(directory)
    if not target.is_dir():
        return []
    images = [
        path
        for path in target.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda path: path.name.lower())


def count_images(directory: str | Path) -> int:
    """统计目录下的图片数量。"""
    return len(list_images(directory))


def next_face_index(directory: str | Path, *, prefix: str = "face_") -> int:
    """返回下一个可用的样本序号，避免重复采集时覆盖旧文件。"""
    max_index = -1
    target = Path(directory)
    if target.is_dir():
        for path in target.glob(f"{prefix}*"):
            match = _FACE_FILE_RE.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def class_names(root: str | Path = DATASET_DIR) -> list[str]:
    """列出数据集中出现过的所有类别名。"""
    names: set[str] = set()
    for split in ("train", "val"):
        split_dir = Path(root) / split
        if split_dir.is_dir():
            names.update(path.name for path in split_dir.iterdir() if path.is_dir())
    return sorted(names)


def summarize(root: str | Path = DATASET_DIR) -> dict[str, dict[str, int]]:
    """统计每个类别在训练集/验证集中的样本数。"""
    root_path = Path(root)
    return {
        name: {
            "train": count_images(root_path / "train" / name),
            "val": count_images(root_path / "val" / name),
        }
        for name in class_names(root_path)
    }


@dataclass(frozen=True)
class SplitResult:
    """一次训练/验证集划分的结果。"""

    name: str
    moved: tuple[str, ...] = ()
    train_count: int = 0
    val_count: int = 0
    #: 未执行划分时的原因说明。
    skipped: str | None = None

    @property
    def moved_count(self) -> int:
        return len(self.moved)


def split_train_val(
    train_dir: str | Path,
    val_dir: str | Path,
    name: str,
    *,
    ratio: float = 0.2,
    seed: int = 42,
    rng: random.Random | None = None,
) -> SplitResult:
    """把 ``train/<name>`` 中约 ``ratio`` 比例的图片移动到 ``val/<name>``。

    与教程中的写法相比有三点改进：

    1. 验证集已有图片时直接跳过，重复执行不会把关联网越用越小；
    2. 用固定种子的随机抽样代替 ``listdir`` 的任意顺序，类别分布更均衡；
    3. 至少给训练集留下 1 张图片，避免样本过少时 Ultralytics 直接报错。
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"验证集比例必须在 [0, 1) 之间，当前为 {ratio}")

    train_class = Path(train_dir) / name
    val_class = Path(val_dir) / name
    val_class.mkdir(parents=True, exist_ok=True)

    existing_val = list_images(val_class)
    images = list_images(train_class)
    if existing_val:
        return SplitResult(
            name=name,
            train_count=len(images),
            val_count=len(existing_val),
            skipped="验证集已有图片，跳过划分",
        )
    if len(images) < 2:
        return SplitResult(
            name=name,
            train_count=len(images),
            skipped="训练样本不足 2 张，跳过划分",
        )

    count = max(1, int(round(len(images) * ratio)))
    count = min(count, len(images) - 1)

    picker = rng or random.Random(seed)
    picked = picker.sample(images, count)
    moved: list[str] = []
    for source in picked:
        destination = val_class / source.name
        shutil.move(str(source), str(destination))
        moved.append(source.name)

    return SplitResult(
        name=name,
        moved=tuple(moved),
        train_count=len(images) - len(moved),
        val_count=len(moved),
    )


def split_all(
    root: str | Path = DATASET_DIR,
    *,
    ratio: float = 0.2,
    seed: int = 42,
) -> list[SplitResult]:
    """对数据集中的每个类别执行一次划分。"""
    train_dir, val_dir = ensure_layout(root)
    rng = random.Random(seed)
    return [
        split_train_val(train_dir, val_dir, name, ratio=ratio, rng=rng)
        for name in class_names(root)
    ]


def validate_for_training(
    root: str | Path = DATASET_DIR,
    *,
    min_train_per_class: int = 5,
    min_val_per_class: int = 1,
    min_classes: int = 2,
) -> tuple[list[str], list[str]]:
    """检查数据集是否满足训练要求，返回 ``(错误, 警告)``。"""
    root_path = Path(root)
    train_dir = root_path / "train"
    val_dir = root_path / "val"

    if not train_dir.is_dir():
        return ([f"找不到训练集目录：{train_dir}"], [])

    stats = summarize(root_path)
    if not stats:
        return ([f"训练集 {train_dir} 为空，请先运行采集脚本收集人脸样本"], [])

    errors: list[str] = []
    warnings: list[str] = []

    if len(stats) < min_classes:
        warnings.append(
            f"数据集中只有 {len(stats)} 个类别，分类任务至少需要 {min_classes} 个人才能体现效果"
        )

    for name, counts in stats.items():
        if counts["train"] + counts["val"] == 0:
            errors.append(f"类别 {name} 没有任何图片")
            continue
        if counts["train"] < min_train_per_class:
            warnings.append(
                f"类别 {name} 训练集只有 {counts['train']} 张（建议不少于 {min_train_per_class} 张）"
            )
        if counts["val"] < min_val_per_class:
            warnings.append(f"类别 {name} 验证集只有 {counts['val']} 张")

    if not val_dir.is_dir() or not list_images(val_dir):
        warnings.append("验证集为空，训练前会尝试自动划分")

    return errors, warnings
