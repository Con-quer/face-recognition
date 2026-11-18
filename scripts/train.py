"""训练模型：用 YOLO11 分类模型训练人脸识别器（对应教程中的 train.py）。

用法::

    python scripts/train.py                 # 默认 30 轮，CPU
    python scripts/train.py --epochs 50 --device 0

全部参数见：python scripts/train.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facerec.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
