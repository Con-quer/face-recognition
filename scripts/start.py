"""开始识别：加载训练好的权重，打开摄像头实时识别（对应教程中的 start.py）。

用法::

    python scripts/start.py
    python scripts/start.py --weights runs/classify/face_recognize/weights/best.pt

全部参数见：python scripts/start.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facerec.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["recognize", *sys.argv[1:]]))
