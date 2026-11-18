"""准备数据：从摄像头采集人脸样本（对应教程中的 get_picture.py）。

用法::

    python scripts/get_picture.py                          # 弹窗输入姓名
    python scripts/get_picture.py --name 张三 --target 40   # 采集 40 张后自动结束

全部参数见：python scripts/get_picture.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facerec.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["collect", *sys.argv[1:]]))
