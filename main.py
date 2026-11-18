"""项目入口。

用法::

    python main.py collect --name 张三 --target 40
    python main.py train --epochs 50
    python main.py recognize
    python main.py doctor

等价于安装后执行 ``facerec <子命令>``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 未安装成包时，把 src 加入模块搜索路径，直接 python main.py 即可运行。
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from facerec.cli import main  # noqa: E402  必须在调整 sys.path 之后再导入

if __name__ == "__main__":
    raise SystemExit(main())
