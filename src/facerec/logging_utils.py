"""统一的日志配置。

Windows 控制台默认使用 GBK 编码，中文日志容易触发 ``UnicodeEncodeError``，
这里在首次配置时把标准输出尽量切换到 UTF-8。
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_ROOT_NAME = "facerec"
_configured = False


def _enable_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 某些 IDE 或输出重定向场景下 stdout 不支持重新配置，忽略即可。
            pass


def get_logger(
    name: str = _ROOT_NAME,
    level: int | str = logging.INFO,
    *,
    reconfigure_console: bool = True,
) -> logging.Logger:
    """获取项目日志器，首次调用时完成基本配置。"""
    global _configured
    if not _configured:
        if reconfigure_console:
            _enable_utf8_console()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        package_logger = logging.getLogger(_ROOT_NAME)
        package_logger.handlers.clear()
        package_logger.addHandler(handler)
        package_logger.propagate = False
        _configured = True

    logger = logging.getLogger(name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}")
    logger.setLevel(level)
    return logger
