"""识别阶段：加载训练好的模型，实时识别人脸（对应教程 ``start.py``）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import DEFAULT_WEIGHTS, RecognizeConfig
from .drawing import put_text
from .gui import destroy_all, ensure_display, show, wait_key
from .logging_utils import get_logger
from .trainer import require_ultralytics

logger = get_logger(__name__)

ESC_KEY = 27


@dataclass(frozen=True)
class Prediction:
    """一次分类的结果。"""

    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...] = ()


class FaceClassifier:
    """封装 Ultralytics 分类模型，输出 ``(标签, 置信度)``。"""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS) -> None:
        weights_path = Path(weights)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"找不到模型权重：{weights_path}\n"
                "请先训练模型（python main.py train），或通过 --weights 指定已有的 best.pt。"
            )

        YOLO = require_ultralytics()
        logger.info("加载模型：%s", weights_path)
        self.weights = weights_path
        self.model = YOLO(str(weights_path))
        self.class_names = list(getattr(self.model, "names", {}).values())

    def predict(
        self,
        frame: np.ndarray,
        *,
        top_k: int = 3,
        plot: bool = False,
    ) -> tuple[Prediction, np.ndarray | None]:
        """对整帧画面做分类，返回预测结果与可选的标注图。"""
        results = self.model(frame, verbose=False)
        result = results[0]
        probabilities = result.probs

        if probabilities is None:
            return Prediction(label="unknown", confidence=0.0), None

        values = probabilities.data.tolist()
        names = result.names
        order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)

        best_index = order[0]
        prediction = Prediction(
            label=str(names.get(best_index, best_index)),
            confidence=float(values[best_index]),
            top_k=tuple((str(names.get(i, i)), float(values[i])) for i in order[: max(1, top_k)]),
        )
        plotted = result.plot() if plot else None
        return prediction, plotted


def run(config: RecognizeConfig | None = None) -> None:
    """打开摄像头，持续输出识别结果。按 ESC / Q 退出。"""
    config = config or RecognizeConfig()
    classifier = FaceClassifier(config.weights)

    window = config.window_name
    ensure_display(window, config.width, config.height)

    mirror = config.mirror
    show_hud = config.show_hud
    previous = time.perf_counter()
    fps = 0.0

    logger.info("开始识别，ESC / Q 退出，M 切换镜像，H 隐藏信息")
    try:
        with Camera(config.camera_index, config.width, config.height) as camera:
            while True:
                frame = camera.read()
                if mirror:
                    frame = cv2.flip(frame, 1)

                prediction, plotted = classifier.predict(frame, plot=config.use_plot)
                display = plotted if plotted is not None else frame

                if show_hud:
                    now = time.perf_counter()
                    elapsed = now - previous
                    previous = now
                    if elapsed > 0:
                        # 简单指数平滑，避免数字剧烈跳动。
                        fps = 0.8 * fps + 0.2 * (1.0 / elapsed)

                    if prediction.confidence >= config.conf_threshold:
                        color = (0, 255, 0)
                        text = f"{prediction.label}  {prediction.confidence:.2f}"
                    else:
                        color = (0, 165, 255)
                        text = f"unknown  {prediction.confidence:.2f}"

                    put_text(
                        display,
                        text,
                        (12, 12),
                        color=color,
                        font_size=30,
                        background=(30, 30, 30),
                    )
                    put_text(
                        display,
                        f"FPS {fps:5.1f}",
                        (12, 56),
                        color=(200, 200, 200),
                        font_size=20,
                        background=(30, 30, 30),
                    )
                    details = "  ".join(
                        f"{label}:{score:.2f}" for label, score in prediction.top_k[1:]
                    )
                    if details:
                        put_text(display, details, (12, 88), color=(180, 180, 180), font_size=18)
                    put_text(
                        display,
                        "ESC/Q quit   M mirror   H hud",
                        (12, display.shape[0] - 32),
                        color=(200, 200, 200),
                        font_size=18,
                    )

                show(window, display)
                key = wait_key(1)
                if key in (ESC_KEY, ord("q"), ord("Q")):
                    break
                if key in (ord("m"), ord("M")):
                    mirror = not mirror
                elif key in (ord("h"), ord("H")):
                    show_hud = not show_hud
    finally:
        destroy_all()
