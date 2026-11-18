"""采集阶段：从摄像头截取人脸并保存为训练样本（对应教程 ``get_picture.py``）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .camera import Camera
from .config import CollectConfig
from .dataset import (
    SplitResult,
    count_images,
    ensure_layout,
    next_face_index,
    sanitize_name,
    split_train_val,
)
from .detector import FaceDetector, blur_score, crop_face, is_blurry
from .drawing import put_text
from .gui import destroy_all, ensure_display, show, wait_key
from .logging_utils import get_logger

logger = get_logger(__name__)

ESC_KEY = 27


class CollectionCancelled(RuntimeError):
    """用户在弹窗或命令行中取消了采集。"""


@dataclass
class CollectResult:
    """一次采集的结果统计。"""

    name: str
    save_dir: Path
    saved: int
    total: int
    skipped_blurry: int = 0
    split: SplitResult | None = None


def prompt_name_dialog() -> str | None:
    """弹窗询问姓名，取消或环境不支持时返回 ``None``。"""
    try:
        import tkinter as tk
        from tkinter import messagebox, simpledialog
    except ImportError:  # 无 tkinter 的解释器
        return None

    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:  # 无图形界面
        return None

    try:
        while True:
            value = simpledialog.askstring("命名", "请输入姓名", parent=root)
            if value is None:
                return None
            try:
                return sanitize_name(value)
            except ValueError as exc:
                messagebox.showwarning("命名", str(exc), parent=root)
    finally:
        root.destroy()


def resolve_user_name(raw: str | None = None, *, use_gui: bool = True) -> str:
    """确定本次采集使用的姓名：命令行参数 > 弹窗 > 命令行输入。"""
    if raw is not None and str(raw).strip():
        return sanitize_name(raw)

    if use_gui:
        name = prompt_name_dialog()
        if name is not None:
            return name
        logger.info("未使用弹窗输入，改为在命令行输入姓名")

    while True:
        try:
            value = input("请输入姓名（直接回车取消）：").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise CollectionCancelled("已取消采集") from exc
        if not value:
            raise CollectionCancelled("没有输入姓名，已取消采集")
        try:
            return sanitize_name(value)
        except ValueError as exc:
            print(exc)


def _save_face(path: Path, image, jpeg_quality: int) -> None:
    """保存人脸图片。

    这里刻意使用 ``cv2.imencode`` + Python 写文件，而不是 ``cv2.imwrite``：
    OpenCV 在 Windows 上使用窄字符接口打开文件，路径或姓名含中文时会静默失败
    （``imwrite`` 只返回 ``False`` 而不抛异常）。本项目姓名通常是中文，
    项目目录本身也可能带中文，因此必须绕开该限制。
    """
    suffix = path.suffix or ".jpg"
    params: list[int] = []
    if suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"编码图片失败：{path.name}")
    path.write_bytes(buffer.tobytes())


def collect(
    name: str | None = None,
    config: CollectConfig | None = None,
    *,
    detector: FaceDetector | None = None,
) -> CollectResult:
    """打开摄像头采集指定姓名的人脸样本，并按比例划分验证集。"""
    config = config or CollectConfig()
    user_name = resolve_user_name(name, use_gui=config.use_gui_prompt)
    logger.info("本次采集：%s", user_name)

    train_dir, val_dir = ensure_layout(config.data_root)
    save_dir = train_dir / user_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 接着已有文件的编号继续，重复采集不会覆盖旧样本。
    index = next_face_index(save_dir)
    if index:
        logger.info("检测到已有 %s 张样本，从 face_%04d 继续编号", index, index)

    face_detector = detector or FaceDetector(config.cascade_path, min_size=config.min_face_size)
    logger.debug("使用级联文件：%s", face_detector.cascade_path)

    window = config.window_name
    ensure_display(window, config.width, config.height)

    saved = 0
    skipped_blurry = 0
    extension = config.image_format.lower().lstrip(".")

    logger.info("按 S 保存样本，ESC / Q 结束采集")
    if config.target_count:
        logger.info("目标样本数：%s 张", config.target_count)

    try:
        with Camera(config.camera_index, config.width, config.height) as camera:
            while True:
                frame = camera.read()
                # 在干净的副本上裁剪，避免把提示用的绿框截进样本里。
                clean_frame = frame.copy()
                faces = face_detector.detect(frame)

                for position, face in enumerate(faces):
                    color = (0, 255, 0) if position == 0 else (255, 170, 0)
                    cv2.rectangle(
                        frame,
                        (face.x, face.y),
                        (face.x + face.w, face.y + face.h),
                        color,
                        2,
                    )

                put_text(
                    frame,
                    f"{user_name}  saved: {saved}  faces: {len(faces)}",
                    (12, 12),
                    font_size=26,
                    background=(40, 40, 40),
                )
                put_text(
                    frame,
                    "S: save    ESC/Q: quit",
                    (12, 52),
                    color=(200, 200, 200),
                    font_size=20,
                )

                show(window, frame)
                key = wait_key(1)

                if key in (ESC_KEY, ord("q"), ord("Q")):
                    logger.info("结束采集")
                    break

                if key not in (ord("s"), ord("S")):
                    continue

                if not faces:
                    logger.warning("画面中没有检测到人脸，本次未保存")
                    continue

                targets = faces if config.save_all_faces else faces[:1]
                for face in targets:
                    crop = crop_face(clean_frame, face, size=config.face_size)
                    if is_blurry(crop, config.blur_threshold):
                        skipped_blurry += 1
                        logger.info(
                            "样本偏模糊（清晰度 %.1f < %.1f），已跳过",
                            blur_score(crop),
                            config.blur_threshold,
                        )
                        continue

                    face_path = save_dir / f"face_{index:04d}.{extension}"
                    _save_face(face_path, crop, config.jpeg_quality)
                    logger.info("已保存 %s（%sx%s）", face_path.name, face.w, face.h)
                    index += 1
                    saved += 1

                if config.target_count and saved >= config.target_count:
                    logger.info("已达到目标样本数 %s 张，自动结束采集", config.target_count)
                    break
    finally:
        destroy_all()

    total = count_images(save_dir)
    split_result: SplitResult | None = None
    if config.auto_split:
        split_result = split_train_val(
            train_dir,
            val_dir,
            user_name,
            ratio=config.val_ratio,
            seed=config.seed,
        )
        if split_result.skipped:
            logger.info("划分跳过：%s", split_result.skipped)
        else:
            logger.info(
                "划分完成：验证集移出 %s 张，训练集剩余 %s 张",
                split_result.moved_count,
                split_result.train_count,
            )

    logger.info("%s 的样本总数：%s 张（本次新增 %s 张）", user_name, total, saved)
    return CollectResult(
        name=user_name,
        save_dir=save_dir,
        saved=saved,
        total=total,
        skipped_blurry=skipped_blurry,
        split=split_result,
    )
