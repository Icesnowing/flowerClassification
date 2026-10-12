# -*- coding: utf-8 -*-
"""
从 PC 端 TFLite 导出 Golden 向量，供 Android GoldenParityTest 对照。

每张图写一个 JSON（默认 golden/<stem>.json），并生成 golden/manifest.json 索引。
预处理与 val_tflite.py 一致：get_transform(train=False) + NHWC float32。
默认把 Android asset 存成 PC 端 Resize(256) 后的 PNG，再对该 PNG 算 golden，
避免手机对超大原图做 Bitmap 缩放与 torchvision 抗锯齿 Resize 不一致。

用法示例:
  python export_golden.py --model-path saved_model/ft14_float32.tflite \\
      --image-path flower14/val/rose/xxx.jpg --image-path flower14/val/tulip/yyy.jpg

  python export_golden.py --model-path saved_model/ft14_float32.tflite \\
      --image-path path/to/golden_dir

  python export_golden.py --model-path saved_model/ft14_float32.tflite \\
      --dataset flower14 --data-root flower14 --limit 5 \\
      --copy-images android_app/app/src/androidTest/assets/golden_images \\
      --copy-json android_app/app/src/androidTest/assets/golden
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision.transforms import functional as TF

import data_utils
from val_tflite import (
    build_transform,
    load_test_data,
    prepare_input_nhwc,
    resolve_model_path,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLDEN_DIR = os.path.join(ROOT, "golden")
INPUT_HEAD_LEN = 12
SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return safe or "image"


def _unique_asset_name(stem: str, used: set[str], ext: str) -> str:
    name = stem + ext
    if name not in used:
        used.add(name)
        return name
    i = 2
    while True:
        candidate = f"{stem}_{i}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def _resize_size_from_config(config_json: str | None) -> int:
    if config_json and os.path.isfile(config_json):
        with open(config_json, encoding="utf-8") as f:
            cfg = json.load(f)
        return int(cfg.get("resize_size", 256))
    return 256


def _pil_resize_short_side(pil_image: Image.Image, short_side: int) -> Image.Image:
    """与 data_utils.get_transform(train=False) 中的 Resize 一致（torchvision / PIL）。"""
    return TF.resize(pil_image, short_side)


def _write_android_asset(
    pil_image: Image.Image,
    dest_path: str,
    asset_mode: str,
    short_side: int,
) -> None:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    if asset_mode == "raw_copy":
        raise ValueError("raw_copy 应在 export_images 中直接 shutil.copy2")
    if asset_mode == "resized256_png":
        resized = _pil_resize_short_side(pil_image, short_side)
        resized.save(dest_path, format="PNG", compress_level=3)
        return
    raise ValueError(f"未知 asset_mode: {asset_mode}")


def run_one(
    interpreter: tf.lite.Interpreter,
    input_detail: dict,
    output_detail: dict,
    image_path: str,
    transform,
) -> tuple[list[float], int, int]:
    img = Image.open(image_path).convert("RGB")
    x = prepare_input_nhwc(img, transform, input_detail)
    flat = x.reshape(-1).astype(np.float32)
    head = flat[:INPUT_HEAD_LEN].tolist()

    input_idx = input_detail["index"]
    output_idx = output_detail["index"]
    interpreter.set_tensor(input_idx, x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_idx)[0]
    if output_detail["dtype"] == np.int8:
        scale, zero_point = output_detail["quantization"]
        if scale != 0:
            out = (out.astype(np.float32) - zero_point) * scale
    argmax = int(np.argmax(out))
    num_logits = int(out.shape[0])
    return head, argmax, num_logits


def write_case_json(
    out_path: str,
    *,
    source_path: str,
    asset_name: str,
    input_head: list[float],
    argmax: int,
    num_logits: int,
    model_path: str,
    float_abs_tol: float,
    asset_mode: str,
    golden_image_path: str,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_path": os.path.normpath(source_path),
        "golden_image_path": os.path.normpath(golden_image_path),
        "asset_name": asset_name,
        "asset_mode": asset_mode,
        "input_head_len": INPUT_HEAD_LEN,
        "input_head": input_head,
        "argmax": argmax,
        "num_logits": num_logits,
        "float_abs_tol": float_abs_tol,
        "model_basename": os.path.basename(model_path),
        "model_sha256": _file_sha256(model_path),
        "preprocess": "Resize(256)+CenterCrop(224)+RGB/255+ImageNet mean/std, NHWC float32",
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def export_images(
    model_path: str,
    image_paths: list[str],
    output_dir: str,
    config_json: str | None,
    float_abs_tol: float,
    copy_images_dir: str | None,
    copy_json_dir: str | None,
    asset_mode: str,
) -> None:
    transform = build_transform(config_json)
    short_side = _resize_size_from_config(config_json)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    os.makedirs(output_dir, exist_ok=True)
    used_assets: set[str] = set()
    manifest_cases: list[dict] = []

    for image_path in image_paths:
        abs_image = os.path.abspath(image_path)
        if not os.path.isfile(abs_image):
            raise FileNotFoundError(f"找不到图片: {abs_image}")

        stem = _safe_stem(abs_image)
        ext = ".png" if asset_mode == "resized256_png" else os.path.splitext(abs_image)[1].lower()
        asset_name = _unique_asset_name(stem, used_assets, ext)
        json_name = os.path.splitext(asset_name)[0] + ".json"
        json_path = os.path.join(output_dir, json_name)

        pil = Image.open(abs_image).convert("RGB")
        asset_work_path = abs_image
        if asset_mode == "resized256_png":
            asset_work_path = os.path.join(output_dir, "_work_" + asset_name)
            _write_android_asset(pil, asset_work_path, asset_mode, short_side)
        elif asset_mode != "raw_copy":
            raise ValueError(f"未知 asset_mode: {asset_mode}")

        head, argmax, num_logits = run_one(
            interpreter, input_detail, output_detail, asset_work_path, transform
        )
        write_case_json(
            json_path,
            source_path=abs_image,
            asset_name=asset_name,
            input_head=head,
            argmax=argmax,
            num_logits=num_logits,
            model_path=model_path,
            float_abs_tol=float_abs_tol,
            asset_mode=asset_mode,
            golden_image_path=asset_work_path,
        )

        if copy_images_dir:
            os.makedirs(copy_images_dir, exist_ok=True)
            dest = os.path.join(copy_images_dir, asset_name)
            if asset_mode == "raw_copy":
                shutil.copy2(abs_image, dest)
            else:
                _write_android_asset(pil, dest, asset_mode, short_side)

        if asset_mode == "resized256_png" and asset_work_path != abs_image:
            try:
                os.remove(asset_work_path)
            except OSError:
                pass

        if copy_json_dir:
            os.makedirs(copy_json_dir, exist_ok=True)
            shutil.copy2(json_path, os.path.join(copy_json_dir, json_name))

        manifest_cases.append(
            {
                "id": os.path.splitext(asset_name)[0],
                "asset_name": asset_name,
                "json_name": json_name,
            }
        )
        print(
            f"{os.path.basename(abs_image)} -> {json_name} "
            f"argmax={argmax} head[0]={head[0]:.8f}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input_head_len": INPUT_HEAD_LEN,
        "float_abs_tol": float_abs_tol,
        "model_basename": os.path.basename(model_path),
        "model_sha256": _file_sha256(model_path),
        "cases": manifest_cases,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if copy_json_dir:
        shutil.copy2(manifest_path, os.path.join(copy_json_dir, "manifest.json"))

    print(f"已写入 {len(manifest_cases)} 条 golden -> {output_dir}")
    print(f"manifest: {manifest_path}")


def expand_image_path(path: str) -> list[str]:
    """单张图片路径，或目录下（含子目录）全部支持的图片，按路径排序。"""
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        return [abs_path]
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"找不到图片或目录: {abs_path}")

    found: list[str] = []
    for root, _dirs, files in os.walk(abs_path):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                found.append(os.path.join(root, fname))
    if not found:
        raise FileNotFoundError(f"目录内没有支持的图片: {abs_path}")
    return found


def collect_image_paths(args: argparse.Namespace) -> list[str]:
    if args.image_path:
        paths: list[str] = []
        for entry in args.image_path:
            paths.extend(expand_image_path(entry))
        return paths
    paths, _, _ = load_test_data(args.dataset, args.data_root, limit=args.limit)
    return list(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 PC TFLite golden（input 前 12 float + argmax）")
    parser.add_argument("--model-path", type=str, required=True, help=".tflite 路径或 saved_model/ 下文件名")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_GOLDEN_DIR,
        help=f"JSON 输出目录（默认 {DEFAULT_GOLDEN_DIR}）",
    )
    parser.add_argument("--config-json", type=str, default=None, help="与 val_tflite 相同的预处理 config")
    parser.add_argument(
        "--float-abs-tol",
        type=float,
        default=1e-4,
        help="写入 JSON 的 Android 浮点绝对误差阈值",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        default=None,
        help="可重复；每张图路径，或目录（递归收集 jpg/jpeg/png/bmp/webp）",
    )
    parser.add_argument("--dataset", choices=("oxford102", "flower14"), default="flower14")
    parser.add_argument("--data-root", type=str, default=ROOT)
    parser.add_argument("--limit", type=int, default=None, help="从 dataset 划分取前 N 张")
    parser.add_argument(
        "--copy-images",
        type=str,
        default=None,
        help="同步拷贝图片到 Android assets 目录（如 android_test/assets/golden_images）",
    )
    parser.add_argument(
        "--copy-json",
        type=str,
        default=None,
        help="同步拷贝 JSON + manifest 到 Android assets/golden",
    )
    parser.add_argument(
        "--asset-mode",
        choices=("resized256_png", "raw_copy"),
        default="resized256_png",
        help="resized256_png：asset 为 PC Resize(256) 的 PNG（推荐，与 GoldenParityTest 对齐）",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    model_path = resolve_model_path(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到 TFLite 模型: {model_path}")

    image_paths = collect_image_paths(args)
    if not image_paths:
        raise SystemExit("请通过 --image-path 或 --dataset + --limit 指定至少一张图")

    export_images(
        model_path=model_path,
        image_paths=image_paths,
        output_dir=os.path.abspath(args.output_dir),
        config_json=args.config_json,
        float_abs_tol=args.float_abs_tol,
        copy_images_dir=args.copy_images,
        copy_json_dir=args.copy_json,
        asset_mode=args.asset_mode,
    )


if __name__ == "__main__":
    main()
