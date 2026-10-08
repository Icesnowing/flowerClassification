# -*- coding: utf-8 -*-
"""
在测试集上评估 TFLite 模型（Oxford 102 或 flower14）。

用法:
  # Oxford 102 官方 test 集（6149 张，data_root 含 102flowers/）
  python val_tflite.py --dataset oxford102 --model-path saved_model/best_mobilenet_v3_large_float32.tflite

  # flower14 官方 val/ 作测试集（98 张）
  python val_tflite.py --dataset flower14 --data-root flower14 --model-path saved_model/ft14_float32.tflite

  # 单张图
  python val_tflite.py --dataset flower14 --model-path saved_model/ft14.tflite --image-path flower14/val/rose/xxx.jpg
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import data_utils

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVED_MODEL = os.path.join(ROOT, "saved_model")


def resolve_model_path(path):
    if os.path.isabs(path) or os.path.isfile(path):
        return os.path.abspath(path)
    candidate = os.path.join(DEFAULT_SAVED_MODEL, path)
    if os.path.isfile(candidate):
        return candidate
    return os.path.abspath(path)


def load_class_names_txt(path):
    if not path or not os.path.isfile(path):
        return None
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            idx_str, name = line.split("\t", 1)
            names.append((int(idx_str), name))
    if not names:
        return None
    names.sort(key=lambda x: x[0])
    return [n for _, n in names]


def _load_imagefolder_split(root_dir):
    """与 torchvision ImageFolder 相同规则：按类名排序，0-based 标签。"""
    class_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    if not class_names:
        raise FileNotFoundError(f"未找到类别子目录: {root_dir}")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, targets = [], []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            paths.append(os.path.join(class_dir, fname))
            targets.append(class_to_idx[class_name])
    return paths, targets, class_names


def load_test_data(dataset, data_root, limit=None):
    """返回 (paths, targets) ；targets 为 0-based 类别索引。"""
    if dataset == "oxford102":
        paths, targets = data_utils.load_split(data_root, "test")
        class_names = None
    elif dataset == "flower14":
        test_root = os.path.join(data_root, "val")
        if not os.path.isdir(test_root):
            raise FileNotFoundError(f"flower14 测试集目录不存在: {test_root}")
        paths, targets, class_names = _load_imagefolder_split(test_root)
    else:
        raise ValueError(f"未知 dataset: {dataset}")

    if limit is not None:
        paths = paths[:limit]
        targets = targets[:limit]

    if dataset == "flower14":
        return paths, np.asarray(targets, dtype=np.int64), class_names
    return paths, np.asarray(targets, dtype=np.int64), None


def build_transform(config_json=None):
    if config_json and os.path.isfile(config_json):
        with open(config_json, encoding="utf-8") as f:
            cfg = json.load(f)
        return data_utils.get_eval_transform_from_config(cfg)
    return data_utils.get_transform(train=False)


def prepare_input_nhwc(pil_image, transform, input_detail):
    x = transform(pil_image).unsqueeze(0).numpy().transpose(0, 2, 3, 1).astype(np.float32) # NCHW --> NHWC
    if input_detail["dtype"] == np.int8:
        scale, zero_point = input_detail["quantization"]
        if scale == 0:
            raise ValueError("INT8 输入缺少有效的 quantization scale")
        x = np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)
    return x


def run_inference(interpreter, input_detail, output_detail, x_nhwc):
    input_idx = input_detail["index"]
    output_idx = output_detail["index"]
    interpreter.set_tensor(input_idx, x_nhwc)
    interpreter.invoke()
    out = interpreter.get_tensor(output_idx)[0]
    if output_detail["dtype"] == np.int8:
        scale, zero_point = output_detail["quantization"]
        if scale != 0:
            out = (out.astype(np.float32) - zero_point) * scale
    return int(np.argmax(out))


def evaluate_tflite(model_path, test_paths, targets, transform, limit=None):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    num_out = int(np.prod(output_detail["shape"]))
    if num_out < int(targets.max()) + 1 if len(targets) else 0:
        raise ValueError(
            f"模型输出维度 {num_out} 小于数据类别数；请确认 --dataset 与 TFLite 模型一致")

    indices = range(len(test_paths)) if limit is None else range(min(limit, len(test_paths)))
    correct = 0
    for i in indices:
        img = Image.open(test_paths[i]).convert("RGB")
        x = prepare_input_nhwc(img, transform, input_detail)
        pred = run_inference(interpreter, input_detail, output_detail, x)
        if pred == int(targets[i]):
            correct += 1
    return correct / len(indices)


def single_image_predict(model_path, image_path, transform, class_names=None):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    img = Image.open(image_path).convert("RGB")
    x = prepare_input_nhwc(img, transform, input_detail)
    pred = run_inference(interpreter, input_detail, output_detail, x)

    out = interpreter.get_tensor(output_detail["index"])[0]
    if output_detail["dtype"] == np.int8:
        scale, zero_point = output_detail["quantization"]
        if scale != 0:
            out = (out.astype(np.float32) - zero_point) * scale
    probs = softmax(out)
    confidence = float(np.max(probs))

    label = class_names[pred] if class_names and pred < len(class_names) else str(pred)
    print(f"图片: {image_path}")
    print(f"预测类别索引: {pred} ({label})")
    print(f"置信度: {confidence:.4f}")
    print(f"模型输出维度: {len(out)}")


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def main():
    parser = argparse.ArgumentParser(description="TFLite 模型测试集评估（Oxford102 / flower14）")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("oxford102", "flower14"),
        default="oxford102",
        help="oxford102: 102flowers 官方 test；flower14: data-root/val",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=ROOT,
        help="oxford102 时为项目根（含 102flowers）；flower14 时为 flower14 目录",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help=".tflite 路径，或相对 saved_model/ 的文件名",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="推理预处理 JSON（与 export 产物一致；缺省为 ImageNet 默认）",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="flower14 类名文件（默认与 model 同目录的 class_names.txt）",
    )
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 张（调试）")
    parser.add_argument("--image-path", type=str, default=None, help="单张图片推理")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    model_path = resolve_model_path(args.model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到 TFLite 模型: {model_path}")

    transform = build_transform(args.config_json)
    class_names = None
    if args.dataset == "flower14":
        names_path = args.class_names
        if not names_path:
            names_path = os.path.join(os.path.dirname(model_path), "class_names.txt")
        class_names = load_class_names_txt(names_path)

    if args.image_path:
        single_image_predict(model_path, args.image_path, transform, class_names=class_names)
        return

    paths, targets, folder_class_names = load_test_data(args.dataset, args.data_root, limit=args.limit)
    if class_names is None and folder_class_names is not None:
        class_names = folder_class_names

    print(f"数据集: {args.dataset} | 样本数: {len(paths)} | 模型: {model_path}")
    acc = evaluate_tflite(model_path, paths, targets, transform, limit=None)
    print(f"Top-1 准确率: {acc * 100:.2f}% ({acc:.4f})")
    if class_names:
        print(f"类别数: {len(class_names)} ({', '.join(class_names[:5])} ...)")


if __name__ == "__main__":
    main()
