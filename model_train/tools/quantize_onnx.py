# -*- coding: utf-8 -*-
"""
Flowers102 MobileNetV3-Large → ONNX INT8 静态量化脚本

作用：
    读取已导出的 FP32 ONNX（如 best_mobilenet_v3_large.onnx），用校准集做
    静态量化，输出 INT8 模型（QDQ 格式），并做量化前后一致性检查。

用法（在项目根目录运行）:
    # 最简（自动读 checkpoints/best_mobilenet_v3_large.onnx，Entropy，per-channel）
    python tools/quantize_onnx.py

    # 指定输入输出与校准数量
    python tools/quantize_onnx.py \
        --input checkpoints/best_mobilenet_v3_large.onnx \
        --output checkpoints/flower_mobilenet_int8.onnx \
        --calib-count 150

    # 调试：只取 4 张图快速验证链路可跑通
    python tools/quantize_onnx.py --calib-count 4 --output checkpoints/_smoke_int8.onnx

产物：
    flower_mobilenet_int8.onnx  # INT8 量化模型（目标 <=5MB）

关键点（与方案 3.3 对应）：
    1. 校准集从训练/验证集抽样，**不要用测试集**（避免数据泄漏）。
    2. 预处理复用 data_utils 的 eval transform，与训练侧严格一致。
    3. 采用静态量化（权重 + 激活），推荐 Entropy + per_channel。
    4. 量化后做 FP32 vs INT8 的 Top-1 一致性检查。
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 让脚本能 import 项目根目录下的 data_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_utils  # noqa: E402

from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_input_output(args):
    """确定输入/输出 ONNX 路径，缺省时用 checkpoints 下 MobileNetV3 模型。"""
    input_path = args.input or os.path.join(
        os.getcwd(), "checkpoints", "best_mobilenet_v3_large.onnx"
    )
    output_path = args.output or os.path.join(
        os.getcwd(), "checkpoints", "flower_mobilenet_int8.onnx"
    )
    if not os.path.exists(input_path):
        sys.exit(f"[错误] 输入模型不存在: {input_path}")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return input_path, output_path


def load_config(config_path):
    """读取 checkpoint 旁的 config.json，返回预处理参数；缺失时回退 ImageNet 默认。"""
    cfg = {}
    if config_path and os.path.exists(config_path):
        import json

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    return {
        "mean": cfg.get("normalize_mean", IMAGENET_MEAN),
        "std": cfg.get("normalize_std", IMAGENET_STD),
        "input_size": cfg.get("input_size", 224),
        "resize_size": cfg.get("resize_size", 256),
    }


def build_calibration_samples(data_root, split, transform, calib_count, batch_size, seed):
    """从训练/验证集抽样，构建校准数据（numpy 列表，NCHW float32）。"""
    paths, _ = data_utils.load_split(data_root, split)
    if calib_count and calib_count < len(paths):
        rng = np.random.default_rng(seed)
        paths = [paths[i] for i in rng.choice(len(paths), size=calib_count, replace=False)]
    elif calib_count:
        paths = paths[:calib_count]
    else:
        paths = paths[:200]  # 未指定时默认最多 200 张

    samples = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensor = transform(img)          # (3, H, W) float32，已 Normalize
        samples.append(tensor.numpy()[np.newaxis])  # (1,3,224,224)
        if len(samples) >= calib_count:
            break
    return samples


class FlowerCalibReader(CalibrationDataReader):
    """onnxruntime 要求的校准数据 reader：迭代返回 {input_name: ndarray}。

    注意：新版 onnxruntime 的 CalibrationDataReader 抽象方法是 get_next()，
    quantize_static 会反复调用它直到返回 None；batch 内多张图拼成 (B,C,H,W)。
    """

    def __init__(self, samples, input_name="input", batch_size=1):
        self.iterator = iter(samples)
        self.input_name = input_name
        self.batch_size = batch_size

    def get_next(self):
        batch = []
        for _ in range(self.batch_size):
            s = next(self.iterator, None)
            if s is None:
                break
            batch.append(s)
        if not batch:
            return None
        data = np.concatenate(batch, axis=0)   # (B,3,224,224)
        return {self.input_name: data}


def collect_sensitive_nodes(model_path):
    """找出对 INT8 量化敏感的算子节点名，用于保持 FP32。

    MobileNetV3-Large 的 HardSwish / HardSigmoid 激活在静态量化下会显著掉点，
    把它们的算子节点排除出量化（nodes_to_exclude）可保住精度。
    """
    try:
        import onnx

        model = onnx.load(model_path)
        sensitive_ops = {"HardSwish", "HardSigmoid"}
        names = [n.name for n in model.graph.node if n.op_type in sensitive_ops]
        print(f"[信息] 识别到 {len(names)} 个敏感算子保持 FP32: "
              f"{'HardSwish/HardSigmoid' if names else '无'}")
        return names
    except Exception as e:
        print(f"[警告] 读取算子失败，无法排除敏感节点: {e}")
        return []


def check_consistency(input_path, output_path, samples, input_size, tolerance=0.02):
    """量化前后 Top-1 一致性抽查：对校准样本对比 FP32 与 INT8 的 argmax。"""
    try:
        sess_fp32 = ort.InferenceSession(input_path, providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[警告] 一致性检查加载模型失败，跳过: {e}")
        return None

    in_name = sess_fp32.get_inputs()[0].name
    total, same = 0, 0
    for s in samples:
        o1 = sess_fp32.run(None, {in_name: s})[0].argmax(axis=1)
        o2 = sess_int8.run(None, {in_name: s})[0].argmax(axis=1)
        total += len(o1)
        same += int((o1 == o2).sum())
    agree = same / total if total else 0.0
    print(f"[一致性] FP32 vs INT8 抽查 {same}/{total} 一致 ({agree*100:.1f}%)")
    if agree < (1 - tolerance):
        print(f"[警告] 一致性低于 {tolerance*100:.0f}%，请检查校准集预处理或校准方法。")
    return agree


def main():
    parser = argparse.ArgumentParser(description="Flowers102 ONNX INT8 静态量化")
    parser.add_argument("--input", type=str, default=None, help="输入 FP32 ONNX 路径")
    parser.add_argument("--output", type=str, default=None, help="输出 INT8 ONNX 路径")
    parser.add_argument("--data-root", type=str, default=None,
                        help="项目根目录（含 102flowers/）；默认取脚本目录上级")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="校准集来源（默认 val；勿用 test 避免数据泄漏）")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json 路径（默认 checkpoints/best_mobilenet_v3_large_config.json）")
    parser.add_argument("--calib-count", type=int, default=150,
                        help="校准图片数量（默认 150；建议 100-200）")
    parser.add_argument("--method", type=str, default="entropy",
                        choices=["entropy", "minmax"], help="校准方法（默认 entropy）")
    parser.add_argument("--per-channel", action="store_true", default=True,
                        help="逐通道量化（CNN 推荐，默认开）")
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--batch-size", type=int, default=1, help="校准分批大小")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--exclude-sensitive", action="store_true",
                        help="把 HardSwish/HardSigmoid 节点排除出量化、保持 FP32。"
                             "注意：实验表明对 MobileNetV3 通常反而掉点，默认不排除；"
                             "仅当全量化精度异常时尝试打开对比。")
    args = parser.parse_args()

    input_path, output_path = resolve_input_output(args)
    data_root = args.data_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config or os.path.join(
        data_root, "checkpoints", "best_mobilenet_v3_large_config.json"
    )

    # 1) 构建与训练一致的预处理
    cfg = load_config(config_path)
    transform = data_utils.get_transform(
        train=False,
        mean=cfg["mean"],
        std=cfg["std"],
        input_size=cfg["input_size"],
        resize_size=cfg["resize_size"],
    )
    print(f"[信息] 预处理: Resize({cfg['resize_size']})->CenterCrop({cfg['input_size']})->Normalize "
          f"mean={cfg['mean']} std={cfg['std']}")

    # 2) 抽样校准集
    samples = build_calibration_samples(
        data_root, args.split, transform, args.calib_count, args.batch_size, args.seed
    )
    if not samples:
        sys.exit("[错误] 校准集为空，请检查 --data-root 与 --split")
    print(f"[信息] 校准集: {len(samples)} 张 (来自 {args.split})，分批大小 {args.batch_size}")

    reader = FlowerCalibReader(samples, batch_size=args.batch_size)
    method = CalibrationMethod.Entropy if args.method == "entropy" else CalibrationMethod.MinMax

    # 3) 可选：把 HardSwish/HardSigmoid 排除出量化（默认不排除——实验表明对
    #    MobileNetV3 通常反而掉点，仅当全量化精度异常时用 --exclude-sensitive 对比）
    nodes_to_exclude = collect_sensitive_nodes(input_path) if args.exclude_sensitive else []

    # 4) 执行静态量化
    print(f"[量化] 开始 ... 方法={args.method} per_channel={args.per_channel} "
          f"排除敏感算子={len(nodes_to_exclude)}")
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        nodes_to_exclude=nodes_to_exclude,
    )

    # 5) 结果报告
    in_mb = os.path.getsize(input_path) / 1024 / 1024
    out_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[体积] FP32 {in_mb:.1f}MB -> INT8 {out_mb:.1f}MB (压缩 {in_mb/out_mb:.1f}x)")

    # 6) 一致性检查
    check_consistency(input_path, output_path, samples[: min(len(samples), 20)], cfg["input_size"])
    print(f"[完成] 量化模型已保存 -> {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
