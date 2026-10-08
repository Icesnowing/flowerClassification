# -*- coding: utf-8 -*-
"""
模型导出（P2-2）：将训练 checkpoint 导出为 ONNX，并用 onnxruntime 校验数值一致性。

用法示例:
    python export.py --checkpoint checkpoints/best_model.pth
    python export.py --checkpoint checkpoints/best_model.pth --output deploy/flower.onnx --opset 18
    python export.py --checkpoint checkpoints/best_model.pth --image flower_01.jpg   # 真实图片双后端对比
    python export.py --checkpoint checkpoints/best_model.pth --no-verify             # 只导出不校验

产物（--output 缺省时与 checkpoint 同目录、同名换后缀）:
    best_model.onnx     # ONNX 模型（动态 batch，opset 默认 18）
    export.log          # 导出与校验日志（checkpoint 同目录）

ONNX 模型约定（接入 C++ / ONNX Runtime / Triton 等推理栈时必读）:
    输入:  name="input", shape=(N, 3, 224, 224), float32, NCHW, RGB 通道序
          预处理 = Resize(resize_size) -> CenterCrop(input_size) -> 归一化到 [0,1]
                   -> Normalize(mean, std)，参数以 checkpoint 旁 *_config.json 为准
                   （同时写入 ONNX metadata_props，部署侧可直接读取）
    输出:  name="logits", shape=(N, num_classes), float32，未过 softmax；
          类别 id = argmax(logits) + 1（1-102）

验收标准（PRODUCTION_ROADMAP P2-2）:
    onnxruntime 与 PyTorch 单张图输出 max abs diff < 1e-4（--tol 可调）。
"""
import argparse
import inspect
import os
import sys

import numpy as np
import torch

import data_utils
import logging_utils
from model import (
    ARCH_CHOICES,
    cli_flag_provided,
    describe_model,
    load_from_checkpoint,
    load_inference_config,
)

_logger = logging_utils.get_logger("export")


def _dynamo_export_by_default():
    """torch.onnx.export 是否默认走 dynamo 导出器（torch 2.9+ 起 dynamo=True 为默认）。

    老版本 torch（<=2.8）无 dynamo 参数或默认 False，走 TorchScript 导出器，
    两者动态 batch 的传参方式不同（dynamic_shapes vs dynamic_axes）。
    """
    try:
        param = inspect.signature(torch.onnx.export).parameters.get("dynamo")
    except (TypeError, ValueError):
        return False
    return bool(param is not None and param.default)


def export_onnx(model, output_path, *, input_size, opset, dynamic_batch=True,
                use_torchscript=True):
    """导出 ONNX 模型（CPU、eval 模式、固定 1 张 dummy 输入，batch 维动态）。

    Args:
        model: 已加载权重的 ResNet/MobileNetV3（任意 device，导出前会转到 CPU，导出后恢复原 device）
        output_path: 输出 .onnx 路径
        input_size: 输入边长（方形，来自推理 config.json）
        opset: ONNX opset 版本（与部署目标对齐，默认 18 兼容主流推理栈）
        dynamic_batch: True 时 batch 维动态（部署侧可任意 batch 推理）
        use_torchscript: True 时强制用 TorchScript 导出器（dynamo=False），
                         生成的图 shape 干净、与 onnxruntime 量化（QDQ）兼容；
                         False 时用当前 torch 默认导出器（dynamo）。

    Returns:
        dummy input（numpy, shape=(1,3,H,W)，固定 seed，供数值校验复用）
    """
    device = next(model.parameters()).device
    model = model.to("cpu").eval()

    # 固定 seed 的 dummy input：导出与后续 PyTorch/ORT 对比共用同一份数据
    rng = np.random.default_rng(42)
    dummy = rng.standard_normal((1, 3, input_size, input_size), dtype=np.float32)
    dummy_t = torch.from_numpy(dummy)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,   # 常量折叠：BN 等参数预先合并，减小图体积
        export_params=True,
    )
    if use_torchscript:
        # TorchScript 导出器：图 shape 干净、与 onnxruntime 静态量化兼容
        export_kwargs["dynamo"] = False
        if dynamic_batch:
            export_kwargs["dynamic_axes"] = {"input": {0: "batch"}, "logits": {0: "batch"}}
    else:
        # 当前 torch 默认（dynamo）导出器
        if dynamic_batch:
            export_kwargs["dynamic_shapes"] = {"x": {0: torch.export.Dim("batch")}}
    with torch.no_grad():
        torch.onnx.export(model, (dummy_t,), output_path, **export_kwargs)
    model.to(device)   # 恢复原 device，避免影响调用方后续使用
    _logger.info("ONNX 已导出 -> %s | opset=%d | input=(N,3,%d,%d) | dynamic_batch=%s | torchscript=%s",
                 output_path, opset, input_size, input_size, dynamic_batch, use_torchscript)
    return dummy


def embed_metadata(onnx_path, *, depth, num_classes, infer_cfg, arch="resnet"):
    """把推理配置写入 ONNX metadata_props（部署侧单一事实来源，无需再读 config.json）。"""
    import onnx

    m = onnx.load(onnx_path)
    infer_cfg = infer_cfg or {}
    meta = {
        "arch": arch,
        "depth": depth,
        "num_classes": num_classes,
        "input_size": infer_cfg.get("input_size", 224),
        "resize_size": infer_cfg.get("resize_size", 256),
        "normalize_mean": ",".join(str(v) for v in infer_cfg.get("normalize_mean", data_utils.IMAGENET_MEAN)),
        "normalize_std": ",".join(str(v) for v in infer_cfg.get("normalize_std", data_utils.IMAGENET_STD)),
        "layout": "NCHW",
        "color_order": "RGB",
        "label_offset": "class_id = argmax(logits) + 1  (1-102)",
    }
    for k, v in meta.items():
        entry = m.metadata_props.add()
        entry.key = str(k)
        entry.value = str(v)
    onnx.save(m, onnx_path)
    _logger.info("推理配置已写入 ONNX metadata_props: %s",
                 {k: meta[k] for k in ("arch", "depth", "num_classes", "input_size", "resize_size")})


@torch.no_grad()
def _torch_logits(model, x_np):
    model.eval()
    return model(torch.from_numpy(x_np)).numpy()


def _ort_logits(session, x_np):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: x_np})[0]


def _compare(tag, torch_out, ort_out, tol):
    """对比两个后端输出，返回是否通过；日志打印 max/mean abs diff 与 Top-1 是否一致"""
    diff = np.abs(torch_out - ort_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    top1_match = bool((torch_out.argmax(axis=1) == ort_out.argmax(axis=1)).all())
    ok = max_diff < tol and top1_match
    _logger.info("[%s] max abs diff=%.3e | mean abs diff=%.3e | Top-1 一致=%s | 阈值=%.0e -> %s",
                 tag, max_diff, mean_diff, top1_match, tol, "通过" if ok else "未通过")
    return ok


def verify_onnx(model, onnx_path, dummy, *, tol, dynamic_batch=True,
                image_path=None, transform=None):
    """onnxruntime 校验：固定 dummy（batch 1；动态 batch 时再跑 batch 4）+ 可选真实图片。

    Returns:
        True 表示全部校验通过
    """
    try:
        import onnxruntime as ort
    except ImportError:
        _logger.warning("未安装 onnxruntime，跳过数值校验（pip install onnxruntime）")
        return True

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_shape = sess.get_inputs()[0].shape
    out_shape = sess.get_outputs()[0].shape
    _logger.info("onnxruntime 加载成功 | 输入 %s %s | 输出 %s %s",
                 sess.get_inputs()[0].name, in_shape, sess.get_outputs()[0].name, out_shape)

    all_ok = True
    # 1) 固定 dummy，batch=1（验收标准：单张图 max diff < 1e-4 量级）
    all_ok &= _compare("dummy batch=1", _torch_logits(model, dummy),
                       _ort_logits(sess, dummy), tol)
    # 2) batch=4 验证动态 batch 维可用（静态 batch 导出跳过）
    if dynamic_batch:
        rng = np.random.default_rng(7)
        dummy4 = rng.standard_normal((4,) + dummy.shape[1:], dtype=np.float32)
        all_ok &= _compare("dummy batch=4", _torch_logits(model, dummy4),
                           _ort_logits(sess, dummy4), tol)

    # 3) 可选：真实图片走完整预处理，双后端 Top-5 对比
    if image_path:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).numpy() if transform is not None else None
        if x is None:
            x = data_utils.get_transform(train=False)(img).unsqueeze(0).numpy()
        t_out, o_out = _torch_logits(model, x), _ort_logits(sess, x)
        all_ok &= _compare(f"image {os.path.basename(image_path)}", t_out, o_out, tol)
        for name, logits in (("PyTorch", t_out), ("ORT", o_out)):
            top5 = np.argsort(logits[0])[::-1][:5]
            _logger.info("  %s Top-5: %s", name,
                         ", ".join(f"类{c + 1}({logits[0][c]:.2f})" for c in top5))
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Flowers102 ResNet -> ONNX 导出（P2-2）")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径（.pth）")
    parser.add_argument("--output", type=str, default=None,
                        help="ONNX 输出路径；默认与 checkpoint 同目录同名（best_model.pth -> best_model.onnx）")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset 版本（与部署目标对齐，默认 18；torch 2.9+ dynamo 导出器"
                             "原生支持 >=18，请求更低版本时靠版本转换、失败则保留原生版本）")
    parser.add_argument("--input-size", type=int, default=None,
                        help="输入边长；默认读 checkpoint 旁 *_config.json 的 input_size（224）")
    parser.add_argument("--static-batch", action="store_true",
                        help="固定 batch=1 导出（默认动态 batch，部署侧可任意 batch）")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="数值校验阈值：onnxruntime 与 PyTorch 输出 max abs diff（默认 1e-4）")
    parser.add_argument("--no-verify", action="store_true", help="只导出，不做 onnxruntime 数值校验")
    parser.add_argument("--no-torchscript", action="store_true",
                        help="使用 torch 默认（dynamo）导出器；默认用 TorchScript 导出器（dynamo=False，"
                             "图 shape 干净、与 onnxruntime 量化兼容）")
    parser.add_argument("--image", type=str, default=None,
                        help="可选：真实图片路径，导出后走完整预处理做双后端 Top-5 对比")
    parser.add_argument("--arch", type=str, default=None, choices=list(ARCH_CHOICES),
                        help="网络结构；默认从 checkpoint 的 args / config.json 读取")
    parser.add_argument("--depth", type=int, default=None, choices=[18, 34, 50],
                        help="ResNet 深度；默认从 checkpoint 的 args / config.json 读取")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="类别数；默认从 checkpoint 的 args / config.json 读取")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="若 weights_only=True 加载失败，允许 weights_only=False（仅用于可信来源）")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    logger = logging_utils.setup_logging(os.path.join(ckpt_dir, "export.log"))
    logger.debug("命令行参数: %s", vars(args))

    if args.output is None:
        args.output = os.path.splitext(args.checkpoint)[0] + ".onnx"

    # 与 test.py 同一套加载逻辑：checkpoint args > CLI > config.json > 内置默认（P0-4）
    infer_cfg = load_inference_config(args.checkpoint)
    cfg_defaults = None
    if infer_cfg:
        cfg_defaults = {
            "arch": infer_cfg.get("arch"),
            "depth": infer_cfg.get("depth"),
            "num_classes": infer_cfg.get("num_classes"),
        }
    # 导出固定走 CPU：ONNX 与设备无关，且避免 GPU 算子差异，保证可复现
    device = torch.device("cpu")
    model, ckpt, model_cfg = load_from_checkpoint(
        args.checkpoint,
        device,
        arch=args.arch,
        depth=args.depth,
        num_classes=args.num_classes,
        arch_from_cli=cli_flag_provided("--arch"),
        depth_from_cli=cli_flag_provided("--depth"),
        num_classes_from_cli=cli_flag_provided("--num-classes"),
        trust_checkpoint=args.trust_checkpoint,
        defaults=cfg_defaults,
    )
    input_size = args.input_size or (infer_cfg or {}).get("input_size", 224)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        logger.info("加载 %s (epoch=%s | best_acc1=%s) | %s | classes=%d | input=%d",
                    args.checkpoint, ckpt.get("epoch", "?"), ckpt.get("best_acc1", "?"),
                    describe_model(model_cfg["arch"], model_cfg["depth"]),
                    model_cfg["num_classes"], input_size)
    else:
        logger.info("加载 %s (plain state_dict) | %s | classes=%d | input=%d",
                    args.checkpoint, describe_model(model_cfg["arch"], model_cfg["depth"]),
                    model_cfg["num_classes"], input_size)

    dummy = export_onnx(model, args.output, input_size=input_size,
                        opset=args.opset, dynamic_batch=not args.static_batch,
                        use_torchscript=not args.no_torchscript)
    embed_metadata(args.output, depth=model_cfg["depth"],
                   num_classes=model_cfg["num_classes"], infer_cfg=infer_cfg,
                   arch=model_cfg["arch"])
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    logger.info("模型大小 %.1f MB", size_mb)

    if args.no_verify:
        logger.info("--no-verify：跳过 onnxruntime 数值校验")
        return

    transform = data_utils.get_eval_transform_from_config(infer_cfg)
    ok = verify_onnx(model, args.output, dummy, tol=args.tol,
                     dynamic_batch=not args.static_batch,
                     image_path=args.image, transform=transform)
    if not ok:
        logger.error("数值校验未通过（max abs diff 超过 %.0e 或 Top-1 不一致），"
                     "请检查 opset 与推理栈兼容性", args.tol)
        sys.exit(1)
    logger.info("导出完成并通过校验 -> %s", os.path.abspath(args.output))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 异常栈写入 export.log（P1-1）
        logging_utils.get_logger().exception("导出失败（未捕获异常）")
        raise
