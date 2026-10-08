# -*- coding: utf-8 -*-
"""
Oxford 102 Flowers 测试/评估脚本（ResNet）

用法示例:
    python test.py --checkpoint checkpoints/best_model.pth
    python test.py --checkpoint checkpoints/best_model.pth --limit-test 200   # 快速抽查
    python test.py --checkpoint checkpoints/best_model.pth --single-image 102flowers/jpg/image_00001.jpg

产物:
    test_results.csv        # 各类别准确率明细（默认输出到 checkpoint 所在目录）
    misclassified.csv       # 可选：误判样本路径与 true/pred 类别

用法（导出误判样本，便于按路径打开图片）:
    python test.py --checkpoint checkpoints/best_model.pth --export-misclassified checkpoints/misclassified.csv
    python test.py --checkpoint checkpoints/best_model.pth --export-misclassified out.csv --misclassified-true 51 --misclassified-pred 86
    python test.py --checkpoint checkpoints/best_model.pth --copy-misclassified-to checkpoints/review_51 --misclassified-true 51
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import data_utils
import logging_utils
from model import (
    ARCH_CHOICES,
    cli_flag_provided,
    config_path_for_checkpoint,
    describe_model,
    load_from_checkpoint,
    load_inference_config,
)

_logger = logging_utils.get_logger("test")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """在测试集上评估，返回总体 Top-1/Top-5 与各类别统计"""
    model.eval()
    total = correct1 = correct5 = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, pred = outputs.topk(5, 1, True, True)          # 取前 5
        pred1 = pred[:, 0]
        correct1 += (pred1 == targets).sum().item()
        correct5 += (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        for t, p in zip(targets.tolist(), pred1.tolist()):
            per_class_total[t] += 1
            confusion[t, p] += 1
            if t == p:
                per_class_correct[t] += 1
        total += images.size(0)
    return {
        "total": total,
        "acc1": correct1 / total,
        "acc5": correct5 / total,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "confusion": confusion,
    }


def print_confusion_for_classes(confusion, class_ids, topk=5):
    """打印指定真实类别被 Top-1 误判为哪些类（按次数降序）"""
    num_classes = confusion.shape[0]
    for cid in class_ids:
        if cid < 1 or cid > num_classes:
            _logger.warning("跳过类别 %d：超出范围 1-%d", cid, num_classes)
            continue
        true_idx = cid - 1
        row = confusion[true_idx]
        total = int(row.sum())
        if total == 0:
            _logger.info("类别 %d 测试集中无样本", cid)
            continue
        correct = int(row[true_idx])
        _logger.info("类别 %d 测试样本 %d | 正确 %d (%.1f%%)",
                     cid, total, correct, correct / total * 100)
        _logger.info("  误判去向 (Top-1):")
        wrong = [(j, int(row[j])) for j in range(num_classes) if j != true_idx and row[j] > 0]
        wrong.sort(key=lambda x: x[1], reverse=True)
        for j, cnt in wrong[:topk]:
            _logger.info("    -> 类别 %3d: %3d 次 (%.1f%%)", j + 1, cnt, cnt / total * 100)
        if not wrong:
            _logger.info("    （无误判）")


@torch.no_grad()
def collect_misclassified(model, loader, paths, device, true_filter=None, pred_filter=None):
    """逐张推理，筛出 Top-1 误判样本（paths 顺序须与 loader 一致，shuffle=False）"""
    model.eval()
    rows = []
    offset = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        vals, pred = probs.topk(5, 1, True, True)
        pred1 = pred[:, 0]
        top1_prob = vals[:, 0]
        bs = images.size(0)
        batch_paths = paths[offset:offset + bs]
        offset += bs
        for path, t, p, p_prob, top5_idx, top5_val in zip(
                batch_paths,
                targets.tolist(),
                pred1.tolist(),
                top1_prob.tolist(),
                pred.tolist(),
                vals.tolist()):
            if t == p:
                continue
            true_id = t + 1
            pred_id = p + 1
            if true_filter is not None and true_id not in true_filter:
                continue
            if pred_filter is not None and pred_id not in pred_filter:
                continue
            top5_str = ";".join(
                f"{idx + 1}:{prob * 100:.1f}%" for idx, prob in zip(top5_idx, top5_val))
            rows.append({
                "path": path,
                "filename": os.path.basename(path),
                "true_class": true_id,
                "pred_class": pred_id,
                "top1_prob": f"{p_prob * 100:.2f}%",
                "top5": top5_str,
            })
    return rows


def save_misclassified_csv(rows, csv_path):
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "filename", "true_class", "pred_class", "top1_prob", "top5"])
        w.writeheader()
        w.writerows(rows)


def copy_misclassified_images(rows, dest_dir):
    """将误判样本复制到目标文件夹，文件名含 true/pred 便于浏览"""
    os.makedirs(dest_dir, exist_ok=True)
    for r in rows:
        dst_name = f"true{r['true_class']:03d}_pred{r['pred_class']:03d}_{r['filename']}"
        shutil.copy2(r["path"], os.path.join(dest_dir, dst_name))


@torch.no_grad()
def predict_single(model, img_path, device, topk=5, transform=None):
    """对单张图片预测，返回 (PIL.Image, 概率列表, 类别列表)。

    transform 缺省时用默认评估预处理；调用方应传入由 config.json 构建的 transform，
    保证与 batch 测试预处理一致（P0-4）。
    """
    if transform is None:
        transform = data_utils.get_transform(train=False)
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    vals, inds = torch.topk(probs, k=min(topk, probs.numel()))
    return img, vals.tolist(), inds.tolist()


def main():
    parser = argparse.ArgumentParser(description="Flowers102 ResNet 测试脚本")
    parser.add_argument("--data-root", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="项目根目录（含 102flowers 子目录）")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="若 weights_only=True 加载失败，允许 weights_only=False（仅用于可信来源）")
    parser.add_argument("--arch", type=str, default=None, choices=list(ARCH_CHOICES),
                        help="网络结构；默认从 checkpoint 的 args / config.json 读取")
    parser.add_argument("--depth", type=int, default=None, choices=[18, 34, 50],
                        help="ResNet 深度；默认从 checkpoint 的 args 读取")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="类别数；默认从 checkpoint 的 args 读取")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-test", type=int, default=None, help="调试：只取前 N 张测试图")
    parser.add_argument("--single-image", type=str, default=None, help="对单张图片做预测")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="各类别准确率 CSV 输出路径（默认 checkpoint 同目录）")
    parser.add_argument("--confusion-for", type=int, nargs="+", default=None,
                        help="分析指定真实类别(1-102)的 Top-1 误判去向，如: --confusion-for 51 84 96 98")
    parser.add_argument("--confusion-topk", type=int, default=5,
                        help="每个类别展示前 K 个最常误判的目标类")
    parser.add_argument("--export-misclassified", type=str, default=None,
                        help="导出误判样本 CSV（含图片绝对路径、真实/预测类别）")
    parser.add_argument("--misclassified-true", type=int, nargs="+", default=None,
                        help="只导出真实类别为指定 id 的误判，如: --misclassified-true 51")
    parser.add_argument("--misclassified-pred", type=int, nargs="+", default=None,
                        help="只导出被预测为指定 id 的误判，如: --misclassified-pred 86")
    parser.add_argument("--copy-misclassified-to", type=str, default=None,
                        help="将误判图片复制到指定文件夹（可与 --export-misclassified 联用）")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 结构化日志：控制台 INFO + 文件 DEBUG -> checkpoint 同目录 test.log（P1-1）
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    logger = logging_utils.setup_logging(os.path.join(ckpt_dir, "test.log"))
    logger.debug("命令行参数: %s", vars(args))

    device = get_device()
    # P0-4：推理配置单一事实来源 —— checkpoint 旁的 *_config.json
    infer_cfg = load_inference_config(args.checkpoint)
    cfg_defaults = None
    if infer_cfg:
        cfg_defaults = {
            "arch": infer_cfg.get("arch"),
            "depth": infer_cfg.get("depth"),
            "num_classes": infer_cfg.get("num_classes"),
        }
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
    args.arch = model_cfg["arch"]
    args.depth = model_cfg["depth"]
    args.num_classes = model_cfg["num_classes"]
    # 单图与 batch 测试共用同一预处理（来自 config.json；缺失时回退默认 ImageNet 配置）
    eval_transform = data_utils.get_eval_transform_from_config(infer_cfg)
    if infer_cfg:
        logger.info("推理配置 %s | input=%s resize=%s",
                    config_path_for_checkpoint(args.checkpoint),
                    infer_cfg.get('input_size', 224), infer_cfg.get('resize_size', 256))
    else:
        logger.info("未找到推理配置 JSON，使用默认预处理 (Resize 256 + CenterCrop 224 + ImageNet 归一化)")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        info = (f"epoch={ckpt.get('epoch', '?')} | best_acc1="
                f"{ckpt.get('best_acc1', '?')}")
    else:
        info = "plain state_dict"
    logger.info("加载 %s (%s) | %s | classes=%d | device=%s",
                args.checkpoint, info, describe_model(args.arch, args.depth),
                args.num_classes, device)

    if args.single_image:
        img, probs, inds = predict_single(model, args.single_image, device,
                                          transform=eval_transform)
        logger.info("预测 %s", args.single_image)
        for p, c in zip(probs, inds):
            logger.info("    类别 %3d（1-102）  概率 %.2f%%", c + 1, p * 100)
        return

    paths, _ = data_utils.load_split(args.data_root, "test")
    if args.limit_test is not None:
        paths = paths[:args.limit_test]
    test_loader = data_utils.build_loader(args.data_root, "test", args.batch_size,
                                          args.num_workers, limit=args.limit_test,
                                          transform=eval_transform)
    res = evaluate(model, test_loader, device, args.num_classes)
    logger.info("测试样本 %d | Top-1 %.2f%% | Top-5 %.2f%%",
                res['total'], res['acc1'] * 100, res['acc5'] * 100)

    if args.save_csv is None:
        args.save_csv = os.path.join(os.path.dirname(args.checkpoint), "test_results.csv")
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id(1-102)", "samples", "correct", "accuracy"])
        for c in range(args.num_classes):
            total_c = int(res["per_class_total"][c])
            if total_c == 0:
                continue
            corr = int(res["per_class_correct"][c])
            w.writerow([c + 1, total_c, corr, f"{corr / total_c * 100:.2f}%"])
    logger.info("各类别准确率明细 -> %s", args.save_csv)
    accs = res["per_class_correct"] / np.maximum(res["per_class_total"], 1)
    has_data = res["per_class_total"] > 0
    sorted_idx = np.argsort(accs)
    order = sorted_idx[has_data[sorted_idx]]   # 只看有测试样本的类别
    logger.info("最差 10 类 class_id: accuracy")
    for c in order[:10]:
        logger.info("    类别 %3d: %.1f%% (%d/%d)",
                    c + 1, accs[c] * 100,
                    int(res['per_class_correct'][c]), int(res['per_class_total'][c]))

    if args.confusion_for:
        print_confusion_for_classes(
            res["confusion"], args.confusion_for, topk=args.confusion_topk)

    if args.export_misclassified or args.copy_misclassified_to:
        true_f = set(args.misclassified_true) if args.misclassified_true else None
        pred_f = set(args.misclassified_pred) if args.misclassified_pred else None
        wrong_rows = collect_misclassified(
            model, test_loader, paths, device, true_filter=true_f, pred_filter=pred_f)
        if args.export_misclassified:
            save_misclassified_csv(wrong_rows, args.export_misclassified)
            logger.info("误判样本 %d 条 -> %s", len(wrong_rows), args.export_misclassified)
        if args.copy_misclassified_to:
            copy_misclassified_images(wrong_rows, args.copy_misclassified_to)
            logger.info("误判图片 %d 张 -> %s", len(wrong_rows),
                        os.path.abspath(args.copy_misclassified_to))
        if wrong_rows[:3]:
            logger.info("前 3 条误判示例:")
            for r in wrong_rows[:3]:
                logger.info("    %s  true=%d pred=%d  %s",
                            r['filename'], r['true_class'], r['pred_class'], r['path'])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 异常栈写入 test.log（P1-1）
        logging_utils.get_logger().exception("测试失败（未捕获异常）")
        raise
