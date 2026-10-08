# -*- coding: utf-8 -*-
"""
flower14 文件夹测试集评估（官方 val/ 或任意 ImageFolder 布局）

与 train_folder.py 配套：训练时从 train 划验证集选模型，本脚本在 held-out 测试集上评估。

用法示例（项目根目录）:
  python test_folder.py --checkpoint checkpoints_ft14/best_mobilenet_v3_large.pth --data-root flower14
  python test_folder.py --checkpoint checkpoints_ft14/best_resnet18.pth --data-root flower14 --single-image flower14/val/rose/xxx.jpg
  python test_folder.py --checkpoint checkpoints_ft14/best_resnet18.pth --test-root flower14/val --save-csv checkpoints_ft14/test_results.csv

产物（默认写入 checkpoint 同目录）:
  test_results.csv      # 各类别准确率
  test_results.json     # 总体 Top-1/Top-5 与路径信息
  test.log
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

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
from test import collect_misclassified, copy_misclassified_images, predict_single, save_misclassified_csv


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names(checkpoint_path, fallback):
    """读取 train_folder 保存的 class_names.txt；缺失时用 ImageFolder 类名。"""
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    path = os.path.join(ckpt_dir, "class_names.txt")
    if os.path.isfile(path):
        names = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                idx_str, name = line.split("\t", 1)
                names.append((int(idx_str), name))
        if names:
            names.sort(key=lambda x: x[0])
            return [n for _, n in names]
    return list(fallback)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, use_amp=False):
    model.eval()
    total = correct1 = correct5 = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        _, pred = outputs.topk(5, 1, True, True)
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
        "acc1": correct1 / total if total else 0.0,
        "acc5": correct5 / total if total else 0.0,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "confusion": confusion,
    }


def image_paths_in_order(dataset):
    """Subset 或 ImageFolder 的样本路径，顺序与 DataLoader(shuffle=False) 一致。"""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
        return [base.samples[i][0] for i in indices]
    return [p for p, _ in dataset.samples]


def main():
    parser = argparse.ArgumentParser(description="flower14 文件夹测试集评估")
    parser.add_argument("--data-root", type=str, default=None,
                        help="flower14 根目录（含 val/）；与 --test-root 二选一")
    parser.add_argument("--test-root", type=str, default=None,
                        help="测试集目录（ImageFolder，默认 <data-root>/val）")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--arch", type=str, default=None, choices=list(ARCH_CHOICES))
    parser.add_argument("--depth", type=int, default=None, choices=[18, 34, 50])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--single-image", type=str, default=None)
    parser.add_argument("--save-csv", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--export-misclassified", type=str, default=None)
    parser.add_argument("--copy-misclassified-to", type=str, default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    logger = logging_utils.setup_logging(os.path.join(ckpt_dir, "test.log"))
    logger.debug("命令行参数: %s", vars(args))

    device = get_device()
    use_amp = args.amp if args.amp is not None else device.type == "cuda"
    if use_amp and device.type != "cuda":
        use_amp = False

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
    num_classes = model_cfg["num_classes"]
    eval_transform = data_utils.get_eval_transform_from_config(infer_cfg)
    if infer_cfg:
        logger.info("推理配置 %s", config_path_for_checkpoint(args.checkpoint))
    logger.info("加载 %s | %s | classes=%d | device=%s",
                args.checkpoint, describe_model(model_cfg["arch"], model_cfg["depth"]),
                num_classes, device)

    if args.single_image:
        img_path = args.single_image
        _, probs, inds = predict_single(model, img_path, device, transform=eval_transform)
        names_path = os.path.join(ckpt_dir, "class_names.txt")
        id_to_name = {}
        if os.path.isfile(names_path):
            with open(names_path, encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        i, n = line.strip().split("\t", 1)
                        id_to_name[int(i)] = n
        logger.info("预测 %s", img_path)
        for p, c in zip(probs, inds):
            label = id_to_name.get(c, f"class_{c}")
            logger.info("    [%2d] %-20s  概率 %.2f%%", c, label, p * 100)
        return

    if args.test_root:
        test_root = args.test_root
    elif args.data_root:
        test_root = os.path.join(args.data_root, "val")
    else:
        raise ValueError("请指定 --data-root flower14 或 --test-root <测试集目录>")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"测试集目录不存在: {test_root}")

    test_ds = datasets.ImageFolder(test_root, transform=eval_transform)
    class_names = load_class_names(args.checkpoint, test_ds.classes)
    if len(class_names) != num_classes:
        logger.warning("class_names 数量 (%d) 与模型 num_classes (%d) 不一致，CSV 仍按索引输出",
                        len(class_names), num_classes)
    if args.limit_test is not None:
        test_ds = Subset(test_ds, range(min(args.limit_test, len(test_ds))))

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    paths = image_paths_in_order(test_ds)
    res = evaluate(model, test_loader, device, num_classes, use_amp=use_amp)
    logger.info("测试集 %s | 样本 %d | Top-1 %.2f%% | Top-5 %.2f%%",
                test_root, res["total"], res["acc1"] * 100, res["acc5"] * 100)

    save_csv = args.save_csv or os.path.join(ckpt_dir, "test_results.csv")
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_index", "class_name", "samples", "correct", "accuracy"])
        for c in range(num_classes):
            total_c = int(res["per_class_total"][c])
            if total_c == 0:
                continue
            corr = int(res["per_class_correct"][c])
            name = class_names[c] if c < len(class_names) else ""
            w.writerow([c, name, total_c, corr, f"{corr / total_c * 100:.2f}%"])
    logger.info("各类别准确率 -> %s", save_csv)

    save_json = args.save_json or os.path.join(ckpt_dir, "test_results.json")
    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "test_root": os.path.abspath(test_root),
        "num_samples": int(res["total"]),
        "acc1": res["acc1"],
        "acc5": res["acc5"],
        "per_class_csv": os.path.abspath(save_csv),
    }
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("总体指标 -> %s", save_json)

    if args.export_misclassified or args.copy_misclassified_to:
        wrong_rows = collect_misclassified(model, test_loader, paths, device)
        for row in wrong_rows:
            ti, pi = row["true_class"] - 1, row["pred_class"] - 1
            if 0 <= ti < len(class_names):
                row["true_name"] = class_names[ti]
            if 0 <= pi < len(class_names):
                row["pred_name"] = class_names[pi]
        if args.export_misclassified:
            save_misclassified_csv(wrong_rows, args.export_misclassified)
            logger.info("误判样本 %d 条 -> %s", len(wrong_rows), args.export_misclassified)
        if args.copy_misclassified_to:
            copy_misclassified_images(wrong_rows, args.copy_misclassified_to)
            logger.info("误判图片 -> %s", os.path.abspath(args.copy_misclassified_to))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging_utils.get_logger().exception("测试失败（未捕获异常）")
        raise
