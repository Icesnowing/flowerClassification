# -*- coding: utf-8 -*-
"""
按路线A：用 flower14 直接微调现有 102 类模型 → 输出 14 类新模型

与 train.py 的区别（本脚本为新增独立文件，不修改 train.py / model.py / data_utils.py）：
  - 数据加载：ImageFolder 读 flower14/train、flower14/val；默认从 train 按类分层 90% 训练 / 10% 验证（选模型），
    官方 val 目录仅作测试集（训练过程中不参与选 best / early stopping）。
  - 初始化：加载现有 102 类 checkpoint（如 checkpoints/best_model.pth）的“主干”权重，
    把分类头换成 14 类（随机初始化），再在 flower14 上微调。

用法示例（在项目根目录运行）:
  python train_folder.py --data-root flower14 --checkpoint checkpoints\\best_model.pth
  python train_folder.py --data-root flower14 --checkpoint checkpoints\\best_model.pth --epochs 20 --batch-size 32 --lr 1e-3 --save-dir checkpoints_ft14

产物（--save-dir，默认 ./checkpoints_ft14）:
  best_<网络>.pth   # 验证集 Top-1 最高的 14 类权重（文件名带训练网络，如 best_resnet34.pth、best_mobilenet_v3_large.pth）
  last_model.pth   # 最后一轮权重
  best_<网络>_config.json # 推理配置（arch / depth / num_classes=14 / 归一化参数），供 test.py / export.py 读取
  class_names.txt  # 类别索引 -> 花名映射（0 astilbe ... 13 water_lily），推理时对照
  train_history.csv / train.log
  测试集评估请用 test_folder.py（官方 val/）
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import data_utils          # 仅复用 get_transform / IMAGENET 归一化，不改它
import logging_utils
from model import (
    ARCH_CHOICES,
    best_checkpoint_name,
    build_model,
    cli_flag_provided,
    count_parameters,
    describe_model,
    head_parameter_names,
    load_checkpoint_file,
    load_inference_config,
    resolve_model_hparams,
    save_inference_config,
    split_backbone_head,
)


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("KMP_DYNAMIC", "FALSE")
        os.environ.setdefault("OMP_DYNAMIC", "FALSE")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def topk_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).item() for k in topk]


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total = 0
    loss_sum = 0.0
    correct1 = correct5 = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss_sum += criterion(outputs, targets).item() * images.size(0)
        c1, c5 = topk_accuracy(outputs, targets)
        correct1 += c1
        correct5 += c5
        total += images.size(0)
    return {"loss": loss_sum / total, "acc1": correct1 / total, "acc5": correct5 / total}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc1, args, scaler=None,
                    epochs_no_improve=0):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_acc1": best_acc1,
        "epochs_no_improve": epochs_no_improve,
        "args": vars(args),
    }, path)


def load_backbone_weights(checkpoint_path, model, device, arch="resnet",
                          trust_checkpoint=False, ckpt=None):
    """把已有 checkpoint 的主干权重装载进 model，分类头保持随机。"""
    if ckpt is None:
        ckpt, _ = load_checkpoint_file(checkpoint_path, device,
                                       trust_checkpoint=trust_checkpoint)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"checkpoint 不是可用的训练产物: {checkpoint_path}")
    sd = ckpt["model_state_dict"]
    head_names = head_parameter_names(model, arch)
    model_sd = model.state_dict()
    to_load = {}
    for key, value in sd.items():
        if key in head_names or key.startswith("fc."):
            continue
        if key not in model_sd or model_sd[key].shape != value.shape:
            continue
        to_load[key] = value
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    backbone_missing = sorted(set(missing) - head_names)
    if backbone_missing:
        sample = ", ".join(backbone_missing[:6])
        extra = f" ... 共 {len(backbone_missing)} 个" if len(backbone_missing) > 6 else ""
        raise RuntimeError(
            f"主干权重未能对齐（仍有参数为随机初始化）。"
            f"请确认 --checkpoint 与 --arch / --depth 与训练该权重时一致；"
            f"若未显式指定，脚本会从 checkpoint 旁 *_config.json 或 checkpoint.args 自动读取。"
            f"缺失示例: {sample}{extra}")
    if unexpected:
        raise RuntimeError(f"加载 checkpoint 时出现未预期的键: {unexpected}")
    return ckpt


def stratified_train_val_indices(targets, val_fraction, seed):
    """按类别分层划分 train 索引：约 (1-val_fraction) 训练、(val_fraction) 验证。"""
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        rng.shuffle(class_indices)
        n_val = int(round(len(class_indices) * val_fraction))
        n_val = min(max(n_val, 1), len(class_indices) - 1) if len(class_indices) > 1 else 0
        val_idx.extend(class_indices[:n_val].tolist())
        train_idx.extend(class_indices[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser(description="按路线A：flower14 直接微调现有 102 模型 → 14 类")
    parser.add_argument("--data-root", type=str, required=True,
                        help="flower14 目录，须含 train/ 和 val/，其下按类别名分子文件夹")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="现有 102 类模型权重路径，如 checkpoints/best_model.pth")
    parser.add_argument("--arch", type=str, default="resnet", choices=list(ARCH_CHOICES),
                        help="网络结构，须与 --checkpoint 的网络一致")
    parser.add_argument("--depth", type=int, default=18, choices=[18, 34, 50],
                        help="ResNet 深度；仅 --arch resnet 时生效；未指定时优先从 checkpoint 读取")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="主干学习率；分类头自动用 10 倍")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="val acc1 连续 N 轮无提升则提前停训；0 关闭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints_ft14")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="从 flower14/train 分层划出验证集比例（选模型/早停）；默认 0.1")
    parser.add_argument("--limit-train", type=int, default=None, help="调试：训练子集只取前 N 张")
    parser.add_argument("--limit-val", type=int, default=None, help="调试：验证子集只取前 N 张")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.makedirs(args.save_dir, exist_ok=True)
    logger = logging_utils.setup_logging(os.path.join(args.save_dir, "train.log"))
    logger.debug("命令行参数: %s", vars(args))
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError(f"--val-fraction 须在 (0, 1) 内，收到: {args.val_fraction}")

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    use_amp = args.amp if args.amp is not None else device.type == "cuda"
    if use_amp and device.type != "cuda":
        logger.warning("--amp 仅在 CUDA 生效，当前 device=cpu，已自动关闭")
        use_amp = False

    # 1) train 分层 90/10 做训练/验证；官方 val 目录仅作测试
    train_root = os.path.join(args.data_root, "train")
    test_root = os.path.join(args.data_root, "val")
    if not (os.path.isdir(train_root) and os.path.isdir(test_root)):
        raise FileNotFoundError(f"--data-root 下应含 train/ 和 val/ 子目录: {args.data_root}")
    train_source = datasets.ImageFolder(train_root, transform=data_utils.get_transform(train=True))
    val_source = datasets.ImageFolder(train_root, transform=data_utils.get_transform(train=False))
    test_folder_ds = datasets.ImageFolder(test_root, transform=data_utils.get_transform(train=False))
    if train_source.classes != test_folder_ds.classes:
        raise ValueError("train 与 val（测试集）的类别不一致")
    num_classes = len(train_source.classes)
    class_names = list(train_source.classes)
    train_idx, val_idx = stratified_train_val_indices(
        train_source.targets, args.val_fraction, args.seed)
    train_ds = Subset(train_source, train_idx)
    val_ds = Subset(val_source, val_idx)
    split_meta = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "class_names": class_names,
    }
    with open(os.path.join(args.save_dir, "split_indices.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    logger.info("从文件夹读取到 %d 个类别: %s", num_classes, ", ".join(class_names))
    logger.info(
        "数据划分: train=%d (%.0f%% of train 目录) | val=%d (%.0f%%，选模型) | test=%d (官方 val/) | device=%s",
        len(train_ds), (1 - args.val_fraction) * 100,
        len(val_ds), args.val_fraction * 100,
        len(test_folder_ds), device,
    )

    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_cuda,
                            persistent_workers=args.num_workers > 0)

    # 2) 从 checkpoint / 旁路 config 解析 arch、depth（输出仍为 flower14 的 num_classes）
    source_ckpt, _ = load_checkpoint_file(
        args.checkpoint, device, trust_checkpoint=args.trust_checkpoint)
    infer_defaults = load_inference_config(args.checkpoint) or {}
    hparams, hparam_warnings, _ = resolve_model_hparams(
        source_ckpt,
        arch=args.arch,
        depth=args.depth,
        num_classes=num_classes,
        arch_from_cli=cli_flag_provided("--arch"),
        depth_from_cli=cli_flag_provided("--depth"),
        num_classes_from_cli=True,
        defaults=infer_defaults,
    )
    args.arch = hparams["arch"]
    args.depth = hparams["depth"]
    for msg in hparam_warnings:
        logger.warning(msg)
    logger.info("源模型结构: %s（来自 checkpoint / config，与 14 类新头组合微调）",
                describe_model(args.arch, args.depth))

    model = build_model(args.arch, depth=args.depth, num_classes=num_classes,
                        pretrained=False).to(device)
    load_backbone_weights(args.checkpoint, model, device, arch=args.arch,
                          trust_checkpoint=args.trust_checkpoint, ckpt=source_ckpt)
    logger.info("已从 %s 装载主干权重到 %s 类新模型，分类头已随机初始化",
                args.checkpoint, num_classes)
    logger.info("%s | classes=%d | 参数量=%.2fM",
                describe_model(args.arch, args.depth), num_classes,
                count_parameters(model) / 1e6)

    criterion = nn.CrossEntropyLoss()

    # 3) 主干小学习率、新分类头 10 倍学习率（复用 train.py 的微调策略）
    backbone_params, head_params = split_backbone_head(model, args.arch)
    param_groups = [
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10.0},
    ]
    if args.optimizer == "sgd":
        optimizer = SGD(param_groups, lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_acc1, run_best_acc1, epochs_no_improve = -1.0, -1.0, 0
    history_path = os.path.join(args.save_dir, "train_history.csv")
    if not os.path.exists(history_path):
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc1", "val_loss", "val_acc1", "val_acc5", "lr"])
    best_name = best_checkpoint_name(args.arch, args.depth)
    best_path = os.path.join(args.save_dir, best_name)
    last_path = os.path.join(args.save_dir, "last_model.pth")

    logger.info("主干 lr=%.5f / 分类头 lr=%.5f | optimizer=%s | batch=%d | epochs=%d | amp=%s",
                args.lr, args.lr * 10.0, args.optimizer, args.batch_size, args.epochs, use_amp)

    stopped_early = False
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_c1, epoch_total, start = 0.0, 0, 0, time.time()
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = images.size(0)
            epoch_loss += loss.item() * bs
            c1, _ = topk_accuracy(outputs, targets)
            epoch_c1 += c1
            epoch_total += bs
            if (step + 1) % args.log_interval == 0 or step + 1 == len(train_loader):
                logger.info("Epoch %d/%d [%d/%d] loss %.4f | acc1 %.2f%% | lr %.2e",
                            epoch + 1, args.epochs, step + 1, len(train_loader),
                            loss.item(), epoch_c1 / epoch_total * 100,
                            optimizer.param_groups[0]['lr'])

        train_loss = epoch_loss / epoch_total
        train_acc1 = epoch_c1 / epoch_total
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d 耗时 %.1fs | train loss %.4f acc1 %.2f%% | "
                    "val loss %.4f acc1 %.2f%% acc5 %.2f%% | lr %.2e",
                    epoch + 1, args.epochs, time.time() - start,
                    train_loss, train_acc1 * 100,
                    val_metrics['loss'], val_metrics['acc1'] * 100,
                    val_metrics['acc5'] * 100, lr_now)

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.4f}", f"{train_acc1 * 100:.2f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['acc1'] * 100:.2f}",
                f"{val_metrics['acc5'] * 100:.2f}", f"{lr_now:.2e}"])

        is_best = val_metrics["acc1"] > best_acc1
        if is_best:
            best_acc1 = val_metrics["acc1"]
        if val_metrics["acc1"] > run_best_acc1:
            run_best_acc1 = val_metrics["acc1"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop_patience > 0:
                logger.info("val acc1 无提升（连续 %d/%d 轮，patience 用尽将提前停训）",
                            epochs_no_improve, args.early_stop_patience)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_acc1, args,
                        scaler=scaler, epochs_no_improve=epochs_no_improve)
        save_inference_config(last_path, arch=args.arch, depth=args.depth,
                              num_classes=num_classes, pretrained=False,
                              normalize_mean=data_utils.IMAGENET_MEAN,
                              normalize_std=data_utils.IMAGENET_STD,
                              input_size=224, resize_size=256)
        if is_best:
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_acc1, args,
                            scaler=scaler, epochs_no_improve=epochs_no_improve)
            save_inference_config(best_path, arch=args.arch, depth=args.depth,
                                  num_classes=num_classes, pretrained=False,
                                  normalize_mean=data_utils.IMAGENET_MEAN,
                                  normalize_std=data_utils.IMAGENET_STD,
                                  input_size=224, resize_size=256)
            logger.info("新最佳模型已保存 (val acc1 = %.2f%%) | 权重 -> %s",
                        best_acc1 * 100, best_path)
        scheduler.step()

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            stopped_early = True
            logger.info("Early stopping 触发（epoch %d/%d）| 最佳 val acc1=%.2f%% 保留在 %s",
                        epoch + 1, args.epochs, best_acc1 * 100, best_name)
            break

    # 4) 保存类别名映射（推理时对照 0-13 -> 花名）
    names_path = os.path.join(args.save_dir, "class_names.txt")
    with open(names_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}\t{name}\n")
    logger.info("类别名映射 -> %s", names_path)
    logger.info("训练结束（%s）| 最佳 val acc1 = %.2f%% | 权重: %s",
                "early stopping 提前停训" if stopped_early else "跑完全部 epoch",
                best_acc1 * 100, best_path)
    eval_ckpt = best_path if os.path.isfile(best_path) else last_path
    logger.info("在官方 val/ 测试集上评估: python test_folder.py --checkpoint %s --data-root %s",
                eval_ckpt, args.data_root)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging_utils.get_logger().exception("训练失败（未捕获异常）")
        raise
