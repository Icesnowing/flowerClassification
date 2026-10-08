# -*- coding: utf-8 -*-
"""
Oxford 102 Flowers 图像分类训练脚本（ResNet / MobileNetV3）

数据布局（--data-root 默认本脚本所在目录）:
    data_root/
      102flowers/
        jpg/image_00001.jpg ~ image_08189.jpg
        imagelabels.mat      # 标签 1-102
        setid.mat            # trnid(1020) / valid(1020) / tstid(6149)

用法示例:
    python train.py                                    # ResNet18 + AdamW + 10 epochs（从零训练）
    python train.py --pretrained                       # 预训练权重微调（生产常用，主干低学习率）
    python train.py --depth 34 --epochs 50 --batch-size 32
    python train.py --arch mobilenet_v3_large --pretrained          # MobileNetV3-Large 微调
    python train.py --arch mobilenet_v3_small --epochs 30           # MobileNetV3-Small 从零训练
    python train.py --optimizer adamw --lr 1e-3
    python train.py --resume checkpoints/best_resnet18.pth   # 断点续训
    python train.py --pretrained --oversample 51:5 --save-dir checkpoints/os51   # 训练集类别 51 过采样 5 倍
    python train.py --pretrained --epochs 50 --early-stop-patience 5   # val acc1 连续 5 轮无提升则提前停训
    python train.py --seed 42 --deterministic                          # 可复现模式（相同 seed 严格复现，略降速）
    python train.py --run-name exp1                                    # 实验追踪：runs/exp1（tensorboard --logdir runs 查看）
    python train.py --wandb-project flowers                            # 同时启用 W&B 云端追踪（需 pip install wandb）

产物（--save-dir，默认 ./checkpoints）:
    best_<网络>.pth         # 验证集 Top-1 最高的权重，文件名带训练网络
                            # （如 best_resnet18.pth、best_mobilenet_v3_large.pth）；
                            # 再次训练同一网络时若该文件已存在且验证配置可比，
                            # 继承其历史最佳 acc1，仅当新实验超越历史最佳才覆盖（防误覆盖）
    last_model.pth          # 最后一轮权重
    train_history.csv     # 每轮 train/val 指标与学习率（追加写入，跨实验共用同一文件）

实验追踪产物（--log-dir，默认 ./runs）:
    runs/<run_name>/      # TensorBoard 事件文件（scalars + hparams）；--no-tensorboard 可关闭
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import data_utils
import experiment_tracker
import logging_utils
from model import (
    ARCH_CHOICES,
    best_checkpoint_name,
    build_model,
    cli_flag_provided,
    count_parameters,
    describe_model,
    load_checkpoint_file,
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
        # P1-5：cuDNN 卷积改用确定性算法、关闭 benchmark 自动调优（可能略降速，换取可复现）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def topk_accuracy(output, target, topk=(1, 5)):
    """
    返回各 topk 下预测正确的样本数列表
    Top-K 的含义是：真实标签出现在前 K 个预测中即算正确。
    """
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).item() for k in topk]


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    """在给定 DataLoader 上评估，返回 {loss, acc1, acc5}；use_amp 时以前向 autocast 评估"""
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
    return {
        "loss": loss_sum / total,
        "acc1": correct1 / total,
        "acc5": correct5 / total,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc1, args,
                    scaler=None, epochs_no_improve=0):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_acc1": best_acc1,
        "epochs_no_improve": epochs_no_improve,   # P1-4：early stopping 计数，供 resume 恢复
        "args": vars(args),
    }, path)


def save_inference_config_for(ckpt_path, args):
    """在 checkpoint 旁导出推理配置 JSON（P0-4：推理与训练配置单一事实来源）。"""
    return save_inference_config(
        ckpt_path,
        arch=args.arch,
        depth=args.depth,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        normalize_mean=data_utils.IMAGENET_MEAN,
        normalize_std=data_utils.IMAGENET_STD,
        input_size=224,
        resize_size=256,
    )


def main():
    parser = argparse.ArgumentParser(description="Flowers102 训练脚本（ResNet / MobileNetV3）")
    parser.add_argument("--data-root", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="项目根目录（含 102flowers 子目录）")
    parser.add_argument("--arch", type=str, default="resnet", choices=list(ARCH_CHOICES),
                        help="网络结构：resnet（配合 --depth）或 MobileNetV3 large/small")
    parser.add_argument("--depth", type=int, default=18, choices=[18, 34, 50],
                        help="ResNet 深度；仅 --arch resnet 时生效")
    parser.add_argument("--pretrained", action="store_true",
                        help="加载 torchvision ImageNet 预训练权重微调（首次运行需联网下载）")
    parser.add_argument("--num-classes", type=int, default=102)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率；不填时：从零训练 SGD 默认 0.1（按 batch/256 缩放）、"
                             "AdamW 默认 1e-3；预训练微调 SGD 默认 0.01、AdamW 默认 1e-3")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 子进程数")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="SGD 线性预热轮数")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Early stopping：val acc1 连续 N 轮无提升则提前停训"
                             "（best_<网络>.pth 保留历史最优）；0 表示关闭")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="混合精度训练（autocast + GradScaler，仅 CUDA 生效）；"
                             "默认 CUDA 开、CPU 关，可用 --amp / --no-amp 显式控制")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="可复现模式：cudnn.deterministic=True 且 cudnn.benchmark=False，"
                             "相同 seed 与数据下结果可严格复现（可能略降速）")
    parser.add_argument("--save-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
                        help="checkpoint 保存目录；若目录下已有同网络的 best_<网络>.pth 且验证配置可比，"
                             "继承其历史最佳 acc1，仅当新实验超越历史最佳才覆盖（防误覆盖）")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint 路径，继续训练")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="resume 时若 weights_only=True 失败，允许 weights_only=False（仅用于可信来源）")
    parser.add_argument("--log-interval", type=int, default=10, help="每多少 batch 打印一次")
    parser.add_argument("--limit-train", type=int, default=None, help="调试：只取前 N 张训练图")
    parser.add_argument("--limit-val", type=int, default=None, help="调试：只取前 N 张验证图")
    parser.add_argument("--oversample", type=str, nargs="+", default=None,
                        help="训练过采样，格式 类别id:倍数，可多个，如: --oversample 51:5")
    # P2-1：实验追踪（TensorBoard 默认开；W&B 可选）
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
                        help="TensorBoard 日志根目录（每次训练创建 <log-dir>/<run-name> 子目录，"
                             "多实验对比: tensorboard --logdir <log-dir>）")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                        help="TensorBoard 实验追踪（默认开；未安装 tensorboard 时 warning 并自动跳过，"
                             "不影响训练与 train_history.csv）")
    parser.add_argument("--run-name", type=str, default=None,
                        help="实验 run 名（TensorBoard 子目录名 / W&B 默认 run 名）；"
                             "默认 时间戳+关键超参，如 20260921_093600_resnet18_adamw_lr0.001_bs8"
                             " 或 20260922_110600_mobilenet_v3_large_adamw_lr0.001_bs8_ft")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B 项目名（传入即启用 W&B 云端追踪，需 pip install wandb 并登录）")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run 名；默认与 --run-name 相同")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 结构化日志：控制台 INFO + 文件 DEBUG -> save_dir/train.log（P1-1）
    os.makedirs(args.save_dir, exist_ok=True)
    logger = logging_utils.setup_logging(os.path.join(args.save_dir, "train.log"))
    logger.debug("命令行参数: %s", vars(args))

    if args.early_stop_patience < 0:
        logger.warning("--early-stop-patience=%d 非法，已按 0（关闭）处理",
                       args.early_stop_patience)
        args.early_stop_patience = 0

    # P1-5：--deterministic 时一并设置 cuDNN 确定性标志（须在创建模型/训练前生效）
    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device()
    if args.deterministic:
        # CPU 严格复现：OpenMP 动态线程调整会让并行浮点归约顺序随运行变化（实测同 seed 结果分叉），
        # 须在任何并行计算前关闭——OMP 运行时在首个并行区才读取环境变量，此处设置仍生效；
        # setdefault 尊重用户显式设置的外部值
        os.environ.setdefault("KMP_DYNAMIC", "FALSE")   # Intel OpenMP（Windows 版 torch）
        os.environ.setdefault("OMP_DYNAMIC", "FALSE")   # GNU OpenMP（Linux 版 torch）
        logger.info("deterministic 模式已开启：cudnn.deterministic=%s, cudnn.benchmark=%s（可能略降速）",
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        logger.info("已设 KMP_DYNAMIC=%s / OMP_DYNAMIC=%s：固定 CPU 并行归约顺序",
                    os.environ["KMP_DYNAMIC"], os.environ["OMP_DYNAMIC"])

    resume_ckpt = None
    if args.resume:
        resume_ckpt, resume_load_mode = load_checkpoint_file(
            args.resume, device, trust_checkpoint=args.trust_checkpoint)
        if resume_load_mode == "weights_only":
            logger.info("resume checkpoint（weights_only=True）: %s", args.resume)
        hparams, hp_warnings, _ = resolve_model_hparams(
            resume_ckpt,
            arch=args.arch,
            depth=args.depth,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            arch_from_cli=cli_flag_provided("--arch"),
            depth_from_cli=cli_flag_provided("--depth"),
            num_classes_from_cli=cli_flag_provided("--num-classes"),
            pretrained_from_cli=cli_flag_provided("--pretrained"),
        )
        for msg in hp_warnings:
            logger.warning("%s", msg)
        args.arch = hparams["arch"]
        args.depth = hparams["depth"]
        args.num_classes = hparams["num_classes"]
        args.pretrained = hparams["pretrained"]

    if args.arch != "resnet" and cli_flag_provided("--depth"):
        logger.info("--depth 仅对 ResNet 生效，当前 %s 忽略 --depth=%d",
                    describe_model(args.arch, args.depth), args.depth)

    if args.lr is None:
        if args.pretrained:
            # 微调默认用小学习率（主干权重已收敛，过大 lr 会破坏已学特征）
            args.lr = 1e-3 if args.optimizer == "adamw" else 0.01
        else:
            args.lr = 1e-3 if args.optimizer == "adamw" else 0.1
    if args.optimizer == "sgd":
        args.lr = args.lr * args.batch_size / 256.0   # 线性缩放规则：batch 32 时 0.01 -> 0.00125

    oversample = data_utils.parse_oversample(args.oversample)
    # P1-2：CUDA 时启用 pin_memory 加速 H2D 拷贝；num_workers>0 时复用 worker 进程
    use_cuda = device.type == "cuda"
    # P1-3：AMP 仅 CUDA 启用；--amp/--no-amp 可显式覆盖，默认跟随设备
    use_amp = args.amp if args.amp is not None else use_cuda
    if use_amp and not use_cuda:
        logger.warning("--amp 仅在 CUDA 上生效，当前 device=cpu，已自动关闭")
        use_amp = False
    persistent = args.num_workers > 0
    logger.debug("DataLoader: pin_memory=%s persistent_workers=%s num_workers=%d",
                 use_cuda, persistent, args.num_workers)
    train_loader = data_utils.build_loader(
        args.data_root, "train", args.batch_size, args.num_workers,
        limit=args.limit_train, oversample=oversample or None,
        pin_memory=use_cuda, persistent_workers=persistent)
    val_loader = data_utils.build_loader(
        args.data_root, "val", args.batch_size, args.num_workers, limit=args.limit_val,
        pin_memory=use_cuda, persistent_workers=persistent)

    model = build_model(args.arch, depth=args.depth, num_classes=args.num_classes,
                        pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    # 参数分组：微调时主干用小学习率，新随机初始化的分类头用 10 倍学习率
    if args.pretrained:
        backbone_params, head_params = split_backbone_head(model, args.arch)
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 10.0},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]

    if args.optimizer == "sgd":
        optimizer = SGD(param_groups, lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度：SGD 用线性预热 + 余弦退火；AdamW 直接用余弦退火
    warmup = min(args.warmup_epochs, max(args.epochs - 1, 0))
    if args.optimizer == "sgd" and warmup > 0 and args.epochs > warmup:
        scheduler = SequentialLR(
            optimizer,
            [LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
             CosineAnnealingLR(optimizer, T_max=args.epochs - warmup)],
            milestones=[warmup])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # P1-3：AMP。enabled=False 时 scale/step/update 均为透明直通，CPU 路径零分支
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_acc1 = 0, -1.0   # best_acc1=-1 保证无历史最佳时首轮一定保存
    run_best_acc1 = -1.0               # 本次训练自身的最佳（early stopping / hparams 用，与历史最佳区分）
    epochs_no_improve = 0              # P1-4：val acc1 连续无提升轮数（early stopping 计数）
    if args.resume:
        ckpt = resume_ckpt
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc1 = ckpt.get("best_acc1", -1.0)
        run_best_acc1 = best_acc1      # 续训：本次最佳与历史最佳同源，保持原有 early stopping 语义
        # P1-4：恢复 early stopping 计数；旧 checkpoint 无该键则从 0 重新计（向后兼容）
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        logger.info("从 %s 继续，epoch %d 起，历史最佳 acc1=%.4f",
                    args.resume, start_epoch, best_acc1)
        logger.debug("early stopping 计数恢复: epochs_no_improve=%d", epochs_no_improve)

    # 防覆盖：save_dir 已有同网络的 best_<网络>.pth 且验证配置可比时，继承其历史最佳 acc1，
    # 仅当本轮 val acc1 超越历史最佳才覆盖（不同网络各存各的，互不覆盖）
    best_name = best_checkpoint_name(args.arch, args.depth)
    best_path = os.path.join(args.save_dir, best_name)
    if os.path.exists(best_path):
        try:
            old_ckpt, _ = load_checkpoint_file(best_path, device,
                                               trust_checkpoint=args.trust_checkpoint)
            old_args = old_ckpt.get("args") or {}
            old_best = float(old_ckpt.get("best_acc1", -1.0))
            # limit_val 只取验证集子集、num_classes/data_root 不同则验证集不同，acc1 均不可比
            same_depth = (args.arch != "resnet"
                          or old_args.get("depth", args.depth) == args.depth)
            comparable = (
                old_args.get("arch", "resnet") == args.arch
                and same_depth
                and old_args.get("num_classes", args.num_classes) == args.num_classes
                and os.path.abspath(old_args.get("data_root", args.data_root))
                    == os.path.abspath(args.data_root)
                and old_args.get("limit_val") is None
                and args.limit_val is None
            )
            if comparable and old_best > best_acc1:
                best_acc1 = old_best
                logger.info("检测到已有 %s（历史最佳 val acc1=%.2f%%），"
                            "仅当本轮训练超越该成绩时才覆盖", best_name, old_best * 100)
            elif not comparable:
                logger.warning("已有 %s 与当前配置不可比"
                               "（arch/depth/num_classes/data_root/limit_val 不同），不继承其 best_acc1，"
                               "首轮可能覆盖它；如需保留请备份或更换 --save-dir", best_name)
        except Exception as e:
            logger.warning("读取已有 %s 失败（%s），不继承历史最佳，"
                           "首轮可能覆盖它；如需保留请备份或更换 --save-dir", best_name, e)

    history_path = os.path.join(args.save_dir, "train_history.csv")
    if not os.path.exists(history_path):
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc1", "val_loss", "val_acc1", "val_acc5", "lr"])

    lr_info = f"lr={args.lr:.5f}"
    if args.pretrained:
        lr_info += f"（主干 {args.lr:.5f} / 分类头 {args.lr * 10.0:.5f}）"
    logger.info("%s | pretrained=%s | classes=%d | device=%s | optimizer=%s | "
                "%s | batch=%d | epochs=%d | amp=%s",
                describe_model(args.arch, args.depth), args.pretrained, args.num_classes,
                device, args.optimizer, lr_info, args.batch_size, args.epochs, use_amp)
    logger.info("train=%d | val=%d | 参数量=%.2fM",
                len(train_loader.dataset), len(val_loader.dataset),
                count_parameters(model) / 1e6)
    if args.early_stop_patience > 0:
        logger.info("Early stopping 启用：patience=%d（以 val acc1 为准，连续无提升则停训，"
                    "%s 保留历史最优）", args.early_stop_patience, best_name)
    if oversample:
        _, train_targets = data_utils.load_split(args.data_root, "train")
        if args.limit_train is not None:
            train_targets = train_targets[:args.limit_train]
        total_w = float(data_utils.make_oversample_weights(train_targets, oversample).sum())
        parts = []
        for cid0, mult in sorted(oversample.items()):
            n = int((train_targets == cid0).sum())
            share = n * mult / total_w * 100.0
            parts.append(f"类{cid0 + 1}×{mult:g}({n}张≈{share:.1f}%/epoch)")
        logger.info("过采样: %s", " | ".join(parts))

    # P2-1：实验追踪。TensorBoard 默认开（--no-tensorboard 关闭）；--wandb-project 可选启用 W&B；
    # 依赖缺失或初始化失败时降级为 warning，不影响训练与 train_history.csv
    run_name = args.run_name or experiment_tracker.default_run_name(args)
    tracker = experiment_tracker.ExperimentTracker(
        log_dir=args.log_dir if args.tensorboard else None,
        run_name=run_name,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        config=vars(args),
    )

    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_c1, epoch_total = 0.0, 0, 0
        start = time.time()
        for step, (images, targets) in enumerate(train_loader):
            # non_blocking：配合 pin_memory 实现 H2D 异步拷贝（CPU 上为 no-op）
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            # P1-3：autocast 下前向用 FP16/BF16，GradScaler 防止梯度下溢
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            epoch_loss += loss.item() * bs
            c1, _ = topk_accuracy(outputs, targets)  # top1正确的样本数
            epoch_c1 += c1  
            epoch_total += bs
            if (step + 1) % args.log_interval == 0 or step + 1 == len(train_loader):
                logger.info("Epoch %d/%d [%d/%d] loss %.4f | acc1 %.2f%% | lr %.2e",
                            epoch + 1, args.epochs, step + 1, len(train_loader),
                            loss.item(), epoch_c1 / epoch_total * 100,
                            optimizer.param_groups[0]['lr'])
                # P2-1：batch 级曲线（全局 step，跨 epoch 单调递增；resume 时从 start_epoch 续接）
                tracker.log_batch({
                    "train/batch_loss": loss.item(),
                    "train/batch_acc1": epoch_c1 / epoch_total,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=epoch * len(train_loader) + step + 1)

        train_loss = epoch_loss / epoch_total
        train_acc1 = epoch_c1 / epoch_total
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start
        # 关键指标一行摘要（epoch、val acc1、lr），便于 grep / 日志采集
        logger.info("Epoch %d/%d 耗时 %.1fs | train loss %.4f acc1 %.2f%% | "
                    "val loss %.4f acc1 %.2f%% acc5 %.2f%% | lr %.2e",
                    epoch + 1, args.epochs, epoch_time,
                    train_loss, train_acc1 * 100,
                    val_metrics['loss'], val_metrics['acc1'] * 100,
                    val_metrics['acc5'] * 100, lr_now)

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.4f}", f"{train_acc1 * 100:.2f}",
                f"{val_metrics['loss']:.4f}", f"{val_metrics['acc1'] * 100:.2f}",
                f"{val_metrics['acc5'] * 100:.2f}", f"{lr_now:.2e}"])

        # is_best 与历史最佳比较（决定是否覆盖 best_<网络>.pth）；
        # early stopping 与本次训练自身最佳比较——继承历史最佳后，
        # 若仍按“未破纪录”计无提升，稳步进步的实验也会被误判提前停训
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

        # P2-1：epoch 级曲线（放在 best 更新之后，val/best_acc1 为历史最优轨迹）
        tracker.log_epoch({
            "train/loss": train_loss,
            "train/acc1": train_acc1,
            "val/loss": val_metrics["loss"],
            "val/acc1": val_metrics["acc1"],
            "val/acc5": val_metrics["acc5"],
            "val/best_acc1": best_acc1,
            "lr": lr_now,
            "epoch_time_sec": epoch_time,
        }, epoch=epoch + 1)

        last_path = os.path.join(args.save_dir, "last_model.pth")
        save_checkpoint(last_path, model, optimizer,
                        scheduler, epoch, best_acc1, args, scaler=scaler,
                        epochs_no_improve=epochs_no_improve)
        save_inference_config_for(last_path, args)
        if is_best:
            save_checkpoint(best_path, model, optimizer,
                            scheduler, epoch, best_acc1, args, scaler=scaler,
                            epochs_no_improve=epochs_no_improve)
            cfg_path = save_inference_config_for(best_path, args)
            logger.info("新最佳模型已保存 (val acc1 = %.2f%%) | 权重 -> %s | 推理配置 -> %s",
                        best_acc1 * 100, best_path, cfg_path)
        scheduler.step()

        # P1-4：patience 轮内 val acc1 无提升则提前停训；best_<网络>.pth 不被覆盖，保留历史最优
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            stopped_early = True
            logger.info("Early stopping 触发：val acc1 连续 %d 轮无提升（patience=%d），"
                        "于 epoch %d/%d 停止训练 | 最佳 val acc1=%.2f%% 权重保留在 %s",
                        epochs_no_improve, args.early_stop_patience,
                        epoch + 1, args.epochs, best_acc1 * 100, best_name)
            break

    end_reason = "early stopping 提前停训" if stopped_early else "跑完全部 epoch"
    if best_acc1 > run_best_acc1:
        logger.info("训练结束（%s）| 本次最佳 val acc1 = %.2f%% | "
                    "%s 保留历史最佳 %.2f%%（未被覆盖）",
                    end_reason, run_best_acc1 * 100, best_name, best_acc1 * 100)
    else:
        logger.info("训练结束（%s），最佳 val acc1 = %.2f%% | 权重: %s",
                    end_reason, best_acc1 * 100, best_path)

    # P2-1：超参与最终指标（TensorBoard HPARAMS 面板 / W&B summary），随后关闭后端 flush 缓冲
    # hparam/best_acc1 记录本次训练自身的最佳（而非继承的历史最佳），保证跨实验对比可比
    tracker.log_hparams(
        {"arch": args.arch, "depth": args.depth, "optimizer": args.optimizer, "lr": args.lr,
         "batch_size": args.batch_size, "epochs": args.epochs,
         "pretrained": args.pretrained, "weight_decay": args.weight_decay,
         "seed": args.seed},
        {"hparam/best_acc1": run_best_acc1, "hparam/stopped_early": int(stopped_early)})
    tracker.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # 异常栈写入 train.log（P1-1 验收：异常栈写入日志）
        logging_utils.get_logger().exception("训练失败（未捕获异常）")
        raise
