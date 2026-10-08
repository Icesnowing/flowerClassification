# -*- coding: utf-8 -*-
"""实验追踪（P2-1）：TensorBoard / W&B 双后端，统一封装标量与超参记录。

策略:
  - TensorBoard 默认启用：`--log-dir` 为根目录，每次训练创建一个 run 子目录
    （默认 run 名 = 时间戳 + 关键超参，便于 `tensorboard --logdir runs` 多实验对比）。
  - W&B 可选启用：`--wandb-project` 指定项目名即开启；未安装 wandb 或初始化失败时
    warning 并跳过，不影响训练。
  - 任一后端依赖缺失或初始化失败时降级为 warning，训练主流程与 train_history.csv
    始终不受影响。

用法（train.py）:
    tracker = ExperimentTracker(log_dir=args.log_dir, run_name=run_name,
                                wandb_project=args.wandb_project, config=vars(args))
    tracker.log_batch({"train/batch_loss": loss}, step=global_step)
    tracker.log_epoch({"train/loss": ..., "val/acc1": ..., "lr": ...}, epoch=epoch)
    tracker.log_hparams(hparam_dict, {"hparam/best_acc1": best_acc1})
    tracker.close()
"""
import atexit
import os
import time

import logging_utils

_logger = logging_utils.get_logger("tracker")


def default_run_name(args):
    """默认 run 名：时间戳 + 关键超参。

    例: 20260921_093600_resnet18_adamw_lr0.001_bs8_ft（ft 表示预训练微调）
        20260922_110600_mobilenet_v3_large_adamw_lr0.001_bs8_ft
    """
    from model import network_tag

    ts = time.strftime("%Y%m%d_%H%M%S")
    parts = [ts, network_tag(getattr(args, "arch", "resnet"), args.depth), args.optimizer,
             f"lr{args.lr:g}", f"bs{args.batch_size}"]
    if args.pretrained:
        parts.append("ft")
    return "_".join(parts)


class ExperimentTracker:
    """TensorBoard / W&B 统一封装；两者都不可用时退化为 no-op（不影响训练）。"""

    def __init__(self, *, log_dir=None, run_name=None, wandb_project=None,
                 wandb_run_name=None, config=None):
        self._tb_writer = None
        self._wandb_run = None
        self._closed = False
        self.run_name = run_name or time.strftime("%Y%m%d_%H%M%S")

        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                _logger.warning("未安装 tensorboard，跳过 TensorBoard 记录"
                                "（pip install tensorboard 后可用）；train_history.csv 不受影响")
            else:
                try:
                    run_dir = os.path.join(log_dir, self.run_name)
                    self._tb_writer = SummaryWriter(log_dir=run_dir)
                    _logger.info("TensorBoard 已启用: %s（查看: tensorboard --logdir %s）",
                                 run_dir, log_dir)
                except Exception as e:
                    _logger.warning("TensorBoard 初始化失败（跳过，不影响训练）: %s", e)

        if wandb_project:
            try:
                import wandb
            except ImportError:
                _logger.warning("--wandb-project 已指定但未安装 wandb，跳过 W&B 记录"
                                "（pip install wandb 后可用）")
            else:
                try:
                    self._wandb_run = wandb.init(
                        project=wandb_project,
                        name=wandb_run_name or self.run_name,
                        config=config or {},
                    )
                    _logger.info("W&B 已启用: project=%s run=%s（%s）",
                                 wandb_project, self._wandb_run.name,
                                 self._wandb_run.url)
                except Exception as e:
                    # 未登录 / 网络异常等：降级为仅本地追踪
                    _logger.warning("W&B 初始化失败（跳过，不影响训练）: %s", e)

        if not self.enabled:
            _logger.info("实验追踪未启用（TensorBoard/W&B 均不可用），仅保留 train_history.csv")

        # 异常退出（未捕获异常 / 键盘中断）时也能 flush 并关闭，避免事件文件数据滞留缓冲区
        atexit.register(self.close)

    @property
    def enabled(self):
        return self._tb_writer is not None or self._wandb_run is not None

    @property
    def tb_run_dir(self):
        """TensorBoard 当前 run 目录；未启用时为 None。"""
        return self._tb_writer.log_dir if self._tb_writer is not None else None

    def log_batch(self, metrics, step):
        """batch 级标量（如 train/batch_loss），step 为全局 batch 序号（跨 epoch 单调递增）。"""
        if self._tb_writer is not None:
            for tag, value in metrics.items():
                self._tb_writer.add_scalar(tag, value, step)
        if self._wandb_run is not None:
            try:
                # W&B 的 step 需单调递增，batch/epoch 两套步长混用会告警；
                # 改为普通字段记录，图表可自选 x 轴
                self._wandb_run.log({"batch_step": step, **metrics})
            except Exception as e:
                _logger.warning("W&B log 失败（忽略本步）: %s", e)

    def log_epoch(self, metrics, epoch):
        """epoch 级标量（train/val loss、acc1/acc5、lr 等），epoch 从 1 起。"""
        if self._tb_writer is not None:
            for tag, value in metrics.items():
                self._tb_writer.add_scalar(tag, value, epoch)
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({"epoch": epoch, **metrics})
            except Exception as e:
                _logger.warning("W&B log 失败（忽略本步）: %s", e)

    def log_hparams(self, hparams, metrics):
        """训练结束时记录超参与最终指标。

        TensorBoard 侧写入 HPARAMS 面板（可按超参筛选/排序对比多实验）；
        W&B 侧写入 run summary。metrics 需非空且数值化（add_hparams 要求）。
        """
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_hparams(hparams, metric_dict=metrics)
            except Exception as e:
                _logger.warning("TensorBoard add_hparams 失败（忽略）: %s", e)
        if self._wandb_run is not None:
            try:
                for k, v in metrics.items():
                    self._wandb_run.summary[k.replace("hparam/", "")] = v
            except Exception as e:
                _logger.warning("W&B summary 写入失败（忽略）: %s", e)

    def close(self):
        """关闭后端并 flush 缓冲区（幂等，可重复调用）。"""
        if self._closed:
            return
        self._closed = True
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
