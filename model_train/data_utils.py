# -*- coding: utf-8 -*-
"""
Oxford 102 Flowers 数据集加载与预处理

数据布局（data_root 为项目根目录）:
    data_root/
      102flowers/
        jpg/image_00001.jpg ~ image_08189.jpg   # 8189 张花卉图片
        imagelabels.mat                         # labels: 1x8189 uint8, 值 1-102
        setid.mat                               # trnid(1020) / valid(1020) / tstid(6149)，1-based 下标

说明:
  - MATLAB 下标从 1 开始，这里统一转换为 0-based
  - 类别标签 1-102 在 Dataset 内部转换为 0-101
  - 图片与类别对应关系: image_00001.jpg 对应 imagelabels 第 1 个元素；
    train/val/test 用 setid.mat 里的下标子集，见 load_split()
"""
import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ImageNet 归一化参数（通用做法）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SPLIT_KEY = {"train": "trnid", "val": "valid", "test": "tstid"}


def load_split(data_root, split):
    """读取某个划分的图片路径与标签。

    Returns:
        paths   (list[str])      图片绝对路径，长度 = 该划分样本数
        targets (np.ndarray)     int64 标签数组，0-101（内部 0-based）
    """
    base = os.path.join(data_root, "102flowers")
    labels = sio.loadmat(os.path.join(base, "imagelabels.mat"))["labels"].flatten()
    setid = sio.loadmat(os.path.join(base, "setid.mat"))
    idx = setid[SPLIT_KEY[split]].flatten().astype(np.int64) - 1  # 1-based -> 0-based
    jpg_dir = os.path.join(base, "jpg")
    paths = [os.path.join(jpg_dir, "image_%05d.jpg" % (i + 1)) for i in idx]
    targets = labels[idx].astype(np.int64) - 1                     # 1-102 -> 0-101
    return paths, targets


class FlowerDataset(Dataset):
    """Flowers102 数据集：按路径列表 + 标签数组构建"""

    def __init__(self, paths, targets, transform=None):
        assert len(paths) == len(targets)
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"图片读取失败: {self.paths[idx]} ({e})")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def get_transform(train=True, mean=None, std=None, input_size=224, resize_size=256):
    """数据增强 / 标准化。

    训练: 随机裁剪 + 水平翻转 + 轻微色彩抖动（提高泛化）
    验证/测试: Resize(resize_size) + CenterCrop(input_size)，与训练输入尺寸一致

    mean/std 缺省时使用 ImageNet 归一化参数；推理侧通常由 config.json 提供。
    """
    mean = IMAGENET_MEAN if mean is None else mean
    std = IMAGENET_STD if std is None else std
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_eval_transform_from_config(config):
    """按推理配置 JSON（model.load_inference_config 的返回值）构建验证/测试预处理。

    config 为 None 时回退默认 ImageNet 配置，保证旧 checkpoint 行为不变。
    """
    config = config or {}
    return get_transform(
        train=False,
        mean=config.get("normalize_mean"),
        std=config.get("normalize_std"),
        input_size=config.get("input_size", 224),
        resize_size=config.get("resize_size", 256),
    )


def parse_oversample(specs):
    """解析过采样配置，如 ['51:5'] -> {50: 5.0}（内部 0-based 类别 -> 权重倍数）"""
    if not specs:
        return {}
    out = {}
    for item in specs:
        part = item.strip()
        if ":" not in part:
            raise ValueError(f"过采样格式应为 类别id:倍数，例如 51:5，收到: {item!r}")
        cid_str, mult_str = part.split(":", 1)
        cid = int(cid_str)
        mult = float(mult_str)
        if cid < 1 or cid > 102:
            raise ValueError(f"类别 id 须在 1-102，收到: {cid}")
        if mult <= 0:
            raise ValueError(f"过采样倍数须 > 0，收到: {mult}")
        out[cid - 1] = mult
    return out


def make_oversample_weights(targets, oversample):
    """按类别倍数生成 WeightedRandomSampler 用的样本权重"""
    weights = np.ones(len(targets), dtype=np.float64)
    for class_idx, mult in oversample.items():
        weights[targets == class_idx] = mult
    return weights


def build_loader(data_root, split, batch_size, num_workers=4, shuffle=None,
                 limit=None, pin_memory=False, oversample=None, transform=None,
                 persistent_workers=False):
    """构建 DataLoader。

    Args:
        split: 'train' / 'val' / 'test'
        limit: 调试用，只取前 N 张
        oversample: 0-based 类别 -> 采样权重倍数，仅对 train 生效（WeightedRandomSampler）
        transform: 可选，覆盖默认预处理（推理侧传入 config.json 构建的 transform）
        pin_memory: CUDA 训练时置 True，加速主机->设备拷贝（P1-2）
        persistent_workers: num_workers > 0 时可置 True，避免每个 epoch 重建 worker 进程（P1-2）
    """
    paths, targets = load_split(data_root, split)
    if limit is not None:
        paths, targets = paths[:limit], targets[:limit]
    if transform is None:
        transform = get_transform(train=(split == "train"))
    dataset = FlowerDataset(paths, targets, transform=transform)
    if shuffle is None:
        shuffle = (split == "train")

    sampler = None
    if split == "train" and oversample:
        sample_weights = make_oversample_weights(targets, oversample)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # num_workers=0 时 persistent_workers 非法（DataLoader 会抛 ValueError）
        persistent_workers=persistent_workers and num_workers > 0,
    )
