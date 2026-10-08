# -*- coding: utf-8 -*-
"""
ResNet 残差网络（从零实现，不依赖 torchvision 预训练权重）

结构要点（论文 "Deep Residual Learning for Image Recognition", He et al. 2016）:
  - stem: 7x7 卷积(64) -> BN -> ReLU -> 3x3 MaxPool
  - 四个残差阶段: 64/128/256/512 通道，每个阶段末尾 stride=2 下采样
  - 残差块: y = F(x) + x，维度不匹配时用 1x1 卷积 shortcut 下采样
  - 分类头: 全局平均池化 -> 全连接(102)

支持深度:
  - 18 / 34 : BasicBlock（两个 3x3 卷积）
  - 50     : Bottleneck（1x1 -> 3x3 -> 1x1，expansion=4）

两种训练模式（build_model / build_resnet 的 pretrained 参数 / train.py 的 --pretrained）:
  - pretrained=False: 从零初始化训练（学习用途）
  - pretrained=True : 加载 torchvision ImageNet 权重到主干，仅分类头随机初始化（生产常用）

另外支持 torchvision MobileNetV3（--arch mobilenet_v3_large / mobilenet_v3_small）:
  结构使用官方实现，预训练权重与分类头替换方式与 ResNet 相同。
"""
import json
import os
import sys

import torch
import torch.nn as nn

import logging_utils

_logger = logging_utils.get_logger("model")

ARCH_CHOICES = ("resnet", "mobilenet_v3_large", "mobilenet_v3_small")

_DEFAULT_MODEL_HPARAMS = {
    "arch": "resnet",
    "depth": 18,
    "num_classes": 102,
    "pretrained": False,
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积，padding=1，stride=1 时保持空间尺寸"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet 基本残差块（用于 18/34 层）"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity          # 残差连接: F(x) + x
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈残差块（用于 50 层及以上）"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=102):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet(depth=18, num_classes=102, pretrained=False):
    """按深度构建 ResNet。

    Args:
        depth: 18 / 34 / 50
        num_classes: 输出类别数（Flowers102 为 102）
        pretrained: True 时把 torchvision 的 ImageNet 预训练权重加载到主干，
                   分类头 fc 保持随机初始化（102 类）；首次运行需联网下载权重。
                   False 时从零初始化（本文件自实现结构）。

    说明:
      - torchvision 的 ResNet 结构与本文件自实现结构一致，
        因此两种模式共用同一个 ResNet 类，只差权重来源。
      - 预训练权重只覆盖 conv/bn 主干，fc 因类别数不同而重新随机初始化。
    """
    arch = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
    }
    if depth not in arch:
        raise ValueError(f"不支持的深度: {depth}，可选 18/34/50")
    block, layers = arch[depth]
    model = ResNet(block, layers, num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights(model, depth)
    return model


def _load_pretrained_weights(model, depth):
    """从 torchvision 加载 ImageNet 预训练权重到 model 主干。

    说明:
      - 仅加载 fc 以外的权重（fc 输出维度不同，保持随机初始化）
      - 首次调用自动联网下载权重，缓存在 torch 的 hub 目录
      - 下载失败会抛出带提示的 RuntimeError
    """
    import torchvision.models as tv_models

    ctor = {18: tv_models.resnet18, 34: tv_models.resnet34, 50: tv_models.resnet50}[depth]
    weights = {
        18: tv_models.ResNet18_Weights.DEFAULT,
        34: tv_models.ResNet34_Weights.DEFAULT,
        50: tv_models.ResNet50_Weights.DEFAULT,
    }[depth]
    try:
        tv_model = ctor(weights=weights)
    except Exception as e:
        raise RuntimeError(
            "加载 torchvision 预训练权重失败（需要联网下载，缓存在 torch hub 目录）。"
            "请检查网络后重试，或改用 pretrained=False 从零训练。"
            f"原始错误: {e}") from e

    sd = tv_model.state_dict()
    sd.pop("fc.weight", None)
    sd.pop("fc.bias", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) <= {"fc.weight", "fc.bias"}, f"意外的缺失键: {missing}"
    assert not unexpected, f"意外的多余键: {unexpected}"
    _logger.info("已加载 torchvision ResNet-%d ImageNet 权重到主干，"
                 "分类头 fc 已替换为 %d 类（随机初始化）", depth, model.fc.out_features)


def build_mobilenet_v3(variant="large", num_classes=102, pretrained=False):
    """构建 torchvision MobileNetV3，并把最后一层换成 num_classes。

    variant: "large" 或 "small"。
    pretrained=True 时加载 ImageNet 权重，仅最后一层线性分类头随机初始化。
    """
    import torchvision.models as tv_models

    spec = {
        "large": (tv_models.mobilenet_v3_large, tv_models.MobileNet_V3_Large_Weights),
        "small": (tv_models.mobilenet_v3_small, tv_models.MobileNet_V3_Small_Weights),
    }
    if variant not in spec:
        raise ValueError(f"不支持的 MobileNetV3 规格: {variant}，可选 large/small")
    ctor, weights_enum = spec[variant]
    weights = weights_enum.DEFAULT if pretrained else None
    try:
        model = ctor(weights=weights)
    except Exception as e:
        if not pretrained:
            raise
        raise RuntimeError(
            "加载 torchvision MobileNetV3 预训练权重失败（需要联网下载，缓存在 torch hub 目录）。"
            "请检查网络后重试，或改用 pretrained=False 从零训练。"
            f"原始错误: {e}") from e
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if pretrained:
        _logger.info("已加载 torchvision MobileNetV3-%s ImageNet 权重，"
                     "分类头最后一层已替换为 %d 类（随机初始化）",
                     variant, num_classes)
    return model


def build_model(arch="resnet", depth=18, num_classes=102, pretrained=False):
    """按 --arch 构建分类网络。arch=resnet 时 depth 取 18/34/50。"""
    arch = arch or "resnet"
    if arch == "resnet":
        return build_resnet(depth, num_classes=num_classes, pretrained=pretrained)
    if arch == "mobilenet_v3_large":
        return build_mobilenet_v3("large", num_classes=num_classes, pretrained=pretrained)
    if arch == "mobilenet_v3_small":
        return build_mobilenet_v3("small", num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"不支持的网络: {arch}，可选 {', '.join(ARCH_CHOICES)}")


def network_tag(arch="resnet", depth=18):
    """训练网络短名，用于最佳权重文件名与实验 run 名。"""
    arch = arch or "resnet"
    if arch == "resnet":
        return f"resnet{int(depth)}"
    if arch in ("mobilenet_v3_large", "mobilenet_v3_small"):
        return arch
    raise ValueError(f"不支持的网络: {arch}，可选 {', '.join(ARCH_CHOICES)}")


def describe_model(arch="resnet", depth=18):
    """日志用的网络名称。"""
    arch = arch or "resnet"
    if arch == "resnet":
        return f"ResNet-{int(depth)}"
    labels = {
        "mobilenet_v3_large": "MobileNetV3-Large",
        "mobilenet_v3_small": "MobileNetV3-Small",
    }
    if arch not in labels:
        raise ValueError(f"不支持的网络: {arch}，可选 {', '.join(ARCH_CHOICES)}")
    return labels[arch]


def best_checkpoint_name(arch="resnet", depth=18):
    """最佳权重文件名，带训练网络名。

    例: resnet depth=18 -> best_resnet18.pth
        mobilenet_v3_large -> best_mobilenet_v3_large.pth
    """
    return f"best_{network_tag(arch, depth)}.pth"


def head_parameter_names(model, arch="resnet"):
    """新初始化的分类头参数名。ResNet 为 fc.*；MobileNetV3 为 classifier 最后一层。"""
    arch = arch or "resnet"
    if arch == "resnet":
        return {name for name, _ in model.named_parameters() if name.startswith("fc.")}
    last = model.classifier[-1]
    prefix = None
    for name, module in model.named_modules():
        if module is last:
            prefix = name + "."
            break
    if not prefix:
        raise ValueError("找不到 MobileNetV3 分类头最后一层")
    names = {name for name, _ in model.named_parameters() if name.startswith(prefix)}
    if not names:
        raise ValueError(f"分类头 {prefix} 没有可训练参数")
    return names


def split_backbone_head(model, arch="resnet"):
    """拆出主干参数与分类头参数（微调时分类头用更大的学习率）。"""
    head_names = head_parameter_names(model, arch)
    backbone, head = [], []
    for name, param in model.named_parameters():
        (head if name in head_names else backbone).append(param)
    return backbone, head


def count_parameters(model):
    """返回可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cli_flag_provided(flag):
    """命令行是否显式传入某 flag（支持 --flag 与 --flag=value）。"""
    prefix = flag + "="
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(prefix):
            return True
    return False


def resolve_model_hparams(
    ckpt,
    *,
    arch=None,
    depth=None,
    num_classes=None,
    pretrained=None,
    arch_from_cli=False,
    depth_from_cli=False,
    num_classes_from_cli=False,
    pretrained_from_cli=False,
    defaults=None,
):
    """解析模型结构超参。

    优先级（高 -> 低）:
        checkpoint 内 args  >  显式 CLI  >  defaults（如推理 config.json）  >  内置默认

    Returns:
        (hparams dict, warnings list, used_checkpoint_args bool)
    """
    base = dict(_DEFAULT_MODEL_HPARAMS)
    if defaults:
        base.update({k: v for k, v in defaults.items() if v is not None})

    ckpt_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    has_args = isinstance(ckpt_args, dict) and bool(ckpt_args)
    warnings = []

    if not has_args:
        warnings.append(
            "[警告] checkpoint 不含训练配置 (args)，将使用命令行或默认值 "
            f"(arch={arch if arch_from_cli and arch is not None else base['arch']}, "
            f"depth={depth if depth_from_cli and depth is not None else base['depth']}, "
            f"num_classes={num_classes if num_classes_from_cli and num_classes is not None else base['num_classes']})。")

    hparams = {}
    cli_map = {
        "arch": (arch, arch_from_cli),
        "depth": (depth, depth_from_cli),
        "num_classes": (num_classes, num_classes_from_cli),
        "pretrained": (pretrained, pretrained_from_cli),
    }
    flag_names = {
        "arch": "--arch",
        "depth": "--depth",
        "num_classes": "--num-classes",
        "pretrained": "--pretrained",
    }
    for key, (cli_val, from_cli) in cli_map.items():
        ckpt_val = ckpt_args.get(key, base[key]) if has_args else base[key]
        if has_args:
            chosen = ckpt_val
        elif from_cli and cli_val is not None:
            chosen = cli_val
        else:
            chosen = base[key]
        if from_cli and has_args and cli_val is not None and cli_val != ckpt_val:
            warnings.append(
                f"[警告] 命令行 {flag_names[key]}={cli_val} 与 checkpoint 中 {ckpt_val} 不一致，"
                f"以 checkpoint 为准。")
        hparams[key] = chosen

    return hparams, warnings, has_args


def load_checkpoint_file(path, map_location, *, trust_checkpoint=False):
    """安全加载 checkpoint（PyTorch 2.x 优先 `weights_only=True`）。

    策略:
      - 默认先 `weights_only=True`，限制 pickle 反序列化面。
      - 若失败且未声明可信来源，抛出错误并提示 `--trust-checkpoint`。
      - 仅当 `trust_checkpoint=True` 时回退 `weights_only=False`（完整 pickle，需自行信任文件来源）。
      - PyTorch 无 `weights_only` 参数时回退为传统 `torch.load`（旧版本兼容）。

    Returns:
        (checkpoint, load_mode) 其中 load_mode 为 "weights_only" | "unsafe" | "legacy"
    """
    def _load(weights_only):
        if weights_only is None:
            return torch.load(path, map_location=map_location)
        return torch.load(path, map_location=map_location, weights_only=weights_only)

    try:
        ckpt = _load(True)
        return ckpt, "weights_only"
    except TypeError:
        ckpt = _load(None)
        return ckpt, "legacy"
    except Exception as e:
        if not trust_checkpoint:
            raise RuntimeError(
                f"无法以安全模式 (weights_only=True) 加载 checkpoint: {path}\n"
                f"原因: {e}\n"
                "若该文件来自本仓库训练产物或你完全信任的来源，请加上 --trust-checkpoint 后重试。"
                "对外部不可信文件请勿使用 --trust-checkpoint，应仅分发/加载纯 model_state_dict。"
            ) from e
        ckpt = _load(False)
        _logger.warning("已使用 weights_only=False 加载（--trust-checkpoint）: %s", path)
        return ckpt, "unsafe"


def config_path_for_checkpoint(checkpoint_path):
    """推理配置 JSON 路径：best_model.pth -> best_config.json；其他 -> <stem>_config.json"""
    directory = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path)
    if filename.endswith("_model.pth"):
        filename = filename[: -len("_model.pth")] + "_config.json"
    else:
        filename = os.path.splitext(filename)[0] + "_config.json"
    return os.path.join(directory, filename) if directory else filename


def save_inference_config(checkpoint_path, *, depth, num_classes, pretrained,
                          normalize_mean, normalize_std, input_size=224, resize_size=256,
                          arch="resnet"):
    """在 checkpoint 旁导出推理配置 JSON（推理侧单一事实来源）。

    Returns:
        写入的 JSON 路径
    """
    config = {
        "arch": str(arch or "resnet"),
        "depth": int(depth),
        "num_classes": int(num_classes),
        "pretrained": bool(pretrained),
        "input_size": int(input_size),
        "resize_size": int(resize_size),
        "normalize_mean": [float(v) for v in normalize_mean],
        "normalize_std": [float(v) for v in normalize_std],
    }
    path = config_path_for_checkpoint(checkpoint_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def load_inference_config(checkpoint_path):
    """读取 checkpoint 旁的推理配置 JSON；不存在或损坏时返回 None（调用方回退默认）。"""
    path = config_path_for_checkpoint(checkpoint_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        _logger.warning("推理配置读取失败（忽略，回退默认预处理）: %s (%s)", path, e)
        return None
    if not isinstance(cfg, dict):
        _logger.warning("推理配置格式异常（忽略）: %s", path)
        return None
    return cfg


def load_from_checkpoint(
    checkpoint_path,
    device,
    *,
    arch=None,
    depth=None,
    num_classes=None,
    arch_from_cli=False,
    depth_from_cli=False,
    num_classes_from_cli=False,
    trust_checkpoint=False,
    defaults=None,
):
    """加载训练 checkpoint 或纯 state_dict，并按 checkpoint args 构建网络（推理用，不拉取 ImageNet 权重）。

    Args:
        defaults: 可选 dict，无 checkpoint args 且未显式 CLI 时的回退值（如推理 config.json）。

    Returns:
        model, ckpt, hparams dict (arch, depth, num_classes)
    """
    ckpt, _load_mode = load_checkpoint_file(
        checkpoint_path, device, trust_checkpoint=trust_checkpoint)
    hparams, warnings, _ = resolve_model_hparams(
        ckpt,
        arch=arch,
        depth=depth,
        num_classes=num_classes,
        arch_from_cli=arch_from_cli,
        depth_from_cli=depth_from_cli,
        num_classes_from_cli=num_classes_from_cli,
        pretrained=False,
        pretrained_from_cli=False,
        defaults=defaults,
    )
    for msg in warnings:
        _logger.warning("%s", msg)

    model = build_model(
        hparams["arch"], depth=hparams["depth"],
        num_classes=hparams["num_classes"], pretrained=False)
    model = model.to(device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        if not isinstance(ckpt, dict):
            raise ValueError("无法识别的 checkpoint 格式（非 dict 且无 model_state_dict）")
        model.load_state_dict(ckpt)
    return model, ckpt, {
        "arch": hparams["arch"],
        "depth": hparams["depth"],
        "num_classes": hparams["num_classes"],
    }


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    for d in (18, 34, 50):
        m = build_model("resnet", depth=d, num_classes=102)
        y = m(x)
        print(f"{describe_model('resnet', d)}: 输出 {tuple(y.shape)} | "
              f"参数量 {count_parameters(m) / 1e6:.2f} M | {best_checkpoint_name('resnet', d)}")
    for arch in ("mobilenet_v3_large", "mobilenet_v3_small"):
        m = build_model(arch, num_classes=102)
        y = m(x)
        print(f"{describe_model(arch)}: 输出 {tuple(y.shape)} | "
              f"参数量 {count_parameters(m) / 1e6:.2f} M | {best_checkpoint_name(arch)}")
