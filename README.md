# 花卉图像分类（PyTorch → TFLite → Android）

端到端项目：在 **Oxford 102 Flowers** 上训练/微调 CNN（ResNet、torchvision MobileNetV3），可选在 **Flowers14** 上迁移到 14 类，并导出 **TensorFlow Lite** 供 **Android** 真机识别。

| 模块 | 路径 | 说明 |
|---|---|---|
| 训练与导出 | [`model_train/`](model_train/) | 数据加载、训练、测试、ONNX/TFLite、精度校验 |
| 安卓 App | [`android_app/`](android_app/) | 相机/相册选图，TFLite 推理，14 类中英标签 |
| 详细文档 | [`model_train/docs/`](model_train/docs/) | TFLite 转换 |

> **运行 Python 脚本时请先进入 `model_train` 目录**（下文命令均默认在该目录下执行）。

---

## 环境

- **推荐 Python**：3.10 – 3.13（训练脚本）；**TFLite 转换**建议单独使用 **Python 3.12** + `.venv-tf`（TensorFlow / onnx2tf，见 [`model_train/docs/TFLite_转换命令.md`](model_train/docs/TFLite_转换命令.md)）
- 训练依赖：[`model_train/requirements.txt`](model_train/requirements.txt)；可选 [`model_train/environment.yml`](model_train/environment.yml)（conda）
- 无 GPU 时自动使用 CPU；安装 CUDA 版 PyTorch 后脚本自动用 GPU，无需改代码

### pip（训练环境）

```powershell
cd model_train
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt

# 冒烟：1 轮、少量样本
python train.py --limit-train 32 --limit-val 32 --epochs 1
```

### CUDA（可选）

先按本机 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配的 `torch` / `torchvision`，再执行 `pip install -r requirements.txt`。

---

## 数据布局

### Oxford 102（102 类，官方 mat 划分）

将数据放在 `model_train/102flowers/`（或通过 `--data-root` 指定含该子目录的路径）：

```
model_train/
  102flowers/
    jpg/image_00001.jpg ~ image_08189.jpg
    imagelabels.mat        # 标签 1–102
    setid.mat              # trnid(1020) / valid(1020) / tstid(6149)
```

### Flowers14（14 类，ImageFolder）

仓库内示例路径：`model_train/flower14/`（可按 [`model_train/flower14_files/README.md`](model_train/flower14_files/README.md) 从 ModelScope 等来源获取完整数据）：

```
model_train/flower14/
  train/<类别名>/*.jpg     # 约 1.36 万张
  val/<类别名>/*.jpg       # 官方验证 98 张（每类 7 张）
```

14 类与 Oxford 102 中部分类别同源，适合在 102 类预训练主干上做 **14 类迁移** 或 **MobileNet 从头/微调**（见下文）。

---

## 快速开始

### 1. Oxford 102 — 训练与测试

```powershell
cd model_train

# ResNet18 + ImageNet 预训练微调（生产常用）
python train.py --pretrained --depth 34 --epochs 30

# MobileNetV3-Large 微调（移动端/安卓路线常用）
python train.py --arch mobilenet_v3_large --pretrained --epochs 15

# 测试（checkpoint 名随网络变化，如 best_resnet34.pth / best_mobilenet_v3_large.pth）
python test.py --checkpoint checkpoints\best_mobilenet_v3_large.pth

# 单张图
python test.py --checkpoint checkpoints\best_mobilenet_v3_large.pth `
    --single-image 102flowers\jpg\image_00001.jpg

# 导出 ONNX（部署中间格式）
python export.py --checkpoint checkpoints\best_mobilenet_v3_large.pth
```

最佳权重文件名规则：`best_<网络标签>.pth`（例如 `best_resnet18.pth`、`best_mobilenet_v3_large.pth`），旁路生成对应的 `*_config.json` 供 `test.py` / `export.py` 自动读取。

### 2. Flowers14 — 两条路线

| 路线 | 脚本 | 适用场景 |
|---|---|---|
| **A：102 主干 → 14 类头** | `train_folder.py` | 已有 102 类 checkpoint，只识别 14 种花 |
| **B：MobileNet 直接在 14 类上训** | `train_mobilenet_flower14.py` | 不依赖 102 权重，轻量模型 + 安卓部署 |

```powershell
cd model_train

# 路线 A：从 102 类 best 权重迁移（示例 ResNet-34）
python train_folder.py --data-root flower14 --checkpoint checkpoints\best_resnet34.pth `
    --depth 34 --epochs 20 --save-dir checkpoints_ft14

# 路线 B：MobileNetV3-Large + flower14
python train_mobilenet_flower14.py --data-root flower14 --pretrained --epochs 30 `
    --save-dir checkpoints_flower14_mnv3

# 14 类测试（官方 val/ 作测试集）
python test_folder.py --data-root flower14 --checkpoint checkpoints_ft14\best_resnet34.pth
python test_folder.py --data-root flower14 --checkpoint checkpoints_flower14_mnv3\best_mobilenet_v3_large.pth
```

数据划分说明：`train/` 按类分层划出约 10% 作训练内验证（选 best / early stopping）；`val/` **不参与选模型**，仅用于 `test_folder.py` 评估。

### 3. TFLite 与安卓

转换链（细节与命令见 [`model_train/docs/TFLite_转换命令.md`](model_train/docs/TFLite_转换命令.md)）：

```
PyTorch checkpoint → export.py (ONNX) → onnx2tf (SavedModel) → TFLiteConverter
```

- **推荐部署**：动态量化 TFLite（权重 INT8、激活 float），精度损失约 1 个点以内；全 INT8 PTQ 在本项目上易崩塌。
- 校验：`val_tflite.py`（Oxford 102 或 flower14）、`val_tflite_int8.py`。

**Android App**（[`android_app/`](android_app/)）：

- 运行时模型：`app/src/main/assets/model_flower14.tflite`（需自行从 14 类训练产物转换后拷贝，体积较大通常不入库）
- 标签：`labels_flower14.txt`（英文类名 + 中文展示名）
- 输入：224×224 RGB，ImageNet 归一化 float32 **NHWC**；输出 14 维 **logits**（与 `FlowerInterpreter.java` 一致）

```powershell
cd android_app
.\gradlew.bat assembleRelease
# 安装：adb install -r app\build\outputs\apk\release\app-release.apk
```

---

## 目录与脚本说明

| 路径 | 作用 |
|---|---|
| `model_train/model.py` | ResNet（18/34/50，从零实现）+ MobileNetV3；`build_model(..., pretrained=True)` 加载 ImageNet 权重 |
| `model_train/data_utils.py` | Oxford 102 的 mat 划分；transform / 归一化常量 |
| `model_train/train.py` | Oxford 102 训练：SGD/AdamW、预热、余弦退火、AMP、early stopping、TensorBoard/W&B |
| `model_train/test.py` | Oxford 102 测试：Top-1/Top-5、per-class CSV、单图、误判导出 |
| `model_train/train_folder.py` | 102 checkpoint → 14 类 ImageFolder 微调 |
| `model_train/train_mobilenet_flower14.py` | flower14 上训练 MobileNetV3-Large |
| `model_train/test_folder.py` | flower14 测试（含单图、误判导出等） |
| `model_train/export.py` | checkpoint → ONNX + onnxruntime 数值校验 |
| `model_train/tflite_int8_quant.py` | SavedModel → TFLite 量化（需 TF 环境） |
| `model_train/val_tflite.py` / `val_tflite_int8.py` | TFLite 在测试集/单图上的精度核对 |
| `model_train/logging_utils.py` | 控制台 + 文件日志 |
| `model_train/experiment_tracker.py` | TensorBoard / W&B 封装 |
| `model_train/tools/` | INT8/ONNX 诊断与辅助脚本 |
| `model_train/checkpoints/`、`checkpoints_ft14/` 等 | 训练产物（运行时生成） |
| `model_train/saved_model/` | onnx2tf 输出的 SavedModel / `.tflite` |
| `model_train/runs/` | TensorBoard 事件文件 |
| `android_app/app/src/main/java/.../tflite/` | 预处理、Interpreter、调度与结果解析 |

---

## 常用训练参数（`train.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-root` | `model_train` 目录 | 须含 `102flowers/` |
| `--arch` | `resnet` | `resnet` / `mobilenet_v3_large` / `mobilenet_v3_small` |
| `--depth` | 18 | 仅 ResNet：18 / 34 / 50 |
| `--pretrained` | 关 | ImageNet 预训练微调（首次需联网下载权重） |
| `--epochs` | 10 | 微调常用 15–30 |
| `--batch-size` | 8 | 可按显存调整 |
| `--optimizer` | `adamw` | 或 `sgd`（带 warmup + 余弦退火） |
| `--early-stop-patience` | 0（关） | 验证 Top-1 连续 N 轮无提升则停训 |
| `--oversample` | 无 | 训练集指定类别过采样，如 `51:5` |
| `--resume` | 无 | 断点续训 |
| `--tensorboard` / `--no-tensorboard` | 开 | 日志在 `runs/<run-name>/` |
| `--wandb-project` | 无 | 传入即启用 W&B |
| `--limit-train` / `--limit-val` | 无 | 调试子集 |

测试 `test.py` / `test_folder.py`：`--checkpoint` 必填；配置优先从 checkpoint 内 `args` 与 `*_config.json` 恢复；不可信 checkpoint 勿使用 `--trust-checkpoint`。

---

## 训练产物

- `best_<网络>.pth`：验证集 Top-1 最高权重（含 optimizer 等，可 `--resume`）
- `best_<网络>_config.json`（或历史命名 `best_config.json`）：`arch` / `depth` / `num_classes` / 输入尺寸 / ImageNet 归一化
- `last_model.pth`、`train_history.csv`、`train.log` / `test.log`
- `test_results.csv`、可选 `misclassified.csv`
- `*.onnx`：`export.py` 生成，动态 batch，metadata 含预处理说明
- `saved_model/*.tflite`：安卓与 `val_tflite.py` 使用

---

## ONNX 部署约定（摘要）

| 项 | 值 |
|---|---|
| 输入 | `input`，`(N, 3, 224, 224)` float32 **NCHW RGB** |
| 预处理 | `Resize(256)` → `CenterCrop(224)` → `[0,1]` → ImageNet `Normalize`（以 `*_config.json` 为准） |
| 输出 | `logits`，`(N, num_classes)`，**未 softmax**；Oxford 102 类别 id = `argmax + 1`（1–102）；flower14 为 0–13 索引 |

---

## 实验追踪

```powershell
cd model_train
python train.py --pretrained --run-name ft_mnv3_large
tensorboard --logdir runs
```

可选 W&B：`pip install wandb` → `wandb login` → `python train.py --wandb-project flowers`。

---

## 常见问题

- **`No module named 'torch'`**：在 `model_train` 下激活虚拟环境并 `pip install -r requirements.txt`。
- **找不到 `102flowers`**：下载 Oxford 102 数据并放到 `model_train/102flowers/`，或设置 `--data-root`。
- **`--pretrained` 下载失败**：需联网；权重缓存在用户目录 `.cache/torch/hub/checkpoints`。
- **checkpoint 与 `--depth` / `--arch` 不一致**：测试/导出以 checkpoint 内保存的配置为准；续训须与上次一致。
- **安卓 APK 无模型或推理报错**：确认 `assets/model_flower14.tflite` 已放入且与 14 类训练/转换产物一致；标签文件与类顺序对齐 `class_names.txt`。
- **TFLite 全 INT8 精度暴跌**：预期现象，改用动态量化或 FP16。
