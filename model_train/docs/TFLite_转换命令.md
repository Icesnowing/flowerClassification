# onnx2tf → SavedModel → TFLite INT8 · 转换命令（可执行）

> 前置：`checkpoints/best_mobilenet_ts.onnx`（16.5MB，已校验 92.6%）已就绪。
> 目标：转换成 `flower_mobilenet_int8.tflite`（≤5MB），供安卓 TFLite 部署。
> 以下命令为可直接复制的完整流程，含环境搭建（因为本机尚未装 TF/onnx2tf）。

---

## 0. 关键结论：只执行命令能完成转换吗？

**能，但有前提：命令必须包含"建环境 + 装依赖"这一步。** 单独复制转换那几条命令是跑不通的，因为：

1. **本机没有可用的 TensorFlow 环境**（Python 3.13 装不了 TF；3.12 也已确认未装 TF/onnx2tf）。
2. onnx2tf、TensorFlow、onnx 都需先安装。

所以下面给的是**完整可执行流程**（含环境准备 → 转换 → 量化 → 校验），照顺序跑即可。若你的机器之后已装好环境，可直接从「第 3 节」开始。

**本机可用 Python 版本**：`C:\Users\23896\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe`（Python 3.12.14，满足 TF/onnx2tf 要求）。

---

## 1. 建虚拟环境 + 装依赖（一次性）

在项目根目录 `E:\AI\Pytorch\flowerClassification` 下操作：

```powershell
# 用 Python 3.12 创建独立虚拟环境（避免污染训练环境）
C:\Users\23896\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe -m venv .venv-tf

# 激活（PowerShell）
.\.venv-tf\Scripts\Activate.ps1

# 安装依赖（CPU 版 TensorFlow 即可，onnx2tf 自动带 onnx）
E:\AI\Pytorch\flowerClassification\.venv-tf\Scripts\python.exe -m pip install --upgrade pip
 pip install -U "onnx2tf[tensorflow]"
```

> 说明：
> - `tensorflow-cpu` 为 CPU 版（转换不需要 GPU）；若装 `tensorflow` 全量版也可。
> - onnx2tf 会自动安装其依赖（onnx、onnxruntime 等）。
> - 首次安装较慢（TF 约数百 MB），属正常。

---

## 2. 确认输入（已有）

```powershell
# 确认 ONNX 存在（应约 16.5MB）
Get-Item checkpoints\best_mobilenet_ts.onnx
```

---

## 3. onnx2tf 转 SavedModel

```powershell
# 输出到 saved_model 目录；固定输入形状，去掉动态 batch（量化友好）
onnx2tf -i checkpoints/best_mobilenet_v3_large.onnx -o saved_model --overwrite_input_shape "input:1,3,224,224" --output_signaturedefs --tflite_backend tf_converter
```

> 关键参数说明：
> - `-o saved_model`：输出目录。
> - `--overwrite_input_shape "input:1,3,224,224"`：输入形状是NCHW。
> - `--output_signaturedefs`：生成签名，便于后续 TFLiteConverter 加载。
> - 若报 shape 名不匹配，先用 `python -c "import onnx; m=onnx.load('checkpoints/best_mobilenet_ts.onnx'); print(m.graph.input[0].name)"` 确认输入名。

**产出**：`saved_model/` 目录（含 `saved_model.pb` + `variables/`）。

产物：

![image-20260923145748964](C:\Users\23896\AppData\Roaming\Typora\typora-user-images\image-20260923145748964.png)



---



---

## 4. FP32 精度验证（门禁：Top-1 ≥ 90%）

```powershell
# 用 TFLite Interpreter 完整跑测试集，确认转换没破坏精度
python -c "
import numpy as np, tensorflow as tf
from PIL import Image
import sys, os
sys.path.insert(0, '.')
import data_utils

interp = tf.lite.Interpreter(model_path='checkpoints/flower_mobilenet_fp32.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]

paths, targets = data_utils.load_split('.', 'test')
rng = np.random.default_rng(0)
idx = rng.choice(len(paths), size=200, replace=False)
tf_img = data_utils.get_transform(train=False)

ok = 0
for i in idx:
    img = Image.open(paths[i]).convert('RGB')
    x = tf_img(img).unsqueeze(0).numpy()          # (1,3,224,224) NCHW 归一化
    x = x.transpose(0,2,3,1)                      # -> NHWC (1,224,224,3)
    interp.set_tensor(inp['index'], x.astype(np.float32))
    interp.invoke()
    pred = np.argmax(interp.get_tensor(out['index'])[0])
    if pred == targets[i]: ok += 1
print(f'FP32 TFLite Top-1: {ok/len(idx)*100:.2f}%')
"
```

> **关键**：FP32 TFLite 输入是 **NHWC**，且这里手动做了 `Normalize`（因为 `get_transform(train=False)` 含 Normalize）。TFLite 模型是否已含归一化取决于第 3 步转换，需要实际验证后确定——这正是「归一化位置」要确认的点。

**门禁**：若 Top-1 < 90%，说明 ONNX→TF 转换有算子问题，先解决再量化。

---

## 6. INT8 量化（含代表数据集）

```powershell
python -c "
import numpy as np, tensorflow as tf
from PIL import Image
import sys, os
sys.path.insert(0, '.')
import data_utils

# 代表数据集：从训练/验证集抽样（勿用测试集）
paths, _ = data_utils.load_split('.', 'val')
rng = np.random.default_rng(42)
idx = rng.choice(len(paths), size=150, replace=False)

def representative_dataset():
    tf_img = data_utils.get_transform(train=False)
    for i in idx:
        img = Image.open(paths[i]).convert('RGB')
        x = tf_img(img).unsqueeze(0).numpy().transpose(0,2,3,1)  # NHWC
        yield [x.astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_saved_model('saved_model')
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_dataset
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.uint8   # 或 tf.float32
conv.inference_output_type = tf.uint8
tflite_model = conv.convert()
open('checkpoints/flower_mobilenet_int8.tflite','wb').write(tflite_model)
import os
print('INT8 tflite 生成，大小', round(os.path.getsize('checkpoints/flower_mobilenet_int8.tflite')/1048576,2), 'MB')
"
```

**产出**：`checkpoints/flower_mobilenet_int8.tflite`（目标 ≤5MB）。

---

## 7. INT8 精度验证（最终门禁：Top-1 ≥ 90%）

```powershell
python -c "
import numpy as np, tensorflow as tf
from PIL import Image
import sys
sys.path.insert(0, '.')
import data_utils

interp = tf.lite.Interpreter(model_path='checkpoints/flower_mobilenet_int8.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
print('输入 type:', inp['dtype'], 'shape:', inp['shape'])
print('输出 type:', out['dtype'], 'shape:', out['shape'])

paths, targets = data_utils.load_split('.', 'test')
rng = np.random.default_rng(0)
idx = rng.choice(len(paths), size=500, replace=False)
tf_img = data_utils.get_transform(train=False)

ok = 0
for i in idx:
    img = Image.open(paths[i]).convert('RGB')
    x = tf_img(img).unsqueeze(0).numpy().transpose(0,2,3,1).astype(np.float32)
    if inp['dtype'] == np.uint8:
        x = (x * 255).astype(np.uint8)   # 若输入是 uint8，需反归一化到 0-255
    interp.set_tensor(inp['index'], x)
    interp.invoke()
    pred = np.argmax(interp.get_tensor(out['index'])[0])
    if pred == targets[i]: ok += 1
print(f'INT8 TFLite Top-1: {ok/len(idx)*100:.2f}%')
"
```

**验收**：Top-1 ≥ 90%（相对 FP32 92.62% 损失 ≤2%）。

---

## 8. 产出物清单

| 文件 | 说明 |
|---|---|
| `saved_model/` | TensorFlow 中间产物（可删） |
| `checkpoints/flower_mobilenet_fp32.tflite` | FP32 验证用（约 16MB） |
| `checkpoints/flower_mobilenet_int8.tflite` | **部署用（≤5MB）** |
| `labels.txt` | 102 行类别标签（与输出索引对齐，第 0 行=类1） |

---

## 9. 常见问题

| 问题 | 处理 |
|---|---|
| 输入名不是 `input` | 用 onnx 读取实际输入名，改 `--overwrite_input_shape` 里的名字 |
| onnx2tf 报算子不支持 | 确认是最新版 onnx2tf；hard-swish 一般支持良好 |
| INT8 精度仍掉点 | 检查代表数据集预处理（是否归一化、是否用测试集）；或 `inference_input_type=tf.float32` 只量化权重 |
| TF 安装慢 | 首次下载约数百 MB，耐心等待 |
