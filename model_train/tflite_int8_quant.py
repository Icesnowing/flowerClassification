import tensorflow as tf
import numpy as np
import os
from PIL import Image
import data_utils

# ====== 配置区 ======
SAVED_MODEL_DIR = "saved_model"          # 你已有的 SavedModel 目录
OUTPUT_TFLITE = "saved_model/best_mobilenet_v3_large_int8.tflite"
NUM_CALIBRATION = 1000                     # 校准样本数，100~500 之间
INPUT_SIZE = (224, 224)
# ====================
paths, targets = data_utils.load_split('.', 'val')
rng = np.random.default_rng(0)
idx = rng.choice(len(paths), size=NUM_CALIBRATION, replace=False)

def representative_dataset():
    """
    生成器：每次 yield 一个 batch 的输入数据。
    这里使用目录中的真实图片进行校准。
    """
    tf_img = data_utils.get_transform(train=False)
    for i in idx:
        img =  Image.open(paths[i]).convert('RGB')
        x = tf_img(img).unsqueeze(0).numpy()          # (1,3,224,224) NCHW 归一化
        x = x.transpose(0,2,3,1)                      # -> NHWC (1,224,224,3)
        # 增加 batch 维度: (1, 224, 224, 3)
        yield [x.astype(np.float32)]

# 加载 SavedModel 并配置量化
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

# 启用默认优化（包含权重量化）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 设置代表性数据集（激活值校准）
converter.representative_dataset = representative_dataset

# 强制全整数量化：确保所有算子都是 int8 实现
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 输入输出也设为 int8（部署到 Edge TPU 等纯整数硬件时必须）
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

# 执行转换
tflite_int8_model = converter.convert()

# 保存
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_int8_model)

print(f"INT8 模型已保存至: {OUTPUT_TFLITE}")
print(f"文件大小: {os.path.getsize(OUTPUT_TFLITE) / 1024:.1f} KB")