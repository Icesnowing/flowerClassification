# -*- coding: utf-8 -*-
"""临时验证：对比 FP32 与 INT8 量化模型在测试集上的 Top-1 精度。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import onnxruntime as ort
from PIL import Image
import data_utils

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP32 = os.path.join(DATA_ROOT, "checkpoints", "best_mobilenet_v3_large.onnx")
INT8 = os.path.join(DATA_ROOT, "checkpoints", "flower_dyn_int8.onnx")

paths, targets = data_utils.load_split(DATA_ROOT, "test")
# 随机抽 500 张做对比（全量 6149 张在 CPU 上过慢；500 张统计足够可靠）
rng = np.random.default_rng(0)
idx = rng.choice(len(paths), size=500, replace=False)
paths = [paths[i] for i in idx]
targets = [targets[i] for i in idx]
transform = data_utils.get_transform(train=False)

sess_fp32 = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])
in_name = sess_fp32.get_inputs()[0].name

correct = {"fp32": 0, "int8": 0}
agree = 0
total = 0
for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    x = transform(img).unsqueeze(0).numpy()
    t = int(targets[i])
    o1 = sess_fp32.run(None, {in_name: x})[0].argmax(axis=1)[0]
    o2 = sess_int8.run(None, {in_name: x})[0].argmax(axis=1)[0]
    if o1 == t:
        correct["fp32"] += 1
    if o2 == t:
        correct["int8"] += 1
    if o1 == o2:
        agree += 1
    total += 1

print(f"测试集样本数: {total}")
print(f"FP32 Top-1: {correct['fp32']/total*100:.2f}%")
print(f"INT8 Top-1: {correct['int8']/total*100:.2f}%")
print(f"INT8 相对 FP32 损失: {(correct['fp32']-correct['int8'])/total*100:.2f}%")
print(f"argmax 一致率: {agree/total*100:.2f}%")
