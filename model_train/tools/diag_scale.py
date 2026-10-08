# -*- coding: utf-8 -*-
"""打印 INT8 模型所有量化张量的 scale，定位退化层。"""
import os
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INT8 = os.path.join(ROOT, "saved_model", "best_mobilenet_v3_large_int8.tflite")

interp = tf.lite.Interpreter(model_path=INT8)
interp.allocate_tensors()
tensors = interp.get_tensor_details()
q = []
for t in tensors:
    qp = t['quantization_parameters']
    sc = qp.get('scales')
    if sc is not None and np.asarray(sc).size > 0:
        q.append((t['name'], float(np.asarray(sc)[0]), int(qp['zero_points'][0])))
q.sort(key=lambda x: x[1])
print("=== 最小 scale 的 12 个张量（疑似退化层）===")
for n, s, z in q[:12]:
    print(f"  {n[:75]:75s} scale={s:.3e} zp={z}")
print("\n=== 最大 scale 的 8 个张量 ===")
for n, s, z in q[-8:]:
    print(f"  {n[:75]:75s} scale={s:.3e} zp={z}")
