# -*- coding: utf-8 -*-
"""INT8 深层诊断：全测试集准确率 + 关键张量量化参数 + 逐层输出数值范围。"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "saved_model")
FP32 = os.path.join(SM, "best_mobilenet_v3_large_float32.tflite")
INT8 = os.path.join(SM, "best_mobilenet_v3_large_int8.tflite")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, resize_size=256, input_size=224):
    img = img.resize((resize_size, resize_size), Image.BILINEAR)
    x = (resize_size - input_size) // 2
    img = img.crop((x, x, x + input_size, x + input_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr[None, ...]

def load_test_paths():
    import scipy.io as sio
    base = os.path.join(ROOT, "102flowers")
    labels = sio.loadmat(os.path.join(base, "imagelabels.mat"))["labels"].flatten()
    setid = sio.loadmat(os.path.join(base, "setid.mat"))
    idx = setid["tstid"].flatten().astype(np.int64) - 1
    return [os.path.join(base, "jpg", "image_%05d.jpg" % (i + 1)) for i in idx], labels[idx].astype(np.int64) - 1

def full_test(path, paths, targets, desc):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    correct = 0
    N = 400  # 先跑 400 张看量级
    for i in range(N):
        x = preprocess(Image.open(paths[i]).convert('RGB'))
        if inp['dtype'] == np.int8:
            s, z = inp['quantization']
            x = np.clip(np.round(x/s)+z, -128, 127).astype(np.int8)
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out['index'])[0]))
        correct += (pred == int(targets[i]))
    print(f"  {desc}: {correct}/{N} = {correct/N*100:.2f}%")

def quant_snapshot(path, paths):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    tensors = interp.get_tensor_details()
    print(f"\n  {os.path.basename(path)} 张量数={len(tensors)}")
    # 找卷积/全连接输出张量（按 name 含关键字），打印量化 scale 范围
    interesting = [t for t in tensors if t['name'] not in ('',) ]
    q = [t for t in tensors if (t['quantization_parameters'].get('scales') is not None and np.asarray(t['quantization_parameters'].get('scales')).size > 0)]
    scales = [t['quantization_parameters']['scales'][0] for t in q if np.asarray(t['quantization_parameters']['scales']).size > 0]
    print(f"  已量化张量: {len(q)} 个 | scale 范围 [{min(scales):.4g}, {max(scales):.4g}]")
    # 输出张量与最后几个全连接层
    out_name = interp.get_output_details()[0]['name']
    for t in q:
        qp = t['quantization_parameters']
        nm = t['name'].lower()
        if 'fc' in nm or 'classifier' in nm or 'dense' in nm or t['name'] == out_name:
            print(f"    {t['name'][:70]:70s} shape={str(t['shape']):22s} scale={qp['scales'][0]:.6g} zp={qp['zero_points'][0]}")
    print(f"    OUTPUT[{out_name}]: scale={interp.get_output_details()[0]['quantization_parameters']}")

def main():
    paths, targets = load_test_paths()
    print("===== 全测试集抽样准确率（400张）=====")
    full_test(FP32, paths, targets, "FP32")
    full_test(INT8, paths, targets, "INT8")

    print("\n===== 量化参数快照 =====")
    quant_snapshot(INT8, paths)

if __name__ == "__main__":
    main()
