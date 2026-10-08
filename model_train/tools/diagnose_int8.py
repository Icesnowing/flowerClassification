# -*- coding: utf-8 -*-
"""INT8 量化崩溃诊断脚本：对比 fp32/int8 输出，检查算子类型与输入量化参数。"""
import sys, os, glob
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, resize_size=256, input_size=224):
    """Resize -> CenterCrop -> ToTensor -> Normalize(ImageNet)，返回 NHWC float32 单张。"""
    img = img.resize((resize_size, resize_size), Image.BILINEAR)
    x = (resize_size - input_size) // 2
    img = img.crop((x, x, x + input_size, x + input_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    arr = (arr - MEAN) / STD                                  # HWC 归一化
    return arr[None, ...]                                     # (1,H,W,3)

def load_test_paths():
    import scipy.io as sio
    base = os.path.join(ROOT, "102flowers")
    labels = sio.loadmat(os.path.join(base, "imagelabels.mat"))["labels"].flatten()
    setid = sio.loadmat(os.path.join(base, "setid.mat"))
    idx = setid["tstid"].flatten().astype(np.int64) - 1
    paths = [os.path.join(base, "jpg", "image_%05d.jpg" % (i + 1)) for i in idx]
    targets = labels[idx].astype(np.int64) - 1
    return paths, targets
SM = os.path.join(ROOT, "saved_model")
FP32 = os.path.join(SM, "best_mobilenet_v3_large_float32.tflite")
FP16 = os.path.join(SM, "best_mobilenet_v3_large_float16.tflite")
INT8 = os.path.join(SM, "best_mobilenet_v3_large_int8.tflite")

def model_info(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    ops = interp._get_ops_details()
    print(f"  {os.path.basename(path)}")
    print(f"    input : dtype={inp['dtype']} shape={inp['shape']} quant={inp['quantization']}")
    print(f"    output: dtype={out['dtype']} shape={out['shape']} quant={out['quantization']}")
    from collections import Counter
    c = Counter(op['op_name'] for op in ops)
    print(f"    ops({len(ops)}): {dict(c)}")
    return interp

def run_one(interp, x):
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp['index'], x)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]['index'])[0]

def main():
    print("===== 模型结构信息 =====")
    for p in (FP32, FP16, INT8):
        model_info(p)

    paths, targets = load_test_paths()

    print("\n===== fp32 vs int8 输出对比（前 20 张测试图）=====")
    fp32_i = tf.lite.Interpreter(model_path=FP32); fp32_i.allocate_tensors()
    fp32_in = fp32_i.get_input_details()[0]
    int8_i = tf.lite.Interpreter(model_path=INT8); int8_i.allocate_tensors()
    int8_in = int8_i.get_input_details()[0]

    match = 0; diff_max = 0.0; diff_mean = 0.0
    for i in range(20):
        img = Image.open(paths[i]).convert('RGB')
        x = preprocess(img)
        if int8_in['dtype'] == np.int8:
            scale, zp = int8_in['quantization']
            xi = np.clip(np.round(x/scale)+zp, -128, 127).astype(np.int8)
            l_int8 = run_one(int8_i, xi)
        else:
            l_int8 = run_one(int8_i, x)
        l_fp32 = run_one(fp32_i, x)
        d = np.abs(l_fp32 - l_int8)
        diff_max = max(diff_max, d.max()); diff_mean += d.mean()
        p32 = np.argmax(l_fp32); p8 = np.argmax(l_int8)
        same = (p32 == p8); match += same
        print(f"  img{i:04d} label={targets[i]:3d} fp32top={p32:3d}({l_fp32[p32]:.3f}) int8top={p8:3d}({l_int8[p8]:.3f}) {'OK' if same else 'MISMATCH'}")
    print(f"\n  Top1 一致率: {match}/20 | max abs diff={diff_max:.4f} | mean abs diff={diff_mean/20:.4f}")

if __name__ == "__main__":
    main()
