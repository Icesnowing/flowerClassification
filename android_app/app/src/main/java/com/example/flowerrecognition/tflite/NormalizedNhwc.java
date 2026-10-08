package com.example.flowerrecognition.tflite;

/**
 * 送进 {@link FlowerInterpreter#run} 的 float32 NHWC。
 * 只能由 {@link ImagePreprocessor#toNhwc} / {@link ImagePreprocessor#pixelsToNhwc} 生成，
 * 里面已经是 ImageNet 归一化结果。
 */
public final class NormalizedNhwc {
    public final float[] values;

    public NormalizedNhwc(float[] values) {
        if (values.length != ImagePreprocessor.INPUT_FLOATS) {
            throw new IllegalArgumentException(
                    "送进 run 的必须是归一化 NHWC float，长度 " + ImagePreprocessor.INPUT_FLOATS
                            + "，实际 " + values.length);
        }
        this.values = values;
    }
}
