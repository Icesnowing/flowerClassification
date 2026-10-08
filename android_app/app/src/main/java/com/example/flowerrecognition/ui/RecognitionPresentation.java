package com.example.flowerrecognition.ui;

import com.example.flowerrecognition.util.IntPair;

import java.util.Locale;

/**
 * 第 6 节的展示规则。
 * 低置信度看的是 Top-1 softmax，比较的是原始概率，不是四舍五入后的百分比。
 */
public final class RecognitionPresentation {
    /** 方案：Top-1 softmax < 50% 才提示无法确定。等于 50% 仍展示为可确定。 */
    public static final float CONFIDENCE_THRESHOLD = 0.5f;

    public static final String UNCERTAIN_MESSAGE = "无法确定，请换一张更清晰、主体更完整的照片";

    /** 界面上的原图长边上限。推理仍使用解码后的 Bitmap，不看这张预览。 */
    public static final int PREVIEW_MAX_SIDE = 1280;

    private RecognitionPresentation() {
    }

    public static boolean isUncertain(float top1Probability) {
        return top1Probability < CONFIDENCE_THRESHOLD;
    }

    public static String confidenceText(float probability) {
        return String.format(Locale.US, "%.2f%%", probability * 100.0);
    }

    /**
     * 预览长边不超过 maxSide，另一边按整数比例缩小。
     * 不够大的图保持原尺寸，避免把测试图再缩一遍。
     */
    public static IntPair previewSize(int width, int height, int maxSide) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("图片尺寸无效：" + width + "x" + height);
        }
        if (maxSide <= 0) {
            throw new IllegalArgumentException("预览长边必须大于 0");
        }
        int longSide = Math.max(width, height);
        if (longSide <= maxSide) {
            return new IntPair(width, height);
        }
        int scaledWidth = Math.max((int) ((long) width * maxSide / longSide), 1);
        int scaledHeight = Math.max((int) ((long) height * maxSide / longSide), 1);
        return new IntPair(scaledWidth, scaledHeight);
    }

    public static IntPair previewSize(int width, int height) {
        return previewSize(width, height, PREVIEW_MAX_SIDE);
    }
}
