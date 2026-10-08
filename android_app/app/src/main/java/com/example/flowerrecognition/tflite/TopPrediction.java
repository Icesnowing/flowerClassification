package com.example.flowerrecognition.tflite;

/**
 * 一条按概率排序的类别。
 * index 是 logits / labels 的下标（0–13），不是 1-based 的 classId。
 */
public final class TopPrediction {
    public final int index;
    public final String english;
    public final String chinese;
    public final float probability;

    public TopPrediction(int index, String english, String chinese, float probability) {
        this.index = index;
        this.english = english;
        this.chinese = chinese;
        this.probability = probability;
    }

    /** 展示用，保留 4 位小数。对齐时比较的是 index，不是这一位之后的误差。 */
    public String probabilityText() {
        return InferenceResult.formatProbability(probability);
    }
}
