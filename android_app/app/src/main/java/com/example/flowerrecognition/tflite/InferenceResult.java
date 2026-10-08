package com.example.flowerrecognition.tflite;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import android.util.Log;

/**
 * 一次推理的解析结果。
 * index 是 logits 的 argmax（并列取第一个）。top5 只用于显示概率，不改类别。
 * classId 等于 index + 1，只写日志，不能拿去取 LabelTable（查表用 index）。
 */
public final class InferenceResult {
    public static final int TOP_K = 5;
    public static final String LOG_TAG = "FlowerInterpreter";
    private static final String TAG = LOG_TAG;
    public final int index;
    public final String english;
    public final String chinese;
    public final List<TopPrediction> top5;

    public InferenceResult(int index, String english, String chinese, List<TopPrediction> top5) {
        if (index < 0 || index >= LabelTable.EXPECTED_SIZE) {
            throw new IllegalArgumentException(
                    "argmax 下标必须在 0–" + (LabelTable.EXPECTED_SIZE - 1) + "，实际 " + index);
        }
        if (top5.size() != TOP_K) {
            throw new IllegalArgumentException("Top-" + TOP_K + " 应为 " + TOP_K + " 条，实际 " + top5.size());
        }
        if (top5.get(0).index != index) {
            throw new IllegalArgumentException("Top-1 下标必须等于 argmax");
        }
        this.index = index;
        this.english = english;
        this.chinese = chinese;
        this.top5 = top5;
    }

    public int getClassId() {
        return index + 1;
    }

    /** Top-1 的 softmax 概率，用于展示与低置信度判断。 */
    public float top1Probability() {
        return top5.get(0).probability;
    }

    /**
     * 从未经 softmax 的 logits 解析类别。
     * 先取 argmax，再做数值稳定的 softmax，按概率从大到小取 TOP_K。
     * softmax 不改变大小顺序，因此 Top-1 下标与 argmax 相同。
     */
    public static InferenceResult fromLogits(float[] logits, LabelTable labels) {
        if (logits.length != LabelTable.EXPECTED_SIZE) {
            throw new IllegalArgumentException(
                    "logits 长度应为 " + LabelTable.EXPECTED_SIZE + "，实际 " + logits.length);
        }
        if (labels.size() != LabelTable.EXPECTED_SIZE) {
            throw new IllegalArgumentException(
                    "标签应为 " + LabelTable.EXPECTED_SIZE + " 类，实际 " + labels.size());
        }
        for (float logit : logits) {
            if (!Float.isFinite(logit)) {
                throw new IllegalArgumentException("logits 含有非有限值");
            }
        }


        float[] probabilities = softmax(logits);
        int index = argmax(probabilities);
        List<Integer> ranked = new ArrayList<>(logits.length);
        for (int i = 0; i < logits.length; i++) {
            ranked.add(i);
        }
        ranked.sort((a, b) -> {
            int cmp = Float.compare(probabilities[b], probabilities[a]);
            if (cmp != 0) {
                return cmp;
            }
            cmp = Float.compare(logits[b], logits[a]);
            if (cmp != 0) {
                return cmp;
            }
            return Integer.compare(a, b);
        });
        List<TopPrediction> top = new ArrayList<>(TOP_K);
        for (int k = 0; k < TOP_K; k++) {
            int i = ranked.get(k);
            FlowerLabel label = labels.get(i);
            top.add(new TopPrediction(i, label.english, label.chinese, probabilities[i]));
        }
        FlowerLabel topLabel = labels.get(index);
        return new InferenceResult(index, topLabel.english, topLabel.chinese, top);
    }

    /** 最大值的下标。并列时保留先出现的那个。 */
    public static int argmax(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * 数值稳定的 softmax：先减去最大值再 exp，避免溢出。
     * 最大值那一项的 exp 是 1，分母至少为 1。
     */
    public static float[] softmax(float[] logits) {
        float maxLogit = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
            }
        }
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - maxLogit);
        }
        float sum = 0f;
        for (float value : exps) {
            sum += value;
        }
        for (int i = 0; i < exps.length; i++) {
            exps[i] = exps[i] / sum;
        }
        return exps;
    }

    public static String formatProbability(float probability) {
        return String.format(Locale.US, "%.4f", probability);
    }
}
