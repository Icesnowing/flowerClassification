package com.example.flowerrecognition.util;

import java.util.Arrays;
import java.util.Locale;

/** 延迟样本的分位数（nearest-rank，与仪器化 benchmark 口径一致）。 */
public final class LatencyStats {
    private LatencyStats() {
    }

    public static long percentile(long[] samples, int percentile) {
        if (samples.length == 0) {
            throw new IllegalArgumentException("samples 不能为空");
        }
        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("percentile 须在 0..100");
        }
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        index = Math.max(0, Math.min(sorted.length - 1, index));
        return sorted[index];
    }

    public static String formatSegment(String name, long[] samplesMs) {
        return String.format(
                Locale.US,
                "%s: p50=%d p90=%d p95=%d p99=%d ms",
                name,
                percentile(samplesMs, 50),
                percentile(samplesMs, 90),
                percentile(samplesMs, 95),
                percentile(samplesMs, 99));
    }
}
