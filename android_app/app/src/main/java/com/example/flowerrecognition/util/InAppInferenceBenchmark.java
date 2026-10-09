package com.example.flowerrecognition.util;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.util.Log;

import com.example.flowerrecognition.tflite.FlowerInterpreter;
import com.example.flowerrecognition.tflite.ImagePreprocessor;
import com.example.flowerrecognition.tflite.InferenceScheduler;
import com.example.flowerrecognition.tflite.NormalizedNhwc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * Release 包内嵌推理压测：固定 assets 图，预热 {@link #WARMUP} 次，采样 {@link #RUNS} 次，
 * 统计冷启动（{@link FlowerInterpreter#load} + 首次 {@link FlowerInterpreter#run}）、
 * 预处理 / TFLite 推理 / 结果解析（{@link FlowerInterpreter#recognizeLogits}）及合计的 p50、p90、p95、p99。
 */
public final class InAppInferenceBenchmark {
    public static final String LOG_TAG = "InferenceBenchmark";
    public static final String REPORT_FILE_NAME = "benchmark_report.txt";
    /** 与 PC {@code val_tflite.py} 对齐时可替换为同一张图。 */
    public static final String BENCHMARK_ASSET = "benchmark.png";
    public static final int WARMUP = 30;
    public static final int RUNS = 300;

    private InAppInferenceBenchmark() {
    }

    /** 冷启动分段：模型加载 + Interpreter/张量初始化，以及首次 invoke（含 arena、XNNPACK 等）。 */
    public static final class ColdStartupTiming {
        public final long modelLoadMs;
        public final long firstInvokeMs;
        public final long coldStartupTotalMs;

        ColdStartupTiming(long modelLoadMs, long firstInvokeMs) {
            this.modelLoadMs = modelLoadMs;
            this.firstInvokeMs = firstInvokeMs;
            this.coldStartupTotalMs = modelLoadMs + firstInvokeMs;
        }
    }

    public static File reportFile(Context context) {
        return new File(context.getApplicationContext().getFilesDir(), REPORT_FILE_NAME);
    }

    /**
     * 从 assets 解码 benchmark 图。请将 {@value #BENCHMARK_ASSET} 放在 {@code src/main/assets/}。
     */
    public static Bitmap loadBenchmarkBitmap(Context context) throws IOException {
        try (InputStream input = context.getAssets().open(BENCHMARK_ASSET)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                throw new IOException("无法解码 assets/" + BENCHMARK_ASSET);
            }
            return bitmap;
        }
    }

    /**
     * 在推理线程上新建 Interpreter，测量冷启动至首次 {@link FlowerInterpreter#run} 结束。
     * {@code input} 须为固定 benchmark 图预处理结果，仅用于触发首次 invoke，不计入启动耗时。
     */
    public static ColdStartupTiming measureColdStartup(Context context, NormalizedNhwc input)
            throws IOException {
        long loadStart = System.nanoTime();
        FlowerInterpreter cold = FlowerInterpreter.load(context.getApplicationContext());
        long modelLoadMs = (System.nanoTime() - loadStart) / 1_000_000L;
        try {
            long invokeStart = System.nanoTime();
            cold.run(input);
            long firstInvokeMs = (System.nanoTime() - invokeStart) / 1_000_000L;
            return new ColdStartupTiming(modelLoadMs, firstInvokeMs);
        } finally {
            cold.close();
        }
    }

    /**
     * 在推理线程调用。{@code progress} 可为 null；会传入简短进度文案。
     */
    public static String run(
            Context context,
            FlowerInterpreter model,
            Bitmap decoded,
            long decodeOnceMs,
            Consumer<String> progress) throws IOException {
        if (progress != null) {
            progress.accept("冷启动测量…");
        }
        NormalizedNhwc coldInvokeInput = ImagePreprocessor.toNhwc(decoded);
        ColdStartupTiming coldStartup = measureColdStartup(context, coldInvokeInput);

        for (int i = 0; i < WARMUP; i++) {
            if (progress != null && (i == 0 || i == WARMUP - 1 || (i + 1) % 10 == 0)) {
                progress.accept(String.format(Locale.US, "预热 %d / %d", i + 1, WARMUP));
            }
            NormalizedNhwc warmup = ImagePreprocessor.toNhwc(decoded);
            float[] warmupLogits = model.run(warmup);
            model.recognizeLogits(warmupLogits, warmup);
        }

        long[] preprocessMs = new long[RUNS];
        long[] runMs = new long[RUNS];
        long[] postprocessMs = new long[RUNS];
        long[] totalMs = new long[RUNS];
        for (int i = 0; i < RUNS; i++) {
            if (progress != null && (i == 0 || i == RUNS - 1 || (i + 1) % 10 == 0)) {
                progress.accept(String.format(Locale.US, "采样 %d / %d", i + 1, RUNS));
            }
            long t0 = System.nanoTime();
            NormalizedNhwc input = ImagePreprocessor.toNhwc(decoded);
            preprocessMs[i] = (System.nanoTime() - t0) / 1_000_000L;

            long t1 = System.nanoTime();
            float[] logits = model.run(input);
            runMs[i] = (System.nanoTime() - t1) / 1_000_000L;

            long t2 = System.nanoTime();
            model.recognizeLogits(logits, input);
            postprocessMs[i] = (System.nanoTime() - t2) / 1_000_000L;
            totalMs[i] = preprocessMs[i] + runMs[i] + postprocessMs[i];
        }
        long warmInvokeP50 = LatencyStats.percentile(runMs, 50);
        long deltaColdStartup = coldStartup.coldStartupTotalMs - warmInvokeP50;
        String report = String.format(
                Locale.US,
                "device=%s sdk=%d warmup=%d runs=%d thread=%s asset=%s%n"
                        + "decode_once=%d ms (不计入下方分位)%n"
                        + "cold_startup_total_ms=%d%n"
                        + "code_startup_total_minus_p50=%d ms%n"
                        + "%s%n"
                        + "%s%n"
                        + "%s%n"
                        + "%s%n",
                Build.MODEL,
                Build.VERSION.SDK_INT,
                WARMUP,
                RUNS,
                InferenceScheduler.THREAD_NAME,
                BENCHMARK_ASSET,
                decodeOnceMs,
                coldStartup.coldStartupTotalMs,
                deltaColdStartup,
                LatencyStats.formatSegment("preprocess", preprocessMs),
                LatencyStats.formatSegment("inference (tflite run)", runMs),
                LatencyStats.formatSegment("postprocess (recognizeLogits)", postprocessMs),
                LatencyStats.formatSegment("total(per-iter)", totalMs));
        Log.i(LOG_TAG, report.replace('\n', '|'));
        return report;
    }

    public static void writeReport(Context context, String report) {
        File file = reportFile(context);
        synchronized (InAppInferenceBenchmark.class) {
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(file, false), StandardCharsets.UTF_8))) {
                writer.write(report);
            } catch (IOException e) {
                Log.w(LOG_TAG, "无法写入 " + file.getAbsolutePath(), e);
            }
        }
        Log.i(LOG_TAG, "报告已写入 " + file.getAbsolutePath());
    }
}
