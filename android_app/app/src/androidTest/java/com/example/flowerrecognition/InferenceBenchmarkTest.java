package com.example.flowerrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.example.flowerrecognition.tflite.FlowerInterpreter;
import com.example.flowerrecognition.tflite.ImagePreprocessor;
import com.example.flowerrecognition.tflite.InferenceScheduler;
import com.example.flowerrecognition.tflite.NormalizedNhwc;

import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * 真机单张推理耗时（方式二）：固定图 → 预热 → 多次 toNhwc + run，输出 median / p90。
 * <p>
 * 运行（PowerShell 必须把 -P 参数加引号，否则 {@code -Pandroid} 会被当成开关）：
 * {@code .\gradlew connectedDebugAndroidTest "-Pandroid.testInstrumentationRunnerArguments.class=com.example.flowerrecognition.InferenceBenchmarkTest"}
 * 或只跑 androidTest：{@code .\gradlew connectedDebugAndroidTest}（会跑全部 instrumented test，目前含本类与 ExampleInstrumentedTest）。
 * Android Studio：对类 Run connected test（需 USB 真机）。
 * <p>
 * 指标口径与验收文档一致：预处理 {@link ImagePreprocessor#toNhwc} + {@link FlowerInterpreter#run}，
 * 不含相册解码。可将 {@code androidTest/assets/benchmark.jpg} 换成与 PC {@code val_tflite.py} 同图以便对齐 Top-1。
 */
@RunWith(AndroidJUnit4.class)
public class InferenceBenchmarkTest {
    private static final String TAG = "InferenceBenchmark";
    private static final String BENCHMARK_ASSET = "rose2.jpg";
    private static final int WARMUP = 10;
    private static final int RUNS = 30;

    @Test
    public void reportsPreprocessAndRunLatency() throws Exception {
        Assume.assumeFalse(
                "请在物理真机上运行；模拟器耗时不作验收依据",
                isProbablyEmulator());

        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

        long decodeStart = System.nanoTime();
        Bitmap decoded = loadBenchmarkBitmap(testContext);
        long decodeMs = (System.nanoTime() - decodeStart) / 1_000_000L;

        runOnTfliteThread(() -> {
            FlowerInterpreter model = FlowerInterpreter.load(appContext);
            try {
                for (int i = 0; i < WARMUP; i++) {
                    NormalizedNhwc input = ImagePreprocessor.toNhwc(decoded);
                    model.run(input);
                }

                long[] preprocessMs = new long[RUNS];
                long[] runMs = new long[RUNS];
                for (int i = 0; i < RUNS; i++) {
                    long t0 = System.nanoTime();
                    NormalizedNhwc input = ImagePreprocessor.toNhwc(decoded);
                    preprocessMs[i] = (System.nanoTime() - t0) / 1_000_000L;

                    long t1 = System.nanoTime();
                    model.run(input);
                    runMs[i] = (System.nanoTime() - t1) / 1_000_000L;
                }

                long preprocessMedian = percentile(preprocessMs, 50);
                long preprocessP90 = percentile(preprocessMs, 90);
                long runMedian = percentile(runMs, 50);
                long runP90 = percentile(runMs, 90);
                long totalMedian = preprocessMedian + runMedian;
                long totalP90 = preprocessP90 + runP90;

                String summary = String.format(
                        Locale.US,
                        "device=%s sdk=%d runs=%d warmup=%d | decode_once=%dms | "
                                + "preprocess median=%d p90=%d | run median=%d p90=%d | "
                                + "preprocess+run median=%d p90=%d (target <=50ms)",
                        Build.MODEL,
                        Build.VERSION.SDK_INT,
                        RUNS,
                        WARMUP,
                        decodeMs,
                        preprocessMedian,
                        preprocessP90,
                        runMedian,
                        runP90,
                        totalMedian,
                        totalP90);
                Log.i(TAG, summary);
            } finally {
                model.close();
                if (!decoded.isRecycled()) {
                    decoded.recycle();
                }
            }
            return null;
        });
    }

    private static Bitmap loadBenchmarkBitmap(Context testContext) throws IOException {
        try (InputStream input = testContext.getAssets().open(BENCHMARK_ASSET)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                throw new IOException("无法解码 androidTest/assets/" + BENCHMARK_ASSET);
            }
            return bitmap;
        }
    }

    private static <T> T runOnTfliteThread(Callable<T> work) throws Exception {
        var executor = Executors.newSingleThreadExecutor(
                runnable -> new Thread(runnable, InferenceScheduler.THREAD_NAME));
        try {
            Future<T> future = executor.submit(work);
            return future.get(5, TimeUnit.MINUTES);
        } finally {
            executor.shutdown();
        }
    }

    private static long percentile(long[] samples, int percentile) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        index = Math.max(0, Math.min(sorted.length - 1, index));
        return sorted[index];
    }

    private static boolean isProbablyEmulator() {
        String fingerprint = Build.FINGERPRINT.toLowerCase(Locale.US);
        String model = Build.MODEL.toLowerCase(Locale.US);
        String hardware = Build.HARDWARE.toLowerCase(Locale.US);
        String product = Build.PRODUCT.toLowerCase(Locale.US);
        return fingerprint.startsWith("generic")
                || fingerprint.contains("emulator")
                || fingerprint.contains("sdk_gphone")
                || model.contains("emulator")
                || model.contains("android sdk built for x86")
                || hardware.contains("goldfish")
                || hardware.contains("ranchu")
                || product.contains("sdk_gphone");
    }
}
