package com.example.flowerrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.example.flowerrecognition.tflite.FlowerInterpreter;
import com.example.flowerrecognition.tflite.InferenceScheduler;
import com.example.flowerrecognition.util.InAppInferenceBenchmark;

import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * 真机单张推理耗时（adb）：固定图 → 预热 → 多次 toNhwc + run，输出 p50 / p90 / p95 / p99。
 * <p>
 * 无 adb 时在 release 包内长按首页模型状态，走 {@link InAppInferenceBenchmark}。
 */
@RunWith(AndroidJUnit4.class)
public class InferenceBenchmarkTest {
    private static final String TAG = InAppInferenceBenchmark.LOG_TAG;
    @Test
    public void reportsPreprocessAndRunLatency() throws Exception {
        Assume.assumeFalse(
                "请在物理真机上运行；模拟器耗时不作验收依据",
                isProbablyEmulator());

        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();

        long decodeStart = System.nanoTime();
        Bitmap decoded = InAppInferenceBenchmark.loadBenchmarkBitmap(appContext);
        long decodeMs = (System.nanoTime() - decodeStart) / 1_000_000L;

        runOnTfliteThread(() -> {
            FlowerInterpreter model = FlowerInterpreter.load(appContext);
            try {
                String report = InAppInferenceBenchmark.run(appContext, model, decoded, decodeMs, null);
                Log.i(TAG, report.replace('\n', '|'));
            } finally {
                model.close();
                if (!decoded.isRecycled()) {
                    decoded.recycle();
                }
            }
            return null;
        });
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
