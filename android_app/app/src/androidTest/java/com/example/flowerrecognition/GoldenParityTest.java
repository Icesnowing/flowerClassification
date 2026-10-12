package com.example.flowerrecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import com.example.flowerrecognition.tflite.FlowerInterpreter;
import com.example.flowerrecognition.tflite.ImagePreprocessor;
import com.example.flowerrecognition.tflite.InferenceResult;
import com.example.flowerrecognition.tflite.InferenceScheduler;
import com.example.flowerrecognition.tflite.NormalizedNhwc;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * PC {@code export_golden.py} 与 Android 预处理 + TFLite 的数值对齐。
 * <p>
 * 资源布局（由 export_golden.py --copy-images / --copy-json 同步）:
 * <ul>
 *   <li>{@code assets/golden/manifest.json}</li>
 *   <li>{@code assets/golden/&lt;case&gt;.json}</li>
 *   <li>{@code assets/golden_images/&lt;asset_name&gt;.jpg}</li>
 * </ul>
 * 相册 / EXIF 不在此覆盖；asset 图应无方向标签，与 PC PIL 一致。
 */
@RunWith(AndroidJUnit4.class)
public class GoldenParityTest {
    private static final String MANIFEST_ASSET = "golden/manifest.json";
    private static final String GOLDEN_JSON_DIR = "golden/";
    private static final String GOLDEN_IMAGE_DIR = "golden_images/";

    @Test
    public void preprocessAndArgmaxMatchPcGolden() throws Exception {
        Context targetContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        Assume.assumeTrue(
                "未找到 " + MANIFEST_ASSET + "，请先运行 model_train/export_golden.py 并 --copy-json/--copy-images",
                assetExists(testContext, MANIFEST_ASSET));

        JSONObject manifest = readJsonAsset(testContext, MANIFEST_ASSET);
        JSONArray cases = manifest.getJSONArray("cases");
        Assume.assumeTrue("manifest.cases 为空", cases.length() > 0);

        runOnTfliteThread(() -> {
            FlowerInterpreter model = FlowerInterpreter.load(targetContext);
            try {
                for (int i = 0; i < cases.length(); i++) {
                    JSONObject ref = cases.getJSONObject(i);
                    String jsonName = ref.getString("json_name");
                    runOneCase(testContext, model, jsonName);
                }
            } finally {
                model.close();
            }
            return null;
        });
    }

    private static void runOneCase(Context context, FlowerInterpreter model, String jsonName)
            throws Exception {
        JSONObject golden = readJsonAsset(context, GOLDEN_JSON_DIR + jsonName);
        String assetName = golden.getString("asset_name");
        int expectedArgmax = golden.getInt("argmax");
        float absTol = (float) golden.optDouble("float_abs_tol", 1e-4);
        JSONArray headJson = golden.getJSONArray("input_head");
        int headLen = golden.optInt("input_head_len", headJson.length());

        Bitmap decoded = decodeAssetBitmap(context, GOLDEN_IMAGE_DIR + assetName);
        try {
            NormalizedNhwc input = ImagePreprocessor.toNhwc(decoded);
            assertInputHead(input.values, headJson, headLen, absTol, jsonName);

            float[] logits = model.run(input);
            int actualArgmax = InferenceResult.argmax(logits);
            Assert.assertEquals(
                    "argmax 不一致: " + jsonName + " asset=" + assetName,
                    expectedArgmax,
                    actualArgmax);
        } finally {
            if (!decoded.isRecycled()) {
                decoded.recycle();
            }
        }
    }

    private static void assertInputHead(
            float[] values,
            JSONArray expectedHead,
            int headLen,
            float absTol,
            String caseId) throws Exception {
        Assert.assertTrue(
                "input 长度不足 " + headLen + " (" + caseId + ")",
                values.length >= headLen);
        int n = Math.min(headLen, expectedHead.length());
        List<String> mismatches = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float expected = (float) expectedHead.getDouble(i);
            float actual = values[i];
            if (Math.abs(actual - expected) > absTol) {
                mismatches.add(String.format(
                        Locale.US,
                        "[%d] expected=%.8f actual=%.8f diff=%.8f",
                        i,
                        expected,
                        actual,
                        actual - expected));
            }
        }
        if (!mismatches.isEmpty()) {
            Assert.fail(
                    caseId + " input_head 超出 tol=" + absTol + ": "
                            + String.join("; ", mismatches)
                            + " | head=" + FlowerInterpreter.formatInputHead(values));
        }
    }

    private static Bitmap decodeAssetBitmap(Context context, String assetPath) throws IOException {
        try (InputStream input = context.getAssets().open(assetPath)) {
            Bitmap bitmap = BitmapFactory.decodeStream(input);
            if (bitmap == null) {
                throw new IOException("无法解码 assets/" + assetPath);
            }
            return bitmap;
        }
    }

    private static boolean assetExists(Context context, String assetPath) {
        try (InputStream ignored = context.getAssets().open(assetPath)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private static JSONObject readJsonAsset(Context context, String assetPath) throws Exception {
        try (InputStream input = context.getAssets().open(assetPath);
                BufferedReader reader = new BufferedReader(
                        new InputStreamReader(input, StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
            return new JSONObject(sb.toString());
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
}
