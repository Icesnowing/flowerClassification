package com.example.flowerrecognition.util;

import android.content.Context;
import android.util.Log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

/**
 * 将解码 / 预处理 / 推理 / 结果处理耗时追加写入应用私有目录下的日志文件（debug 与 release 均写入）。
 */
public final class InferenceTimingLog {
    public static final String LOG_TAG = "InferenceTimingLog";
    public static final String FILE_NAME = "inference_timings.log";

    private static final DateTimeFormatter TIMESTAMP =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX", Locale.US);
    private static final Object LOCK = new Object();

    private InferenceTimingLog() {
    }

    public static File logFile(Context context) {
        return new File(context.getApplicationContext().getFilesDir(), FILE_NAME);
    }

    /**
     * 追加一行耗时记录，并同步打 Logcat（tag {@link #LOG_TAG}）。
     */
    public static void append(
            Context context,
            long decodeMs,
            long preprocessMs,
            long inferenceMs,
            long postprocessMs,
            String source) {
        String line = formatLine(decodeMs, preprocessMs, inferenceMs, postprocessMs, source);
        Log.i(LOG_TAG, line);
        synchronized (LOCK) {
            File file = logFile(context);
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(file, true), StandardCharsets.UTF_8))) {
                writer.write(line);
                writer.newLine();
            } catch (IOException e) {
                Log.w(LOG_TAG, "无法写入 " + file.getAbsolutePath(), e);
            }
        }
    }

    static String formatLine(
            long decodeMs,
            long preprocessMs,
            long inferenceMs,
            long postprocessMs,
            String source) {
        long totalMs = decodeMs + preprocessMs + inferenceMs + postprocessMs;
        String timestamp = ZonedDateTime.now().format(TIMESTAMP);
        return timestamp
                + " decode_ms=" + decodeMs
                + " preprocess_ms=" + preprocessMs
                + " inference_ms=" + inferenceMs
                + " postprocess_ms=" + postprocessMs
                + " total_ms=" + totalMs
                + " source=" + source;
    }
}
