package com.example.flowerrecognition.tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Locale;

public final class FlowerInterpreter implements Closeable {
    public static final String LOG_TAG = "FlowerInterpreter";
    private static final String TAG = LOG_TAG;
    private static final String MODEL_ASSET = "model_flower14.tflite";
    private static final int[] INPUT_SHAPE = {1, 224, 224, 3};
    private static final int[] OUTPUT_SHAPE = {1, 14};
    private static final int INPUT_BYTES = 224 * 224 * 3 * 4;

    private final Interpreter interpreter;
    private final AssetFileDescriptor modelFd;
    @SuppressWarnings("unused")
    private final FileInputStream modelStream;
    public final LabelTable labels;

    private FlowerInterpreter(
            Interpreter interpreter,
            AssetFileDescriptor modelFd,
            FileInputStream modelStream,
            LabelTable labels) {
        this.interpreter = interpreter;
        this.modelFd = modelFd;
        this.modelStream = modelStream;
        this.labels = labels;
    }

    /**
     * 输入必须是 {@link ImagePreprocessor} 产出的归一化 float32。
     * 按 float 写入，不做 0–255 整型、不做只 /255，也不按 INT8 的 scale / zero_point 量化。
     * 返回的 14 个数是 logits，没有经过 softmax。
     *
     * 只在名为 tflite 的那一个线程上调用，不要在这次调用之外再开线程碰同一个 Interpreter。
     */
    public float[] run(NormalizedNhwc input) {
        ensureTfliteThread();
        if (interpreter.getInputTensor(0).dataType() != DataType.FLOAT32) {
            throw new IllegalStateException("输入必须是归一化 float32，不能按 INT8 的 scale/zero_point 送入");
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(INPUT_BYTES).order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(input.values);
        buffer.rewind();
        float[][] logits = new float[1][OUTPUT_SHAPE[1]];
        interpreter.run(buffer, logits);
        return logits[0];
    }

    /**
     * 在推理线程里跑 {@link #run}，再解析 argmax 和 Top-5。
     * {@link InferenceResult#getClassId()} 只出现在日志里，查表用的是 argmax 下标。
     */
    public InferenceResult recognize(NormalizedNhwc input) {
        return recognizeLogits(run(input), input);
    }

    /** 在已得到 logits 时解析结果并写日志（供界面分段计时的 run 之后调用）。 */
    public InferenceResult recognizeLogits(float[] logits, NormalizedNhwc input) {
        InferenceResult result = InferenceResult.fromLogits(logits, labels);
        Log.i(TAG, "input[0:12]=" + formatInputHead(input.values));
        StringBuilder top5Text = new StringBuilder("top5=");
        for (int i = 0; i < result.top5.size(); i++) {
            if (i > 0) {
                top5Text.append(", ");
            }
            TopPrediction p = result.top5.get(i);
            top5Text.append('#').append(p.index).append(' ')
                    .append(p.chinese).append(' ').append(p.probabilityText());
        }
        Log.i(TAG, "classId=" + result.getClassId() + " index=" + result.index + " "
                + result.english + " / " + result.chinese + " " + top5Text);
        return result;
    }

    public String summary() {
        FlowerLabel first = labels.get(0);
        return "模型已加载\n"
                + "输入 float32 [1, 224, 224, 3]\n"
                + "输出 float32 [1, 14]\n"
                + "标签 " + labels.size() + " 类，第 0 行：" + first.chinese + " / " + first.english;
    }

    @Override
    public void close() throws IOException {
        ensureTfliteThread();
        interpreter.close();
        modelFd.close();
    }

    private void ensureTfliteThread() {
        if (!Thread.currentThread().getName().equals(InferenceScheduler.THREAD_NAME)) {
            throw new IllegalStateException(
                    "Interpreter 只能在 " + InferenceScheduler.THREAD_NAME + " 线程上使用");
        }
    }

    public static FlowerInterpreter load(Context context) throws IOException {
        if (!Thread.currentThread().getName().equals(InferenceScheduler.THREAD_NAME)) {
            throw new IllegalStateException(
                    "模型只能在 " + InferenceScheduler.THREAD_NAME + " 线程上加载");
        }
        LabelTable labels = LabelTable.load(context);
        AssetFileDescriptor modelFd = openModel(context);
        FileInputStream modelStream = new FileInputStream(modelFd.getFileDescriptor());
        Interpreter interpreter;
        try {
            ByteBuffer buffer = modelStream.getChannel().map(
                    FileChannel.MapMode.READ_ONLY,
                    modelFd.getStartOffset(),
                    modelFd.getDeclaredLength());
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            Interpreter created = new Interpreter(buffer, options);
            try {
                created.allocateTensors();
                assertTensorSpec(created);
                interpreter = created;
            } catch (Throwable e) {
                created.close();
                throw e;
            }
        } catch (Throwable e) {
            modelStream.close();
            modelFd.close();
            throw e;
        }
        return new FlowerInterpreter(interpreter, modelFd, modelStream, labels);
    }

    private static AssetFileDescriptor openModel(Context context) {
        try {
            return context.getAssets().openFd(MODEL_ASSET);
        } catch (Exception e) {
            throw new IllegalStateException(
                    "无法映射 assets/" + MODEL_ASSET + "。确认文件在 assets 中，且未压缩进 APK。", e);
        }
    }

    /** 裁剪图左上角 4 个像素的 RGB，用来和 val_tflite.py 的前 12 个输入 float 对照。 */
    public static String formatInputHead(float[] values) {
        int count = Math.min(12, values.length);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(String.format(Locale.US, "%.8f", values[i]));
        }
        return sb.toString();
    }

    private static void assertTensorSpec(Interpreter interpreter) {
        org.tensorflow.lite.Tensor input = interpreter.getInputTensor(0);
        org.tensorflow.lite.Tensor output = interpreter.getOutputTensor(0);
        if (input.dataType() != DataType.FLOAT32 || !Arrays.equals(input.shape(), INPUT_SHAPE)) {
            throw new IllegalStateException(
                    "输入应为 float32 [1, 224, 224, 3]，实际 " + input.dataType() + " "
                            + Arrays.toString(input.shape()));
        }
        if (output.dataType() != DataType.FLOAT32 || !Arrays.equals(output.shape(), OUTPUT_SHAPE)) {
            throw new IllegalStateException(
                    "输出应为 float32 [1, 14]，实际 " + output.dataType() + " "
                            + Arrays.toString(output.shape()));
        }
    }
}
