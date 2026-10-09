package com.example.flowerrecognition.ui;

import android.graphics.Bitmap;

import com.example.flowerrecognition.tflite.InferenceResult;

/** 界面状态。字段直接可变，每次改完调用 MainActivity 的 render() 刷新视图。 */
public final class RecognitionUiState {
    public String modelStatus = "正在加载模型…";
    public boolean modelReady = false;
    public boolean modelFailed = false;
    public boolean running = false;
    public String notice = null;
    public NoticeTone noticeTone = NoticeTone.NEUTRAL;
    public boolean showCameraSettings = false;
    public Bitmap preview = null;
    public InferenceResult result = null;
    public long decodeMs = 0L;
    public long preprocessMs = 0L;
    public long inferenceMs = 0L;
    public long postprocessMs = 0L;
    public boolean benchmarkRunning = false;
    public String benchmarkProgress = null;
    public String benchmarkReport = null;

    public RecognitionUiState() {
    }

    /** 整体复制一份，用于提交识别前保存现场、未被接收时恢复。 */
    public RecognitionUiState copy() {
        RecognitionUiState copy = new RecognitionUiState();
        copy.modelStatus = modelStatus;
        copy.modelReady = modelReady;
        copy.modelFailed = modelFailed;
        copy.running = running;
        copy.notice = notice;
        copy.noticeTone = noticeTone;
        copy.showCameraSettings = showCameraSettings;
        copy.preview = preview;
        copy.result = result;
        copy.decodeMs = decodeMs;
        copy.preprocessMs = preprocessMs;
        copy.inferenceMs = inferenceMs;
        copy.postprocessMs = postprocessMs;
        copy.benchmarkRunning = benchmarkRunning;
        copy.benchmarkProgress = benchmarkProgress;
        copy.benchmarkReport = benchmarkReport;
        return copy;
    }
}
