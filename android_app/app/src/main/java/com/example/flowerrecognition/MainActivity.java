package com.example.flowerrecognition;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.activity.OnBackPressedCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.example.flowerrecognition.tflite.FlowerInterpreter;
import com.example.flowerrecognition.tflite.ImagePreprocessor;
import com.example.flowerrecognition.tflite.InferenceResult;
import com.example.flowerrecognition.tflite.InferenceScheduler;
import com.example.flowerrecognition.tflite.NormalizedNhwc;
import com.example.flowerrecognition.tflite.Outcome;
import com.example.flowerrecognition.ui.ImageSource;
import com.example.flowerrecognition.ui.NoticeTone;
import com.example.flowerrecognition.ui.PreviewBitmap;
import com.example.flowerrecognition.ui.RecognitionPresentation;
import com.example.flowerrecognition.ui.RecognitionUiState;
import com.example.flowerrecognition.util.InferenceTimingLog;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class MainActivity extends ComponentActivity {
    private static final String KEY_CAMERA_URI = "camera_uri";
    private static final String CAPTURE_DIR = "camera";
    private static final String CAPTURE_FILE = "capture.jpg";

    private final ActivityResultLauncher<PickVisualMediaRequest> pickImage =
            registerForActivityResult(new ActivityResultContracts.PickVisualMedia(), this::onImagePicked);
    private final ActivityResultLauncher<String> requestCameraPermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(),
                    this::onCameraPermissionResult);
    private final ActivityResultLauncher<Intent> captureImage =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                    result -> onPhotoTaken(result.getResultCode()));

    private final InferenceScheduler scheduler = new InferenceScheduler(this::runOnUiThread);
    private FlowerInterpreter flowerInterpreter = null;
    private RecognitionUiState uiState = new RecognitionUiState();
    private Uri cameraUri = null;
    private PendingImage pendingImage = null;

    private Button backButton;
    private LinearLayout homeContent;
    private Button takePhotoButton;
    private Button pickGalleryButton;
    private Button openSettingsButton;
    private TextView noticeBanner;
    private TextView modelStatus;
    private LinearLayout runningRow;
    private LinearLayout resultContent;
    private ImageView previewImage;
    private TextView resultChinese;
    private TextView resultConfidence;
    private TextView resultTimings;
    private TextView uncertainBanner;

    private final OnBackPressedCallback backCallback = new OnBackPressedCallback(false) {
        @Override
        public void handleOnBackPressed() {
            backToHome();
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        androidx.activity.EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        bindViews();
        getOnBackPressedDispatcher().addCallback(this, backCallback);

        if (savedInstanceState != null) {
            String saved = savedInstanceState.getString(KEY_CAMERA_URI);
            cameraUri = saved != null ? Uri.parse(saved) : null;
        }
        uiState.modelStatus = getString(R.string.loading_model);
        render();
        scheduler.execute(() -> {
            String status;
            boolean ready;
            try {
                FlowerInterpreter loaded = FlowerInterpreter.load(getApplicationContext());
                flowerInterpreter = loaded;
                Log.i(FlowerInterpreter.LOG_TAG, loaded.summary());
                Log.i(
                        InferenceTimingLog.LOG_TAG,
                        "耗时日志文件: "
                                + InferenceTimingLog.logFile(getApplicationContext()).getAbsolutePath());
                status = getString(R.string.model_ready);
                ready = true;
            } catch (Exception e) {
                status = e.getMessage() != null ? e.getMessage() : e.toString();
                ready = false;
            }
            postStatus(status, ready);
        });
    }

    private void bindViews() {
        backButton = findViewById(R.id.backButton);
        homeContent = findViewById(R.id.homeContent);
        takePhotoButton = findViewById(R.id.takePhotoButton);
        pickGalleryButton = findViewById(R.id.pickGalleryButton);
        openSettingsButton = findViewById(R.id.openSettingsButton);
        noticeBanner = findViewById(R.id.noticeBanner);
        modelStatus = findViewById(R.id.modelStatus);
        runningRow = findViewById(R.id.runningRow);
        resultContent = findViewById(R.id.resultContent);
        previewImage = findViewById(R.id.previewImage);
        resultChinese = findViewById(R.id.resultChinese);
        resultConfidence = findViewById(R.id.resultConfidence);
        resultTimings = findViewById(R.id.resultTimings);
        uncertainBanner = findViewById(R.id.uncertainBanner);

        takePhotoButton.setOnClickListener(v -> onTakePhoto());
        pickGalleryButton.setOnClickListener(v -> onPickGallery());
        openSettingsButton.setOnClickListener(v -> openAppSettings());
        backButton.setOnClickListener(v -> backToHome());
        uncertainBanner.setText(RecognitionPresentation.UNCERTAIN_MESSAGE);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return insets;
        });
    }

    /** 把 uiState 渲染到视图。Compose 的重组在这里就是手工刷新。 */
    private void render() {
        boolean shown = !uiState.running
                && uiState.result != null
                && uiState.preview != null
                && !uiState.preview.isRecycled();
        backCallback.setEnabled(shown);
        backButton.setVisibility(shown ? View.VISIBLE : View.GONE);
        homeContent.setVisibility(shown ? View.GONE : View.VISIBLE);
        resultContent.setVisibility(shown ? View.VISIBLE : View.GONE);
        if (shown) {
            bindResult(uiState);
        } else {
            bindHome();
        }
    }

    private void bindHome() {
        boolean enabled = uiState.modelReady && !uiState.running;
        takePhotoButton.setEnabled(enabled);
        pickGalleryButton.setEnabled(enabled);
        openSettingsButton.setVisibility(uiState.showCameraSettings ? View.VISIBLE : View.GONE);

        if (uiState.notice != null) {
            noticeBanner.setText(uiState.notice);
            applyNoticeTone(uiState.noticeTone);
            noticeBanner.setVisibility(View.VISIBLE);
        } else {
            noticeBanner.setVisibility(View.GONE);
        }

        boolean showStatus = uiState.notice == null
                && (!uiState.modelReady || (uiState.result == null && !uiState.running));
        modelStatus.setVisibility(showStatus ? View.VISIBLE : View.GONE);
        if (showStatus) {
            modelStatus.setText(uiState.modelStatus);
            modelStatus.setTextColor(ContextCompat.getColor(this,
                    uiState.modelFailed ? R.color.text_error : R.color.text_secondary));
        }

        runningRow.setVisibility(uiState.running ? View.VISIBLE : View.GONE);
    }

    private void applyNoticeTone(NoticeTone tone) {
        int background;
        int textColor;
        switch (tone) {
            case WARNING:
                background = R.drawable.bg_banner_warning;
                textColor = R.color.banner_warning_text;
                break;
            case ERROR:
                background = R.drawable.bg_banner_error;
                textColor = R.color.banner_error_text;
                break;
            case NEUTRAL:
            default:
                background = R.drawable.bg_banner_neutral;
                textColor = R.color.banner_neutral_text;
                break;
        }
        noticeBanner.setBackgroundResource(background);
        noticeBanner.setTextColor(ContextCompat.getColor(this, textColor));
    }

    private void bindResult(RecognitionUiState state) {
        previewImage.setImageBitmap(state.preview);
        InferenceResult result = state.result;
        resultChinese.setText(result.chinese);
        resultConfidence.setText(getString(R.string.confidence,
                RecognitionPresentation.confidenceText(result.top1Probability())));
        resultTimings.setText(getString(R.string.inference_timings,
                state.decodeMs, state.preprocessMs, state.inferenceMs));
        uncertainBanner.setVisibility(
                RecognitionPresentation.isUncertain(result.top1Probability())
                        ? View.VISIBLE
                        : View.GONE);
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (cameraUri != null) {
            outState.putString(KEY_CAMERA_URI, cameraUri.toString());
        }
    }

    private void onPickGallery() {
        if (!canAcceptImage()) {
            return;
        }
        try {
            pickImage.launch(new PickVisualMediaRequest.Builder()
                    .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                    .build());
        } catch (Exception e) {
            showNotice(e.getMessage() != null ? e.getMessage() : e.toString(), NoticeTone.ERROR);
        }
    }

    /**
     * 系统照片选择器取消时 URI 为空。只提示，不进入推理。
     * 相册继续走 {@link ImagePreprocessor#decode}，不降采样，方便和电脑对同一张图。
     */
    private void onImagePicked(Uri uri) {
        if (uri == null) {
            showNotice(getString(R.string.pick_cancelled), NoticeTone.NEUTRAL);
            return;
        }
        submitImage(uri, ImageSource.GALLERY);
    }

    private void onTakePhoto() {
        if (!canAcceptImage()) {
            return;
        }
        boolean granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
        if (granted) {
            uiState.showCameraSettings = false;
            render();
            launchCamera();
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA);
        }
    }

    private void onCameraPermissionResult(boolean granted) {
        if (granted) {
            uiState.showCameraSettings = false;
            uiState.notice = null;
            render();
            launchCamera();
            return;
        }
        uiState.showCameraSettings = true;
        uiState.notice = getString(R.string.camera_permission_denied);
        uiState.noticeTone = NoticeTone.ERROR;
        render();
    }

    /**
     * 相机把全尺寸 JPEG 写到 cache 里的 content URI，不用 Intent 里的缩略图。
     * 部分系统要求 ClipData 才会把 FileProvider 的写权限交给相机。
     */
    private void launchCamera() {
        Uri uri;
        try {
            uri = createCaptureUri();
        } catch (Exception e) {
            showNotice(e.getMessage() != null ? e.getMessage() : e.toString(), NoticeTone.ERROR);
            return;
        }
        cameraUri = uri;
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        intent.setClipData(ClipData.newUri(getContentResolver(), "capture", uri));
        List<ResolveInfo> activities = queryCameraActivities(intent);
        if (activities.isEmpty()) {
            showNotice(getString(R.string.no_camera_app), NoticeTone.ERROR);
            return;
        }
        for (ResolveInfo info : activities) {
            grantUriPermission(
                    info.activityInfo.packageName,
                    uri,
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
        }
        try {
            captureImage.launch(intent);
        } catch (ActivityNotFoundException e) {
            releaseCameraGrant();
            showNotice(getString(R.string.no_camera_app), NoticeTone.ERROR);
        }
    }

    private void onPhotoTaken(int resultCode) {
        releaseCameraGrant();
        Uri uri = cameraUri;
        if (resultCode != RESULT_OK || uri == null) {
            deleteCaptureFile();
            showNotice(getString(R.string.capture_cancelled), NoticeTone.NEUTRAL);
            return;
        }
        submitImage(uri, ImageSource.CAMERA);
    }

    private void openAppSettings() {
        Intent intent = new Intent(
                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.fromParts("package", getPackageName(), null));
        startActivity(intent);
    }

    private boolean canAcceptImage() {
        return uiState.modelReady && !uiState.running;
    }

    private void submitImage(Uri uri, ImageSource source) {
        if (isDestroyed()) {
            return;
        }
        if (!uiState.modelReady) {
            pendingImage = new PendingImage(uri, source);
            showNotice(getString(R.string.waiting_for_model), NoticeTone.NEUTRAL);
            return;
        }
        if (uiState.running) {
            return;
        }
        RecognitionUiState previous = uiState.copy();
        uiState.running = true;
        uiState.result = null;
        uiState.decodeMs = 0L;
        uiState.preprocessMs = 0L;
        uiState.inferenceMs = 0L;
        uiState.notice = null;
        uiState.noticeTone = NoticeTone.NEUTRAL;
        render();
        boolean accepted = scheduler.submit(
                () -> recognize(uri, source),
                this::onRecognized);
        if (!accepted) {
            uiState = previous;
            render();
        }
    }

    private void onRecognized(Outcome<RecognizedImage> outcome) {
        if (isDestroyed()) {
            RecognizedImage image = outcome.getOrNull();
            if (image != null && image.preview != null && !image.preview.isRecycled()) {
                image.preview.recycle();
            }
            return;
        }
        outcome.fold(
                image -> {
                    uiState.running = false;
                    uiState.preview = image.preview;
                    uiState.result = image.result;
                    uiState.decodeMs = image.decodeMs;
                    uiState.preprocessMs = image.preprocessMs;
                    uiState.inferenceMs = image.inferenceMs;
                    uiState.notice = null;
                    uiState.noticeTone = NoticeTone.NEUTRAL;
                    render();
                },
                error -> {
                    Log.e(FlowerInterpreter.LOG_TAG, "recognize failed", error);
                    uiState.running = false;
                    uiState.result = null;
                    uiState.notice = error.getMessage() != null
                            ? error.getMessage()
                            : getString(R.string.recognize_failed);
                    uiState.noticeTone = NoticeTone.ERROR;
                    render();
                });
    }

    /**
     * 相册：完整解码。相机：先按短边 256 做 inSampleSize，再进同一套预处理。
     * 预览是界面上的原图，长边超过 1280 才缩小；送进模型的仍是解码结果。
     */
    private RecognizedImage recognize(Uri uri, ImageSource source) {
        Log.i(FlowerInterpreter.LOG_TAG, "uri=" + uri + " source=" + source);
        FlowerInterpreter model = flowerInterpreter;
        if (model == null) {
            throw new IllegalStateException("模型尚未加载");
        }
        try {
            long decodeStart = System.nanoTime();
            Bitmap decoded = source == ImageSource.GALLERY
                    ? ImagePreprocessor.decode(getContentResolver(), uri)
                    : ImagePreprocessor.decodeCamera(getContentResolver(), uri);
            long decodeMs = (System.nanoTime() - decodeStart) / 1_000_000L;
            try {
                long preprocessStart = System.nanoTime();
                NormalizedNhwc input = ImagePreprocessor.toNhwc(decoded);
                long preprocessMs = (System.nanoTime() - preprocessStart) / 1_000_000L;
                long runStart = System.nanoTime();
                float[] logits = model.run(input);
                long inferenceMs = (System.nanoTime() - runStart) / 1_000_000L;
                InferenceResult result = model.recognizeLogits(logits, input);
                InferenceTimingLog.append(
                        getApplicationContext(),
                        decodeMs,
                        preprocessMs,
                        inferenceMs,
                        source.name());
                Bitmap shown = PreviewBitmap.scalePreview(decoded);
                if (shown != decoded && !decoded.isRecycled()) {
                    decoded.recycle();
                }
                return new RecognizedImage(shown, result, decodeMs, preprocessMs, inferenceMs);
            } catch (Throwable error) {
                if (!decoded.isRecycled()) {
                    decoded.recycle();
                }
                throw error;
            }
        } finally {
            if (source == ImageSource.CAMERA) {
                deleteCaptureFile();
            }
        }
    }

    private void postStatus(String text, boolean ready) {
        scheduler.postToMain(() -> {
            if (isDestroyed()) {
                return;
            }
            uiState.modelReady = ready;
            uiState.modelFailed = !ready;
            uiState.modelStatus = text;
            if (!ready) {
                pendingImage = null;
                render();
                return;
            }
            render();
            PendingImage pending = pendingImage;
            if (pending == null) {
                return;
            }
            pendingImage = null;
            submitImage(pending.uri, pending.source);
        });
    }

    /** 结果页的返回：清掉这次识别，回到拍照 / 相册。模型保持已加载。 */
    private void backToHome() {
        RecognitionUiState state = uiState;
        if (state.running || state.result == null) {
            return;
        }
        state.preview = null;
        state.result = null;
        state.decodeMs = 0L;
        state.preprocessMs = 0L;
        state.inferenceMs = 0L;
        state.notice = null;
        state.noticeTone = NoticeTone.NEUTRAL;
        render();
    }

    private void showNotice(String text, NoticeTone tone) {
        uiState.notice = text;
        uiState.noticeTone = tone;
        render();
    }

    private Uri createCaptureUri() throws IOException {
        File dir = new File(getCacheDir(), CAPTURE_DIR);
        dir.mkdirs();
        File file = new File(dir, CAPTURE_FILE);
        if (file.exists() && !file.delete()) {
            throw new IllegalStateException("无法准备拍照文件");
        }
        if (!file.createNewFile()) {
            throw new IllegalStateException("无法创建拍照文件");
        }
        return FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", file);
    }

    private void deleteCaptureFile() {
        new File(getApplicationContext().getCacheDir(), CAPTURE_DIR + "/" + CAPTURE_FILE).delete();
    }

    private void releaseCameraGrant() {
        Uri uri = cameraUri;
        if (uri == null) {
            return;
        }
        revokeUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
    }

    @SuppressWarnings("deprecation")
    private List<ResolveInfo> queryCameraActivities(Intent intent) {
        return getPackageManager().queryIntentActivities(intent, PackageManager.MATCH_DEFAULT_ONLY);
    }

    @Override
    protected void onDestroy() {
        scheduler.shutdown(() -> {
            if (flowerInterpreter != null) {
                try {
                    flowerInterpreter.close();
                } catch (IOException e) {
                    Log.e(FlowerInterpreter.LOG_TAG, "close failed", e);
                }
                flowerInterpreter = null;
            }
        });
        super.onDestroy();
    }

    private static final class PendingImage {
        final Uri uri;
        final ImageSource source;

        PendingImage(Uri uri, ImageSource source) {
            this.uri = uri;
            this.source = source;
        }
    }

    private static final class RecognizedImage {
        final Bitmap preview;
        final InferenceResult result;
        final long decodeMs;
        final long preprocessMs;
        final long inferenceMs;

        RecognizedImage(
                Bitmap preview,
                InferenceResult result,
                long decodeMs,
                long preprocessMs,
                long inferenceMs) {
            this.preview = preview;
            this.result = result;
            this.decodeMs = decodeMs;
            this.preprocessMs = preprocessMs;
            this.inferenceMs = inferenceMs;
        }
    }
}
