package com.example.flowerrecognition.ui;

import android.graphics.Bitmap;

import com.example.flowerrecognition.util.IntPair;

public final class PreviewBitmap {
    private PreviewBitmap() {
    }

    /**
     * 给界面看的原图。长边已不超过 {@link RecognitionPresentation#PREVIEW_MAX_SIDE} 时返回同一张 Bitmap。
     * 调用方只回收返回值不同的那张全尺寸图。
     */
    public static Bitmap scalePreview(Bitmap bitmap) {
        IntPair size = RecognitionPresentation.previewSize(bitmap.getWidth(), bitmap.getHeight());
        if (size.first == bitmap.getWidth() && size.second == bitmap.getHeight()) {
            return bitmap;
        }
        return Bitmap.createScaledBitmap(bitmap, size.first, size.second, true);
    }
}
