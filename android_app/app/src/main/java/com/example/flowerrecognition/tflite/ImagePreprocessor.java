package com.example.flowerrecognition.tflite;

import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;

import com.example.flowerrecognition.util.IntPair;

import java.io.IOException;
import java.io.InputStream;

public final class ImagePreprocessor {
    /** 与 torchvision Resize(256) 的短边一致。相机图降采样后短边不能低于这个值。 */
    public static final int TARGET_SHORT_SIDE = 256;

    /** torchvision CenterCrop(224)。 */
    public static final int CROP_SIZE = 224;

    public static final int INPUT_FLOATS = 1 * CROP_SIZE * CROP_SIZE * 3;

    private static final float MEAN_R = 0.485f;
    private static final float MEAN_G = 0.456f;
    private static final float MEAN_B = 0.406f;
    private static final float STD_R = 0.229f;
    private static final float STD_G = 0.224f;
    private static final float STD_B = 0.225f;

    private ImagePreprocessor() {
    }

    /**
     * 相册对齐图完整解码。不设 inSampleSize，像素路径与 PC 一致。
     * 返回前已按 EXIF 转正一次，不要再对结果调用 {@link #applyExifOrientation}。
     */
    public static Bitmap decode(ContentResolver resolver, Uri uri) {
        return decodeWith(resolver, uri, options ->
                options.inPreferredConfig = Bitmap.Config.ARGB_8888);
    }

    /**
     * 相机全尺寸 JPEG 的解码。先读边界，再按短边 TARGET_SHORT_SIDE 设置 inSampleSize，
     * 避免 ARGB_8888 整图解码 OOM。输入是 EXTRA_OUTPUT 写出的 content URI，不是 Intent 缩略图。
     * 返回前已按 EXIF 转正一次，不要再对结果调用 {@link #applyExifOrientation}。
     */
    public static Bitmap decodeCamera(ContentResolver resolver, Uri uri) {
        IntPair bounds = readBounds(resolver, uri);
        int sampleSize = inSampleSizeForShortSide(bounds.first, bounds.second, TARGET_SHORT_SIDE);
        return decodeWith(resolver, uri, options -> {
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            options.inSampleSize = sampleSize;
        });
    }

    /**
     * 2 的幂抽样，使 min(width, height) / sample 仍不小于 targetShortSide。
     * 比较用整数除法：short / (sample * 2) >= targetShortSide 时才把 sample 再乘 2。
     */
    public static int inSampleSizeForShortSide(int width, int height, int targetShortSide) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("图片尺寸无效：" + width + "x" + height);
        }
        if (targetShortSide <= 0) {
            throw new IllegalArgumentException("目标短边必须大于 0");
        }
        long shortSide = Math.min(width, height);
        int sample = 1;
        while (sample < (1 << 30) && shortSide / (sample * 2L) >= targetShortSide) {
            sample *= 2;
        }
        return sample;
    }

    public static int inSampleSizeForShortSide(int width, int height) {
        return inSampleSizeForShortSide(width, height, TARGET_SHORT_SIDE);
    }

    /**
     * 缩放之前把 EXIF 方向转到 0°。只处理 90°、180°、270°。
     * 未写方向、0°，以及镜像、转置标签都不旋转：Oxford 测试图和 PC 上已经是正方向的文件落在这里。
     * 对同一张图只调用一次，再转会和 val_tflite.py 的方向不一致。
     */
    public static Bitmap applyExifOrientation(ContentResolver resolver, Uri uri, Bitmap bitmap) {
        int degrees = rotationDegrees(readOrientation(resolver, uri));
        if (degrees == 0) {
            return bitmap;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        Bitmap rotated = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, false);
        if (rotated != bitmap) {
            bitmap.recycle();
        }
        return rotated;
    }

    /**
     * 已转正的 Bitmap → 归一化 NHWC float32，形状 (1, 224, 224, 3)。
     * 顺序是 Resize(256) → CenterCrop(224) → /255 → ImageNet mean/std。
     * 这是送进 {@link FlowerInterpreter#run} 的唯一输入。
     */
    public static NormalizedNhwc toNhwc(Bitmap bitmap) {
        IntPair size = resizedSize(bitmap.getWidth(), bitmap.getHeight());
        Bitmap scaled = Bitmap.createScaledBitmap(bitmap, size.first, size.second, true);
        Bitmap cropped;
        try {
            IntPair origin = centerCropOrigin(size.first, size.second);
            cropped = Bitmap.createBitmap(scaled, origin.first, origin.second, CROP_SIZE, CROP_SIZE);
        } catch (Throwable error) {
            recycleUnless(scaled, bitmap);
            throw error;
        }
        // 裁剪图和缩放图可能是同一对象。那种情况下必须等像素读完再回收。
        if (scaled != cropped) {
            recycleUnless(scaled, bitmap);
        }
        try {
            int[] pixels = new int[CROP_SIZE * CROP_SIZE];
            cropped.getPixels(pixels, 0, CROP_SIZE, 0, 0, CROP_SIZE, CROP_SIZE);
            return pixelsToNhwc(pixels);
        } finally {
            recycleUnless(cropped, bitmap);
        }
    }

    /**
     * torchvision Resize(256)：短边变为 shortSide，长边 shortSide * long / short 整数截断。
     * 短边小于 shortSide 时同样放大。
     */
    public static IntPair resizedSize(int width, int height, int shortSide) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("图片尺寸无效：" + width + "x" + height);
        }
        if (shortSide <= 0) {
            throw new IllegalArgumentException("目标短边必须大于 0");
        }
        if (width <= height) {
            return new IntPair(shortSide, (int) (((long) shortSide * height) / width));
        } else {
            return new IntPair((int) (((long) shortSide * width) / height), shortSide);
        }
    }

    public static IntPair resizedSize(int width, int height) {
        return resizedSize(width, height, TARGET_SHORT_SIDE);
    }

    /** torchvision CenterCrop 的左上角。round 与 Python 一致：半数向偶数取整。 */
    public static IntPair centerCropOrigin(int width, int height, int crop) {
        if (width < crop || height < crop) {
            throw new IllegalArgumentException("裁剪前尺寸 " + width + "x" + height + " 小于 " + crop);
        }
        int left = roundHalfToEven((width - crop) / 2.0);
        int top = roundHalfToEven((height - crop) / 2.0);
        return new IntPair(left, top);
    }

    public static IntPair centerCropOrigin(int width, int height) {
        return centerCropOrigin(width, height, CROP_SIZE);
    }

    /** Python 3 round：恰好 .5 时向偶数取整。 */
    public static int roundHalfToEven(double value) {
        double base = Math.floor(value);
        double diff = value - base;
        int floored = (int) base;
        if (diff < 0.5) {
            return floored;
        } else if (diff > 0.5) {
            return floored + 1;
        } else if (floored % 2 == 0) {
            return floored;
        } else {
            return floored + 1;
        }
    }

    /**
     * 224×224 的 ARGB 像素排成归一化 NHWC。忽略 Alpha。
     * 每个通道先 /255，再减 mean、除 std。结果不是 0–255，也不是只做了 /255 的 0–1。
     */
    public static NormalizedNhwc pixelsToNhwc(int[] pixels, int side) {
        if (pixels.length != side * side) {
            throw new IllegalArgumentException("像素数量应为 " + (side * side) + "，实际 " + pixels.length);
        }
        float[] out = new float[side * side * 3];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                int pixel = pixels[y * side + x];
                int index = (y * side + x) * 3;
                out[index] = normalizeChannel((pixel >> 16) & 0xFF, MEAN_R, STD_R);
                out[index + 1] = normalizeChannel((pixel >> 8) & 0xFF, MEAN_G, STD_G);
                out[index + 2] = normalizeChannel(pixel & 0xFF, MEAN_B, STD_B);
            }
        }
        return new NormalizedNhwc(out);
    }

    public static NormalizedNhwc pixelsToNhwc(int[] pixels) {
        return pixelsToNhwc(pixels, CROP_SIZE);
    }

    private static void recycleUnless(Bitmap bitmap, Bitmap keep) {
        if (bitmap != keep && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    public static float normalizeChannel(int value, float mean, float std) {
        float unit = value / 255f;
        return (unit - mean) / std;
    }

    /** EXIF 方向对应的顺时针角度。只返回 90、180、270，其余为 0。 */
    public static int rotationDegrees(int orientation) {
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                return 90;
            case ExifInterface.ORIENTATION_ROTATE_180:
                return 180;
            case ExifInterface.ORIENTATION_ROTATE_270:
                return 270;
            default:
                return 0;
        }
    }

    private static int readOrientation(ContentResolver resolver, Uri uri) {
        try (InputStream input = openStream(resolver, uri)) {
            return new ExifInterface(input).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL);
        } catch (IOException e) {
            return ExifInterface.ORIENTATION_NORMAL;
        }
    }

    private static IntPair readBounds(ContentResolver resolver, Uri uri) {
        try (InputStream input = openStream(resolver, uri)) {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            BitmapFactory.decodeStream(input, null, bounds);
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                throw new IllegalArgumentException("无法读取图片尺寸：" + uri);
            }
            return new IntPair(bounds.outWidth, bounds.outHeight);
        } catch (IOException e) {
            throw new IllegalArgumentException("无法读取图片尺寸：" + uri, e);
        }
    }

    private interface OptionsConfigurer {
        void configure(BitmapFactory.Options options);
    }

    private static Bitmap decodeWith(
            ContentResolver resolver,
            Uri uri,
            OptionsConfigurer configure) {
        Bitmap bitmap;
        try (InputStream input = openStream(resolver, uri)) {
            BitmapFactory.Options options = new BitmapFactory.Options();
            configure.configure(options);
            Bitmap decoded = BitmapFactory.decodeStream(input, null, options);
            if (decoded == null) {
                throw new IllegalArgumentException("无法解码图片：" + uri);
            }
            bitmap = ensureArgb8888(decoded);
        } catch (IOException e) {
            throw new IllegalArgumentException("无法解码图片：" + uri, e);
        }
        return applyExifOrientation(resolver, uri, bitmap);
    }

    private static InputStream openStream(ContentResolver resolver, Uri uri) {
        InputStream input;
        try {
            input = resolver.openInputStream(uri);
        } catch (Exception e) {
            throw new IllegalArgumentException("无法打开图片：" + uri, e);
        }
        if (input == null) {
            throw new IllegalArgumentException("无法打开图片：" + uri);
        }
        return input;
    }

    private static Bitmap ensureArgb8888(Bitmap bitmap) {
        if (bitmap.getConfig() == Bitmap.Config.ARGB_8888) {
            return bitmap;
        }
        Bitmap argb = bitmap.copy(Bitmap.Config.ARGB_8888, false);
        if (argb == null) {
            throw new IllegalArgumentException("无法转为 ARGB_8888");
        }
        if (argb != bitmap) {
            bitmap.recycle();
        }
        return argb;
    }
}
