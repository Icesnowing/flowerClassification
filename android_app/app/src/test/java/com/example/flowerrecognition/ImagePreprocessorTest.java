package com.example.flowerrecognition;

import android.media.ExifInterface;

import com.example.flowerrecognition.tflite.ImagePreprocessor;
import com.example.flowerrecognition.tflite.NormalizedNhwc;
import com.example.flowerrecognition.util.IntPair;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;

public class ImagePreprocessorTest {
    @Test
    public void keepsFullResolutionWhenShortSideWouldDropBelow256() {
        assertEquals(1, ImagePreprocessor.inSampleSizeForShortSide(256, 256));
        assertEquals(1, ImagePreprocessor.inSampleSizeForShortSide(400, 300));
        assertEquals(1, ImagePreprocessor.inSampleSizeForShortSide(511, 400));
    }

    @Test
    public void usesLargestPowerOfTwoThatKeepsShortSideAtLeast256() {
        assertEquals(2, ImagePreprocessor.inSampleSizeForShortSide(512, 512));
        assertEquals(2, ImagePreprocessor.inSampleSizeForShortSide(800, 600));
        assertEquals(8, ImagePreprocessor.inSampleSizeForShortSide(4032, 3024));
        assertEquals(8, ImagePreprocessor.inSampleSizeForShortSide(3024, 4032));
        assertEquals(16, ImagePreprocessor.inSampleSizeForShortSide(10000, 8000));
    }

    @Test
    public void rotationDegreesOnlyForRightAngles() {
        assertEquals(90, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_ROTATE_90));
        assertEquals(180, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_ROTATE_180));
        assertEquals(270, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_ROTATE_270));
        assertEquals(0, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_NORMAL));
        assertEquals(0, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_UNDEFINED));
        assertEquals(0, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_FLIP_HORIZONTAL));
        assertEquals(0, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_FLIP_VERTICAL));
        assertEquals(0, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_TRANSPOSE));
        assertEquals(0, ImagePreprocessor.rotationDegrees(ExifInterface.ORIENTATION_TRANSVERSE));
    }

    @Test
    public void resizeUsesTruncatingIntegerDivisionAndStillUpscales() {
        assertEquals(new IntPair(256, 409), ImagePreprocessor.resizedSize(500, 800));
        assertEquals(new IntPair(409, 256), ImagePreprocessor.resizedSize(800, 500));
        assertEquals(new IntPair(256, 256), ImagePreprocessor.resizedSize(256, 256));
        assertEquals(new IntPair(256, 512), ImagePreprocessor.resizedSize(100, 200));
        assertEquals(new IntPair(256, 329), ImagePreprocessor.resizedSize(7, 9));
    }

    @Test
    public void centerCropUsesPythonRoundHalfToEven() {
        assertEquals(0, ImagePreprocessor.roundHalfToEven(0.5));
        assertEquals(2, ImagePreprocessor.roundHalfToEven(1.5));
        assertEquals(52, ImagePreprocessor.roundHalfToEven(52.5));
        assertEquals(54, ImagePreprocessor.roundHalfToEven(53.5));
        assertEquals(new IntPair(16, 16), ImagePreprocessor.centerCropOrigin(256, 256));
        assertEquals(new IntPair(16, 52), ImagePreprocessor.centerCropOrigin(256, 329));
        assertEquals(new IntPair(52, 16), ImagePreprocessor.centerCropOrigin(329, 256));
    }

    @Test
    public void packsNormalizedRgbAsNhwcAndIgnoresAlpha() {
        int[] pixels = new int[ImagePreprocessor.CROP_SIZE * ImagePreprocessor.CROP_SIZE];
        pixels[0] = 0x80FF0000;
        pixels[1] = 0xFF00FF00;
        pixels[ImagePreprocessor.CROP_SIZE] = 0xFF0000FF;

        float[] nhwc = ImagePreprocessor.pixelsToNhwc(pixels).values;

        assertEquals(ImagePreprocessor.INPUT_FLOATS, nhwc.length);
        assertEquals(2.2489083f, nhwc[0], 1e-5f);
        assertNotEquals(255f, nhwc[0]);
        assertNotEquals(1f, nhwc[0]);
        assertEquals(ImagePreprocessor.normalizeChannel(0, 0.456f, 0.224f), nhwc[1], 0f);
        assertEquals(ImagePreprocessor.normalizeChannel(0, 0.406f, 0.225f), nhwc[2], 0f);
        assertEquals(ImagePreprocessor.normalizeChannel(255, 0.456f, 0.224f), nhwc[4], 0f);
        int row = ImagePreprocessor.CROP_SIZE * 3;
        assertEquals(ImagePreprocessor.normalizeChannel(255, 0.406f, 0.225f), nhwc[row + 2], 0f);
        assertEquals((0f - 0.485f) / 0.229f, nhwc[row], 0f);
    }

    @Test
    public void runInputMustBeTheFullNormalizedBuffer() {
        assertThrows(IllegalArgumentException.class, () -> new NormalizedNhwc(new float[3]));
    }
}
