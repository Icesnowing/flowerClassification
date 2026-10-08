package com.example.flowerrecognition;

import com.example.flowerrecognition.ui.RecognitionPresentation;
import com.example.flowerrecognition.util.IntPair;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class RecognitionPresentationTest {
    @Test
    public void uncertainOnlyWhenTop1IsBelow50Percent() {
        assertTrue(RecognitionPresentation.isUncertain(0f));
        assertTrue(RecognitionPresentation.isUncertain(0.4999f));
        assertFalse(RecognitionPresentation.isUncertain(0.5f));
        assertFalse(RecognitionPresentation.isUncertain(0.9234f));
    }

    @Test
    public void uncertainCopyMatchesTheSpec() {
        assertEquals(
                "无法确定，请换一张更清晰、主体更完整的照片",
                RecognitionPresentation.UNCERTAIN_MESSAGE);
    }

    @Test
    public void confidenceTextUsesTwoDecimalPercent() {
        assertEquals("25.00%", RecognitionPresentation.confidenceText(0.25f));
        assertEquals("50.00%", RecognitionPresentation.confidenceText(0.5f));
        assertEquals("100.00%", RecognitionPresentation.confidenceText(1f));
    }

    @Test
    public void previewKeepsSmallImagesAndCapsTheLongSide() {
        assertEquals(new IntPair(800, 600), RecognitionPresentation.previewSize(800, 600));
        assertEquals(new IntPair(1280, 960), RecognitionPresentation.previewSize(4032, 3024));
        assertEquals(new IntPair(64, 1280), RecognitionPresentation.previewSize(100, 2000));
        assertEquals(new IntPair(1280, 1280), RecognitionPresentation.previewSize(1280, 1280));
    }

    @Test
    public void previewSizeRejectsEmptyImages() {
        assertThrows(IllegalArgumentException.class, () ->
                RecognitionPresentation.previewSize(0, 10));
    }
}
