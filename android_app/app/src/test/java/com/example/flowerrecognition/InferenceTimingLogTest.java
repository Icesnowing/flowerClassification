package com.example.flowerrecognition;

import com.example.flowerrecognition.util.InferenceTimingLog;

import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class InferenceTimingLogTest {
    @Test
    public void formatLineIncludesSegmentTimingsAndSource() {
        String line = InferenceTimingLog.formatLine(10, 20, 30, 5, "GALLERY");
        assertTrue(line.contains("decode_ms=10"));
        assertTrue(line.contains("preprocess_ms=20"));
        assertTrue(line.contains("inference_ms=30"));
        assertTrue(line.contains("postprocess_ms=5"));
        assertTrue(line.contains("total_ms=65"));
        assertTrue(line.contains("source=GALLERY"));
    }
}
