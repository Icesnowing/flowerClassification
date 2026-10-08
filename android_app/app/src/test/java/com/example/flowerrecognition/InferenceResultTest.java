package com.example.flowerrecognition;

import com.example.flowerrecognition.tflite.FlowerInterpreter;
import com.example.flowerrecognition.tflite.InferenceResult;
import com.example.flowerrecognition.tflite.LabelTable;
import com.example.flowerrecognition.tflite.TopPrediction;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class InferenceResultTest {
    private final LabelTable labels = LabelTable.fromLines(makeLines());

    private static List<String> makeLines() {
        List<String> lines = new ArrayList<>(LabelTable.EXPECTED_SIZE);
        for (int i = 0; i < LabelTable.EXPECTED_SIZE; i++) {
            lines.add("flower " + i + " 花" + i);
        }
        return lines;
    }

    @Test
    public void argmaxKeepsTheFirstIndexWhenTied() {
        float[] logits = new float[LabelTable.EXPECTED_SIZE];
        logits[3] = 4f;
        logits[7] = 4f;
        logits[1] = 1f;

        InferenceResult result = InferenceResult.fromLogits(logits, labels);

        assertEquals(3, result.index);
        assertEquals(4, result.getClassId());
        assertEquals("flower 3", result.english);
        assertEquals("花3", result.chinese);
        assertEquals(3, result.top5.get(0).index);
    }

    @Test
    public void top5FollowsSoftmaxOrderAndMatchesArgmax() {
        float[] logits = new float[LabelTable.EXPECTED_SIZE];
        Arrays.fill(logits, -1f);
        logits[10] = 3f;
        logits[4] = 2f;
        logits[8] = 2f;
        logits[1] = 0.5f;
        logits[12] = 0f;
        logits[6] = -0.2f;

        InferenceResult result = InferenceResult.fromLogits(logits, labels);
        float[] probabilities = InferenceResult.softmax(logits);
        float sum = 0f;
        for (float p : probabilities) {
            sum += p;
        }

        assertEquals(10, InferenceResult.argmax(logits));
        assertEquals(10, result.index);
        assertEquals(10, result.top5.get(0).index);
        List<Integer> indices = new ArrayList<>(result.top5.size());
        for (TopPrediction p : result.top5) {
            indices.add(p.index);
        }
        assertEquals(Arrays.asList(10, 4, 8, 1, 12), indices);
        assertTrue(sum > 0.999f && sum < 1.001f);
        for (int i = 0; i + 1 < result.top5.size(); i++) {
            assertTrue(result.top5.get(i).probability >= result.top5.get(i + 1).probability);
        }
        assertEquals("花10", result.top5.get(0).chinese);
        assertEquals("flower 10", result.top5.get(0).english);
    }

    @Test
    public void softmaxSubtractsMaxSoLargeLogitsStayFinite() {
        float[] logits = new float[LabelTable.EXPECTED_SIZE];
        logits[2] = 1000f;
        logits[9] = 1000f;

        float[] probabilities = InferenceResult.softmax(logits);
        InferenceResult result = InferenceResult.fromLogits(logits, labels);

        for (float p : probabilities) {
            assertTrue(Float.isFinite(p));
        }
        assertEquals(0.5f, probabilities[2], 1e-5f);
        assertEquals(0.5f, probabilities[9], 1e-5f);
        assertEquals(2, result.index);
        assertEquals(2, result.top5.get(0).index);
        assertEquals("0.5000", result.top5.get(0).probabilityText());
        assertEquals("0.5000", result.top5.get(1).probabilityText());
    }

    @Test
    public void probabilityTextKeepsFourDecimalPlaces() {
        assertEquals("0.3333", InferenceResult.formatProbability(1f / 3f));
        assertEquals("1.0000", InferenceResult.formatProbability(1f));
    }

    @Test
    public void inputHeadPrintsTheFirstTwelveFloatsAtEightDecimals() {
        String text = FlowerInterpreter.formatInputHead(new float[]{1.5f, -0.25f});
        assertEquals("1.50000000, -0.25000000", text);
    }

    @Test
    public void rejectsLogitsThatAreNotExpectedLength() {
        assertThrows(IllegalArgumentException.class, () ->
                InferenceResult.fromLogits(new float[LabelTable.EXPECTED_SIZE - 1], labels));
    }
}
