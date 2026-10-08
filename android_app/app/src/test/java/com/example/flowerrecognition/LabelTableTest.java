package com.example.flowerrecognition;

import com.example.flowerrecognition.tflite.FlowerLabel;
import com.example.flowerrecognition.tflite.LabelTable;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class LabelTableTest {
    @Test
    public void splitsEnglishAndChineseOnFirstHanCharacter() {
        FlowerLabel label = LabelTable.parseFlowerLabel("pink primrose 粉红报春");
        assertEquals("pink primrose", label.english);
        assertEquals("粉红报春", label.chinese);
    }

    @Test
    public void rejectsLineWithoutChineseName() {
        assertThrows(IllegalArgumentException.class, () ->
                LabelTable.parseFlowerLabel("pink primrose"));
    }
}
