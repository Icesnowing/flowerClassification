package com.example.flowerrecognition.tflite;

/** 一行标签：英文名 + 中文名。 */
public final class FlowerLabel {
    public final String english;
    public final String chinese;

    public FlowerLabel(String english, String chinese) {
        this.english = english;
        this.chinese = chinese;
    }
}
