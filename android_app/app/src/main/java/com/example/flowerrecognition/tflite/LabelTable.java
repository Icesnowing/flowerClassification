package com.example.flowerrecognition.tflite;

import android.content.Context;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public final class LabelTable {
    public static final int EXPECTED_SIZE = 14;
    private static final String ASSET_NAME = "labels_flower14.txt";
    private static final String BOM = "\uFEFF";

    private final List<FlowerLabel> labels;

    private LabelTable(List<FlowerLabel> labels) {
        this.labels = labels;
    }

    public int size() {
        return labels.size();
    }

    public FlowerLabel get(int index) {
        return labels.get(index);
    }

    public static LabelTable load(Context context) {
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(ASSET_NAME), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            throw new IllegalStateException("无法读取 assets/" + ASSET_NAME, e);
        }
        return fromLines(lines);
    }

    /** 按行解析标签。空行丢弃，行数必须是 EXPECTED_SIZE。 */
    public static LabelTable fromLines(List<String> lines) {
        List<String> cleaned = new ArrayList<>(lines.size());
        for (String line : lines) {
            String trimmed = stripBom(line.trim());
            if (!trimmed.isEmpty()) {
                cleaned.add(trimmed);
            }
        }
        if (cleaned.size() != EXPECTED_SIZE) {
            throw new IllegalStateException(
                    ASSET_NAME + " 应为 " + EXPECTED_SIZE + " 行，实际 " + cleaned.size() + " 行");
        }
        List<FlowerLabel> parsed = new ArrayList<>(cleaned.size());
        for (String line : cleaned) {
            parsed.add(parseFlowerLabel(line));
        }
        return new LabelTable(parsed);
    }

    public static FlowerLabel parseFlowerLabel(String line) {
        String trimmed = stripBom(line.trim());
        int cjkIndex = -1;
        for (int i = 0; i < trimmed.length(); i++) {
            char c = trimmed.charAt(i);
            if (c >= '\u4E00' && c <= '\u9FFF') {
                cjkIndex = i;
                break;
            }
        }
        if (cjkIndex <= 0) {
            throw new IllegalArgumentException("标签无法拆分中英文：" + trimmed);
        }
        return new FlowerLabel(
                trimmed.substring(0, cjkIndex).trim(),
                trimmed.substring(cjkIndex).trim());
    }

    private static String stripBom(String text) {
        return text.startsWith(BOM) ? text.substring(1) : text;
    }
}
