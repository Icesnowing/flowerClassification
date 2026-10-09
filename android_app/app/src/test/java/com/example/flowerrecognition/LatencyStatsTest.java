package com.example.flowerrecognition;

import com.example.flowerrecognition.util.LatencyStats;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LatencyStatsTest {
    @Test
    public void percentileUsesNearestRank() {
        long[] samples = new long[100];
        for (int i = 0; i < 100; i++) {
            samples[i] = i + 1;
        }
        assertEquals(50, LatencyStats.percentile(samples, 50));
        assertEquals(90, LatencyStats.percentile(samples, 90));
        assertEquals(95, LatencyStats.percentile(samples, 95));
        assertEquals(99, LatencyStats.percentile(samples, 99));
    }
}
