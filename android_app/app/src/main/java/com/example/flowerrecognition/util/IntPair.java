package com.example.flowerrecognition.util;

import java.util.Objects;

/** 两个 int 的不可变组合，替代 Kotlin 的 Pair<Int, Int>。 */
public final class IntPair {
    public final int first;
    public final int second;

    public IntPair(int first, int second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof IntPair)) {
            return false;
        }
        IntPair other = (IntPair) o;
        return first == other.first && second == other.second;
    }

    @Override
    public int hashCode() {
        return Objects.hash(first, second);
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ")";
    }
}
