package com.example.flowerrecognition.tflite;

import java.util.function.Consumer;

/**
 * 一次后台任务的结果：要么成功带值，要么失败带异常。
 * 对应 Kotlin 的 Result，用在推理线程回到主线程的回调里。
 */
public final class Outcome<T> {
    private final T value;
    private final Throwable error;

    private Outcome(T value, Throwable error) {
        this.value = value;
        this.error = error;
    }

    public static <T> Outcome<T> success(T value) {
        return new Outcome<>(value, null);
    }

    public static <T> Outcome<T> failure(Throwable error) {
        if (error == null) {
            throw new IllegalArgumentException("error 不能为 null");
        }
        return new Outcome<>(null, error);
    }

    public boolean isSuccess() {
        return error == null;
    }

    public T getOrNull() {
        return error == null ? value : null;
    }

    public Throwable exceptionOrNull() {
        return error;
    }

    public T getOrThrow() {
        if (error != null) {
            return sneakyThrow(error);
        }
        return value;
    }

    /** 成功走 onSuccess，失败走 onFailure，正好一个分支。 */
    public void fold(Consumer<T> onSuccess, Consumer<Throwable> onFailure) {
        if (error == null) {
            onSuccess.accept(value);
        } else {
            onFailure.accept(error);
        }
    }

    @SuppressWarnings("unchecked")
    private static <R, E extends Throwable> R sneakyThrow(Throwable t) throws E {
        throw (E) t;
    }
}
