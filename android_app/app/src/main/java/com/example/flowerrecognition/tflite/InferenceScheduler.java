package com.example.flowerrecognition.tflite;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * 进程里唯一的推理线程，名字固定为 THREAD_NAME。
 * 模型加载、解码、预处理、{@link FlowerInterpreter#run} 和 close 都提交到这里，
 * 避免换线程碰同一个 Interpreter。
 */
public final class InferenceScheduler {
    public static final String THREAD_NAME = "tflite";

    private final ExecutorService executor;
    private final Consumer<Runnable> deliverToMain;
    private final AtomicBoolean accepting = new AtomicBoolean(true);
    private final AtomicBoolean running = new AtomicBoolean(false);

    public InferenceScheduler(Consumer<Runnable> deliverToMain) {
        this(Executors.newSingleThreadExecutor(runnable -> new Thread(runnable, THREAD_NAME)),
                deliverToMain);
    }

    public InferenceScheduler(ExecutorService executor, Consumer<Runnable> deliverToMain) {
        this.executor = executor;
        this.deliverToMain = deliverToMain;
    }

    /** 把任务排到推理线程。销毁之后不再接收。 */
    public void execute(Runnable task) {
        if (!accepting.get()) {
            return;
        }
        try {
            executor.execute(task);
        } catch (RejectedExecutionException e) {
            // 销毁已经停掉这条线程。
        }
    }

    public void postToMain(Runnable action) {
        deliverToMain.accept(action);
    }

    /**
     * 主线程提交一张图。上一张还没结束时返回 false，不排队。
     * work 在推理线程执行，onResult 回到主线程后再改界面。
     */
    public <T> boolean submit(Callable<T> work, Consumer<Outcome<T>> onResult) {
        if (!accepting.get()) {
            return false;
        }
        if (!running.compareAndSet(false, true)) {
            return false;
        }
        try {
            executor.execute(() -> {
                Outcome<T> result;
                try {
                    result = Outcome.success(work.call());
                } catch (Throwable e) {
                    result = Outcome.failure(e);
                }
                Outcome<T> finalResult = result;
                postToMain(() -> {
                    running.set(false);
                    onResult.accept(finalResult);
                });
            });
            return true;
        } catch (RejectedExecutionException e) {
            running.set(false);
            return false;
        }
    }

    /**
     * 把 release 排进推理线程，让它排在正在跑的那张图后面，然后再停止接收新任务。
     * 先排队再 shutdown：shutdown 之后不能再提交，close 必须留在这条线程上。
     */
    public void shutdown(Runnable release) {
        if (!accepting.compareAndSet(true, false)) {
            return;
        }
        try {
            executor.execute(release);
        } catch (RejectedExecutionException e) {
            // 线程池已停，直接忽略。
        }
        executor.shutdown();
    }
}
