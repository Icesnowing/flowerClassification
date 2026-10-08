package com.example.flowerrecognition;

import com.example.flowerrecognition.tflite.InferenceScheduler;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class InferenceSchedulerTest {
    @Test
    public void runsWorkOnTheTfliteThreadAndResultOnTheMainThread() throws Exception {
        ExecutorService main = Executors.newSingleThreadExecutor(r -> new Thread(r, "main-test"));
        InferenceScheduler scheduler = new InferenceScheduler(main::execute);
        CountDownLatch done = new CountDownLatch(1);
        try {
            assertTrue(scheduler.submit(
                    () -> {
                        assertEquals(InferenceScheduler.THREAD_NAME, Thread.currentThread().getName());
                        return "ok";
                    },
                    result -> {
                        assertEquals("main-test", Thread.currentThread().getName());
                        assertEquals("ok", result.getOrThrow());
                        done.countDown();
                    }));
            assertTrue(done.await(5, TimeUnit.SECONDS));
        } finally {
            scheduler.shutdown(() -> { });
            main.shutdown();
        }
    }

    @Test
    public void ignoresASecondImageUntilTheFirstFinishes() throws Exception {
        ExecutorService main = Executors.newSingleThreadExecutor(r -> new Thread(r, "main-test"));
        InferenceScheduler scheduler = new InferenceScheduler(main::execute);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch finished = new CountDownLatch(1);
        try {
            assertTrue(scheduler.submit(
                    () -> {
                        started.countDown();
                        assertTrue(release.await(5, TimeUnit.SECONDS));
                        return 1;
                    },
                    result -> finished.countDown()));
            assertTrue(started.await(5, TimeUnit.SECONDS));
            assertFalse(scheduler.submit(() -> 2, result -> { }));
            release.countDown();
            assertTrue(finished.await(5, TimeUnit.SECONDS));

            CountDownLatch again = new CountDownLatch(1);
            assertTrue(scheduler.submit(() -> 3, result -> again.countDown()));
            assertTrue(again.await(5, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            scheduler.shutdown(() -> { });
            main.shutdown();
        }
    }

    @Test
    public void failureStillReleasesTheGate() throws Exception {
        ExecutorService main = Executors.newSingleThreadExecutor(r -> new Thread(r, "main-test"));
        InferenceScheduler scheduler = new InferenceScheduler(main::execute);
        CountDownLatch failed = new CountDownLatch(1);
        try {
            assertTrue(scheduler.submit(
                    () -> {
                        throw new IllegalStateException("解码失败");
                    },
                    result -> {
                        assertEquals("解码失败", result.exceptionOrNull().getMessage());
                        failed.countDown();
                    }));
            assertTrue(failed.await(5, TimeUnit.SECONDS));
            CountDownLatch again = new CountDownLatch(1);
            assertTrue(scheduler.submit(() -> "next", result -> again.countDown()));
            assertTrue(again.await(5, TimeUnit.SECONDS));
        } finally {
            scheduler.shutdown(() -> { });
            main.shutdown();
        }
    }

    @Test
    public void shutdownClosesOnTheTfliteThreadAfterTheCurrentImage() throws Exception {
        ExecutorService main = Executors.newSingleThreadExecutor(r -> new Thread(r, "main-test"));
        InferenceScheduler scheduler = new InferenceScheduler(main::execute);
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        CountDownLatch closed = new CountDownLatch(1);
        List<String> order = new ArrayList<>();
        try {
            assertTrue(scheduler.submit(
                    () -> {
                        order.add("work:" + Thread.currentThread().getName());
                        started.countDown();
                        assertTrue(release.await(5, TimeUnit.SECONDS));
                        return null;
                    },
                    result -> { }));
            assertTrue(started.await(5, TimeUnit.SECONDS));
            scheduler.shutdown(() -> {
                order.add("close:" + Thread.currentThread().getName());
                closed.countDown();
            });
            assertFalse(scheduler.submit(() -> null, result -> { }));
            release.countDown();
            assertTrue(closed.await(5, TimeUnit.SECONDS));
            assertEquals(Arrays.asList("work:tflite", "close:tflite"), order);
        } finally {
            release.countDown();
            scheduler.shutdown(() -> { });
            main.shutdown();
        }
    }
}
