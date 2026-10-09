#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    void Submit(const std::function<void()>& task);
    // Includes queued and running tasks. Completion publishes their writes.
    void WaitAll();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    // The queue, stop flag and completion count share one mutex. This prevents
    // notifications racing between a worker's predicate check and its sleep.
    std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable completed;
    bool stop = false;
    size_t tasks_in_progress = 0;

    void WorkerThread();
};

#endif
