#include "threadpool.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

ThreadPool::ThreadPool(size_t num_threads)
{
    if (num_threads == 0)
        throw std::invalid_argument("Number of threads must be greater than zero.");
    workers.reserve(num_threads);
    try {
        for (size_t i = 0; i < num_threads; ++i)
            workers.emplace_back(&ThreadPool::WorkerThread, this);
    } catch (...) {
        // A partially constructed vector of joinable threads would terminate
        // during unwinding. Stop/join successful starts before propagating.
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& worker : workers) worker.join();
        throw;
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    condition.notify_all();
    for (auto& worker : workers) worker.join();
}

void ThreadPool::Submit(const std::function<void()>& task)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (stop) throw std::runtime_error("Cannot submit to a stopped thread pool.");
        tasks.push(task); // If allocation/copy throws, do not increment pending.
        ++tasks_in_progress;
    }
    condition.notify_one();
}

void ThreadPool::WaitAll()
{
    std::unique_lock<std::mutex> lock(mutex);
    completed.wait(lock, [this] { return tasks_in_progress == 0; });
}

void ThreadPool::WorkerThread()
{
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (tasks.empty()) return; // Stopping, after draining queued work.
            task = std::move(tasks.front());
            tasks.pop();
        }
        try {
            task();
        } catch (const std::exception& error) {
            std::cerr << "Exception in task: " << error.what() << '\n';
        } catch (...) {
            std::cerr << "Unknown exception in task.\n";
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (--tasks_in_progress == 0) completed.notify_all();
        }
    }
}
