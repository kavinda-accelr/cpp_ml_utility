#pragma once

#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>

namespace obj_detect
{
    class Thread_Pool
    {
    public:
        Thread_Pool(const unsigned int num_threads);

        void assign(std::function<void()> work);

        void join();

        void wait_until(const unsigned int task_cout);

        ~Thread_Pool();
    private:
        static void thread_work(Thread_Pool* threadPool);

        std::atomic_bool m_join;
        std::mutex m_queue_mutex;
        std::queue<std::function<void()>> m_work_queue;
        unsigned int m_num_threads;
        std::vector<std::thread> m_threads;
        std::atomic_uint16_t m_task_count;
    };
}