#include "Thread_Pool.hpp"

obj_detect::Thread_Pool::Thread_Pool(const unsigned int num_threads = 0) : m_task_count(0), m_join(false)
{
    m_num_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < m_num_threads; i++)
    {
        m_threads.emplace_back(std::thread(Thread_Pool::thread_work, this));
    }
}

void obj_detect::Thread_Pool::assign(std::function<void()> work)
{
    std::unique_lock<std::mutex> queue_lck(m_queue_mutex);
    m_work_queue.push(work);
    queue_lck.unlock();
}

void obj_detect::Thread_Pool::join()
{
    m_join = true;
    for (auto& t : m_threads)
    {
        if (t.joinable()) t.join();
    }
}

void obj_detect::Thread_Pool::wait_until(const unsigned int task_cout)
{
    // wait only if task count not completed and join not called
    while (m_task_count < task_cout && !m_join) std::this_thread::yield();
    m_task_count = 0;
}

obj_detect::Thread_Pool::~Thread_Pool()
{
    join();
}

void obj_detect::Thread_Pool::thread_work(Thread_Pool* threadPool)
{
    std::function<void()> work;
    bool work_assigned = false;
    std::unique_lock<std::mutex> queue_lck(threadPool->m_queue_mutex, std::defer_lock);
    while (!(threadPool->m_join && threadPool->m_work_queue.empty())) //break the loop if only join is called and queue is empty 
    {
        if (threadPool->m_work_queue.empty()) std::this_thread::yield();
        else
        {
            queue_lck.lock();
            if (!threadPool->m_work_queue.empty())
            {
                work = threadPool->m_work_queue.front();
                threadPool->m_work_queue.pop();
                work_assigned = true;
            }
            queue_lck.unlock();
            if (work_assigned)
            {
                work();
                threadPool->m_task_count++;
                work_assigned = false;
            }
        }
    }
}