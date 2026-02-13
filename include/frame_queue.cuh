#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

struct Frame
{
    std::vector<float> x, y, z;
    size_t step{0};
};

class FrameQueue
{
  public:
    explicit FrameQueue(size_t capacity) : m_capacity(capacity)
    {
    }

    void push(Frame frame)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_full.wait(lock, [this] { return m_queue.size() < m_capacity || m_closed; });
        if (m_closed)
            return;
        m_queue.push_back(std::move(frame));
        m_not_empty.notify_one();
    }

    std::optional<Frame> try_pop()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
            return std::nullopt;
        Frame f = std::move(m_queue.front());
        m_queue.pop_front();
        m_not_full.notify_one();
        return f;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_closed = true;
        m_not_full.notify_all();
        m_not_empty.notify_all();
    }

    bool is_closed() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_closed && m_queue.empty();
    }

  private:
    size_t m_capacity;
    std::deque<Frame> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_not_full;
    std::condition_variable m_not_empty;
    bool m_closed{false};
};
