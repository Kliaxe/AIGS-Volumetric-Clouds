#include "Timer.h"

Timer::Timer(std::string label) : m_label(label)
{
    // Start timing upon creation
    Start();
}

void Timer::Start()
{
    m_startTime = std::chrono::high_resolution_clock::now();
}

long long Timer::Stop(TimeUnit unit)
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = endTime - m_startTime;  // Get the duration as a general duration

    // Return the duration based on the specified TimeUnit
    switch (unit)
    {
    case TimeUnit::Seconds:
        m_durationCount = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        break;
    case TimeUnit::Milliseconds:
        m_durationCount = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        break;
    case TimeUnit::Nanoseconds:
        m_durationCount = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        break;
    default:
        throw std::invalid_argument("Invalid TimeUnit");
    }

    m_selectedTimeUnit = unit;

    return m_durationCount;
}

void Timer::Print() const
{
    std::cout << "Time taken to compute " << m_label << ": " << m_durationCount << " ms..." << std::endl;
}

std::string Timer::GetTimeUnitString(TimeUnit unit) const
{
    // Return appropriate time symbol
    switch (unit)
    {
    case TimeUnit::Seconds:
        return "s";
    case TimeUnit::Milliseconds:
        return "ms";
    case TimeUnit::Nanoseconds:
        return "ns";
    default:
        throw std::invalid_argument("Invalid TimeUnit");
    }
}