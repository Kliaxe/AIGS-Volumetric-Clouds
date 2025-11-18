#pragma once

#include <iostream>
#include <chrono>
#include <string>

class Timer
{
public:
    // Enum representing various time units
    enum class TimeUnit
    {
        Seconds,
        Milliseconds,
        Nanoseconds
    };

    Timer(std::string label);

    // Start timing
    void Start();

    // Stop timing and return the duration based on the specified unit
    long long Stop(TimeUnit unit = TimeUnit::Milliseconds);

    // Print the result to console
    void Print() const;

private:
    // Get time unit represented as symbol
    std::string GetTimeUnitString(TimeUnit unit = TimeUnit::Milliseconds) const;

private:

    // Lable for print
    std::string m_label;

    // Saved time unit
    TimeUnit m_selectedTimeUnit;

    // Point of reference
    std::chrono::high_resolution_clock::time_point m_startTime;
    
    // Converted time difference
    long long m_durationCount;
};