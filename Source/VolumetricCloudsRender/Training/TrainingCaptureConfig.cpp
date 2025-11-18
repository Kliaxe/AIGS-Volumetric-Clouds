#include "TrainingCaptureConfig.h"

#include <algorithm>

// ----------------------------------------------------------------------------------------------------------
// ctor
// ----------------------------------------------------------------------------------------------------------
TrainingCaptureConfig::TrainingCaptureConfig()
    : m_minAnimationTimeSeconds(0.0f)
    , m_maxAnimationTimeSeconds(10000.0f)
    , m_targetBatchSize(1024U)
    , m_lowResolutionScale(0.25f)
    , m_lowResolutionMaxStepCount(64)
    , m_fullResolutionMaxStepCount(1024)
    , m_lowResolutionJitterEnabled(true)
    , m_highResolutionJitterEnabled(true)
    , m_outputDirectory(std::filesystem::current_path() / "TrainingCaptures")
    , m_randomSeed(0xC0FFEEULL)
{
}

void TrainingCaptureConfig::SetAnimationTimeRange(float minSeconds, float maxSeconds)
{
    const float a = std::min(minSeconds, maxSeconds);
    const float b = std::max(minSeconds, maxSeconds);
    m_minAnimationTimeSeconds = a;
    m_maxAnimationTimeSeconds = b;
}

float TrainingCaptureConfig::GetMinAnimationTime() const
{
    return m_minAnimationTimeSeconds;
}

float TrainingCaptureConfig::GetMaxAnimationTime() const
{
    return m_maxAnimationTimeSeconds;
}

// ----------------------------------------------------------------------------------------------------------
// Target batch size
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetTargetBatchSize(std::uint32_t pairCount)
{
    m_targetBatchSize = pairCount;
}

std::uint32_t TrainingCaptureConfig::GetTargetBatchSize() const
{
    return m_targetBatchSize;
}

// ----------------------------------------------------------------------------------------------------------
// Resolution scaling
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetLowResolutionScale(float scale)
{
    const float clampedScale = std::clamp(scale, 0.05f, 1.0f);
    m_lowResolutionScale = clampedScale;
}

float TrainingCaptureConfig::GetLowResolutionScale() const
{
    return m_lowResolutionScale;
}

// ----------------------------------------------------------------------------------------------------------
// Iteration budgets
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetLowResolutionMaxStepCount(int stepCount)
{
    m_lowResolutionMaxStepCount = std::max(stepCount, 1);
}

int TrainingCaptureConfig::GetLowResolutionMaxStepCount() const
{
    return m_lowResolutionMaxStepCount;
}

void TrainingCaptureConfig::SetFullResolutionMaxStepCount(int stepCount)
{
    m_fullResolutionMaxStepCount = std::max(stepCount, 1);
}

int TrainingCaptureConfig::GetFullResolutionMaxStepCount() const
{
    return m_fullResolutionMaxStepCount;
}

// ----------------------------------------------------------------------------------------------------------
// Raymarch jitter toggles
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetLowResolutionJitterEnabled(bool enabled)
{
    m_lowResolutionJitterEnabled = enabled;
}

bool TrainingCaptureConfig::GetLowResolutionJitterEnabled() const
{
    return m_lowResolutionJitterEnabled;
}

void TrainingCaptureConfig::SetHighResolutionJitterEnabled(bool enabled)
{
    m_highResolutionJitterEnabled = enabled;
}

bool TrainingCaptureConfig::GetHighResolutionJitterEnabled() const
{
    return m_highResolutionJitterEnabled;
}

// ----------------------------------------------------------------------------------------------------------
// Output directory
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetOutputDirectory(const std::filesystem::path& directory)
{
    m_outputDirectory = directory;
}

const std::filesystem::path& TrainingCaptureConfig::GetOutputDirectory() const
{
    return m_outputDirectory;
}

// ----------------------------------------------------------------------------------------------------------
// Random seed
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetRandomSeed(std::uint64_t seed)
{
    m_randomSeed = seed;
}

std::uint64_t TrainingCaptureConfig::GetRandomSeed() const
{
    return m_randomSeed;
}


