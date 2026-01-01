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
    , m_lowResolutionMaxStepCount(128)
    , m_fullResolutionMaxStepCount(2048)
    , m_lowResolutionJitterEnabled(true)
    , m_highResolutionJitterEnabled(true)
    , m_outputDirectory(std::filesystem::current_path() / "TrainingCaptures")
    , m_randomSeed(0xC0FFEEULL)
    , m_minExtinctionDensityMultiplier(0.01f)
    , m_maxExtinctionDensityMultiplier(0.1f)
    , m_minCoverageAmount(1.0f)
    , m_maxCoverageAmount(2.0f)
    , m_minCoverageMinimum(0.0f)
    , m_maxCoverageMinimum(0.30f)
    , m_minTypeAmount(1.0f)
    , m_maxTypeAmount(1.25f)
    , m_minTypeMinimum(0.0f)
    , m_maxTypeMinimum(0.05f)
    , m_minTotalNoiseScale(0.001f)
    , m_maxTotalNoiseScale(0.0008f)
    , m_minSunDirection(1.0f, 0.0f, 0.0f)
    , m_maxSunDirection(0.0f, 1.0f, 0.0f)
    , m_minSunIlluminance(8.0f, 8.0f, 7.0f)
    , m_maxSunIlluminance(8.0f, 8.0f, 7.0f)
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

// ----------------------------------------------------------------------------------------------------------
// Cloud optical properties
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetExtinctionDensityMultiplierRange(float minMultiplier, float maxMultiplier)
{
    const float a = std::min(minMultiplier, maxMultiplier);
    const float b = std::max(minMultiplier, maxMultiplier);
    m_minExtinctionDensityMultiplier = a;
    m_maxExtinctionDensityMultiplier = b;
}

float TrainingCaptureConfig::GetMinExtinctionDensityMultiplier() const
{
    return m_minExtinctionDensityMultiplier;
}

float TrainingCaptureConfig::GetMaxExtinctionDensityMultiplier() const
{
    return m_maxExtinctionDensityMultiplier;
}

// ----------------------------------------------------------------------------------------------------------
// Cloud density shaping (weather map)
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetCoverageAmountRange(float minAmount, float maxAmount)
{
    const float a = std::min(minAmount, maxAmount);
    const float b = std::max(minAmount, maxAmount);
    m_minCoverageAmount = a;
    m_maxCoverageAmount = b;
}

float TrainingCaptureConfig::GetMinCoverageAmount() const
{
    return m_minCoverageAmount;
}

float TrainingCaptureConfig::GetMaxCoverageAmount() const
{
    return m_maxCoverageAmount;
}

void TrainingCaptureConfig::SetCoverageMinimumRange(float minMinimum, float maxMinimum)
{
    const float a = std::min(minMinimum, maxMinimum);
    const float b = std::max(minMinimum, maxMinimum);
    m_minCoverageMinimum = a;
    m_maxCoverageMinimum = b;
}

float TrainingCaptureConfig::GetMinCoverageMinimum() const
{
    return m_minCoverageMinimum;
}

float TrainingCaptureConfig::GetMaxCoverageMinimum() const
{
    return m_maxCoverageMinimum;
}

void TrainingCaptureConfig::SetTypeAmountRange(float minAmount, float maxAmount)
{
    const float a = std::min(minAmount, maxAmount);
    const float b = std::max(minAmount, maxAmount);
    m_minTypeAmount = a;
    m_maxTypeAmount = b;
}

float TrainingCaptureConfig::GetMinTypeAmount() const
{
    return m_minTypeAmount;
}

float TrainingCaptureConfig::GetMaxTypeAmount() const
{
    return m_maxTypeAmount;
}

void TrainingCaptureConfig::SetTypeMinimumRange(float minMinimum, float maxMinimum)
{
    const float a = std::min(minMinimum, maxMinimum);
    const float b = std::max(minMinimum, maxMinimum);
    m_minTypeMinimum = a;
    m_maxTypeMinimum = b;
}

float TrainingCaptureConfig::GetMinTypeMinimum() const
{
    return m_minTypeMinimum;
}

float TrainingCaptureConfig::GetMaxTypeMinimum() const
{
    return m_maxTypeMinimum;
}

// ----------------------------------------------------------------------------------------------------------
// Cloud structural noise
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetTotalNoiseScaleRange(float minScale, float maxScale)
{
    const float a = std::min(minScale, maxScale);
    const float b = std::max(minScale, maxScale);
    m_minTotalNoiseScale = a;
    m_maxTotalNoiseScale = b;
}

float TrainingCaptureConfig::GetMinTotalNoiseScale() const
{
    return m_minTotalNoiseScale;
}

float TrainingCaptureConfig::GetMaxTotalNoiseScale() const
{
    return m_maxTotalNoiseScale;
}

// ----------------------------------------------------------------------------------------------------------
// Sun lighting
// ----------------------------------------------------------------------------------------------------------
void TrainingCaptureConfig::SetSunDirectionRange(const glm::vec3& minDirection, const glm::vec3& maxDirection)
{
    m_minSunDirection = glm::vec3(
        std::min(minDirection.x, maxDirection.x),
        std::min(minDirection.y, maxDirection.y),
        std::min(minDirection.z, maxDirection.z));

    m_maxSunDirection = glm::vec3(
        std::max(minDirection.x, maxDirection.x),
        std::max(minDirection.y, maxDirection.y),
        std::max(minDirection.z, maxDirection.z));
}

glm::vec3 TrainingCaptureConfig::GetMinSunDirection() const
{
    return m_minSunDirection;
}

glm::vec3 TrainingCaptureConfig::GetMaxSunDirection() const
{
    return m_maxSunDirection;
}

void TrainingCaptureConfig::SetSunIlluminanceRange(const glm::vec3& minIlluminance, const glm::vec3& maxIlluminance)
{
    m_minSunIlluminance = glm::vec3(
        std::min(minIlluminance.x, maxIlluminance.x),
        std::min(minIlluminance.y, maxIlluminance.y),
        std::min(minIlluminance.z, maxIlluminance.z));

    m_maxSunIlluminance = glm::vec3(
        std::max(minIlluminance.x, maxIlluminance.x),
        std::max(minIlluminance.y, maxIlluminance.y),
        std::max(minIlluminance.z, maxIlluminance.z));
}

glm::vec3 TrainingCaptureConfig::GetMinSunIlluminance() const
{
    return m_minSunIlluminance;
}

glm::vec3 TrainingCaptureConfig::GetMaxSunIlluminance() const
{
    return m_maxSunIlluminance;
}