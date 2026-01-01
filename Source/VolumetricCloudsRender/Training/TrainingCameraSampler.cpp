#include "TrainingCameraSampler.h"

#include <algorithm>
#include <glm/gtc/constants.hpp>
#include <glm/glm.hpp>

// Single viewpoint dataset? If defined, the camera will be fixed at the origin.
#define DATASET_SINGLE

// --------------------------------------------------------------------------------------------------
// ctor
// --------------------------------------------------------------------------------------------------
TrainingCameraSampler::TrainingCameraSampler()
    : m_randomEngine(0xC0FFEEULL)
    , m_seed(0xC0FFEEULL)
    , m_focusPoint(0.0f, 0.0f, 0.0f) // Keep the focus point at the origin for now.
#ifdef DATASET_SINGLE
    , m_minRadius(1.0f)
    , m_maxRadius(1.0f)
    , m_minAltitude(0.0f)
    , m_maxAltitude(0.0f)
    , m_minYaw(0.0f)
    , m_maxYaw(0.0f)
    , m_minPitch(0.0f)
    , m_maxPitch(0.0f)
#else
    , m_minRadius(6000.0f)
    , m_maxRadius(12000.0f)
    , m_minAltitude(0.0f)
    , m_maxAltitude(2500.0f)
    , m_minYaw(-glm::pi<float>())
    , m_maxYaw(glm::pi<float>())
    , m_minPitch(glm::radians(-10.0f))
    , m_maxPitch(glm::radians(80.0f))
#endif
{
}

// --------------------------------------------------------------------------------------------------
// Seed management
// --------------------------------------------------------------------------------------------------
void TrainingCameraSampler::SetRandomSeed(std::uint64_t seed)
{
    m_seed = seed;
    m_randomEngine.seed(seed);
}

std::uint64_t TrainingCameraSampler::GetRandomSeed() const
{
    return m_seed;
}

// --------------------------------------------------------------------------------------------------
// Focus point
// --------------------------------------------------------------------------------------------------
void TrainingCameraSampler::SetFocusPoint(const glm::vec3& focusPoint)
{
    m_focusPoint = focusPoint;
}

glm::vec3 TrainingCameraSampler::GetFocusPoint() const
{
    return m_focusPoint;
}

// --------------------------------------------------------------------------------------------------
// Radius range
// --------------------------------------------------------------------------------------------------
void TrainingCameraSampler::SetRadiusRange(float minRadius, float maxRadius)
{
    m_minRadius = std::max(0.0f, std::min(minRadius, maxRadius));
    m_maxRadius = std::max(m_minRadius + 1.0f, maxRadius);
}

float TrainingCameraSampler::GetMinRadius() const
{
    return m_minRadius;
}

float TrainingCameraSampler::GetMaxRadius() const
{
    return m_maxRadius;
}

// --------------------------------------------------------------------------------------------------
// Altitude range
// --------------------------------------------------------------------------------------------------
void TrainingCameraSampler::SetAltitudeRange(float minAltitude, float maxAltitude)
{
    m_minAltitude = std::min(minAltitude, maxAltitude);
    m_maxAltitude = std::max(minAltitude, maxAltitude);
}

float TrainingCameraSampler::GetMinAltitude() const
{
    return m_minAltitude;
}

float TrainingCameraSampler::GetMaxAltitude() const
{
    return m_maxAltitude;
}

// --------------------------------------------------------------------------------------------------
// Yaw range
// --------------------------------------------------------------------------------------------------
void TrainingCameraSampler::SetYawRange(float minYawRadians, float maxYawRadians)
{
    m_minYaw = std::min(minYawRadians, maxYawRadians);
    m_maxYaw = std::max(minYawRadians, maxYawRadians);
}

float TrainingCameraSampler::GetMinYaw() const
{
    return m_minYaw;
}

float TrainingCameraSampler::GetMaxYaw() const
{
    return m_maxYaw;
}

// --------------------------------------------------------------------------------------------------
// Pitch range
// --------------------------------------------------------------------------------------------------
void TrainingCameraSampler::SetPitchRange(float minPitchRadians, float maxPitchRadians)
{
    m_minPitch = std::min(minPitchRadians, maxPitchRadians);
    m_maxPitch = std::max(minPitchRadians, maxPitchRadians);
}

float TrainingCameraSampler::GetMinPitch() const
{
    return m_minPitch;
}

float TrainingCameraSampler::GetMaxPitch() const
{
    return m_maxPitch;
}

// --------------------------------------------------------------------------------------------------
// Sample generation
// --------------------------------------------------------------------------------------------------
TrainingCameraSampler::CameraSample TrainingCameraSampler::GenerateSample()
{
    std::uniform_real_distribution<float> yawDistribution(m_minYaw, m_maxYaw);
    std::uniform_real_distribution<float> pitchDistribution(m_minPitch, m_maxPitch);
    std::uniform_real_distribution<float> radiusDistribution(m_minRadius, m_maxRadius);
    std::uniform_real_distribution<float> altitudeDistribution(m_minAltitude, m_maxAltitude);

    const float radius = radiusDistribution(m_randomEngine);
    const float yaw = yawDistribution(m_randomEngine);
    const float pitch = pitchDistribution(m_randomEngine);
    const float altitudeOffset = altitudeDistribution(m_randomEngine);

    CameraSample sample;

    sample.position.x = m_focusPoint.x + radius * std::cos(yaw);
    sample.position.z = m_focusPoint.z + radius * std::sin(yaw);
    sample.position.y = m_focusPoint.y + altitudeOffset;

    glm::vec3 adjustedFocus = m_focusPoint;
    // If pitch is fixed (forward-only), keep the focus at the camera altitude to avoid any tilt.
    if (m_minPitch == m_maxPitch)
    {
        adjustedFocus.y = sample.position.y;
    }
    else
    {
        adjustedFocus.y += std::tan(pitch) * radius * 0.25f;
    }

    sample.focusPoint = adjustedFocus;
    sample.upVector = glm::vec3(0.0f, 1.0f, 0.0f);

    return sample;
}


