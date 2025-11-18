#pragma once

#include <random>
#include <cstdint>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

// Forward declarations -----------------------------------------------------------------------------
class Camera;

// --------------------------------------------------------------------------------------------------
// TrainingCameraSampler
// --------------------------------------------------------------------------------------------------
//  Generates deterministic-but-configurable camera poses around the area of interest so that
//  each capture pair can cover a diverse set of viewpoints. The sampler owns its RNG state so that
//  future training runs can be reproduced by reusing the same seed.
// --------------------------------------------------------------------------------------------------
class TrainingCameraSampler
{
public:

    struct CameraSample
    {
        glm::vec3 position;
        glm::vec3 focusPoint;
        glm::vec3 upVector;
    };

public:

    TrainingCameraSampler();

    void SetRandomSeed(std::uint64_t seed);
    std::uint64_t GetRandomSeed() const;

    void SetFocusPoint(const glm::vec3& focusPoint);
    glm::vec3 GetFocusPoint() const;

    void SetRadiusRange(float minRadius, float maxRadius);
    float GetMinRadius() const;
    float GetMaxRadius() const;

    void SetAltitudeRange(float minAltitude, float maxAltitude);
    float GetMinAltitude() const;
    float GetMaxAltitude() const;

    void SetYawRange(float minYawRadians, float maxYawRadians);
    float GetMinYaw() const;
    float GetMaxYaw() const;

    void SetPitchRange(float minPitchRadians, float maxPitchRadians);
    float GetMinPitch() const;
    float GetMaxPitch() const;

    CameraSample GenerateSample();

private:

    std::mt19937_64 m_randomEngine;
    std::uint64_t m_seed;

    glm::vec3 m_focusPoint;

    float m_minRadius;
    float m_maxRadius;

    float m_minAltitude;
    float m_maxAltitude;

    float m_minYaw;
    float m_maxYaw;

    float m_minPitch;
    float m_maxPitch;
};


