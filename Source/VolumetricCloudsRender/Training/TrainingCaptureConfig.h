#pragma once

#include <filesystem>
#include <cstdint>
#include <glm/vec3.hpp>

// ----------------------------------------------------------------------------------------------------------
// TrainingCaptureConfig
// ----------------------------------------------------------------------------------------------------------
//  Encapsulates all the tunable knobs for a training capture session.
//  The goal is to keep these parameters in one place so that future runs
//  can tweak behaviour without touching the capture orchestration code.
// ----------------------------------------------------------------------------------------------------------
class TrainingCaptureConfig
{
public:

    TrainingCaptureConfig();

    // --- Animation time override (capture-only) -------------------------------
    // Always enabled during capture. Specify the range used to sample uTime.
    void SetAnimationTimeRange(float minSeconds, float maxSeconds);
    float GetMinAnimationTime() const;
    float GetMaxAnimationTime() const;

    // --- Target batch size ---------------------------------------------------
    void SetTargetBatchSize(std::uint32_t pairCount);
    std::uint32_t GetTargetBatchSize() const;

    // --- Resolution scaling --------------------------------------------------
    void SetLowResolutionScale(float scale);
    float GetLowResolutionScale() const;

    // --- Render iteration budgets -------------------------------------------
    void SetLowResolutionMaxStepCount(int stepCount);
    int GetLowResolutionMaxStepCount() const;

    void SetFullResolutionMaxStepCount(int stepCount);
    int GetFullResolutionMaxStepCount() const;

    // --- Raymarch jitter toggles --------------------------------------------
    // These control whether per-ray white-noise jitter is applied to the
    // marching pattern during training captures.
    void SetLowResolutionJitterEnabled(bool enabled);
    bool GetLowResolutionJitterEnabled() const;

    void SetHighResolutionJitterEnabled(bool enabled);
    bool GetHighResolutionJitterEnabled() const;

    // --- Output path ---------------------------------------------------------
    void SetOutputDirectory(const std::filesystem::path& directory);
    const std::filesystem::path& GetOutputDirectory() const;

    // --- Randomisation seed --------------------------------------------------
    void SetRandomSeed(std::uint64_t seed);
    std::uint64_t GetRandomSeed() const;

    // --- Cloud optical properties -------------------------------------------
    // Scalar density multiplier range used when sampling uExtinctionCoefficientMultiplier.
    void SetExtinctionDensityMultiplierRange(float minMultiplier, float maxMultiplier);
    float GetMinExtinctionDensityMultiplier() const;
    float GetMaxExtinctionDensityMultiplier() const;

    // --- Cloud density shaping (weather map) --------------------------------
    // Coverage amount and minimum used to remap weather map coverage channel.
    void SetCoverageAmountRange(float minAmount, float maxAmount);
    float GetMinCoverageAmount() const;
    float GetMaxCoverageAmount() const;

    void SetCoverageMinimumRange(float minMinimum, float maxMinimum);
    float GetMinCoverageMinimum() const;
    float GetMaxCoverageMinimum() const;

    // Type amount and minimum used to remap weather map type channel.
    void SetTypeAmountRange(float minAmount, float maxAmount);
    float GetMinTypeAmount() const;
    float GetMaxTypeAmount() const;

    void SetTypeMinimumRange(float minMinimum, float maxMinimum);
    float GetMinTypeMinimum() const;
    float GetMaxTypeMinimum() const;

    // --- Cloud structural noise ---------------------------------------------
    // Scalar range used when sampling uTotalNoiseScale.
    void SetTotalNoiseScaleRange(float minScale, float maxScale);
    float GetMinTotalNoiseScale() const;
    float GetMaxTotalNoiseScale() const;

    // --- Sun lighting --------------------------------------------------------
    // Sun direction range used for uSunDirection. Directions are normalised
    // before being sent to the shader.
    void SetSunDirectionRange(const glm::vec3& minDirection, const glm::vec3& maxDirection);
    glm::vec3 GetMinSunDirection() const;
    glm::vec3 GetMaxSunDirection() const;

    // Sun illuminance range used for uSunIlluminance.
    void SetSunIlluminanceRange(const glm::vec3& minIlluminance, const glm::vec3& maxIlluminance);
    glm::vec3 GetMinSunIlluminance() const;
    glm::vec3 GetMaxSunIlluminance() const;

private:

    float m_minAnimationTimeSeconds;
    float m_maxAnimationTimeSeconds;

    std::uint32_t m_targetBatchSize;

    float m_lowResolutionScale;

    int m_lowResolutionMaxStepCount;
    int m_fullResolutionMaxStepCount;

    bool m_lowResolutionJitterEnabled;
    bool m_highResolutionJitterEnabled;

    std::filesystem::path m_outputDirectory;

    std::uint64_t m_randomSeed;

    // Cloud optical properties ------------------------------------------------
    float m_minExtinctionDensityMultiplier;
    float m_maxExtinctionDensityMultiplier;

    // Cloud density shaping (weather map) -------------------------------------
    float m_minCoverageAmount;
    float m_maxCoverageAmount;
    float m_minCoverageMinimum;
    float m_maxCoverageMinimum;

    float m_minTypeAmount;
    float m_maxTypeAmount;
    float m_minTypeMinimum;
    float m_maxTypeMinimum;

    // Cloud structural noise --------------------------------------------------
    float m_minTotalNoiseScale;
    float m_maxTotalNoiseScale;

    // Sun lighting ------------------------------------------------------------
    glm::vec3 m_minSunDirection;
    glm::vec3 m_maxSunDirection;
    glm::vec3 m_minSunIlluminance;
    glm::vec3 m_maxSunIlluminance;
};


