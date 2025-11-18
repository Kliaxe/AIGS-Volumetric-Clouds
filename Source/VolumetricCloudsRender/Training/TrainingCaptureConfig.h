#pragma once

#include <filesystem>
#include <cstdint>

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
};


