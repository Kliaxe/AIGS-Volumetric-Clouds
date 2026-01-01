#pragma once

#include <memory>
#include <cstdint>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

// Forward declarations -----------------------------------------------------------------------------
class VolumetricCloudsRenderApplication;
class TrainingCaptureConfig;
class CameraController;
class VolumetricCloudsRenderPass;
class Texture2DObject;

#include "TrainingCameraSampler.h"
#include "TrainingFrameWriter.h"

// --------------------------------------------------------------------------------------------------
// TrainingCaptureRunner
// --------------------------------------------------------------------------------------------------
//  Centralised state machine for running a training capture session. Handles camera sampling, render
//  orchestration, GPU read-back, and delegating persistence to the frame writer.
// --------------------------------------------------------------------------------------------------
class TrainingCaptureRunner
{
public:

    TrainingCaptureRunner(VolumetricCloudsRenderApplication& owner,
                          TrainingCaptureConfig& config,
                          TrainingCameraSampler& sampler,
                          CameraController& cameraController,
                          VolumetricCloudsRenderPass& cloudsPass);

    void SetActive(bool enabled);
    bool IsActive() const;

    void ProcessFrame();

    std::uint32_t GetCompletedPairs() const;

private:

    void BeginSession();
    void EndSession();

    void CaptureNextPair();

    TrainingFrameWriter::FrameCapture CaptureResolution(const glm::ivec2& resolution,
                                                        int maxStepCount,
                                                        bool jitterEnabled);

    void ApplyCameraSample(const TrainingCameraSampler::CameraSample& sample);

private:

    VolumetricCloudsRenderApplication& m_owner;
    TrainingCaptureConfig& m_config;
    TrainingCameraSampler& m_sampler;
    CameraController& m_cameraController;
    VolumetricCloudsRenderPass& m_cloudsPass;

    std::unique_ptr<TrainingFrameWriter> m_frameWriter;

    bool m_isActive;
    bool m_isSessionInitialized;

    std::uint32_t m_completedPairs;

    glm::ivec4 m_originalViewport;

    glm::mat4 m_originalViewMatrix;
    glm::mat4 m_originalProjectionMatrix;
    bool m_originalCameraControllerState;

    // Original cloud state restored when the capture session ends.
    glm::vec3 m_originalSunDirection;
    glm::vec3 m_originalSunIlluminance;
    float     m_originalExtinctionDensityMultiplier;
    float     m_originalTotalNoiseScale;
    float     m_originalCoverageAmount;
    float     m_originalCoverageMinimum;
    float     m_originalTypeAmount;
    float     m_originalTypeMinimum;
};


