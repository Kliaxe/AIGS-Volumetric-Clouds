#include "TrainingCaptureRunner.h"

#include "TrainingCaptureConfig.h"
#include "TrainingCameraSampler.h"
#include "VolumetricCloudsRenderApplication.h"

#include "Application/Window.h"
#include "Camera/Camera.h"
#include "Camera/CameraController.h"
#include "Scene/SceneCamera.h"
#include "VolumetricCloudsRenderPass.h"
#include "TrainingFrameWriter.h"

#include "Core/DeviceGL.h"
#include "Texture/Texture2DObject.h"
#include "Texture/TextureObject.h"
#include "Core/Data.h"

#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

// --------------------------------------------------------------------------------------------------
// ctor
// --------------------------------------------------------------------------------------------------
TrainingCaptureRunner::TrainingCaptureRunner(VolumetricCloudsRenderApplication& owner,
                                             TrainingCaptureConfig& config,
                                             TrainingCameraSampler& sampler,
                                             CameraController& cameraController,
                                             VolumetricCloudsRenderPass& cloudsPass)
    : m_owner(owner)
    , m_config(config)
    , m_sampler(sampler)
    , m_cameraController(cameraController)
    , m_cloudsPass(cloudsPass)
    , m_frameWriter(std::make_unique<TrainingFrameWriter>())
    , m_isActive(false)
    , m_isSessionInitialized(false)
    , m_completedPairs(0U)
    , m_originalViewport(0, 0, 0, 0)
    , m_originalViewMatrix(1.0f)
    , m_originalProjectionMatrix(1.0f)
    , m_originalCameraControllerState(false)
{
}

// --------------------------------------------------------------------------------------------------
// Activation
// --------------------------------------------------------------------------------------------------
void TrainingCaptureRunner::SetActive(bool enabled)
{
    if (enabled == m_isActive)
    {
        return;
    }

    m_isActive = enabled;

    if (m_isActive)
    {
        BeginSession();
    }
    else
    {
        EndSession();
    }
}

bool TrainingCaptureRunner::IsActive() const
{
    return m_isActive;
}

// --------------------------------------------------------------------------------------------------
// Frame-stepped processing
// --------------------------------------------------------------------------------------------------
void TrainingCaptureRunner::ProcessFrame()
{
    if (!m_isActive)
    {
        return;
    }

    if (!m_isSessionInitialized)
    {
        BeginSession();
        if (!m_isSessionInitialized)
        {
            return;
        }
    }

    if (m_completedPairs >= m_config.GetTargetBatchSize())
    {
        EndSession();
        return;
    }

    CaptureNextPair();

    if (m_completedPairs >= m_config.GetTargetBatchSize())
    {
        EndSession();
    }
}

std::uint32_t TrainingCaptureRunner::GetCompletedPairs() const
{
    return m_completedPairs;
}

// --------------------------------------------------------------------------------------------------
// Session boundaries
// --------------------------------------------------------------------------------------------------
void TrainingCaptureRunner::BeginSession()
{
    if (m_isSessionInitialized)
    {
        return;
    }

    DeviceGL& device = m_owner.GetDevice();

    device.GetViewport(m_originalViewport.x,
                       m_originalViewport.y,
                       m_originalViewport.z,
                       m_originalViewport.w);

    if (auto sceneCamera = m_cameraController.GetCamera())
    {
        if (auto camera = sceneCamera->GetCamera())
        {
            m_originalViewMatrix = camera->GetViewMatrix();
            m_originalProjectionMatrix = camera->GetProjectionMatrix();
        }
    }

    m_originalCameraControllerState = m_cameraController.IsEnabled();
    m_cameraController.SetEnabled(false);

    m_sampler.SetRandomSeed(m_config.GetRandomSeed());

    // Configure and start background frame writer
    m_frameWriter->SetOutputDirectory(m_config.GetOutputDirectory());
    m_frameWriter->FlushAndStop(); // Ensure any previous session is fully drained
    m_frameWriter->StartWorker();

    m_completedPairs = 0U;

    if (m_config.GetTargetBatchSize() == 0U)
    {
        std::cerr << "[TrainingCaptureRunner] Target batch size is zero. Session aborted." << std::endl;
        m_isActive = false;
        m_owner.m_trainingModeRequested = false;
        return;
    }

    m_isSessionInitialized = true;

    std::cout << "[TrainingCaptureRunner] Session started. Target pairs="
              << m_config.GetTargetBatchSize() << std::endl;
}

void TrainingCaptureRunner::EndSession()
{
    if (!m_isSessionInitialized)
    {
        m_isActive = false;
        return;
    }

    m_cloudsPass.SetMaxStepCount(m_config.GetFullResolutionMaxStepCount());
    m_cloudsPass.SetJitterEnabled(m_config.GetHighResolutionJitterEnabled());

    DeviceGL& device = m_owner.GetDevice();
    device.SetViewport(m_originalViewport.x,
                       m_originalViewport.y,
                       m_originalViewport.z,
                       m_originalViewport.w);

    if (auto sceneCamera = m_cameraController.GetCamera())
    {
        if (auto camera = sceneCamera->GetCamera())
        {
            camera->SetViewMatrix(m_originalViewMatrix);
            camera->SetProjectionMatrix(m_originalProjectionMatrix);
            sceneCamera->MatchTransformToCamera();
        }
    }

    m_cameraController.SetEnabled(m_originalCameraControllerState);

    // Drain any pending write tasks before reporting session completion
    m_frameWriter->FlushAndStop();

    m_isSessionInitialized = false;
    m_isActive = false;

    m_owner.m_trainingModeRequested = false;

    std::cout << "[TrainingCaptureRunner] Session finished. Captured pairs="
              << m_completedPairs << std::endl;
}

// --------------------------------------------------------------------------------------------------
// Core capture logic
// --------------------------------------------------------------------------------------------------
void TrainingCaptureRunner::CaptureNextPair()
{
    // --------------------------------------------------------------------------------------------------
    // Timing scope (per pair)
    // --------------------------------------------------------------------------------------------------
    using Clock = std::chrono::steady_clock;
    const auto pairStartTime = Clock::now();

    const TrainingCameraSampler::CameraSample sample = m_sampler.GenerateSample();
    ApplyCameraSample(sample);

    // Determine capture time: always override with configured range during capture
    const float tMin = m_config.GetMinAnimationTime();
    const float tMax = m_config.GetMaxAnimationTime();
    float captureTimeSeconds = m_owner.GetCurrentTime();
    {
        // Deterministic RNG per pair using training seed and pair index
        std::mt19937_64 rng(m_config.GetRandomSeed() ^ (0x9E3779B97F4A7C15ULL + static_cast<std::uint64_t>(m_completedPairs) * 0xBF58476D1CE4E5B9ULL));
        std::uniform_real_distribution<float> dist(tMin, tMax);
        captureTimeSeconds = dist(rng);
        m_cloudsPass.SetTime(captureTimeSeconds);
    }

    int windowWidth = 0;
    int windowHeight = 0;
    m_owner.GetMainWindow().GetDimensions(windowWidth, windowHeight);

    glm::ivec2 fullResolution(windowWidth, windowHeight);

    const float lowScale = m_config.GetLowResolutionScale();
    glm::ivec2 lowResolution(
        std::max(1, static_cast<int>(std::lround(static_cast<float>(fullResolution.x) * lowScale))),
        std::max(1, static_cast<int>(std::lround(static_cast<float>(fullResolution.y) * lowScale))));

    // For training data, ensure that low- and full-resolution renders use the
    // same jitter setting so that the noise statistics of the pair match. If
    // the configuration requests different values, we favour the low-resolution
    // setting and emit a single warning.
    const bool lowJitterEnabled = m_config.GetLowResolutionJitterEnabled();
    const bool highJitterEnabled = m_config.GetHighResolutionJitterEnabled();
    const bool pairJitterEnabled = lowJitterEnabled;

    if (lowJitterEnabled != highJitterEnabled)
    {
        static bool s_warnedJitterMismatch = false;
        if (!s_warnedJitterMismatch)
        {
            std::cerr << "[TrainingCaptureRunner] Warning: low/high jitter settings differ "
                      << "(low=" << (lowJitterEnabled ? "true" : "false")
                      << ", high=" << (highJitterEnabled ? "true" : "false")
                      << "). Using low-resolution setting for both captures to keep "
                      << "training pairs statistically consistent." << std::endl;
            s_warnedJitterMismatch = true;
        }
    }

    const auto lowCaptureStartTime = Clock::now();
    TrainingFrameWriter::FrameCapture lowFrame = CaptureResolution(lowResolution,
                                                                   m_config.GetLowResolutionMaxStepCount(),
                                                                   pairJitterEnabled);
    const auto lowCaptureEndTime = Clock::now();

    const auto fullCaptureStartTime = Clock::now();
    TrainingFrameWriter::FrameCapture highFrame = CaptureResolution(fullResolution,
                                                                    m_config.GetFullResolutionMaxStepCount(),
                                                                    pairJitterEnabled);
    const auto fullCaptureEndTime = Clock::now();

    TrainingFrameWriter::CaptureMetadata metadata{};

    if (auto sceneCamera = m_cameraController.GetCamera())
    {
        if (auto camera = sceneCamera->GetCamera())
        {
            metadata.cameraPosition = camera->ExtractTranslation();
            metadata.cameraFocus = sample.focusPoint;
            metadata.viewMatrix = camera->GetViewMatrix();
            metadata.projectionMatrix = camera->GetProjectionMatrix();
            metadata.captureTimeSeconds = captureTimeSeconds;
        }
    }

    const std::uint32_t pairIndex = m_completedPairs;

    const auto writeStartTime = Clock::now();
    m_frameWriter->EnqueuePair(pairIndex,
                               lowFrame,
                               highFrame,
                               metadata);
    const auto writeEndTime = Clock::now();

    const auto pairEndTime = writeEndTime;

    const auto lowCaptureDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(lowCaptureEndTime - lowCaptureStartTime).count();
    const auto fullCaptureDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(fullCaptureEndTime - fullCaptureStartTime).count();
    const auto writeDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(writeEndTime - writeStartTime).count();
    const auto totalPairDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(pairEndTime - pairStartTime).count();

    std::cout << "[TrainingCaptureRunner] Pair " << pairIndex
              << " timings: lowCapture=" << lowCaptureDurationMs << " ms, "
              << "fullCapture=" << fullCaptureDurationMs << " ms, "
              << "writePair=" << writeDurationMs << " ms, "
              << "total=" << totalPairDurationMs << " ms"
              << std::endl;

    // Count this pair as captured. Disk writes are now handled by the background worker.
    ++m_completedPairs;

    // Restore application-driven time after capture
    m_cloudsPass.SetTime(m_owner.GetCurrentTime());
}

TrainingFrameWriter::FrameCapture TrainingCaptureRunner::CaptureResolution(const glm::ivec2& resolution,
                                                                           int maxStepCount,
                                                                           bool jitterEnabled)
{
    TrainingFrameWriter::FrameCapture capture{};
    capture.resolution = resolution;

    if (resolution.x <= 0 || resolution.y <= 0)
    {
        return capture;
    }

    DeviceGL& device = m_owner.GetDevice();
    device.SetViewport(0, 0, resolution.x, resolution.y);

    m_cloudsPass.SetMaxStepCount(maxStepCount);
    m_cloudsPass.SetJitterEnabled(jitterEnabled);

    m_owner.RenderSceneContent();

    glFinish();

    const std::shared_ptr<Texture2DObject> outputTexture = m_cloudsPass.GetOutputTexture();

    if (!outputTexture)
    {
        std::cerr << "[TrainingCaptureRunner] Output texture unavailable for read-back." << std::endl;
        return capture;
    }

    const std::size_t totalPixels = static_cast<std::size_t>(resolution.x) * resolution.y;

    auto readbackTexture = [totalPixels](const std::shared_ptr<Texture2DObject>& texture, std::vector<float>& destination)
    {
        if (!texture)
        {
            destination.clear();
            return;
        }

        destination.resize(totalPixels * 4U);

        TextureObject::SetActiveTexture(0);
        texture->Bind();
        texture->GetTextureData(0,
                                TextureObject::FormatRGBA,
                                Data::Type::Float,
                                destination.data());
        Texture2DObject::Unbind();
    };

    // ----------------------------------------------------------------------------------------------
    // Read back colour buffer
    // ----------------------------------------------------------------------------------------------
    capture.rgbaColorPixels.resize(totalPixels * 4U);

    TextureObject::SetActiveTexture(0);
    outputTexture->Bind();
    outputTexture->GetTextureData(0,
                                  TextureObject::FormatRGBA,
                                  Data::Type::Float,
                                  capture.rgbaColorPixels.data());

    // ----------------------------------------------------------------------------------------------
    // Read back auxiliary buffers (if available)
    // ----------------------------------------------------------------------------------------------
    readbackTexture(m_cloudsPass.GetDataTexture(), capture.rgbaDataPixels);
    readbackTexture(m_cloudsPass.GetNormalsTexture(), capture.rgbaNormalPixels);
    Texture2DObject::Unbind();

    return capture;
}

// --------------------------------------------------------------------------------------------------
// Camera sample application
// --------------------------------------------------------------------------------------------------
void TrainingCaptureRunner::ApplyCameraSample(const TrainingCameraSampler::CameraSample& sample)
{
    if (auto sceneCamera = m_cameraController.GetCamera())
    {
        if (auto camera = sceneCamera->GetCamera())
        {
            camera->SetViewMatrix(sample.position,
                                  sample.focusPoint,
                                  sample.upVector);

            sceneCamera->MatchTransformToCamera();
        }
    }
}


