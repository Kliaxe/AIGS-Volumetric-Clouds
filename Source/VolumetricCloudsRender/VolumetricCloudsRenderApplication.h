#pragma once

#include "Application/Application.h"

#include "Renderer/Renderer.h"
#include "Camera/CameraController.h"
#include "Utils/DearImGui.h"

#include "Training/TrainingCaptureConfig.h"
#include "Training/TrainingCameraSampler.h"
#include "Training/TrainingCaptureRunner.h"

#include <glm/glm.hpp>

class TrainingCaptureConfig;
class TrainingCameraSampler;
class TrainingCaptureRunner;

class Material;
class FramebufferObject;
class Texture2DObject;
class VolumetricCloudsRenderPass;
class VolumetricCloudsSkyPass;

class VolumetricCloudsRenderApplication : public Application
{
public:
    VolumetricCloudsRenderApplication();

    void SetSunDirection(const glm::vec3& direction);
    const glm::vec3& GetSunDirection() const;

    void SetSunIlluminance(const glm::vec3& illuminance);
    const glm::vec3& GetSunIlluminance() const;

    void SetAmbientLuminance(const glm::vec3& luminance);
    const glm::vec3& GetAmbientLuminance() const;

    void EnableTrainingMode(bool enabled);
    bool IsTrainingModeEnabled() const;

    TrainingCaptureConfig& GetTrainingCaptureConfig();
    const TrainingCaptureConfig& GetTrainingCaptureConfig() const;

    TrainingCameraSampler& GetTrainingCameraSampler();
    const TrainingCameraSampler& GetTrainingCameraSampler() const;

protected:
    void Initialize() override;
    void Update() override;
    void Render() override;
    void Cleanup() override;

private:
    friend class TrainingCaptureRunner;

    void InitializeCamera();
    void InitializeMaterial();
    void InitializeRenderer();
    void InitializeTrainingSupport();

    void RenderSceneContent();

    std::shared_ptr<Material> CreateVolumetricCloudsRenderMaterial();
    std::shared_ptr<Material> CreateTonemappingMaterial();
    void EnsureHdrRenderTargetSized();

    void RenderGUI();
    void UpdateCloudLightingUniforms();

private:
    // Helper object for debug GUI
    DearImGui m_imGui;

    // Camera controller
    CameraController m_cameraController;

    // Renderer
    Renderer m_renderer;

    // Materials
    std::shared_ptr<Material> m_volumetricCloudsRenderMaterial;
    std::shared_ptr<Material> m_tonemappingMaterial;

    VolumetricCloudsRenderPass* m_volumetricCloudsPass = nullptr;
    VolumetricCloudsSkyPass* m_skyPass = nullptr;

    std::shared_ptr<FramebufferObject> m_hdrFramebuffer;
    std::shared_ptr<Texture2DObject> m_hdrColorTexture;
    int m_hdrWidth = 0;
    int m_hdrHeight = 0;

    glm::vec3 m_sunDirection = glm::vec3(0.968, 0.25f, 0.0f);

    const float m_sunIlluminanceMultiplier = 5.0f;
    const float m_ambientLuminanceMultiplier = 0.35f;

    // Fallback values for the sun illuminance and ambient luminance.
    glm::vec3 m_sunIlluminance = glm::vec3(1.0f, 0.9f, 0.8f) * m_sunIlluminanceMultiplier;
    glm::vec3 m_ambientLuminance = glm::vec3(0.2f, 0.5f, 1.0f) * m_ambientLuminanceMultiplier;

    bool m_trainingModeRequested = false;
    TrainingCaptureConfig m_trainingConfig;
    TrainingCameraSampler m_trainingSampler;
    std::unique_ptr<TrainingCaptureRunner> m_trainingRunner;
};
