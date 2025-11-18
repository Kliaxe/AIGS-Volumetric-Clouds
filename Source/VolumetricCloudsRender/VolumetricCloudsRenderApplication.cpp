#include "VolumetricCloudsRenderApplication.h"

#include "Asset/ShaderLoader.h"
#include "Camera/Camera.h"
#include "Scene/SceneCamera.h"
#include "Lighting/DirectionalLight.h"
#include "Shader/Material.h"
#include "Renderer/PostFXRenderPass.h"
#include "Scene/RendererSceneVisitor.h"
#include "VolumetricCloudsRenderPass.h"
#include "VolumetricCloudsSkyPass.h"
#include "Training/TrainingCaptureRunner.h"
#include "Training/TrainingCaptureConfig.h"
#include "Training/TrainingCameraSampler.h"
#include "VolumetricCloudsRenderUtils.h"
#include "Texture/FramebufferObject.h"
#include "Texture/Texture2DObject.h"
#include "Texture/TextureObject.h"
#include <array>
#include <string>
#include <cstdint>
#include <imgui.h>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

VolumetricCloudsRenderApplication::VolumetricCloudsRenderApplication()
 : Application(1280, 720, "Volumetric Clouds Render")
 , m_renderer(GetDevice())
{
    SetSunDirection(m_sunDirection);
}

void VolumetricCloudsRenderApplication::SetSunDirection(const glm::vec3& direction)
{
    const float magnitude = glm::length(direction);
    if (magnitude > 0.0f)
    {
        m_sunDirection = direction / magnitude;
    }
    else
    {
        m_sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    m_sunIlluminance = VolumetricCloudsRenderUtils::ComputeAtmosphericSunLight(m_sunDirection);
    m_sunIlluminance *= m_sunIlluminanceMultiplier;

    UpdateCloudLightingUniforms();
}

const glm::vec3& VolumetricCloudsRenderApplication::GetSunDirection() const
{
    return m_sunDirection;
}

void VolumetricCloudsRenderApplication::SetSunIlluminance(const glm::vec3& illuminance)
{
    m_sunIlluminance = glm::max(illuminance, glm::vec3(0.0f));
    UpdateCloudLightingUniforms();
}

const glm::vec3& VolumetricCloudsRenderApplication::GetSunIlluminance() const
{
    return m_sunIlluminance;
}

void VolumetricCloudsRenderApplication::SetAmbientLuminance(const glm::vec3& luminance)
{
    m_ambientLuminance = glm::max(luminance, glm::vec3(0.0f));
    UpdateCloudLightingUniforms();
}

const glm::vec3& VolumetricCloudsRenderApplication::GetAmbientLuminance() const
{
    return m_ambientLuminance;
}

void VolumetricCloudsRenderApplication::EnableTrainingMode(bool enabled)
{
    m_trainingModeRequested = enabled;

    if (m_trainingRunner)
    {
        m_trainingRunner->SetActive(enabled);
    }
}

bool VolumetricCloudsRenderApplication::IsTrainingModeEnabled() const
{
    return m_trainingRunner && m_trainingRunner->IsActive();
}

TrainingCaptureConfig& VolumetricCloudsRenderApplication::GetTrainingCaptureConfig()
{
    return m_trainingConfig;
}

const TrainingCaptureConfig& VolumetricCloudsRenderApplication::GetTrainingCaptureConfig() const
{
    return m_trainingConfig;
}

TrainingCameraSampler& VolumetricCloudsRenderApplication::GetTrainingCameraSampler()
{
    return m_trainingSampler;
}

const TrainingCameraSampler& VolumetricCloudsRenderApplication::GetTrainingCameraSampler() const
{
    return m_trainingSampler;
}

void VolumetricCloudsRenderApplication::Initialize()
{
    Application::Initialize();

    // Initialize DearImGUI
    m_imGui.Initialize(GetMainWindow());

    InitializeCamera();
    InitializeMaterial();
    InitializeRenderer();
    InitializeTrainingSupport();
}

void VolumetricCloudsRenderApplication::Update()
{
    Application::Update();

    // Update camera controller only when free-roaming mode is active
    if (!IsTrainingModeEnabled())
    {
        m_cameraController.Update(GetMainWindow(), GetDeltaTime());
    }

    if (m_volumetricCloudsPass != nullptr)
    {
        m_volumetricCloudsPass->SetTime(GetCurrentTime());
    }

    // Set renderer camera
    if (auto sceneCamera = m_cameraController.GetCamera())
    {
        if (auto camera = sceneCamera->GetCamera())
        {
            m_renderer.SetCurrentCamera(*camera);
        }
    }
}

void VolumetricCloudsRenderApplication::Render()
{
    Application::Render();

    if (m_trainingRunner && m_trainingRunner->IsActive())
    {
        m_trainingRunner->ProcessFrame();
        return;
    }

    RenderSceneContent();

    // Render the debug user interface
    RenderGUI();
}

void VolumetricCloudsRenderApplication::Cleanup()
{
    // Cleanup DearImGUI
    m_imGui.Cleanup();

    Application::Cleanup();
}

void VolumetricCloudsRenderApplication::InitializeCamera()
{
    // Create the main camera
    std::shared_ptr<Camera> camera = std::make_shared<Camera>();
    camera->SetViewMatrix(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    float fov = 1.0f;
    camera->SetPerspectiveProjectionMatrix(fov, GetMainWindow().GetAspectRatio(), 0.1f, 100000.0f);
    
    // Create a scene node for the camera
    std::shared_ptr<SceneCamera> sceneCamera = std::make_shared<SceneCamera>("camera", camera);

    // Set the camera scene node to be controlled by the camera controller
    m_cameraController.SetCamera(sceneCamera);
}

void VolumetricCloudsRenderApplication::InitializeMaterial()
{
    m_volumetricCloudsRenderMaterial = CreateVolumetricCloudsRenderMaterial();
    m_tonemappingMaterial = CreateTonemappingMaterial();
}

void VolumetricCloudsRenderApplication::InitializeRenderer()
{
    if (!m_hdrFramebuffer)
    {
        m_hdrFramebuffer = std::make_shared<FramebufferObject>();
    }

    EnsureHdrRenderTargetSized();

    std::shared_ptr<const FramebufferObject> hdrTarget = std::static_pointer_cast<const FramebufferObject>(m_hdrFramebuffer);

    // Sky pass -------------------------------------------------------------------
    {
        auto skyPass = std::make_unique<VolumetricCloudsSkyPass>("Content/HDRI/qwantani_puresky_4k.hdr", hdrTarget);

        m_skyPass = skyPass.get();

        m_renderer.AddRenderPass(std::move(skyPass));
    }

    // Volumetric clouds compute pass ---------------------------------------------
    std::shared_ptr<Material> blit = std::make_shared<Material>(m_volumetricCloudsRenderMaterial->GetShaderProgram());

    auto volumetricCloudsPass = std::make_unique<VolumetricCloudsRenderPass>(blit, hdrTarget);

    m_volumetricCloudsPass = volumetricCloudsPass.get();

    m_renderer.AddRenderPass(std::move(volumetricCloudsPass));

    UpdateCloudLightingUniforms();

    // Tonemapping pass ------------------------------------------------------------
    {
        auto tonemapPass = std::make_unique<PostFXRenderPass>(m_tonemappingMaterial, FramebufferObject::GetDefault());
        m_renderer.AddRenderPass(std::move(tonemapPass));
    }
}

void VolumetricCloudsRenderApplication::InitializeTrainingSupport()
{
    if (!m_volumetricCloudsPass)
    {
        return;
    }

    if (!m_trainingRunner)
    {
        m_trainingRunner = std::make_unique<TrainingCaptureRunner>(*this,
                                                                   m_trainingConfig,
                                                                   m_trainingSampler,
                                                                   m_cameraController,
                                                                   *m_volumetricCloudsPass);
    }

    if (m_trainingModeRequested)
    {
        m_trainingRunner->SetActive(true);
    }
}

void VolumetricCloudsRenderApplication::RenderSceneContent()
{
    if (auto sceneCamera = m_cameraController.GetCamera())
    {
        if (auto camera = sceneCamera->GetCamera())
        {
            m_renderer.SetCurrentCamera(*camera);
        }
    }

    EnsureHdrRenderTargetSized();

    GetDevice().Clear(true, Color(0.0f, 0.0f, 0.0f, 1.0f), true, 1.0f);

    m_renderer.Render();
}

std::shared_ptr<Material> VolumetricCloudsRenderApplication::CreateVolumetricCloudsRenderMaterial()
{
    // Build composite material (fullscreen blend of clouds over sky)
    std::vector<const char*> vertexShaderPaths;
    vertexShaderPaths.push_back("shaders/renderer/fullscreen.vert");
    Shader vertexShader = ShaderLoader(Shader::VertexShader).Load(vertexShaderPaths);

    std::vector<const char*> fragmentShaderPaths;
    fragmentShaderPaths.push_back("shaders/volumetricclouds-composite.frag");
    Shader fragmentShader = ShaderLoader(Shader::FragmentShader).Load(fragmentShaderPaths);

    std::shared_ptr<ShaderProgram> shaderProgramPtr = std::make_shared<ShaderProgram>();
    shaderProgramPtr->Build(vertexShader, fragmentShader);

    // Create material
    std::shared_ptr<Material> material = std::make_shared<Material>(shaderProgramPtr);

    return material;
}

std::shared_ptr<Material> VolumetricCloudsRenderApplication::CreateTonemappingMaterial()
{
    std::vector<const char*> vertexShaderPaths;
    vertexShaderPaths.push_back("shaders/renderer/fullscreen.vert");
    Shader vertexShader = ShaderLoader(Shader::VertexShader).Load(vertexShaderPaths);

    std::vector<const char*> fragmentShaderPaths;
    fragmentShaderPaths.push_back("shaders/tonemapping.frag");
    Shader fragmentShader = ShaderLoader(Shader::FragmentShader).Load(fragmentShaderPaths);

    std::shared_ptr<ShaderProgram> shaderProgramPtr = std::make_shared<ShaderProgram>();
    shaderProgramPtr->Build(vertexShader, fragmentShader);

    std::shared_ptr<Material> material = std::make_shared<Material>(shaderProgramPtr);

    material->SetDepthWrite(false);
    material->SetDepthTestFunction(Material::TestFunction::Always);
    material->SetBlendEquation(Material::BlendEquation::None);
    material->SetBlendParams(Material::BlendParam::One, Material::BlendParam::Zero);

    material->SetUniformValue("Exposure", 1.0f);
    material->SetUniformValue("DebugValueA", 0.0f);
    material->SetUniformValue("DebugValueB", 0.0f);

    return material;
}

void VolumetricCloudsRenderApplication::EnsureHdrRenderTargetSized()
{
    GLint viewportX = 0;
    GLint viewportY = 0;
    GLsizei viewportWidth = 0;
    GLsizei viewportHeight = 0;

    GetDevice().GetViewport(viewportX, viewportY, viewportWidth, viewportHeight);

    if (viewportWidth <= 0 || viewportHeight <= 0)
    {
        return;
    }

    if (m_hdrColorTexture && m_hdrWidth == viewportWidth && m_hdrHeight == viewportHeight)
    {
        return;
    }

    m_hdrWidth = viewportWidth;
    m_hdrHeight = viewportHeight;

    if (!m_hdrFramebuffer)
    {
        m_hdrFramebuffer = std::make_shared<FramebufferObject>();
    }

    m_hdrColorTexture = std::make_shared<Texture2DObject>();
    m_hdrColorTexture->Bind();
    m_hdrColorTexture->SetImage(0,
                                viewportWidth,
                                viewportHeight,
                                TextureObject::FormatRGBA,
                                TextureObject::InternalFormatRGBA32F);
    m_hdrColorTexture->SetParameter(TextureObject::ParameterEnum::MinFilter, GL_LINEAR);
    m_hdrColorTexture->SetParameter(TextureObject::ParameterEnum::MagFilter, GL_LINEAR);
    m_hdrColorTexture->SetParameter(TextureObject::ParameterEnum::WrapS, GL_CLAMP_TO_EDGE);
    m_hdrColorTexture->SetParameter(TextureObject::ParameterEnum::WrapT, GL_CLAMP_TO_EDGE);
    Texture2DObject::Unbind();

    m_hdrFramebuffer->Bind(FramebufferObject::Target::Draw);
    m_hdrFramebuffer->SetTexture(FramebufferObject::Target::Draw,
                                 FramebufferObject::Attachment::Color0,
                                 *m_hdrColorTexture);
    const std::array attachments{ FramebufferObject::Attachment::Color0 };
    m_hdrFramebuffer->SetDrawBuffers(attachments);
    FramebufferObject::Unbind(FramebufferObject::Target::Draw);

    if (m_tonemappingMaterial)
    {
        ShaderProgram::Location sourceLocation = m_tonemappingMaterial->GetUniformLocation("SourceTexture");
        if (sourceLocation >= 0)
        {
            std::shared_ptr<const TextureObject> texturePtr = std::static_pointer_cast<const TextureObject>(m_hdrColorTexture);
            m_tonemappingMaterial->SetUniformValue(sourceLocation, texturePtr);
        }
    }
}

void VolumetricCloudsRenderApplication::RenderGUI()
{
    m_imGui.BeginFrame();

    // Draw GUI for camera controller
    m_cameraController.DrawGUI(m_imGui);

    if (auto window = m_imGui.UseWindow("Training Capture"))
    {
        const bool trainingActive = IsTrainingModeEnabled();
        const std::uint32_t completedPairs = (m_trainingRunner ? m_trainingRunner->GetCompletedPairs() : 0U);

        ImGui::Text("Status: %s", trainingActive ? "Capturing" : "Idle");
        ImGui::Text("Completed Pairs: %u", completedPairs);

        int targetPairs = static_cast<int>(m_trainingConfig.GetTargetBatchSize());
        if (ImGui::InputInt("Target Pairs", &targetPairs, 1, 32))
        {
            if (targetPairs < 0)
            {
                targetPairs = 0;
            }
            m_trainingConfig.SetTargetBatchSize(static_cast<std::uint32_t>(targetPairs));
        }

        float lowScale = m_trainingConfig.GetLowResolutionScale();
        if (ImGui::SliderFloat("Low Res Scale", &lowScale, 0.05f, 1.0f))
        {
            m_trainingConfig.SetLowResolutionScale(lowScale);
        }

        int lowSteps = m_trainingConfig.GetLowResolutionMaxStepCount();
        if (ImGui::InputInt("Low Res Steps", &lowSteps, 1, 16))
        {
            m_trainingConfig.SetLowResolutionMaxStepCount(lowSteps);
        }

        int highSteps = m_trainingConfig.GetFullResolutionMaxStepCount();
        if (ImGui::InputInt("High Res Steps", &highSteps, 1, 16))
        {
            m_trainingConfig.SetFullResolutionMaxStepCount(highSteps);
        }

        bool lowResJitterEnabled = m_trainingConfig.GetLowResolutionJitterEnabled();
        if (ImGui::Checkbox("Low Res Jitter", &lowResJitterEnabled))
        {
            m_trainingConfig.SetLowResolutionJitterEnabled(lowResJitterEnabled);
        }

        bool highResJitterEnabled = m_trainingConfig.GetHighResolutionJitterEnabled();
        if (ImGui::Checkbox("High Res Jitter", &highResJitterEnabled))
        {
            m_trainingConfig.SetHighResolutionJitterEnabled(highResJitterEnabled);
        }

        ImGui::Separator();

        const std::string outputDirectoryString = m_trainingConfig.GetOutputDirectory().string();
        ImGui::Text("Output Dir:");
        ImGui::TextWrapped("%s", outputDirectoryString.c_str());

        ImGui::Separator();

        if (!trainingActive)
        {
            if (ImGui::Button("Start Training Capture"))
            {
                EnableTrainingMode(true);
            }
        }
        else
        {
            if (ImGui::Button("Stop"))
            {
                EnableTrainingMode(false);
            }
        }
    }

    m_imGui.EndFrame();
}

void VolumetricCloudsRenderApplication::UpdateCloudLightingUniforms()
{
    if (m_volumetricCloudsPass == nullptr)
    {
        return;
    }

    m_volumetricCloudsPass->SetSunDirection(m_sunDirection);
    m_volumetricCloudsPass->SetSunIlluminance(m_sunIlluminance);
    m_volumetricCloudsPass->SetAmbientLuminance(m_ambientLuminance);
}