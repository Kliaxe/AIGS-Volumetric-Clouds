#include <vector>
#include "VolumetricCloudsRenderPass.h"

#include "Renderer/Renderer.h"
#include "Shader/Material.h"
#include "Asset/ShaderLoader.h"
#include "Texture/Texture2DObject.h"
#include "Texture/Texture3DObject.h"
#include "Texture/FramebufferObject.h"
#include "Camera/Camera.h"

#include <array>
#include <iostream>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

namespace
{
    void LogShaderInfo(GLuint shaderHandle)
    {
        GLint compilationSuccess = GL_FALSE;
        glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compilationSuccess);

        std::array<char, 2048> shaderLog{};
        glGetShaderInfoLog(shaderHandle, static_cast<GLsizei>(shaderLog.size()), nullptr, shaderLog.data());

        std::cerr << "    [Shader] handle=" << shaderHandle
                  << " compiled=" << (compilationSuccess == GL_TRUE ? "true" : "false")
                  << "\n" << shaderLog.data() << std::endl;
    }

    void LogShaderProgramLinkFailure(const char* label, const ShaderProgram& shaderProgram)
    {
    std::array<char, 2048> infoLog{};
        shaderProgram.GetLinkingErrors(infoLog);

        std::cerr << "[ShaderProgram] Link failure in " << label << "\n"
                  << infoLog.data() << std::endl;

        GLint attachedShaderCount = 0;
        glGetProgramiv(shaderProgram.GetHandle(), GL_ATTACHED_SHADERS, &attachedShaderCount);

        if (attachedShaderCount > 0)
        {
            std::vector<GLuint> attachedShaders(static_cast<std::size_t>(attachedShaderCount));
            glGetAttachedShaders(shaderProgram.GetHandle(), attachedShaderCount, nullptr, attachedShaders.data());

            std::cerr << "    Attached shader logs:" << std::endl;
            for (GLuint shaderHandle : attachedShaders)
            {
                LogShaderInfo(shaderHandle);
            }
        }
    }
}

VolumetricCloudsRenderPass::VolumetricCloudsRenderPass(std::shared_ptr<Material> material, std::shared_ptr<const FramebufferObject> framebuffer)
    : RenderPass(framebuffer)
    , m_material(material)
{
    // Build shape noise compute program by concatenating utils + shape-noise compute
    std::vector<const char*> shapePaths;
    shapePaths.push_back("shaders/version.glsl");
	shapePaths.push_back("shaders/volumetricclouds-uniforms.glsl");
    shapePaths.push_back("shaders/volumetricclouds-utils.glsl");
    shapePaths.push_back("shaders/volumetricclouds-shapenoise.comp");
    Shader shapeCompute = ShaderLoader(Shader::ComputeShader).Load(shapePaths);

    if (m_shapeNoiseProgram.Build(shapeCompute))
    {
        m_shapeVolumeSizeLocation = m_shapeNoiseProgram.GetUniformLocation("VolumeSize");
    }
    else
    {
        LogShaderProgramLinkFailure("VolumetricCloudsRenderPass::ShapeNoise", m_shapeNoiseProgram);
        m_isReady = false;
    }

    // Build detail noise compute program (utils + detail-noise compute)
    std::vector<const char*> detailPaths;
    detailPaths.push_back("shaders/version.glsl");
    detailPaths.push_back("shaders/volumetricclouds-uniforms.glsl");
    detailPaths.push_back("shaders/volumetricclouds-utils.glsl");
    detailPaths.push_back("shaders/volumetricclouds-detailnoise.comp");
    Shader detailCompute = ShaderLoader(Shader::ComputeShader).Load(detailPaths);

    if (m_detailNoiseProgram.Build(detailCompute))
    {
        m_detailVolumeSizeLocation = m_detailNoiseProgram.GetUniformLocation("VolumeSize");
    }
    else
    {
        LogShaderProgramLinkFailure("VolumetricCloudsRenderPass::DetailNoise", m_detailNoiseProgram);
        m_isReady = false;
    }

    // Build curl noise compute program (utils + curl-noise compute)
    std::vector<const char*> curlPaths;
    curlPaths.push_back("shaders/version.glsl");
    curlPaths.push_back("shaders/volumetricclouds-uniforms.glsl");
    curlPaths.push_back("shaders/volumetricclouds-utils.glsl");
    curlPaths.push_back("shaders/volumetricclouds-curlnoise.comp");
    Shader curlCompute = ShaderLoader(Shader::ComputeShader).Load(curlPaths);

    if (m_curlNoiseProgram.Build(curlCompute))
    {
        m_curlTextureSizeLocation = m_curlNoiseProgram.GetUniformLocation("TextureSize");
    }
    else
    {
        LogShaderProgramLinkFailure("VolumetricCloudsRenderPass::CurlNoise", m_curlNoiseProgram);
        m_isReady = false;
    }

    // Build weather map compute program (utils + weather-map compute)
    std::vector<const char*> weatherPaths;
    weatherPaths.push_back("shaders/version.glsl");
    weatherPaths.push_back("shaders/volumetricclouds-uniforms.glsl");
    weatherPaths.push_back("shaders/volumetricclouds-utils.glsl");
    weatherPaths.push_back("shaders/volumetricclouds-weathermap.comp");
    Shader weatherCompute = ShaderLoader(Shader::ComputeShader).Load(weatherPaths);

    if (m_weatherMapProgram.Build(weatherCompute))
    {
        m_weatherTextureSizeLocation = m_weatherMapProgram.GetUniformLocation("TextureSize");
    }
    else
    {
        LogShaderProgramLinkFailure("VolumetricCloudsRenderPass::WeatherMap", m_weatherMapProgram);
        m_isReady = false;
    }

    // Build render compute program that samples the generated shape texture
	std::vector<const char*> renderPaths;
	renderPaths.push_back("shaders/version.glsl");
	renderPaths.push_back("shaders/volumetricclouds-uniforms.glsl");
	renderPaths.push_back("shaders/volumetricclouds-utils.glsl");
	renderPaths.push_back("shaders/volumetricclouds-render.comp");
    Shader computeShader = ShaderLoader(Shader::ComputeShader).Load(renderPaths);

    if (m_computeProgram.Build(computeShader))
    {
        m_frameDimensionsLocation = m_computeProgram.GetUniformLocation("FrameDimensions");
        m_shapeSamplerLocation = m_computeProgram.GetUniformLocation("ShapeNoiseTexture");
        m_detailSamplerLocation = m_computeProgram.GetUniformLocation("DetailNoiseTexture");
        m_curlSamplerLocation = m_computeProgram.GetUniformLocation("CurlNoiseTexture");
        m_weatherSamplerLocation = m_computeProgram.GetUniformLocation("WeatherMapTexture");
        m_invViewProjLocation = m_computeProgram.GetUniformLocation("InvViewProjection");
        m_cameraPositionLocation = m_computeProgram.GetUniformLocation("CameraPosition");
        m_timeLocation = m_computeProgram.GetUniformLocation("uTime");
        m_sunDirectionLocation = m_computeProgram.GetUniformLocation("uSunDirection");
        m_sunIlluminanceLocation = m_computeProgram.GetUniformLocation("uSunIlluminance");
        m_ambientLuminanceLocation = m_computeProgram.GetUniformLocation("uAmbientLuminance");
        m_maxStepCountLocation = m_computeProgram.GetUniformLocation("uMaxStepCount");
        m_jitterEnabledLocation = m_computeProgram.GetUniformLocation("uJitterEnabled");
        m_extinctionCoefficientMultiplierLocation = m_computeProgram.GetUniformLocation("uExtinctionCoefficientMultiplier");
        m_totalNoiseScaleLocation = m_computeProgram.GetUniformLocation("uTotalNoiseScale");
        m_coverageAmountLocation = m_computeProgram.GetUniformLocation("uCoverageAmount");
        m_coverageMinimumLocation = m_computeProgram.GetUniformLocation("uCoverageMinimum");
        m_typeAmountLocation = m_computeProgram.GetUniformLocation("uTypeAmount");
        m_typeMinimumLocation = m_computeProgram.GetUniformLocation("uTypeMinimum");
    }
    else
    {
        LogShaderProgramLinkFailure("VolumetricCloudsRenderPass::RenderCompute", m_computeProgram);
        m_isReady = false;
    }

    if (m_material)
    {
        std::shared_ptr<const ShaderProgram> compositeProgram = m_material->GetShaderProgram();

        if (compositeProgram)
        {
            m_compositeCloudSamplerLocation = compositeProgram->GetUniformLocation("CloudsTexture");
        }
    }
}

void VolumetricCloudsRenderPass::SetTime(float timeSeconds)
{
    m_timeSeconds = timeSeconds;
}

void VolumetricCloudsRenderPass::SetSunDirection(const glm::vec3& direction)
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
}

const glm::vec3& VolumetricCloudsRenderPass::GetSunDirection() const
{
    return m_sunDirection;
}

void VolumetricCloudsRenderPass::SetSunIlluminance(const glm::vec3& illuminance)
{
    m_sunIlluminance = glm::max(illuminance, glm::vec3(0.0f));
}

const glm::vec3& VolumetricCloudsRenderPass::GetSunIlluminance() const
{
    return m_sunIlluminance;
}

void VolumetricCloudsRenderPass::SetAmbientLuminance(const glm::vec3& luminance)
{
    m_ambientLuminance = glm::max(luminance, glm::vec3(0.0f));
}

const glm::vec3& VolumetricCloudsRenderPass::GetAmbientLuminance() const
{
    return m_ambientLuminance;
}

void VolumetricCloudsRenderPass::SetExtinctionCoefficientMultiplier(float multiplier)
{
    m_extinctionCoefficientMultiplier = std::max(multiplier, 0.0f);
}

float VolumetricCloudsRenderPass::GetExtinctionCoefficientMultiplier() const
{
    return m_extinctionCoefficientMultiplier;
}

void VolumetricCloudsRenderPass::SetTotalNoiseScale(float scale)
{
    m_totalNoiseScale = std::max(scale, 0.0f);
}

float VolumetricCloudsRenderPass::GetTotalNoiseScale() const
{
    return m_totalNoiseScale;
}

void VolumetricCloudsRenderPass::SetCoverageAmount(float amount)
{
    m_coverageAmount = std::max(amount, 0.0f);
}

float VolumetricCloudsRenderPass::GetCoverageAmount() const
{
    return m_coverageAmount;
}

void VolumetricCloudsRenderPass::SetCoverageMinimum(float minimum)
{
    // Coverage minimum is used as a lower clamp; keep it inside [0, 1].
    m_coverageMinimum = glm::clamp(minimum, 0.0f, 1.0f);
}

float VolumetricCloudsRenderPass::GetCoverageMinimum() const
{
    return m_coverageMinimum;
}

void VolumetricCloudsRenderPass::SetTypeAmount(float amount)
{
    m_typeAmount = std::max(amount, 0.0f);
}

float VolumetricCloudsRenderPass::GetTypeAmount() const
{
    return m_typeAmount;
}

void VolumetricCloudsRenderPass::SetTypeMinimum(float minimum)
{
    m_typeMinimum = glm::clamp(minimum, 0.0f, 1.0f);
}

float VolumetricCloudsRenderPass::GetTypeMinimum() const
{
    return m_typeMinimum;
}

void VolumetricCloudsRenderPass::SetJitterEnabled(bool enabled)
{
    m_jitterEnabled = enabled;
}

bool VolumetricCloudsRenderPass::IsJitterEnabled() const
{
    return m_jitterEnabled;
}

void VolumetricCloudsRenderPass::SetMaxStepCount(int stepCount)
{
    m_maxStepCount = std::max(stepCount, 1);
}

int VolumetricCloudsRenderPass::GetMaxStepCount() const
{
    return m_maxStepCount;
}

std::shared_ptr<Texture2DObject> VolumetricCloudsRenderPass::GetOutputTexture() const
{
    return m_outputTexture;
}

std::shared_ptr<Texture2DObject> VolumetricCloudsRenderPass::GetDataTexture() const
{
    return m_dataTexture;
}

std::shared_ptr<Texture2DObject> VolumetricCloudsRenderPass::GetNormalsTexture() const
{
    return m_normalsTexture;
}

int VolumetricCloudsRenderPass::GetOutputWidth() const
{
    return m_outputWidth;
}

int VolumetricCloudsRenderPass::GetOutputHeight() const
{
    return m_outputHeight;
}

void VolumetricCloudsRenderPass::EnsureOutputTextureSized(int width, int height)
{
    if (m_outputTexture && m_outputWidth == width && m_outputHeight == height)
        return;

    m_outputWidth = width;
    m_outputHeight = height;

    const auto allocateStorageTexture = [width, height]()
    {
        auto texture = std::make_shared<Texture2DObject>();
        texture->Bind();
        texture->SetImage(0, width, height, TextureObject::FormatRGBA, TextureObject::InternalFormatRGBA32F);
        texture->SetParameter(TextureObject::ParameterEnum::MinFilter, GL_NEAREST);
        texture->SetParameter(TextureObject::ParameterEnum::MagFilter, GL_NEAREST);
        texture->SetParameter(TextureObject::ParameterEnum::WrapS, GL_CLAMP_TO_EDGE);
        texture->SetParameter(TextureObject::ParameterEnum::WrapT, GL_CLAMP_TO_EDGE);
        Texture2DObject::Unbind();
        return texture;
    };

    // Colour output -------------------------------------------------------------------------------
    m_outputTexture = allocateStorageTexture();

    // Auxiliary transmittance/depth output --------------------------------------------------------
    m_dataTexture = allocateStorageTexture();

    // Cloud normal output ------------------------------------------------------------------------
    m_normalsTexture = allocateStorageTexture();
}

void VolumetricCloudsRenderPass::EnsureNoiseTextureSized(std::shared_ptr<Texture2DObject>& texture,
                                                         int& currentWidth,
                                                         int& currentHeight,
                                                         int width,
                                                         int height)
{
    if (texture && currentWidth == width && currentHeight == height)
        return;

    currentWidth = width;
    currentHeight = height;

    texture = std::make_shared<Texture2DObject>();
    texture->Bind();
    texture->SetImage(0, width, height, TextureObject::FormatRGBA, TextureObject::InternalFormatRGBA32F);
    texture->SetParameter(TextureObject::ParameterEnum::MinFilter, GL_LINEAR);
    texture->SetParameter(TextureObject::ParameterEnum::MagFilter, GL_LINEAR);
    texture->SetParameter(TextureObject::ParameterEnum::WrapS, GL_REPEAT);
    texture->SetParameter(TextureObject::ParameterEnum::WrapT, GL_REPEAT);
    Texture2DObject::Unbind();
}

void VolumetricCloudsRenderPass::EnsureNoiseVolumeSized(std::shared_ptr<Texture3DObject>& texture,
                                                       int& currentWidth,
                                                       int& currentHeight,
                                                       int& currentDepth,
                                                       int width,
                                                       int height,
                                                       int depth)
{
    if (texture && currentWidth == width && currentHeight == height && currentDepth == depth)
        return;

    currentWidth = width;
    currentHeight = height;
    currentDepth = depth;

    texture = std::make_shared<Texture3DObject>();
    texture->Bind();
    texture->SetImage(0, width, height, depth, TextureObject::FormatRGBA, TextureObject::InternalFormatRGBA32F);
    
    // Reset mip state to a clean base level before the compute pass populates level zero
    texture->SetParameter(TextureObject::ParameterInt::BaseLevel, 0);
    texture->SetParameter(TextureObject::ParameterInt::MaxLevel, 0);
    texture->SetParameter(TextureObject::ParameterFloat::MinLod, 0.0f);
    texture->SetParameter(TextureObject::ParameterFloat::MaxLod, 0.0f);
    texture->SetParameter(TextureObject::ParameterEnum::MinFilter, GL_LINEAR_MIPMAP_LINEAR);
    texture->SetParameter(TextureObject::ParameterEnum::MagFilter, GL_LINEAR);
    texture->SetParameter(TextureObject::ParameterEnum::WrapS, GL_REPEAT);
    texture->SetParameter(TextureObject::ParameterEnum::WrapT, GL_REPEAT);
    texture->SetParameter(TextureObject::ParameterEnum::WrapR, GL_REPEAT);
    Texture3DObject::Unbind();
}

void VolumetricCloudsRenderPass::Render()
{
    if (!m_isReady)
        return;

    Renderer& renderer = GetRenderer();
    DeviceGL& device = renderer.GetDevice();

    if (!renderer.HasCamera())
        return;

    const Camera& camera = renderer.GetCurrentCamera();
    const glm::mat4 viewMatrix = camera.GetViewMatrix();
    const glm::mat4 projMatrix = camera.GetProjectionMatrix();
    const glm::mat4 invViewProj = glm::inverse(projMatrix * viewMatrix);
    const glm::vec3 cameraPosition = camera.ExtractTranslation();

    // Ensure output size matches default framebuffer
    glm::ivec4 viewport;
    device.GetViewport(viewport.x, viewport.y, viewport.z, viewport.w);
    EnsureOutputTextureSized(viewport.z, viewport.w);

    // Build shape noise volume once (boolean lock)
    if (!m_shapeBuilt)
    {
        const int shapeSize = 64;

        EnsureNoiseVolumeSized(m_shapeNoiseVolume,
                               m_shapeWidth,
                               m_shapeHeight,
                               m_shapeDepth,
                               shapeSize,
                               shapeSize,
                               shapeSize);

        m_shapeNoiseProgram.Use();

        if (m_shapeVolumeSizeLocation >= 0)
        {
            m_shapeNoiseProgram.SetUniform(m_shapeVolumeSizeLocation, glm::ivec3(shapeSize, shapeSize, shapeSize));
        }

        m_shapeNoiseVolume->BindImageTexture(0, 0, GL_TRUE, 0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);
        {
            const GLuint gx = (shapeSize + 7U) / 8U;
            const GLuint gy = (shapeSize + 7U) / 8U;
            const GLuint gz = (shapeSize + 7U) / 8U;
            glDispatchCompute(gx, gy, gz);
        }
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        // Generate a full mip chain for the shape noise so that textureLod sampling behaves correctly
        m_shapeNoiseVolume->Bind();
        m_shapeNoiseVolume->GenerateMipChain(TextureObject::MipGenFilter::MIPGENFILTER_LINEAR);
        Texture3DObject::Unbind();

        m_shapeBuilt = true;
    }

    // Build detail noise volume once for debugging view
    if (!m_detailBuilt)
    {
        const int detailSize = 32;

        EnsureNoiseVolumeSized(m_detailNoiseVolume,
                               m_detailWidth,
                               m_detailHeight,
                               m_detailDepth,
                               detailSize,
                               detailSize,
                               detailSize);

        m_detailNoiseProgram.Use();

        if (m_detailVolumeSizeLocation >= 0)
        {
            m_detailNoiseProgram.SetUniform(m_detailVolumeSizeLocation, glm::ivec3(detailSize));
        }

        m_detailNoiseVolume->BindImageTexture(0, 0, GL_TRUE, 0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);
        {
            const GLuint dispatch = (detailSize + 7U) / 8U;
            glDispatchCompute(dispatch, dispatch, dispatch);
        }
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

        // Generate a full mip chain for the detail noise volume as well
        m_detailNoiseVolume->Bind();
        m_detailNoiseVolume->GenerateMipChain(TextureObject::MipGenFilter::MIPGENFILTER_LINEAR);
        Texture3DObject::Unbind();

        m_detailBuilt = true;
    }

    // Build curl noise texture once for debugging view
    if (!m_curlBuilt)
    {
        const int curlSize =128;

        EnsureNoiseTextureSized(m_curlNoiseTexture,
                                m_curlWidth,
                                m_curlHeight,
                                curlSize,
                                curlSize);

        m_curlNoiseProgram.Use();

        if (m_curlTextureSizeLocation >= 0)
        {
            m_curlNoiseProgram.SetUniform(m_curlTextureSizeLocation, glm::ivec2(curlSize, curlSize));
        }

        m_curlNoiseTexture->BindImageTexture(0,0, GL_FALSE,0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);
        {
            const GLuint gx = (curlSize + 15U) / 16U;
            const GLuint gy = (curlSize + 15U) / 16U;
            glDispatchCompute(gx, gy, 1U);
        }
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        m_curlBuilt = true;
    }

    // Build weather map texture once for debugging view
    if (!m_weatherBuilt)
    {
        const int weatherSize = 1024;

        EnsureNoiseTextureSized(m_weatherMapTexture,
                                m_weatherWidth,
                                m_weatherHeight,
                                weatherSize,
                                weatherSize);

        m_weatherMapProgram.Use();

        if (m_weatherTextureSizeLocation >= 0)
        {
            m_weatherMapProgram.SetUniform(m_weatherTextureSizeLocation, glm::ivec2(weatherSize, weatherSize));
        }

        m_weatherMapTexture->BindImageTexture(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);
        {
            const GLuint gx = (weatherSize + 15U) / 16U;
            const GLuint gy = (weatherSize + 15U) / 16U;
            glDispatchCompute(gx, gy, 1U);
        }
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        m_weatherBuilt = true;
    }

    // Run render compute which samples shape-noise volume and writes to output image
    m_computeProgram.Use();

    if (m_frameDimensionsLocation >= 0) m_computeProgram.SetUniform(m_frameDimensionsLocation, glm::vec2(viewport.z, viewport.w));

    if (m_timeLocation >= 0) m_computeProgram.SetUniform(m_timeLocation, m_timeSeconds);

    if (m_sunDirectionLocation >= 0) m_computeProgram.SetUniform(m_sunDirectionLocation, m_sunDirection);

    if (m_sunIlluminanceLocation >= 0) m_computeProgram.SetUniform(m_sunIlluminanceLocation, m_sunIlluminance);

    if (m_ambientLuminanceLocation >= 0) m_computeProgram.SetUniform(m_ambientLuminanceLocation, m_ambientLuminance);

    if (m_extinctionCoefficientMultiplierLocation >= 0) m_computeProgram.SetUniform(m_extinctionCoefficientMultiplierLocation, m_extinctionCoefficientMultiplier);

    if (m_totalNoiseScaleLocation >= 0) m_computeProgram.SetUniform(m_totalNoiseScaleLocation, m_totalNoiseScale);

    if (m_coverageAmountLocation >= 0) m_computeProgram.SetUniform(m_coverageAmountLocation, m_coverageAmount);

    if (m_coverageMinimumLocation >= 0) m_computeProgram.SetUniform(m_coverageMinimumLocation, m_coverageMinimum);

    if (m_typeAmountLocation >= 0) m_computeProgram.SetUniform(m_typeAmountLocation, m_typeAmount);

    if (m_typeMinimumLocation >= 0) m_computeProgram.SetUniform(m_typeMinimumLocation, m_typeMinimum);

    if (m_maxStepCountLocation >= 0) m_computeProgram.SetUniform(m_maxStepCountLocation, m_maxStepCount);

    if (m_jitterEnabledLocation >= 0) m_computeProgram.SetUniform(m_jitterEnabledLocation, m_jitterEnabled ? 1 : 0);

    if (m_invViewProjLocation >= 0) m_computeProgram.SetUniform(m_invViewProjLocation, invViewProj);

    if (m_cameraPositionLocation >= 0) m_computeProgram.SetUniform(m_cameraPositionLocation, cameraPosition);

    constexpr GLuint shapeTextureUnit = 1U;
    constexpr GLuint detailTextureUnit = 2U;
    constexpr GLuint curlTextureUnit = 3U;
    constexpr GLuint weatherTextureUnit = 4U;

    if (m_shapeNoiseVolume && m_shapeSamplerLocation >= 0) m_computeProgram.SetTexture(m_shapeSamplerLocation, shapeTextureUnit, *m_shapeNoiseVolume);

    if (m_detailNoiseVolume && m_detailSamplerLocation >= 0) m_computeProgram.SetTexture(m_detailSamplerLocation, detailTextureUnit, *m_detailNoiseVolume);

    if (m_curlNoiseTexture && m_curlSamplerLocation >= 0) m_computeProgram.SetTexture(m_curlSamplerLocation, curlTextureUnit, *m_curlNoiseTexture);

    if (m_weatherMapTexture && m_weatherSamplerLocation >= 0) m_computeProgram.SetTexture(m_weatherSamplerLocation, weatherTextureUnit, *m_weatherMapTexture);

    m_outputTexture->BindImageTexture(0,0, GL_FALSE,0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);

    if (m_dataTexture)
    {
        m_dataTexture->BindImageTexture(1, 0, GL_FALSE, 0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);
    }
    if (m_normalsTexture)
    {
        m_normalsTexture->BindImageTexture(2, 0, GL_FALSE, 0, GL_WRITE_ONLY, TextureObject::InternalFormatRGBA32F);
    }
    {
        const GLuint gx = (viewport.z + 15) /16;
        const GLuint gy = (viewport.w + 15) /16;
        glDispatchCompute(gx, gy,1);
    }
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    // Composite the volumetric clouds over the existing framebuffer using alpha blending
    assert(m_material);
    m_material->Use();
    std::shared_ptr<const ShaderProgram> compositeProgram = m_material->GetShaderProgram();

    if (compositeProgram)
    {
        constexpr GLuint cloudsTextureUnit = 0U;

        if (m_compositeCloudSamplerLocation >= 0)
        {
            compositeProgram->SetTexture(m_compositeCloudSamplerLocation, cloudsTextureUnit, *m_outputTexture);
        }
    }
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);

    const Mesh* mesh = &renderer.GetFullscreenMesh();
    mesh->DrawSubmesh(0);

    glDisable(GL_BLEND);
}
