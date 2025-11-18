#include "VolumetricCloudsSkyPass.h"

#include "Renderer/Renderer.h"

#include "Camera/Camera.h"
#include "Geometry/Mesh.h"

#include "Asset/ShaderLoader.h"
#include "Asset/Texture2DLoader.h"

#include "Texture/Texture2DObject.h"

#include <glm/gtc/matrix_inverse.hpp>

namespace
{
    constexpr const char* kSkyShaderVertexPath = "shaders/renderer/fullscreen.vert";
    constexpr const char* kSkyShaderFragmentPath = "shaders/sky.frag";

    constexpr const char* kDefaultSkyTexturePath = "Content/HDRI/qwantani_puresky_4k.hdr";
}

VolumetricCloudsSkyPass::VolumetricCloudsSkyPass(const char* hdrPath,
                                                 std::shared_ptr<const FramebufferObject> targetFramebuffer)
    : RenderPass(targetFramebuffer)
{
    InitializeShader();

    const char* texturePath = hdrPath != nullptr ? hdrPath : kDefaultSkyTexturePath;

    LoadSkyTexture(texturePath);
}

std::shared_ptr<Texture2DObject> VolumetricCloudsSkyPass::GetSkyTexture() const
{
    return m_skyTexture;
}

void VolumetricCloudsSkyPass::SetSkyTexture(std::shared_ptr<Texture2DObject> texture)
{
    if (texture)
    {
        ConfigureSkyTexture(*texture);
    }

    m_skyTexture = std::move(texture);
}

void VolumetricCloudsSkyPass::Render()
{
    if (!m_skyTexture)
    {
        return;
    }

    Renderer& renderer = GetRenderer();

    const Camera& camera = renderer.GetCurrentCamera();

    m_shaderProgram.Use();

    if (m_invViewProjLocation >= 0)
    {
        const glm::mat4 invViewProj = glm::inverse(camera.GetViewProjectionMatrix());

        m_shaderProgram.SetUniform(m_invViewProjLocation, invViewProj);
    }

    if (m_cameraPositionLocation >= 0)
    {
        m_shaderProgram.SetUniform(m_cameraPositionLocation, camera.ExtractTranslation());
    }

    if (m_skyTextureLocation >= 0)
    {
        m_shaderProgram.SetTexture(m_skyTextureLocation, 0, *m_skyTexture);
    }

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_ALWAYS);

    const Mesh& fullscreenMesh = renderer.GetFullscreenMesh();
    fullscreenMesh.DrawSubmesh(0);

    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
}

void VolumetricCloudsSkyPass::InitializeShader()
{
    Shader vertexShader = ShaderLoader(Shader::VertexShader).Load(kSkyShaderVertexPath);

    Shader fragmentShader = ShaderLoader(Shader::FragmentShader).Load(kSkyShaderFragmentPath);

    m_shaderProgram.Build(vertexShader, fragmentShader);

    m_invViewProjLocation = m_shaderProgram.GetUniformLocation("InvViewProjMatrix");
    m_cameraPositionLocation = m_shaderProgram.GetUniformLocation("CameraPosition");
    m_skyTextureLocation = m_shaderProgram.GetUniformLocation("SkyTexture");
}

void VolumetricCloudsSkyPass::LoadSkyTexture(const char* hdrPath)
{
    if (hdrPath == nullptr)
    {
        return;
    }

    std::shared_ptr<Texture2DObject> texture = Texture2DLoader::LoadTextureShared(hdrPath,
                                                                                  TextureObject::FormatRGB,
                                                                                  TextureObject::InternalFormatRGB32F,
                                                                                  true,
                                                                                  false);

    if (!texture)
    {
        return;
    }

    ConfigureSkyTexture(*texture);

    m_skyTexture = std::move(texture);
}

void VolumetricCloudsSkyPass::ConfigureSkyTexture(Texture2DObject& texture)
{
    texture.Bind();

    texture.SetParameter(TextureObject::ParameterEnum::WrapS, GL_REPEAT);
    texture.SetParameter(TextureObject::ParameterEnum::WrapT, GL_CLAMP_TO_EDGE);

    texture.SetParameter(TextureObject::ParameterEnum::MinFilter, GL_LINEAR_MIPMAP_LINEAR);
    texture.SetParameter(TextureObject::ParameterEnum::MagFilter, GL_LINEAR);

    Texture2DObject::Unbind();
}


