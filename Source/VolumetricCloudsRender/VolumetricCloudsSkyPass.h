#pragma once

#include "Renderer/RenderPass.h"
#include "Shader/ShaderProgram.h"

#include <memory>

class Texture2DObject;
class FramebufferObject;

class VolumetricCloudsSkyPass : public RenderPass
{
public:
    VolumetricCloudsSkyPass(const char* hdrPath,
                            std::shared_ptr<const FramebufferObject> targetFramebuffer = nullptr);

    std::shared_ptr<Texture2DObject> GetSkyTexture() const;

    void SetSkyTexture(std::shared_ptr<Texture2DObject> texture);

    void Render() override;

private:
    void InitializeShader();

    void LoadSkyTexture(const char* hdrPath);

    void ConfigureSkyTexture(Texture2DObject& texture);

private:
    ShaderProgram m_shaderProgram;

    ShaderProgram::Location m_invViewProjLocation = -1;
    ShaderProgram::Location m_cameraPositionLocation = -1;
    ShaderProgram::Location m_skyTextureLocation = -1;

    std::shared_ptr<Texture2DObject> m_skyTexture;
};


