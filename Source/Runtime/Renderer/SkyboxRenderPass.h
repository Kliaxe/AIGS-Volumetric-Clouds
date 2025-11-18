#pragma once

#include "RenderPass.h"

#include "Shader/ShaderProgram.h"
#include <memory>

class TextureCubemapObject;

class SkyboxRenderPass : public RenderPass
{
public:
    SkyboxRenderPass(std::shared_ptr<TextureCubemapObject> texture);

    std::shared_ptr<TextureCubemapObject> GetTexture() const;
    void SetTexture(std::shared_ptr<TextureCubemapObject> texture);

    void Render() override;

private:
    std::shared_ptr<TextureCubemapObject> m_texture;

    ShaderProgram m_shaderProgram;
    ShaderProgram::Location m_cameraPositionLocation;
    ShaderProgram::Location m_invViewProjMatrixLocation;
    ShaderProgram::Location m_skyboxTextureLocation;
};
