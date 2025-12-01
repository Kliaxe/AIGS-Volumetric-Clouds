#pragma once

#include "Renderer/RenderPass.h"
#include "Shader/ShaderProgram.h"

#include <memory>
#include <glm/glm.hpp>

class Material;
class FramebufferObject;
class Texture2DObject;
class Texture3DObject;

class VolumetricCloudsRenderPass : public RenderPass
{
public:
    VolumetricCloudsRenderPass(std::shared_ptr<Material> blitMaterial, std::shared_ptr<const FramebufferObject> targetFramebuffer = nullptr);

    void Render() override;

    void SetTime(float timeSeconds);

    void SetSunDirection(const glm::vec3& direction);
    const glm::vec3& GetSunDirection() const;

    void SetSunIlluminance(const glm::vec3& illuminance);
    const glm::vec3& GetSunIlluminance() const;

    void SetAmbientLuminance(const glm::vec3& luminance);
    const glm::vec3& GetAmbientLuminance() const;

    // Controls whether per-ray white-noise jitter is applied in the
    // volumetric raymarch (see uJitterEnabled in the compute shader).
    void SetJitterEnabled(bool enabled);
    bool IsJitterEnabled() const;

    void SetMaxStepCount(int stepCount);
    int GetMaxStepCount() const;

    std::shared_ptr<Texture2DObject> GetOutputTexture() const;
    std::shared_ptr<Texture2DObject> GetDataTexture() const;
    std::shared_ptr<Texture2DObject> GetNormalsTexture() const;
    int GetOutputWidth() const;
    int GetOutputHeight() const;

private:
    void EnsureOutputTextureSized(int width, int height);
    void EnsureNoiseTextureSized(std::shared_ptr<Texture2DObject>& texture,
                                 int& currentWidth,
                                 int& currentHeight,
                                 int width,
                                 int height);
    void EnsureNoiseVolumeSized(std::shared_ptr<Texture3DObject>& texture,
                                int& currentWidth,
                                int& currentHeight,
                                int& currentDepth,
                                int width,
                                int height,
                                int depth);

private:
    std::shared_ptr<Material> m_material; // used to blit the compute result

    // Compute program that generates the shape noise volume using utils
    ShaderProgram m_shapeNoiseProgram;
    ShaderProgram::Location m_shapeVolumeSizeLocation = -1;

    // Compute program that generates the detail noise volume
    ShaderProgram m_detailNoiseProgram;
    ShaderProgram::Location m_detailVolumeSizeLocation = -1;

    // Compute program that generates the curl noise texture
    ShaderProgram m_curlNoiseProgram;
    ShaderProgram::Location m_curlTextureSizeLocation = -1;

    // Compute program that generates the weather map texture
    ShaderProgram m_weatherMapProgram;
    ShaderProgram::Location m_weatherTextureSizeLocation = -1;

    // Compute program that renders to the output image and samples the shape noise
    ShaderProgram m_computeProgram;
    ShaderProgram::Location m_frameDimensionsLocation = -1;
    ShaderProgram::Location m_shapeSamplerLocation = -1;
    ShaderProgram::Location m_detailSamplerLocation = -1;
    ShaderProgram::Location m_curlSamplerLocation = -1;
    ShaderProgram::Location m_weatherSamplerLocation = -1;
    ShaderProgram::Location m_invViewProjLocation = -1;
    ShaderProgram::Location m_cameraPositionLocation = -1;
    ShaderProgram::Location m_timeLocation = -1;
    ShaderProgram::Location m_sunDirectionLocation = -1;
    ShaderProgram::Location m_sunIlluminanceLocation = -1;
    ShaderProgram::Location m_ambientLuminanceLocation = -1;
    ShaderProgram::Location m_maxStepCountLocation = -1;
    ShaderProgram::Location m_jitterEnabledLocation = -1;

    // Composite shader uniform locations
    ShaderProgram::Location m_compositeCloudSamplerLocation = -1;

    std::shared_ptr<Texture3DObject> m_shapeNoiseVolume; // generated shape noise3D (RGBA32F)
    std::shared_ptr<Texture3DObject> m_detailNoiseVolume; // generated detail noise3D (RGBA32F)
    std::shared_ptr<Texture2DObject> m_curlNoiseTexture; // generated curl noise2D (RGBA32F)
    std::shared_ptr<Texture2DObject> m_weatherMapTexture; // generated weather map (RGBA32F)

    // Main colour output written by the render compute shader (RGBA32F)
    std::shared_ptr<Texture2DObject> m_outputTexture;

    // Auxiliary data output written alongside the colour buffer. This stores training features
    // such as view transmittance, light transmittance, and linear depth for the current frame.
    std::shared_ptr<Texture2DObject> m_dataTexture;

    // Per-pixel cloud normal output used by the training capture path.
    std::shared_ptr<Texture2DObject> m_normalsTexture;

    int m_outputWidth =0;
    int m_outputHeight =0;
    int m_shapeWidth =0;
    int m_shapeHeight =0;
    int m_shapeDepth =0;
    int m_detailWidth =0;
    int m_detailHeight =0;
    int m_detailDepth =0;
    int m_curlWidth =0;
    int m_curlHeight =0;
    int m_weatherWidth =0;
    int m_weatherHeight =0;
    bool m_shapeBuilt = false; // build the3D volume once
    bool m_detailBuilt = false; // build the detail volume once
    bool m_curlBuilt = false;   // build the curl texture once
    bool m_weatherBuilt = false; // build the weather map once
    bool m_isReady = true;     // guard to skip rendering when initialization fails

    float m_timeSeconds = 0.0f;

    int m_maxStepCount = 256;

    bool m_jitterEnabled = true;

    glm::vec3 m_sunDirection = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 m_sunIlluminance = glm::vec3(8.0f, 8.0f, 7.0f);
    glm::vec3 m_ambientLuminance = glm::vec3(0.5f, 0.6f, 0.7f);
};
