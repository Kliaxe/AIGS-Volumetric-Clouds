#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>
#include <cstdint>

// --------------------------------------------------------------------------------------------------
// TrainingFrameWriter
// --------------------------------------------------------------------------------------------------
//  Responsible for persisting rendered frames and their metadata to disk. Writing is intentionally
//  kept simple (PFM for floating-point image data and a human-readable JSON-like metadata file) so
//  that downstream tooling can be implemented quickly.
// --------------------------------------------------------------------------------------------------
class TrainingFrameWriter
{
public:

    struct FrameCapture
    {
        glm::ivec2 resolution;

        // ----------------------------------------------------------------------------------------------
        // Color buffer
        // ----------------------------------------------------------------------------------------------
        //  Floating-point RGBA buffer used for the main colour output. This matches the render target
        //  format (RGBA32F) and is persisted to disk as a standard PFM file.
        // ----------------------------------------------------------------------------------------------
        std::vector<float> rgbaColorPixels;

        // ----------------------------------------------------------------------------------------------
        // Auxiliary data buffer
        // ----------------------------------------------------------------------------------------------
        //  Optional RGBA buffer containing per-pixel training features. For the volumetric cloud
        //  training captures this is currently laid out as:
        //
        //      R: view-space transmittance (T_view)
        //      G: light-space transmittance / visibility (T_light)
        //      B: linear depth to effective scattering point (t_depth, in world units)
        //      A: reserved (currently 1.0)
        //
        //  This buffer is also written as a PFM file when present, using a separate filename label
        //  so that downstream tools can treat colour and data streams independently.
        // ----------------------------------------------------------------------------------------------
        std::vector<float> rgbaDataPixels;
    };

    struct CaptureMetadata
    {
        glm::vec3 cameraPosition;
        glm::vec3 cameraFocus;
        glm::mat4 viewMatrix;
        glm::mat4 projectionMatrix;
        float captureTimeSeconds;
    };

public:

    TrainingFrameWriter();

    void SetOutputDirectory(const std::filesystem::path& directory);
    const std::filesystem::path& GetOutputDirectory() const;

    bool WritePair(std::uint32_t pairIndex,
                   const FrameCapture& lowResolutionFrame,
                   const FrameCapture& fullResolutionFrame,
                   const CaptureMetadata& metadata);

private:

    bool EnsureOutputDirectory();

    bool WriteFrame(const std::filesystem::path& filePath, const FrameCapture& frame) const;
    bool WriteFramePixels(const std::filesystem::path& filePath,
                          const glm::ivec2& resolution,
                          const std::vector<float>& pixels) const;
    bool WriteMetadata(const std::filesystem::path& filePath, std::uint32_t pairIndex, const CaptureMetadata& metadata) const;

    std::filesystem::path BuildFramePath(std::uint32_t pairIndex, const std::string& label, const std::string& extension) const;
    std::filesystem::path BuildMetadataPath(std::uint32_t pairIndex) const;

private:

    std::filesystem::path m_outputDirectory;
};


