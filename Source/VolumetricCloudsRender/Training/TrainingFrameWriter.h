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
        std::vector<float> rgbaPixels;
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
    bool WriteMetadata(const std::filesystem::path& filePath, std::uint32_t pairIndex, const CaptureMetadata& metadata) const;

    std::filesystem::path BuildFramePath(std::uint32_t pairIndex, const std::string& label, const std::string& extension) const;
    std::filesystem::path BuildMetadataPath(std::uint32_t pairIndex) const;

private:

    std::filesystem::path m_outputDirectory;
};


