#include "TrainingFrameWriter.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

// --------------------------------------------------------------------------------------------------
// ctor
// --------------------------------------------------------------------------------------------------
TrainingFrameWriter::TrainingFrameWriter()
    : m_outputDirectory(std::filesystem::current_path() / "TrainingCaptures")
{
}

// --------------------------------------------------------------------------------------------------
// Directory management
// --------------------------------------------------------------------------------------------------
void TrainingFrameWriter::SetOutputDirectory(const std::filesystem::path& directory)
{
    m_outputDirectory = directory;
}

const std::filesystem::path& TrainingFrameWriter::GetOutputDirectory() const
{
    return m_outputDirectory;
}

bool TrainingFrameWriter::EnsureOutputDirectory()
{
    std::error_code errorCode;
    if (std::filesystem::exists(m_outputDirectory, errorCode))
    {
        return true;
    }

    if (!std::filesystem::create_directories(m_outputDirectory, errorCode))
    {
        std::cerr << "[TrainingFrameWriter] Failed to create output directory: "
                  << m_outputDirectory << " error=" << errorCode.message() << std::endl;
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------------------------------
// Pair write API
// --------------------------------------------------------------------------------------------------
bool TrainingFrameWriter::WritePair(std::uint32_t pairIndex,
                                    const FrameCapture& lowResolutionFrame,
                                    const FrameCapture& fullResolutionFrame,
                                    const CaptureMetadata& metadata)
{
    if (!EnsureOutputDirectory())
    {
        return false;
    }

    // ----------------------------------------------------------------------------------------------
    // Build file paths for the colour and auxiliary data images at both resolutions.
    // Existing naming for the colour images is preserved for backwards compatibility so that
    // previously written consumers which expect only "low/high" PFM files continue to work.
    // The auxiliary data images use explicit labels to make their intent clear.
    // ----------------------------------------------------------------------------------------------
    const std::filesystem::path lowColorPath  = BuildFramePath(pairIndex, "low", ".pfm");
    const std::filesystem::path highColorPath = BuildFramePath(pairIndex, "high", ".pfm");
    const std::filesystem::path lowDataPath   = BuildFramePath(pairIndex, "low_data", ".pfm");
    const std::filesystem::path highDataPath  = BuildFramePath(pairIndex, "high_data", ".pfm");
    const std::filesystem::path metadataPath = BuildMetadataPath(pairIndex);

    bool success = true;

    // Colour images -------------------------------------------------------------------------------
    success &= WriteFrame(lowColorPath, lowResolutionFrame);
    success &= WriteFrame(highColorPath, fullResolutionFrame);

    // Auxiliary data images -----------------------------------------------------------------------
    // It is legal for the data buffers to be empty (for example, legacy captures produced before
    // the data path was introduced). In that case we simply skip writing the data PFM files.
    if (!lowResolutionFrame.rgbaDataPixels.empty())
    {
        success &= WriteFramePixels(lowDataPath,
                                    lowResolutionFrame.resolution,
                                    lowResolutionFrame.rgbaDataPixels);
    }

    if (!fullResolutionFrame.rgbaDataPixels.empty())
    {
        success &= WriteFramePixels(highDataPath,
                                    fullResolutionFrame.resolution,
                                    fullResolutionFrame.rgbaDataPixels);
    }
    success &= WriteMetadata(metadataPath, pairIndex, metadata);

    return success;
}

// --------------------------------------------------------------------------------------------------
// Frame persistence (PFM format)
// --------------------------------------------------------------------------------------------------
bool TrainingFrameWriter::WriteFrame(const std::filesystem::path& filePath,
                                    const FrameCapture& frame) const
{
    return WriteFramePixels(filePath, frame.resolution, frame.rgbaColorPixels);
}

bool TrainingFrameWriter::WriteFramePixels(const std::filesystem::path& filePath,
                                           const glm::ivec2& resolution,
                                           const std::vector<float>& pixels) const
{
    std::ofstream fileStream(filePath, std::ios::binary);

    if (!fileStream)
    {
        std::cerr << "[TrainingFrameWriter] Failed to open frame file: "
                  << filePath << std::endl;
        return false;
    }

    fileStream << "PF\n";
    fileStream << resolution.x << " " << resolution.y << "\n";
    fileStream << "-1.0\n";

    const std::size_t channelCount = 4U;

    if (pixels.size() < static_cast<std::size_t>(resolution.x) * static_cast<std::size_t>(resolution.y) * channelCount)
    {
        std::cerr << "[TrainingFrameWriter] Pixel buffer too small for frame: "
                  << filePath << std::endl;
        return false;
    }

    for (int y = 0; y < resolution.y; ++y)
    {
        const std::size_t rowOffset = static_cast<std::size_t>(y) * resolution.x * channelCount;

        for (int x = 0; x < resolution.x; ++x)
        {
            const std::size_t pixelOffset = rowOffset + static_cast<std::size_t>(x) * channelCount;

            const float r = pixels[pixelOffset + 0U];
            const float g = pixels[pixelOffset + 1U];
            const float b = pixels[pixelOffset + 2U];

            fileStream.write(reinterpret_cast<const char*>(&r), sizeof(float));
            fileStream.write(reinterpret_cast<const char*>(&g), sizeof(float));
            fileStream.write(reinterpret_cast<const char*>(&b), sizeof(float));
        }
    }

    fileStream.flush();

    return fileStream.good();
}

// --------------------------------------------------------------------------------------------------
// Metadata persistence
// --------------------------------------------------------------------------------------------------
bool TrainingFrameWriter::WriteMetadata(const std::filesystem::path& filePath,
                                        std::uint32_t pairIndex,
                                        const CaptureMetadata& metadata) const
{
    std::ofstream fileStream(filePath, std::ios::trunc);

    if (!fileStream)
    {
        std::cerr << "[TrainingFrameWriter] Failed to open metadata file: "
                  << filePath << std::endl;
        return false;
    }

    auto writeVec3 = [&fileStream](const char* label, const glm::vec3& value)
    {
        fileStream << "    \"" << label << "\": ["
                   << value.x << ", "
                   << value.y << ", "
                   << value.z << "]";
    };

    auto writeMat4 = [&fileStream](const char* label, const glm::mat4& value)
    {
        fileStream << "    \"" << label << "\": [";

        for (int row = 0; row < 4; ++row)
        {
            fileStream << "[";
            for (int column = 0; column < 4; ++column)
            {
                fileStream << value[row][column];
                if (column < 3)
                {
                    fileStream << ", ";
                }
            }
            fileStream << "]";
            if (row < 3)
            {
                fileStream << ", ";
            }
        }

        fileStream << "]";
    };

    fileStream << "{\n";
    fileStream << "    \"pairIndex\": " << pairIndex << ",\n";
    fileStream << "    \"captureTimeSeconds\": " << metadata.captureTimeSeconds << ",\n";

    writeVec3("cameraPosition", metadata.cameraPosition);
    fileStream << ",\n";

    writeVec3("cameraFocus", metadata.cameraFocus);
    fileStream << ",\n";

    writeMat4("viewMatrix", metadata.viewMatrix);
    fileStream << ",\n";

    writeMat4("projectionMatrix", metadata.projectionMatrix);
    fileStream << "\n";

    fileStream << "}\n";

    fileStream.flush();

    return fileStream.good();
}

// --------------------------------------------------------------------------------------------------
// Path builders
// --------------------------------------------------------------------------------------------------
std::filesystem::path TrainingFrameWriter::BuildFramePath(std::uint32_t pairIndex,
                                                          const std::string& label,
                                                          const std::string& extension) const
{
    std::ostringstream builder;
    builder << "pair_" << std::setw(6) << std::setfill('0') << pairIndex << "_" << label << extension;
    return m_outputDirectory / builder.str();
}

std::filesystem::path TrainingFrameWriter::BuildMetadataPath(std::uint32_t pairIndex) const
{
    std::ostringstream builder;
    builder << "pair_" << std::setw(6) << std::setfill('0') << pairIndex << "_metadata.json";
    return m_outputDirectory / builder.str();
}


