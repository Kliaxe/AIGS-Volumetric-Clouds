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

    const std::filesystem::path lowFramePath = BuildFramePath(pairIndex, "low", ".pfm");
    const std::filesystem::path highFramePath = BuildFramePath(pairIndex, "high", ".pfm");
    const std::filesystem::path metadataPath = BuildMetadataPath(pairIndex);

    bool success = true;

    success &= WriteFrame(lowFramePath, lowResolutionFrame);
    success &= WriteFrame(highFramePath, fullResolutionFrame);
    success &= WriteMetadata(metadataPath, pairIndex, metadata);

    return success;
}

// --------------------------------------------------------------------------------------------------
// Frame persistence (PFM format)
// --------------------------------------------------------------------------------------------------
bool TrainingFrameWriter::WriteFrame(const std::filesystem::path& filePath, const FrameCapture& frame) const
{
    std::ofstream fileStream(filePath, std::ios::binary);

    if (!fileStream)
    {
        std::cerr << "[TrainingFrameWriter] Failed to open frame file: "
                  << filePath << std::endl;
        return false;
    }

    fileStream << "PF\n";
    fileStream << frame.resolution.x << " " << frame.resolution.y << "\n";
    fileStream << "-1.0\n";

    const std::size_t channelCount = 4U;

    for (int y = 0; y < frame.resolution.y; ++y)
    {
        const std::size_t rowOffset = static_cast<std::size_t>(y) * frame.resolution.x * channelCount;

        for (int x = 0; x < frame.resolution.x; ++x)
        {
            const std::size_t pixelOffset = rowOffset + static_cast<std::size_t>(x) * channelCount;

            const float r = frame.rgbaPixels[pixelOffset + 0U];
            const float g = frame.rgbaPixels[pixelOffset + 1U];
            const float b = frame.rgbaPixels[pixelOffset + 2U];

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


