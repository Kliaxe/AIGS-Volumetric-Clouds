#include "VolumetricCloudsRenderApplication.h"

#include <charconv>
#include <string>
#include <string_view>
#include <filesystem>
#include <iostream>
#include <system_error>

namespace
{
    template <typename T>
    bool ParseUnsigned(std::string_view text, T& value)
    {
        const char* begin = text.data();
        const char* end = begin + text.size();

        std::from_chars_result result = std::from_chars(begin, end, value);
        return result.ec == std::errc() && result.ptr == end;
    }
}

int main(int argc, char** argv)
{
    VolumetricCloudsRenderApplication application;

    bool trainingRequested = false;

    for (int argumentIndex = 1; argumentIndex < argc; ++argumentIndex)
    {
        const std::string argument(argv[argumentIndex]);

        if (argument == "--training")
        {
            trainingRequested = true;
            continue;
        }

        if (argument.rfind("--batch-size=", 0) == 0)
        {
            std::uint32_t batchSize = 0U;
            const std::string_view payload(argument.data() + 13, argument.size() - 13);
            if (ParseUnsigned(payload, batchSize))
            {
                application.GetTrainingCaptureConfig().SetTargetBatchSize(batchSize);
            }
            continue;
        }

        if (argument.rfind("--training-output=", 0) == 0)
        {
            const std::string pathString = argument.substr(18);
            application.GetTrainingCaptureConfig().SetOutputDirectory(std::filesystem::path(pathString));
            continue;
        }

        if (argument.rfind("--training-seed=", 0) == 0)
        {
            std::uint64_t seed = 0ULL;
            const std::string_view payload(argument.data() + 16, argument.size() - 16);
            if (ParseUnsigned(payload, seed))
            {
                application.GetTrainingCaptureConfig().SetRandomSeed(seed);
                application.GetTrainingCameraSampler().SetRandomSeed(seed);
            }
            continue;
        }
    }

    if (trainingRequested)
    {
        application.EnableTrainingMode(true);
    }

    return application.Run();
}
