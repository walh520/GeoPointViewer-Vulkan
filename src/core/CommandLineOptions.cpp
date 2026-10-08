// 命令行选项实现：把控制台参数转换成正式 cache 启动配置。
#include "CommandLineOptions.h"

#include "../cache/SeismicCacheReader.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gpv
{
namespace
{

bool IsCacheSwitch(const std::string& argument)
{
    return argument == "--cache" ||
        argument == "-c";
}

bool IsGpuSwitch(const std::string& argument)
{
    return argument == "--gpu";
}

bool IsGpuNameSwitch(const std::string& argument)
{
    return argument == "--gpu-name";
}

GpuPreference ParseGpuPreference(const std::string& value)
{
    std::string normalized = value;
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });

    if (normalized == "auto")
    {
        return GpuPreference::Auto;
    }
    if (normalized == "integrated" || normalized == "igpu")
    {
        return GpuPreference::Integrated;
    }
    if (normalized == "discrete" || normalized == "dgpu")
    {
        return GpuPreference::Discrete;
    }

    throw std::runtime_error("Unknown --gpu value: " + value + ". Use auto, integrated, or discrete.");
}

} // namespace

CommandLineOptions CommandLineParser::Parse(int argc, char** argv)
{
    CommandLineOptions options;

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];

        if (argument == "--help" || argument == "-h")
        {
            options.showHelp = true;
            continue;
        }

        if (argument == "--smoke-test")
        {
            options.maxFrames = 3;
            continue;
        }

        if (argument == "--async-smoke-test")
        {
            options.maxFrames = 120;
            options.appConfig.asyncSmokeTest = true;
            continue;
        }

        if (argument == "--benchmark")
        {
            options.appConfig.benchmark = true;
            options.appConfig.performanceCsvEnabled = true;
            continue;
        }

        if (argument == "--performance-csv")
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value after --performance-csv.");
            }
            options.appConfig.performanceCsvPath = argv[++index];
            options.appConfig.performanceCsvEnabled = true;
            continue;
        }

        if (argument == "--check-cache")
        {
            options.checkCache = true;
            continue;
        }

        if (argument == "--profile-check")
        {
            options.profileCheck = true;
            continue;
        }

        if (argument == "--start-hidden")
        {
            options.appConfig.startHidden = true;
            continue;
        }

        if (argument == "--profile-mode")
        {
            options.appConfig.startInProfileMode = true;
            continue;
        }

        if (IsCacheSwitch(argument))
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value after " + argument + ".");
            }

            options.appConfig.cacheDirectory = argv[++index];
            options.cacheProvided = true;
            continue;
        }

        if (IsGpuSwitch(argument))
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value after " + argument + ".");
            }

            options.appConfig.gpuPreference = ParseGpuPreference(argv[++index]);
            continue;
        }

        if (IsGpuNameSwitch(argument))
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value after " + argument + ".");
            }

            options.appConfig.gpuNameFilter = argv[++index];
            continue;
        }

        if (argument == "--points" || argument == "--point-count" || argument == "-p")
        {
            throw std::runtime_error("--points is a legacy option. Use --cache <cache_dir>.");
        }

        throw std::runtime_error("Unknown command line argument: " + argument);
    }

    return options;
}

void CommandLineParser::ResolveDefaults(CommandLineOptions& options)
{
    if (options.showHelp || options.cacheProvided)
    {
        return;
    }

    options.appConfig.cacheDirectory = SeismicCacheReader::ResolveDefaultCachePath();
}

void CommandLineParser::PrintUsage()
{
    std::cout
        << "Usage:\n"
        << "  GeoPointViewer.exe --cache <cache_dir>\n"
        << "  GeoPointViewer.exe --cache <cache_dir> --check-cache\n"
        << "  GeoPointViewer.exe --cache <cache_dir> --profile-check\n"
        << "  GeoPointViewer.exe --cache <cache_dir> --smoke-test\n"
        << "  GeoPointViewer.exe --cache <cache_dir> --async-smoke-test\n"
        << "  GeoPointViewer.exe --cache <cache_dir> --benchmark\n"
        << "  GeoPointViewer.exe --cache <cache_dir> --performance-csv <output.csv>\n"
        << "\n"
        << "Options:\n"
        << "  --cache, -c      Python cache directory containing metadata.json.\n"
        << "  --gpu            GPU preference: auto, integrated, or discrete.\n"
        << "  --gpu-name       Select the first suitable GPU whose name contains this text.\n"
        << "  --check-cache    Verify the frozen bbox contract without opening a window.\n"
        << "  --profile-check  Run one deterministic LOD0 profile without opening a window.\n"
        << "  --smoke-test     Render 3 frames and exit.\n"
        << "  --async-smoke-test  Submit one async viewport query, render until it is applied, then exit.\n"
        << "  --benchmark      Run 10 deterministic zooms and 5 drags, write CSV, then exit.\n"
        << "  --performance-csv  Write startup, frame-window, GPU, memory, and transfer metrics.\n"
        << "  --start-hidden   Create the GLFW window hidden for Qt docking.\n"
        << "  --profile-mode   Start an immutable profile-analysis renderer session.\n"
        << "  --help, -h       Show this help.\n"
        << "\n"
        << "Supply --cache <local-cache-directory>. No dataset is bundled.\n";
}

} // namespace gpv
