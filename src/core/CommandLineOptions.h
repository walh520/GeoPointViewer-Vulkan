// 命令行选项声明：解析正式 cache 目录、测试和契约检查开关。
#pragma once

#include "AppConfig.h"

#include <cstdint>

namespace gpv
{

struct CommandLineOptions
{
    AppConfig appConfig;
    std::uint32_t maxFrames = 0;
    bool cacheProvided = false;
    bool checkCache = false;
    bool profileCheck = false;
    bool showHelp = false;
};

class CommandLineParser
{
public:
    static CommandLineOptions Parse(int argc, char** argv);
    static void ResolveDefaults(CommandLineOptions& options);
    static void PrintUsage();
};

} // namespace gpv
