// 程序入口：启动 GLFW 窗口并跑通 Vulkan 初始化与清屏渲染流程。
#include "cache/SeismicCacheReader.h"
#include "core/App.h"
#include "core/CommandLineOptions.h"
#include "profile/ProfileAnalysisService.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    try
    {
        gpv::CommandLineOptions options = gpv::CommandLineParser::Parse(argc, argv);
        if (options.showHelp)
        {
            gpv::CommandLineParser::PrintUsage();
            return EXIT_SUCCESS;
        }

        gpv::CommandLineParser::ResolveDefaults(options);

        if (options.checkCache)
        {
            return gpv::SeismicCacheReader::RunContractCheck(options.appConfig.cacheDirectory)
                ? EXIT_SUCCESS
                : EXIT_FAILURE;
        }

        if (options.profileCheck)
        {
            return gpv::ProfileAnalysisService::RunContractCheck(
                options.appConfig.cacheDirectory)
                ? EXIT_SUCCESS
                : EXIT_FAILURE;
        }

        gpv::App app(options.appConfig);
        app.Run(options.maxFrames);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
