// 应用配置声明：保存正式 cache 目录等启动参数，方便以后前端直接传入。
#pragma once

#include <filesystem>
#include <string>

namespace gpv
{

enum class GpuPreference
{
    Auto,
    Integrated,
    Discrete,
};

struct AppConfig
{
    std::filesystem::path cacheDirectory;
    std::filesystem::path performanceCsvPath;
    GpuPreference gpuPreference = GpuPreference::Auto;
    std::string gpuNameFilter;
    bool asyncSmokeTest = false;
    bool performanceCsvEnabled = false;
    bool benchmark = false;
    bool startHidden = false;
    bool startInProfileMode = false;
};

} // namespace gpv
