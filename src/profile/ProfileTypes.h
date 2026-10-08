// 地质剖面数据结构：定义剖面模式、折线请求、分箱统计和后台任务状态。
#pragma once

#include "../cache/SeismicCacheTypes.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace gpv
{

enum class ProfileModeState
{
    Performance,
    Loading,
    Ready,
    Analyzing,
};

enum class ProfileTaskState
{
    Idle,
    Queued,
    Running,
    Paused,
    Prepared,
    Complete,
    Cancelled,
    Failed,
};

struct ProfileWorldPoint
{
    double x = 0.0;
    double y = 0.0;
};

struct ProfileRequest
{
    std::uint64_t requestId = 0;
    Bounds2D viewport;
    std::vector<ProfileWorldPoint> polyline;
    double halfWidth = 0.0;
    std::uint32_t outputBinCount = 2048;
    bool prepareOnly = false;
};

struct ProfileBin
{
    double distanceStart = 0.0;
    double distanceEnd = 0.0;
    std::uint64_t count = 0;
    std::array<double, MaxCacheAttributes> minimum{};
    std::array<double, MaxCacheAttributes> maximum{};
    std::array<double, MaxCacheAttributes> sum{};
    std::array<std::uint64_t, MaxCacheAttributes> validCount{};
    std::uint64_t representativeSourceId = 0;
    double representativeX = 0.0;
    double representativeY = 0.0;
    double representativeDistanceError = std::numeric_limits<double>::max();
};

struct ProfileResult
{
    std::uint64_t requestId = 0;
    ProfileTaskState state = ProfileTaskState::Idle;
    std::filesystem::path outputPath;
    std::filesystem::path samplePath;
    std::vector<ProfileBin> bins;
    std::string error;
    double totalLength = 0.0;
    double corridorWidth = 0.0;
    double indexMilliseconds = 0.0;
    double ioMilliseconds = 0.0;
    double filterMilliseconds = 0.0;
    double writeMilliseconds = 0.0;
    double totalMilliseconds = 0.0;
    double readMegabytes = 0.0;
    std::uint64_t candidateTileCount = 0;
    std::uint64_t candidatePointCount = 0;
    std::uint64_t testedPointCount = 0;
    std::uint64_t matchedPointCount = 0;
    std::uint64_t samplePointCount = 0;
};

const char* ProfileTaskStateName(ProfileTaskState state);

} // namespace gpv
