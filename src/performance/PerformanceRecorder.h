// 性能记录器声明：计算分位数并输出可直接用于测试报告的 CSV 数据。
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gpv
{

struct DistributionStats
{
    std::size_t sampleCount = 0;
    double average = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double maximum = 0.0;
};

struct PerformanceReportRow
{
    std::string rowType;
    std::string name;
    std::string renderTier;
    std::string queryReason;
    std::string gpuName;
    std::string gpuType;
    std::string appMode;
    std::string profileState;
    double elapsedSeconds = 0.0;
    double responseMilliseconds = 0.0;
    DistributionStats response;
    double fps = 0.0;
    DistributionStats appWork;
    DistributionStats cpuFrame;
    DistributionStats gpuFrame;
    double gpuCullAverageMilliseconds = 0.0;
    double gpuDrawAverageMilliseconds = 0.0;
    double fenceWaitAverageMilliseconds = 0.0;
    double acquireAverageMilliseconds = 0.0;
    double recordAverageMilliseconds = 0.0;
    double submitAverageMilliseconds = 0.0;
    double presentAverageMilliseconds = 0.0;
    double vulkanUploadAverageMilliseconds = 0.0;
    double queryMilliseconds = 0.0;
    double queryTotalMilliseconds = 0.0;
    double cpuBuildMilliseconds = 0.0;
    double mainApplyMilliseconds = 0.0;
    double cacheOpenMilliseconds = 0.0;
    double windowInitMilliseconds = 0.0;
    double vulkanInitMilliseconds = 0.0;
    double initializeMilliseconds = 0.0;
    double lod0IndexReadyMilliseconds = 0.0;
    double gpuMetadataUploadMilliseconds = 0.0;
    double lodPointsPerPixel = 0.0;
    double estimatedPointsPerPixel = 0.0;
    double processMemoryMegabytes = 0.0;
    double processCpuPercent = 0.0;
    double processIoReadMegabytesPerSecond = 0.0;
    double processIoWriteMegabytesPerSecond = 0.0;
    double peakProcessMemoryMegabytes = 0.0;
    double gpuMemoryUsageMegabytes = -1.0;
    double gpuMemoryBudgetMegabytes = -1.0;
    double peakGpuMemoryUsageMegabytes = -1.0;
    double gpuBufferUsedMegabytes = 0.0;
    double gpuBufferCapacityMegabytes = 0.0;
    double pointUploadMegabytes = 0.0;
    double vulkanUploadMegabytes = 0.0;
    double tileCacheMegabytes = 0.0;
    double transferGpuMilliseconds = 0.0;
    double transferThroughputGigabytesPerSecond = 0.0;
    double transferCpuWaitMilliseconds = 0.0;
    double profileCorridorWidth = 0.0;
    double profileTotalMilliseconds = 0.0;
    double profileIoMilliseconds = 0.0;
    double profileFilterMilliseconds = 0.0;
    double profileReadMegabytes = 0.0;
    std::uint64_t renderedPoints = 0;
    std::uint64_t lodPointBudget = 0;
    std::uint64_t inputAssemblyVertices = 0;
    std::uint64_t vertexShaderInvocations = 0;
    std::uint64_t computeShaderInvocations = 0;
    std::uint64_t transferBytes = 0;
    std::uint64_t lod0Points = 0;
    std::uint64_t proxyPoints = 0;
    std::uint64_t overviewPoints = 0;
    std::uint64_t cacheHits = 0;
    std::uint64_t cacheMisses = 0;
    std::uint64_t cacheEvictions = 0;
    std::uint64_t gpuResidencyProgress = 0;
    std::uint64_t gpuResidencyTotal = 0;
    std::uint64_t profileTestedPoints = 0;
    std::uint64_t profileMatchedPoints = 0;
    std::uint64_t profileCandidateTiles = 0;
    std::uint32_t gpuVendorId = 0;
    std::uint32_t gpuDeviceId = 0;
    std::uint32_t gpuDriverVersion = 0;
    std::uint32_t vulkanApiVersion = 0;
    std::uint32_t hitTiles = 0;
    std::uint32_t visiblePages = 0;
    std::uint32_t activePages = 0;
    std::uint32_t residentPages = 0;
    bool complete = false;
};

class PerformanceRecorder
{
public:
    static DistributionStats ComputeDistribution(const std::vector<double>& samples);
    static std::filesystem::path CreateDefaultPath();

    void Open(const std::filesystem::path& path);
    void Write(const PerformanceReportRow& row);
    void Close();
    bool IsOpen() const;
    const std::filesystem::path& Path() const;

private:
    std::filesystem::path path_;
    std::ofstream stream_;
};

} // namespace gpv
