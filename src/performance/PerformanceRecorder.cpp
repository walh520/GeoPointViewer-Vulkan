// 性能记录器实现：输出稳定列结构并对帧耗时样本计算分位数。
#include "PerformanceRecorder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gpv
{
namespace
{

double Percentile(const std::vector<double>& sortedSamples, double percentile)
{
    if (sortedSamples.empty())
    {
        return 0.0;
    }

    const double index = percentile * static_cast<double>(sortedSamples.size() - 1);
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(index));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lowerIndex);
    return
        sortedSamples[lowerIndex] * (1.0 - fraction) +
        sortedSamples[upperIndex] * fraction;
}

std::string EscapeCsv(const std::string& value)
{
    if (value.find_first_of(",\"\r\n") == std::string::npos)
    {
        return value;
    }

    std::string escaped = "\"";
    for (char character : value)
    {
        if (character == '\"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += character;
        }
    }
    escaped += '\"';
    return escaped;
}

} // namespace

DistributionStats PerformanceRecorder::ComputeDistribution(
    const std::vector<double>& samples)
{
    DistributionStats stats{};
    if (samples.empty())
    {
        return stats;
    }

    std::vector<double> sortedSamples = samples;
    std::sort(sortedSamples.begin(), sortedSamples.end());
    double total = 0.0;
    for (double sample : sortedSamples)
    {
        total += sample;
    }

    stats.sampleCount = sortedSamples.size();
    stats.average = total / static_cast<double>(sortedSamples.size());
    stats.p50 = Percentile(sortedSamples, 0.50);
    stats.p95 = Percentile(sortedSamples, 0.95);
    stats.p99 = Percentile(sortedSamples, 0.99);
    stats.maximum = sortedSamples.back();
    return stats;
}

std::filesystem::path PerformanceRecorder::CreateDefaultPath()
{
    const std::time_t currentTime = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm localTime{};
    localtime_s(&localTime, &currentTime);

    std::ostringstream fileName;
    fileName
        << "performance_"
        << std::put_time(&localTime, "%Y%m%d_%H%M%S")
        << ".csv";
    return std::filesystem::path("output") / "performance" / fileName.str();
}

void PerformanceRecorder::Open(const std::filesystem::path& path)
{
    Close();
    path_ = path;
    if (path_.has_parent_path())
    {
        std::filesystem::create_directories(path_.parent_path());
    }

    stream_.open(path_, std::ios::out | std::ios::trunc);
    if (!stream_.is_open())
    {
        throw std::runtime_error("Failed to open the performance CSV file: " + path_.string());
    }

    stream_
        << "row_type,name,elapsed_s,response_ms,response_avg_ms,response_p50_ms,response_p95_ms,"
        << "response_p99_ms,response_max_ms,render_tier,complete,query_reason,gpu_name,gpu_type,"
        << "gpu_vendor_id,gpu_device_id,gpu_driver_version,vulkan_api_version,fps,"
        << "app_work_avg_ms,app_work_p50_ms,app_work_p95_ms,app_work_p99_ms,app_work_max_ms,"
        << "cpu_frame_avg_ms,cpu_frame_p50_ms,cpu_frame_p95_ms,cpu_frame_p99_ms,cpu_frame_max_ms,"
        << "gpu_frame_avg_ms,gpu_frame_p50_ms,gpu_frame_p95_ms,gpu_frame_p99_ms,gpu_frame_max_ms,"
        << "gpu_cull_avg_ms,gpu_draw_avg_ms,fence_wait_avg_ms,acquire_avg_ms,record_avg_ms,"
        << "submit_avg_ms,present_avg_ms,vulkan_upload_avg_ms,query_ms,query_total_ms,"
        << "cpu_build_ms,main_apply_ms,cache_open_ms,window_init_ms,vulkan_init_ms,initialize_ms,"
        << "lod0_index_ready_ms,gpu_metadata_upload_ms,lod_points_per_pixel,"
        << "estimated_points_per_pixel,lod_point_budget,"
        << "rendered_points,lod0_points,proxy_points,"
        << "overview_points,hit_tiles,visible_pages,active_pages,resident_pages,"
        << "ia_vertices,vs_invocations,compute_invocations,process_mb,process_cpu_percent,"
        << "process_io_read_mbps,process_io_write_mbps,gpu_memory_usage_mb,"
        << "peak_process_mb,gpu_memory_budget_mb,peak_gpu_memory_usage_mb,gpu_buffer_used_mb,"
        << "gpu_buffer_capacity_mb,point_upload_mb,vulkan_upload_mb,tile_cache_mb,cache_hits,"
        << "cache_misses,cache_evictions,gpu_residency_progress,"
        << "gpu_residency_total,transfer_bytes,transfer_gpu_ms,transfer_gbps,transfer_cpu_wait_ms,"
        << "app_mode,profile_state,profile_corridor_width,profile_total_ms,profile_io_ms,"
        << "profile_filter_ms,profile_read_mb,profile_tested_points,profile_matched_points,"
        << "profile_candidate_tiles\n";
    stream_.flush();
}

void PerformanceRecorder::Write(const PerformanceReportRow& row)
{
    if (!stream_.is_open())
    {
        return;
    }

    stream_ << std::fixed << std::setprecision(4)
        << EscapeCsv(row.rowType) << ','
        << EscapeCsv(row.name) << ','
        << row.elapsedSeconds << ','
        << row.responseMilliseconds << ','
        << row.response.average << ','
        << row.response.p50 << ','
        << row.response.p95 << ','
        << row.response.p99 << ','
        << row.response.maximum << ','
        << EscapeCsv(row.renderTier) << ','
        << (row.complete ? "yes" : "no") << ','
        << EscapeCsv(row.queryReason) << ','
        << EscapeCsv(row.gpuName) << ','
        << EscapeCsv(row.gpuType) << ','
        << row.gpuVendorId << ','
        << row.gpuDeviceId << ','
        << row.gpuDriverVersion << ','
        << row.vulkanApiVersion << ','
        << row.fps << ','
        << row.appWork.average << ','
        << row.appWork.p50 << ','
        << row.appWork.p95 << ','
        << row.appWork.p99 << ','
        << row.appWork.maximum << ','
        << row.cpuFrame.average << ','
        << row.cpuFrame.p50 << ','
        << row.cpuFrame.p95 << ','
        << row.cpuFrame.p99 << ','
        << row.cpuFrame.maximum << ','
        << row.gpuFrame.average << ','
        << row.gpuFrame.p50 << ','
        << row.gpuFrame.p95 << ','
        << row.gpuFrame.p99 << ','
        << row.gpuFrame.maximum << ','
        << row.gpuCullAverageMilliseconds << ','
        << row.gpuDrawAverageMilliseconds << ','
        << row.fenceWaitAverageMilliseconds << ','
        << row.acquireAverageMilliseconds << ','
        << row.recordAverageMilliseconds << ','
        << row.submitAverageMilliseconds << ','
        << row.presentAverageMilliseconds << ','
        << row.vulkanUploadAverageMilliseconds << ','
        << row.queryMilliseconds << ','
        << row.queryTotalMilliseconds << ','
        << row.cpuBuildMilliseconds << ','
        << row.mainApplyMilliseconds << ','
        << row.cacheOpenMilliseconds << ','
        << row.windowInitMilliseconds << ','
        << row.vulkanInitMilliseconds << ','
        << row.initializeMilliseconds << ','
        << row.lod0IndexReadyMilliseconds << ','
        << row.gpuMetadataUploadMilliseconds << ','
        << row.lodPointsPerPixel << ','
        << row.estimatedPointsPerPixel << ','
        << row.lodPointBudget << ','
        << row.renderedPoints << ','
        << row.lod0Points << ','
        << row.proxyPoints << ','
        << row.overviewPoints << ','
        << row.hitTiles << ','
        << row.visiblePages << ','
        << row.activePages << ','
        << row.residentPages << ','
        << row.inputAssemblyVertices << ','
        << row.vertexShaderInvocations << ','
        << row.computeShaderInvocations << ','
        << row.processMemoryMegabytes << ','
        << row.processCpuPercent << ','
        << row.processIoReadMegabytesPerSecond << ','
        << row.processIoWriteMegabytesPerSecond << ','
        << row.gpuMemoryUsageMegabytes << ','
        << row.peakProcessMemoryMegabytes << ','
        << row.gpuMemoryBudgetMegabytes << ','
        << row.peakGpuMemoryUsageMegabytes << ','
        << row.gpuBufferUsedMegabytes << ','
        << row.gpuBufferCapacityMegabytes << ','
        << row.pointUploadMegabytes << ','
        << row.vulkanUploadMegabytes << ','
        << row.tileCacheMegabytes << ','
        << row.cacheHits << ','
        << row.cacheMisses << ','
        << row.cacheEvictions << ','
        << row.gpuResidencyProgress << ','
        << row.gpuResidencyTotal << ','
        << row.transferBytes << ','
        << row.transferGpuMilliseconds << ','
        << row.transferThroughputGigabytesPerSecond << ','
        << row.transferCpuWaitMilliseconds << ','
        << EscapeCsv(row.appMode) << ','
        << EscapeCsv(row.profileState) << ','
        << row.profileCorridorWidth << ','
        << row.profileTotalMilliseconds << ','
        << row.profileIoMilliseconds << ','
        << row.profileFilterMilliseconds << ','
        << row.profileReadMegabytes << ','
        << row.profileTestedPoints << ','
        << row.profileMatchedPoints << ','
        << row.profileCandidateTiles
        << '\n';
    stream_.flush();
}

void PerformanceRecorder::Close()
{
    if (stream_.is_open())
    {
        stream_.flush();
        stream_.close();
    }
}

bool PerformanceRecorder::IsOpen() const
{
    return stream_.is_open();
}

const std::filesystem::path& PerformanceRecorder::Path() const
{
    return path_;
}

} // namespace gpv
