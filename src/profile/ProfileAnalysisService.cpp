// 地质剖面后台服务实现：低优先级分段读取 LOD0，精确筛选走廊点并输出流式统计。
#include "ProfileAnalysisService.h"

#include "../cache/SeismicCacheReader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace gpv
{
namespace
{

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kMaximumReadBatchBytes = 8ull * 1024ull * 1024ull;
constexpr std::uint64_t kMaximumMergeGapBytes = 4096;
constexpr std::uint32_t kProfileSampleVersion = 1;
constexpr std::uint32_t kProfileSampleFixedHeaderBytes = 64;
constexpr std::uint32_t kProfileSampleAttributeNameBytes = 64;
constexpr std::streamoff kProfileSamplePointCountOffset = 24;
constexpr char kProfileSampleMagic[8] =
{
    'G', 'P', 'V', 'P', 'R', 'F', '1', '\0',
};

double ElapsedMilliseconds(const Clock::time_point& begin, const Clock::time_point& end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

const RecordField& FindField(const CacheMetadata& metadata, const std::string& name)
{
    const auto found = std::find_if(
        metadata.recordFields.begin(),
        metadata.recordFields.end(),
        [&name](const RecordField& field)
        {
            return field.name == name;
        });
    if (found == metadata.recordFields.end())
    {
        throw std::runtime_error("Profile analysis could not find record field: " + name);
    }
    return *found;
}

double ReadNumeric(const char* record, const RecordField& field)
{
    const char* value = record + field.offset;
    if (field.dtype == "float64" || field.dtype == "<f8" || field.dtype == "double")
    {
        double result = 0.0;
        std::memcpy(&result, value, sizeof(result));
        return result;
    }
    if (field.dtype == "float32" || field.dtype == "<f4" || field.dtype == "float")
    {
        float result = 0.0f;
        std::memcpy(&result, value, sizeof(result));
        return static_cast<double>(result);
    }
    if (field.dtype == "int32" || field.dtype == "<i4")
    {
        std::int32_t result = 0;
        std::memcpy(&result, value, sizeof(result));
        return static_cast<double>(result);
    }
    if (field.dtype == "uint32" || field.dtype == "<u4")
    {
        std::uint32_t result = 0;
        std::memcpy(&result, value, sizeof(result));
        return static_cast<double>(result);
    }
    throw std::runtime_error("Profile analysis does not support dtype: " + field.dtype);
}

struct ProfileSegment
{
    ProfileWorldPoint begin;
    ProfileWorldPoint end;
    double length = 0.0;
    double lengthSquared = 0.0;
    double cumulative = 0.0;
};

std::vector<ProfileSegment> BuildSegments(
    const std::vector<ProfileWorldPoint>& polyline,
    double& totalLength)
{
    std::vector<ProfileSegment> segments;
    totalLength = 0.0;
    for (std::size_t index = 1; index < polyline.size(); ++index)
    {
        const double deltaX = polyline[index].x - polyline[index - 1].x;
        const double deltaY = polyline[index].y - polyline[index - 1].y;
        const double lengthSquared = deltaX * deltaX + deltaY * deltaY;
        if (lengthSquared <= 0.0)
        {
            continue;
        }

        const double length = std::sqrt(lengthSquared);
        segments.push_back(ProfileSegment
        {
            polyline[index - 1],
            polyline[index],
            length,
            lengthSquared,
            totalLength,
        });
        totalLength += length;
    }
    return segments;
}

Bounds2D BuildCorridorBounds(
    const std::vector<ProfileWorldPoint>& polyline,
    double halfWidth)
{
    Bounds2D bounds
    {
        polyline.front().x,
        polyline.front().y,
        polyline.front().x,
        polyline.front().y,
    };
    for (const ProfileWorldPoint& point : polyline)
    {
        bounds.minX = std::min(bounds.minX, point.x);
        bounds.minY = std::min(bounds.minY, point.y);
        bounds.maxX = std::max(bounds.maxX, point.x);
        bounds.maxY = std::max(bounds.maxY, point.y);
    }
    bounds.minX -= halfWidth;
    bounds.minY -= halfWidth;
    bounds.maxX += halfWidth;
    bounds.maxY += halfWidth;
    return bounds;
}

bool SegmentIntersectsExpandedBounds(
    const ProfileSegment& segment,
    const Bounds2D& bounds,
    double expansion)
{
    const double minimumX = bounds.minX - expansion;
    const double minimumY = bounds.minY - expansion;
    const double maximumX = bounds.maxX + expansion;
    const double maximumY = bounds.maxY + expansion;
    const double deltaX = segment.end.x - segment.begin.x;
    const double deltaY = segment.end.y - segment.begin.y;
    double parameterMinimum = 0.0;
    double parameterMaximum = 1.0;

    const auto clipAxis = [&parameterMinimum, &parameterMaximum](
        double origin,
        double delta,
        double minimum,
        double maximum)
    {
        if (std::abs(delta) <= std::numeric_limits<double>::epsilon())
        {
            return origin >= minimum && origin <= maximum;
        }

        double entry = (minimum - origin) / delta;
        double exit = (maximum - origin) / delta;
        if (entry > exit)
        {
            std::swap(entry, exit);
        }
        parameterMinimum = std::max(parameterMinimum, entry);
        parameterMaximum = std::min(parameterMaximum, exit);
        return parameterMinimum <= parameterMaximum;
    };

    return clipAxis(segment.begin.x, deltaX, minimumX, maximumX) &&
        clipAxis(segment.begin.y, deltaY, minimumY, maximumY);
}

bool TileTouchesProfileCorridor(
    const TileIndexEntry& tile,
    const std::vector<ProfileSegment>& segments,
    double halfWidth)
{
    return std::any_of(
        segments.begin(),
        segments.end(),
        [&tile, halfWidth](const ProfileSegment& segment)
        {
            return SegmentIntersectsExpandedBounds(segment, tile.bounds, halfWidth);
        });
}

bool ProjectPointToProfile(
    double x,
    double y,
    const std::vector<ProfileSegment>& segments,
    double halfWidthSquared,
    double& chainage,
    double& crosslineOffset)
{
    double closestDistanceSquared = halfWidthSquared;
    bool matched = false;
    for (const ProfileSegment& segment : segments)
    {
        const double deltaX = segment.end.x - segment.begin.x;
        const double deltaY = segment.end.y - segment.begin.y;
        const double relativeX = x - segment.begin.x;
        const double relativeY = y - segment.begin.y;
        const double parameter = std::clamp(
            (relativeX * deltaX + relativeY * deltaY) / segment.lengthSquared,
            0.0,
            1.0);
        const double projectedX = segment.begin.x + parameter * deltaX;
        const double projectedY = segment.begin.y + parameter * deltaY;
        const double errorX = x - projectedX;
        const double errorY = y - projectedY;
        const double distanceSquared = errorX * errorX + errorY * errorY;
        if (distanceSquared <= closestDistanceSquared)
        {
            closestDistanceSquared = distanceSquared;
            chainage = segment.cumulative + parameter * segment.length;
            crosslineOffset =
                (deltaX * relativeY - deltaY * relativeX) / segment.length;
            matched = true;
        }
    }
    return matched;
}

ProfileWorldPoint PointAtChainage(
    const std::vector<ProfileSegment>& segments,
    double chainage)
{
    for (const ProfileSegment& segment : segments)
    {
        if (chainage <= segment.cumulative + segment.length)
        {
            const double parameter = std::clamp(
                (chainage - segment.cumulative) / segment.length,
                0.0,
                1.0);
            return ProfileWorldPoint
            {
                segment.begin.x + (segment.end.x - segment.begin.x) * parameter,
                segment.begin.y + (segment.end.y - segment.begin.y) * parameter,
            };
        }
    }
    return segments.back().end;
}

void InitializeBins(
    std::vector<ProfileBin>& bins,
    double totalLength,
    std::size_t attributeCount)
{
    const double binWidth = totalLength / static_cast<double>(bins.size());
    for (std::size_t binIndex = 0; binIndex < bins.size(); ++binIndex)
    {
        ProfileBin& bin = bins[binIndex];
        bin.distanceStart = static_cast<double>(binIndex) * binWidth;
        bin.distanceEnd = static_cast<double>(binIndex + 1) * binWidth;
        for (std::size_t attrIndex = 0; attrIndex < attributeCount; ++attrIndex)
        {
            bin.minimum[attrIndex] = std::numeric_limits<double>::max();
            bin.maximum[attrIndex] = std::numeric_limits<double>::lowest();
        }
    }
}

struct ProfileOutputPaths
{
    std::filesystem::path statistics;
    std::filesystem::path samples;
    std::filesystem::path samplesTemporary;
};

ProfileOutputPaths CreateProfileOutputPaths(const ProfileRequest& request)
{
    const std::filesystem::path outputDirectory =
        std::filesystem::current_path() / "output" / "profiles";
    std::filesystem::create_directories(outputDirectory);
    const std::uint64_t timestamp = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string stem =
        "profile_" + std::to_string(timestamp) + "_" +
        std::to_string(request.requestId);

    ProfileOutputPaths paths;
    paths.statistics = outputDirectory / (stem + ".csv");
    paths.samples = outputDirectory / (stem + "_samples.bin");
    paths.samplesTemporary = outputDirectory / (stem + "_samples.bin.tmp");
    return paths;
}

template <typename Value>
void WriteBinaryValue(std::ostream& output, const Value& value)
{
    output.write(
        reinterpret_cast<const char*>(&value),
        static_cast<std::streamsize>(sizeof(Value)));
}

void WriteZeroBytes(std::ostream& output, std::size_t count)
{
    const std::array<char, 64> zeros{};
    while (count > 0)
    {
        const std::size_t chunk = std::min(count, zeros.size());
        output.write(zeros.data(), static_cast<std::streamsize>(chunk));
        count -= chunk;
    }
}

void WriteProfileSampleHeader(
    std::ostream& output,
    const CacheMetadata& metadata,
    double totalLength,
    double corridorWidth)
{
    if (metadata.attrFields.size() > MaxCacheAttributes)
    {
        throw std::runtime_error("Profile sample output has too many attributes.");
    }

    const std::uint32_t attributeCount =
        static_cast<std::uint32_t>(metadata.attrFields.size());
    const std::uint32_t headerSize =
        kProfileSampleFixedHeaderBytes +
        attributeCount * kProfileSampleAttributeNameBytes;
    const std::uint32_t recordSize =
        static_cast<std::uint32_t>(
            sizeof(std::uint64_t) +
            sizeof(double) * (4 + attributeCount));
    const std::uint64_t pointCountPlaceholder = 0;
    const std::array<char, 16> reserved{};

    output.write(kProfileSampleMagic, sizeof(kProfileSampleMagic));
    WriteBinaryValue(output, kProfileSampleVersion);
    WriteBinaryValue(output, headerSize);
    WriteBinaryValue(output, recordSize);
    WriteBinaryValue(output, attributeCount);
    WriteBinaryValue(output, pointCountPlaceholder);
    WriteBinaryValue(output, totalLength);
    WriteBinaryValue(output, corridorWidth);
    output.write(reserved.data(), static_cast<std::streamsize>(reserved.size()));

    for (const std::string& attribute : metadata.attrFields)
    {
        if (attribute.empty() || attribute.size() >= kProfileSampleAttributeNameBytes)
        {
            throw std::runtime_error(
                "Profile sample attribute names must contain 1 to 63 UTF-8 bytes.");
        }
        output.write(attribute.data(), static_cast<std::streamsize>(attribute.size()));
        WriteZeroBytes(
            output,
            kProfileSampleAttributeNameBytes - attribute.size());
    }
}

void FinalizeProfileSampleFile(
    std::ofstream& output,
    const ProfileOutputPaths& paths,
    std::uint64_t pointCount)
{
    output.seekp(kProfileSamplePointCountOffset, std::ios::beg);
    WriteBinaryValue(output, pointCount);
    output.flush();
    if (!output)
    {
        throw std::runtime_error("Failed while finalizing exact profile samples.");
    }
    output.close();

    std::error_code error;
    std::filesystem::remove(paths.samples, error);
    error.clear();
    std::filesystem::rename(paths.samplesTemporary, paths.samples, error);
    if (error)
    {
        throw std::runtime_error(
            "Failed to publish exact profile sample file: " + error.message());
    }
}

class ProfileSampleOutputGuard
{
public:
    explicit ProfileSampleOutputGuard(const ProfileOutputPaths& paths)
        : paths_(paths)
    {
    }

    ~ProfileSampleOutputGuard()
    {
        if (released_)
        {
            return;
        }
        std::error_code ignored;
        std::filesystem::remove(paths_.samplesTemporary, ignored);
        ignored.clear();
        std::filesystem::remove(paths_.samples, ignored);
    }

    void Release()
    {
        released_ = true;
    }

private:
    const ProfileOutputPaths& paths_;
    bool released_ = false;
};

std::filesystem::path WriteProfileCsv(
    const std::filesystem::path& outputPath,
    const ProfileResult& result,
    const CacheMetadata& metadata,
    const std::vector<ProfileSegment>& segments)
{
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    if (!output.is_open())
    {
        throw std::runtime_error("Failed to create profile result CSV.");
    }

    output
        << "distance_start,distance_end,distance_center,line_x,line_y,count,"
        << "representative_source_id,representative_x,representative_y";
    for (const std::string& attribute : metadata.attrFields)
    {
        output
            << ',' << attribute << "_min"
            << ',' << attribute << "_max"
            << ',' << attribute << "_mean";
    }
    output << '\n' << std::setprecision(17);

    for (const ProfileBin& bin : result.bins)
    {
        const double distanceCenter = (bin.distanceStart + bin.distanceEnd) * 0.5;
        const ProfileWorldPoint linePoint = PointAtChainage(segments, distanceCenter);
        output
            << bin.distanceStart << ','
            << bin.distanceEnd << ','
            << distanceCenter << ','
            << linePoint.x << ','
            << linePoint.y << ','
            << bin.count << ',';
        if (bin.count > 0)
        {
            output
                << bin.representativeSourceId << ','
                << bin.representativeX << ','
                << bin.representativeY;
        }
        else
        {
            output << ",,";
        }
        for (std::size_t attrIndex = 0; attrIndex < metadata.attrFields.size(); ++attrIndex)
        {
            output << ',';
            if (bin.validCount[attrIndex] > 0)
            {
                output
                    << bin.minimum[attrIndex] << ','
                    << bin.maximum[attrIndex] << ','
                    << bin.sum[attrIndex] /
                        static_cast<double>(bin.validCount[attrIndex]);
            }
            else
            {
                output << ",,";
            }
        }
        output << '\n';
    }
    if (!output)
    {
        throw std::runtime_error("Failed while writing profile result CSV.");
    }
    return outputPath;
}

} // namespace

const char* ProfileTaskStateName(ProfileTaskState state)
{
    switch (state)
    {
    case ProfileTaskState::Idle: return "idle";
    case ProfileTaskState::Queued: return "queued";
    case ProfileTaskState::Running: return "running";
    case ProfileTaskState::Paused: return "paused";
    case ProfileTaskState::Prepared: return "prepared";
    case ProfileTaskState::Complete: return "complete";
    case ProfileTaskState::Cancelled: return "cancelled";
    case ProfileTaskState::Failed: return "failed";
    default: return "unknown";
    }
}

ProfileAnalysisService::~ProfileAnalysisService()
{
    Stop();
}

bool ProfileAnalysisService::RunContractCheck(
    const std::filesystem::path& cacheDirectory)
{
    SeismicCacheReader cacheReader;
    cacheReader.Open(cacheDirectory);
    const Bounds2D world = cacheReader.Metadata().xyRange;
    const double spanX = world.maxX - world.minX;
    const double spanY = world.maxY - world.minY;
    const std::vector<ProfileWorldPoint> polyline
    {
        ProfileWorldPoint{ world.minX + spanX * 0.35, world.minY + spanY * 0.40 },
        ProfileWorldPoint{ world.minX + spanX * 0.65, world.minY + spanY * 0.60 },
    };
    const double halfWidth = std::max(spanX, spanY) / 1920.0 * 10.0;

    ProfileAnalysisService service;
    service.Start(cacheReader);
    const std::uint64_t requestId = service.Analyze(
        world,
        polyline,
        halfWidth,
        2048);
    const auto deadline = Clock::now() + std::chrono::seconds(60);
    while (Clock::now() < deadline)
    {
        const std::optional<ProfileResult> completed = service.PollCompleted();
        if (!completed.has_value())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        const ProfileResult& result = completed.value();
        const bool passed =
            result.requestId == requestId &&
            result.state == ProfileTaskState::Complete &&
            result.matchedPointCount > 0 &&
            result.samplePointCount == result.matchedPointCount &&
            result.bins.size() == 2048 &&
            std::filesystem::is_regular_file(result.outputPath) &&
            std::filesystem::is_regular_file(result.samplePath);
        std::cout
            << "Profile contract check: status=" << (passed ? "passed" : "failed")
            << ", request_id=" << result.requestId
            << ", source=lod0_exact"
            << ", candidate_tiles=" << result.candidateTileCount
            << ", tested_points=" << result.testedPointCount
            << ", matched_points=" << result.matchedPointCount
            << ", read_mb=" << result.readMegabytes
            << ", total_ms=" << result.totalMilliseconds
            << ", path=\"" << std::filesystem::absolute(result.outputPath).string() << "\""
            << ", samples_path=\"" << std::filesystem::absolute(result.samplePath).string() << "\""
            << '\n';
        if (!result.error.empty())
        {
            std::cerr << "Profile contract check error: " << result.error << '\n';
        }
        service.Stop();
        return passed;
    }

    service.Cancel();
    service.Stop();
    std::cerr << "Profile contract check timed out after 60 seconds.\n";
    return false;
}

void ProfileAnalysisService::Start(const SeismicCacheReader& cacheReader)
{
    Stop();
    cacheReader_ = &cacheReader;
    latestRequestId_.store(0, std::memory_order_release);
    state_.store(ProfileTaskState::Idle, std::memory_order_release);
    worker_ = std::jthread(
        [this](std::stop_token stopToken)
        {
            WorkerMain(stopToken);
        });
}

void ProfileAnalysisService::Stop()
{
    if (worker_.joinable())
    {
        worker_.request_stop();
        condition_.notify_all();
        worker_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    pendingRequest_.reset();
    completedResult_.reset();
    cacheReader_ = nullptr;
    state_.store(ProfileTaskState::Idle, std::memory_order_release);
}

std::uint64_t ProfileAnalysisService::PrepareViewport(const Bounds2D& viewport)
{
    ProfileRequest request;
    request.requestId = latestRequestId_.fetch_add(1, std::memory_order_acq_rel) + 1;
    request.viewport = viewport;
    request.prepareOnly = true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingRequest_ = request;
    }
    state_.store(ProfileTaskState::Queued, std::memory_order_release);
    condition_.notify_all();
    return request.requestId;
}

std::uint64_t ProfileAnalysisService::Analyze(
    const Bounds2D& viewport,
    const std::vector<ProfileWorldPoint>& polyline,
    double halfWidth,
    std::uint32_t outputBinCount)
{
    ProfileRequest request;
    request.requestId = latestRequestId_.fetch_add(1, std::memory_order_acq_rel) + 1;
    request.viewport = viewport;
    request.polyline = polyline;
    request.halfWidth = halfWidth;
    request.outputBinCount = std::clamp(outputBinCount, 256u, 8192u);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingRequest_ = request;
    }
    state_.store(ProfileTaskState::Queued, std::memory_order_release);
    condition_.notify_all();
    return request.requestId;
}

void ProfileAnalysisService::Cancel()
{
    latestRequestId_.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingRequest_.reset();
    }
    state_.store(ProfileTaskState::Cancelled, std::memory_order_release);
    condition_.notify_all();
}

void ProfileAnalysisService::SetRenderBusy(bool busy)
{
    renderBusy_.store(busy, std::memory_order_release);
    if (!busy)
    {
        condition_.notify_all();
    }
}

std::optional<ProfileResult> ProfileAnalysisService::PollCompleted()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!completedResult_.has_value())
    {
        return std::nullopt;
    }
    std::optional<ProfileResult> result = std::move(completedResult_);
    completedResult_.reset();
    return result;
}

ProfileTaskState ProfileAnalysisService::State() const
{
    return state_.load(std::memory_order_acquire);
}

void ProfileAnalysisService::WorkerMain(std::stop_token stopToken)
{
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif

    while (!stopToken.stop_requested())
    {
        ProfileRequest request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, stopToken, [this]()
            {
                return pendingRequest_.has_value();
            });
            if (stopToken.stop_requested())
            {
                return;
            }
            request = std::move(pendingRequest_.value());
            pendingRequest_.reset();
        }

        ProfileResult result;
        try
        {
            state_.store(ProfileTaskState::Running, std::memory_order_release);
            result = Execute(request, stopToken);
        }
        catch (const std::exception& error)
        {
            result.requestId = request.requestId;
            result.state = ShouldCancel(request.requestId, stopToken)
                ? ProfileTaskState::Cancelled
                : ProfileTaskState::Failed;
            result.error = error.what();
        }

        if (result.requestId != latestRequestId_.load(std::memory_order_acquire))
        {
            result.state = ProfileTaskState::Cancelled;
        }
        const ProfileTaskState finalState = result.state;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completedResult_ = std::move(result);
        }
        state_.store(finalState, std::memory_order_release);
    }
}

ProfileResult ProfileAnalysisService::Execute(
    const ProfileRequest& request,
    std::stop_token stopToken)
{
    if (cacheReader_ == nullptr)
    {
        throw std::runtime_error("Profile analysis service is not connected to a cache.");
    }

    const auto totalBegin = Clock::now();
    ProfileResult result;
    result.requestId = request.requestId;
    result.corridorWidth = request.halfWidth * 2.0;

    if (!WaitForRenderBudget(request.requestId, stopToken))
    {
        result.state = ProfileTaskState::Cancelled;
        return result;
    }

    const auto indexBegin = Clock::now();
    if (request.prepareOnly)
    {
        const Lod0RegionSummary summary = cacheReader_->SummarizeLod0Region(request.viewport);
        result.candidateTileCount = summary.tileCount;
        result.candidatePointCount = summary.pointCount;
        result.indexMilliseconds = ElapsedMilliseconds(indexBegin, Clock::now());
        result.totalMilliseconds = ElapsedMilliseconds(totalBegin, Clock::now());
        result.state = ProfileTaskState::Prepared;
        return result;
    }

    if (request.polyline.size() < 2 || request.halfWidth <= 0.0)
    {
        throw std::runtime_error("A professional profile requires at least two vertices and a positive corridor width.");
    }

    double totalLength = 0.0;
    const std::vector<ProfileSegment> segments = BuildSegments(request.polyline, totalLength);
    if (segments.empty() || totalLength <= 0.0)
    {
        throw std::runtime_error("Profile polyline has zero length.");
    }
    result.totalLength = totalLength;

    const Bounds2D corridorBounds = BuildCorridorBounds(request.polyline, request.halfWidth);
    std::vector<TileIndexEntry> tiles = cacheReader_->CopyLod0TilesIntersecting(corridorBounds);
    tiles.erase(
        std::remove_if(
            tiles.begin(),
            tiles.end(),
            [&segments, &request](const TileIndexEntry& tile)
            {
                return !TileTouchesProfileCorridor(
                    tile,
                    segments,
                    request.halfWidth);
            }),
        tiles.end());
    std::sort(
        tiles.begin(),
        tiles.end(),
        [](const TileIndexEntry& left, const TileIndexEntry& right)
        {
            return left.offsetBytes < right.offsetBytes;
        });
    result.candidateTileCount = tiles.size();
    result.candidatePointCount = std::accumulate(
        tiles.begin(),
        tiles.end(),
        std::uint64_t{ 0 },
        [](std::uint64_t sum, const TileIndexEntry& tile)
        {
            return sum + tile.lengthPoints;
        });
    result.indexMilliseconds = ElapsedMilliseconds(indexBegin, Clock::now());

    const CacheMetadata& metadata = cacheReader_->Metadata();
    const RecordField& xField = FindField(metadata, "x");
    const RecordField& yField = FindField(metadata, "y");
    std::array<const RecordField*, MaxCacheAttributes> attrFields{};
    for (std::size_t attrIndex = 0; attrIndex < metadata.attrFields.size(); ++attrIndex)
    {
        attrFields[attrIndex] = &FindField(metadata, metadata.attrFields[attrIndex]);
    }

    const ProfileOutputPaths outputPaths = CreateProfileOutputPaths(request);
    ProfileSampleOutputGuard sampleOutputGuard(outputPaths);
    std::vector<char> sampleStreamBuffer(1024 * 1024);
    std::ofstream sampleOutput;
    sampleOutput.rdbuf()->pubsetbuf(
        sampleStreamBuffer.data(),
        static_cast<std::streamsize>(sampleStreamBuffer.size()));
    sampleOutput.open(
        outputPaths.samplesTemporary,
        std::ios::binary | std::ios::trunc);
    if (!sampleOutput.is_open())
    {
        throw std::runtime_error("Failed to create exact profile sample file.");
    }
    WriteProfileSampleHeader(
        sampleOutput,
        metadata,
        totalLength,
        result.corridorWidth);
    if (!sampleOutput)
    {
        throw std::runtime_error("Failed while writing exact profile sample header.");
    }

    result.bins.resize(request.outputBinCount);
    InitializeBins(result.bins, totalLength, metadata.attrFields.size());
    const double halfWidthSquared = request.halfWidth * request.halfWidth;
    const double binScale = static_cast<double>(result.bins.size()) / totalLength;

    std::ifstream input(cacheReader_->ResolveTilesPath(), std::ios::binary);
    if (!input.is_open())
    {
        throw std::runtime_error("Profile analysis failed to open the LOD0 point file.");
    }

    std::size_t firstTile = 0;
    while (firstTile < tiles.size())
    {
        if (!WaitForRenderBudget(request.requestId, stopToken))
        {
            result.state = ProfileTaskState::Cancelled;
            return result;
        }

        const std::uint64_t batchOffset = tiles[firstTile].offsetBytes;
        std::uint64_t batchEnd = batchOffset +
            tiles[firstTile].lengthPoints * metadata.recordSize;
        std::size_t lastTile = firstTile + 1;
        while (lastTile < tiles.size())
        {
            const std::uint64_t tileOffset = tiles[lastTile].offsetBytes;
            const std::uint64_t tileEnd = tileOffset +
                tiles[lastTile].lengthPoints * metadata.recordSize;
            if (tileOffset > batchEnd + kMaximumMergeGapBytes ||
                tileEnd - batchOffset > kMaximumReadBatchBytes)
            {
                break;
            }
            batchEnd = std::max(batchEnd, tileEnd);
            ++lastTile;
        }

        const std::size_t batchBytes = static_cast<std::size_t>(batchEnd - batchOffset);
        std::vector<char> batch(batchBytes);
        const auto ioBegin = Clock::now();
        input.seekg(static_cast<std::streamoff>(batchOffset), std::ios::beg);
        input.read(batch.data(), static_cast<std::streamsize>(batch.size()));
        if (input.gcount() != static_cast<std::streamsize>(batch.size()))
        {
            throw std::runtime_error("Profile analysis encountered a truncated LOD0 read.");
        }
        result.ioMilliseconds += ElapsedMilliseconds(ioBegin, Clock::now());
        result.readMegabytes += static_cast<double>(batchBytes) / (1024.0 * 1024.0);

        const auto filterBegin = Clock::now();
        for (std::size_t tileIndex = firstTile; tileIndex < lastTile; ++tileIndex)
        {
            const TileIndexEntry& tile = tiles[tileIndex];
            const std::size_t tileBase = static_cast<std::size_t>(tile.offsetBytes - batchOffset);
            for (std::uint64_t pointIndex = 0; pointIndex < tile.lengthPoints; ++pointIndex)
            {
                if ((pointIndex & 0xFFFFu) == 0 &&
                    ShouldCancel(request.requestId, stopToken))
                {
                    result.state = ProfileTaskState::Cancelled;
                    return result;
                }

                const char* record = batch.data() + tileBase +
                    static_cast<std::size_t>(pointIndex * metadata.recordSize);
                const double x = ReadNumeric(record, xField);
                const double y = ReadNumeric(record, yField);
                ++result.testedPointCount;
                if (x < corridorBounds.minX || x > corridorBounds.maxX ||
                    y < corridorBounds.minY || y > corridorBounds.maxY)
                {
                    continue;
                }

                double chainage = 0.0;
                double crosslineOffset = 0.0;
                if (!ProjectPointToProfile(
                    x,
                    y,
                    segments,
                    halfWidthSquared,
                    chainage,
                    crosslineOffset))
                {
                    continue;
                }

                const std::size_t binIndex = std::min<std::size_t>(
                    static_cast<std::size_t>(chainage * binScale),
                    result.bins.size() - 1);
                ProfileBin& bin = result.bins[binIndex];
                const double binCenter = (bin.distanceStart + bin.distanceEnd) * 0.5;
                const double chainageError = chainage - binCenter;
                const double representativeError =
                    chainageError * chainageError +
                    crosslineOffset * crosslineOffset;
                const std::uint64_t sourceId =
                    tile.offsetBytes / metadata.recordSize + pointIndex;
                if (representativeError < bin.representativeDistanceError)
                {
                    bin.representativeDistanceError = representativeError;
                    bin.representativeSourceId = sourceId;
                    bin.representativeX = x;
                    bin.representativeY = y;
                }
                ++bin.count;
                ++result.matchedPointCount;

                std::array<double, MaxCacheAttributes> attributeValues{};
                for (std::size_t attrIndex = 0; attrIndex < metadata.attrFields.size(); ++attrIndex)
                {
                    const double value = ReadNumeric(record, *attrFields[attrIndex]);
                    attributeValues[attrIndex] = value;
                    if (!std::isfinite(value))
                    {
                        continue;
                    }
                    bin.minimum[attrIndex] = std::min(bin.minimum[attrIndex], value);
                    bin.maximum[attrIndex] = std::max(bin.maximum[attrIndex], value);
                    bin.sum[attrIndex] += value;
                    ++bin.validCount[attrIndex];
                }

                WriteBinaryValue(sampleOutput, sourceId);
                WriteBinaryValue(sampleOutput, chainage);
                WriteBinaryValue(sampleOutput, crosslineOffset);
                WriteBinaryValue(sampleOutput, x);
                WriteBinaryValue(sampleOutput, y);
                for (std::size_t attrIndex = 0; attrIndex < metadata.attrFields.size(); ++attrIndex)
                {
                    WriteBinaryValue(sampleOutput, attributeValues[attrIndex]);
                }
                if (!sampleOutput)
                {
                    throw std::runtime_error("Failed while writing exact profile samples.");
                }
                ++result.samplePointCount;
            }
        }
        result.filterMilliseconds += ElapsedMilliseconds(filterBegin, Clock::now());
        firstTile = lastTile;
    }

    if (ShouldCancel(request.requestId, stopToken))
    {
        result.state = ProfileTaskState::Cancelled;
        return result;
    }

    const auto writeBegin = Clock::now();
    FinalizeProfileSampleFile(sampleOutput, outputPaths, result.samplePointCount);
    result.samplePath = outputPaths.samples;
    result.outputPath = WriteProfileCsv(
        outputPaths.statistics,
        result,
        metadata,
        segments);
    sampleOutputGuard.Release();
    result.writeMilliseconds = ElapsedMilliseconds(writeBegin, Clock::now());
    result.totalMilliseconds = ElapsedMilliseconds(totalBegin, Clock::now());
    result.state = ProfileTaskState::Complete;
    return result;
}

bool ProfileAnalysisService::ShouldCancel(
    std::uint64_t requestId,
    std::stop_token stopToken) const
{
    return stopToken.stop_requested() ||
        latestRequestId_.load(std::memory_order_acquire) != requestId;
}

bool ProfileAnalysisService::WaitForRenderBudget(
    std::uint64_t requestId,
    std::stop_token stopToken)
{
    while (renderBusy_.load(std::memory_order_acquire))
    {
        if (ShouldCancel(requestId, stopToken))
        {
            return false;
        }
        state_.store(ProfileTaskState::Paused, std::memory_order_release);
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, stopToken, std::chrono::milliseconds(10), [this]()
        {
            return !renderBusy_.load(std::memory_order_acquire);
        });
    }
    if (ShouldCancel(requestId, stopToken))
    {
        return false;
    }
    state_.store(ProfileTaskState::Running, std::memory_order_release);
    return true;
}

} // namespace gpv
