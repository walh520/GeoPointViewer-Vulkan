// GPU Page 缓存实现：校验 GPRPAGE/GPRIDX 协议并提供零拷贝 Page 视图。
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "GpuReadyPageCache.h"
#include "SeismicCacheReader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gpv
{
namespace
{

constexpr std::uint32_t kGpuPageContractVersionLegacy = 1;
constexpr std::uint32_t kGpuPageContractVersionQuantized = 2;
constexpr std::uint32_t kGpuPageHeaderSize = 64;
constexpr std::uint32_t kGpuPageIndexRecordSize = 72;
constexpr std::uint32_t kGpuPagePointCapacity = 65'536;
constexpr std::uint64_t kDefaultViewportPointBudget = 2'000'000;
constexpr std::uint64_t kMaxViewportPoints = 10'000'000;
constexpr double kLodUpgradeBudgetRatio = 0.75;
constexpr double kLodDowngradeBudgetRatio = 1.0;

double EffectiveLodPointsPerPixel(const CacheQueryOptions& options)
{
    return std::clamp(
        options.lodPointsPerPixel,
        InteractiveLodPointsPerPixel,
        StableLodPointsPerPixel);
}

std::uint64_t ComputeViewportPointBudget(const CacheQueryOptions& options)
{
    if (options.framebufferWidth == 0 || options.framebufferHeight == 0)
    {
        return std::min(kMaxViewportPoints, kDefaultViewportPointBudget);
    }

    const long double framebufferPixels =
        static_cast<long double>(options.framebufferWidth) *
        static_cast<long double>(options.framebufferHeight);
    const long double requestedBudget =
        framebufferPixels * EffectiveLodPointsPerPixel(options);
    return static_cast<std::uint64_t>(std::ceil(std::min(
        requestedBudget,
        static_cast<long double>(kMaxViewportPoints))));
}

double ComputeEstimatedPointsPerPixel(
    const CacheQueryOptions& options,
    std::uint64_t estimatedPoints)
{
    if (options.framebufferWidth == 0 || options.framebufferHeight == 0)
    {
        return 0.0;
    }
    const long double framebufferPixels =
        static_cast<long double>(options.framebufferWidth) *
        static_cast<long double>(options.framebufferHeight);
    return static_cast<double>(
        static_cast<long double>(estimatedPoints) / framebufferPixels);
}

template<typename T>
T ReadValue(const std::byte* bytes)
{
    T value{};
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

struct PageHeader
{
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::uint32_t headerSize = 0;
    std::uint32_t recordSize = 0;
    std::uint32_t attrCount = 0;
    std::uint64_t pageCount = 0;
    std::uint64_t pointCount = 0;
    std::uint64_t pageCapacity = 0;
};

PageHeader ReadHeader(const std::byte* data, std::uint64_t size)
{
    if (size < kGpuPageHeaderSize)
    {
        throw std::runtime_error("GPU-ready Page header is truncated.");
    }

    PageHeader header;
    std::memcpy(header.magic.data(), data, header.magic.size());
    header.version = ReadValue<std::uint32_t>(data + 8);
    header.headerSize = ReadValue<std::uint32_t>(data + 12);
    header.recordSize = ReadValue<std::uint32_t>(data + 16);
    header.attrCount = ReadValue<std::uint32_t>(data + 20);
    header.pageCount = ReadValue<std::uint64_t>(data + 24);
    header.pointCount = ReadValue<std::uint64_t>(data + 32);
    header.pageCapacity = ReadValue<std::uint64_t>(data + 40);
    return header;
}

bool MagicEquals(const std::array<char, 8>& magic, const char* expected)
{
    return std::memcmp(magic.data(), expected, magic.size()) == 0;
}

} // namespace

GpuReadyPageCache::~GpuReadyPageCache()
{
    Close();
}

GpuReadyPageCache::MappedFile::~MappedFile()
{
    Close();
}

GpuReadyPageCache::MappedFile::MappedFile(MappedFile&& other) noexcept
{
    *this = std::move(other);
}

GpuReadyPageCache::MappedFile& GpuReadyPageCache::MappedFile::operator=(MappedFile&& other) noexcept
{
    if (this == &other)
    {
        return *this;
    }
    Close();
    fileHandle_ = other.fileHandle_;
    mappingHandle_ = other.mappingHandle_;
    data_ = other.data_;
    size_ = other.size_;
    other.fileHandle_ = nullptr;
    other.mappingHandle_ = nullptr;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void GpuReadyPageCache::MappedFile::Open(const std::filesystem::path& path)
{
    Close();
#ifdef _WIN32
    HANDLE file = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        throw std::runtime_error("Failed to open GPU-ready Page file: " + path.string());
    }

    LARGE_INTEGER fileSize{};
    if (!GetFileSizeEx(file, &fileSize) || fileSize.QuadPart <= 0)
    {
        CloseHandle(file);
        throw std::runtime_error("GPU-ready Page file size is invalid: " + path.string());
    }

    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr)
    {
        CloseHandle(file);
        throw std::runtime_error("Failed to map GPU-ready Page file: " + path.string());
    }
    const void* mapped = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (mapped == nullptr)
    {
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("Failed to view GPU-ready Page file: " + path.string());
    }

    fileHandle_ = file;
    mappingHandle_ = mapping;
    data_ = static_cast<const std::byte*>(mapped);
    size_ = static_cast<std::uint64_t>(fileSize.QuadPart);
#else
    (void)path;
    throw std::runtime_error("GPU-ready Page mapping is only implemented on Windows.");
#endif
}

void GpuReadyPageCache::MappedFile::Close()
{
#ifdef _WIN32
    if (data_ != nullptr)
    {
        UnmapViewOfFile(data_);
    }
    if (mappingHandle_ != nullptr)
    {
        CloseHandle(static_cast<HANDLE>(mappingHandle_));
    }
    if (fileHandle_ != nullptr)
    {
        CloseHandle(static_cast<HANDLE>(fileHandle_));
    }
#endif
    fileHandle_ = nullptr;
    mappingHandle_ = nullptr;
    data_ = nullptr;
    size_ = 0;
}

const std::byte* GpuReadyPageCache::MappedFile::Data() const
{
    return data_;
}

std::uint64_t GpuReadyPageCache::MappedFile::Size() const
{
    return size_;
}

bool GpuReadyPageCache::Open(
    const std::filesystem::path& cacheDirectory,
    const CacheMetadata& metadata)
{
    Close();
    worldBounds_ = metadata.xyRange;
    const std::size_t expectedLevelCount = static_cast<std::size_t>(std::count_if(
        metadata.lodLevels.begin(), metadata.lodLevels.end(), [](const CacheLodLevel& lod)
        {
            return lod.mainDisplayChain &&
                lod.kind != "overview_grid" &&
                !lod.pointPath.empty();
        }));
    const std::size_t configuredLevelCount = static_cast<std::size_t>(std::count_if(
        metadata.lodLevels.begin(), metadata.lodLevels.end(), [](const CacheLodLevel& lod)
        {
            return lod.mainDisplayChain &&
                lod.kind != "overview_grid" &&
                lod.gpuPageContractVersion != 0;
        }));
    if (configuredLevelCount == 0)
    {
        return false;
    }
    if (configuredLevelCount != expectedLevelCount)
    {
        throw std::runtime_error("GPU-ready Page metadata is incomplete for the main LOD chain.");
    }

    for (const CacheLodLevel& lod : metadata.lodLevels)
    {
        if (!lod.mainDisplayChain || lod.kind == "overview_grid" || lod.gpuPageContractVersion == 0)
        {
            continue;
        }
        const bool legacyContract =
            lod.gpuPageContractVersion == kGpuPageContractVersionLegacy &&
            lod.gpuPagePointRecordSize == sizeof(GpuPoint);
        const bool quantizedContract =
            lod.gpuPageContractVersion == kGpuPageContractVersionQuantized &&
            lod.gpuPagePointRecordSize == sizeof(GpuPagePoint);
        if ((!legacyContract && !quantizedContract) ||
            lod.gpuPageIndexRecordSize != kGpuPageIndexRecordSize ||
            lod.gpuPagePointCapacity != kGpuPagePointCapacity)
        {
            throw std::runtime_error("Unsupported GPU-ready Page contract for LOD: " + lod.name);
        }

        MappedFile indexFile;
        indexFile.Open(cacheDirectory / lod.gpuPageIndexPath);
        Level level;
        level.name = lod.name;
        level.lodLevel = lod.level;
        level.pointCount = lod.gpuPagePointCount;
        level.pointFile.Open(cacheDirectory / lod.gpuPagePath);

        const PageHeader dataHeader = ReadHeader(level.pointFile.Data(), level.pointFile.Size());
        const PageHeader indexHeader = ReadHeader(indexFile.Data(), indexFile.Size());
        const bool validLegacyHeaders =
            legacyContract &&
            MagicEquals(dataHeader.magic, "GPRPAGE1") &&
            MagicEquals(indexHeader.magic, "GPRIDX1\0");
        const bool validQuantizedHeaders =
            quantizedContract &&
            MagicEquals(dataHeader.magic, "GPRPAGE2") &&
            MagicEquals(indexHeader.magic, "GPRIDX2\0");
        if ((!validLegacyHeaders && !validQuantizedHeaders) ||
            dataHeader.version != lod.gpuPageContractVersion ||
            indexHeader.version != lod.gpuPageContractVersion ||
            dataHeader.headerSize != kGpuPageHeaderSize ||
            indexHeader.headerSize != kGpuPageHeaderSize ||
            dataHeader.recordSize != lod.gpuPagePointRecordSize ||
            indexHeader.recordSize != kGpuPageIndexRecordSize ||
            dataHeader.attrCount != metadata.attrFields.size() ||
            indexHeader.attrCount != metadata.attrFields.size() ||
            dataHeader.pageCapacity != kGpuPagePointCapacity ||
            indexHeader.pageCapacity != kGpuPagePointCapacity ||
            dataHeader.pageCount != indexHeader.pageCount ||
            dataHeader.pointCount != indexHeader.pointCount ||
            dataHeader.pointCount != lod.gpuPagePointCount ||
            dataHeader.pageCount != lod.gpuPageCount)
        {
            throw std::runtime_error("GPU-ready Page header mismatch for LOD: " + lod.name);
        }

        const std::uint64_t expectedDataBytes =
            kGpuPageHeaderSize + dataHeader.pointCount * dataHeader.recordSize;
        const std::uint64_t expectedIndexBytes =
            kGpuPageHeaderSize + indexHeader.pageCount * kGpuPageIndexRecordSize;
        if (level.pointFile.Size() != expectedDataBytes || indexFile.Size() != expectedIndexBytes)
        {
            throw std::runtime_error("GPU-ready Page file length mismatch for LOD: " + lod.name);
        }

        level.pages.reserve(static_cast<std::size_t>(indexHeader.pageCount));
        const std::byte* pointBase = level.pointFile.Data() + kGpuPageHeaderSize;
        for (std::uint64_t pageIndex = 0; pageIndex < indexHeader.pageCount; ++pageIndex)
        {
            const std::byte* record =
                indexFile.Data() + kGpuPageHeaderSize + pageIndex * kGpuPageIndexRecordSize;
            const std::uint64_t pointOffset = ReadValue<std::uint64_t>(record + 8);
            const std::uint32_t pointCount = ReadValue<std::uint32_t>(record + 16);
            if (pointOffset > dataHeader.pointCount || pointCount > dataHeader.pointCount - pointOffset)
            {
                throw std::runtime_error("GPU-ready Page point range is invalid for LOD: " + lod.name);
            }

            GpuReadyPageView page;
            page.key = ReadValue<std::uint64_t>(record);
            page.pointCount = pointCount;
            page.lodLevel = static_cast<int>(ReadValue<std::uint32_t>(record + 20));
            page.originX = ReadValue<double>(record + 24);
            page.originY = ReadValue<double>(record + 32);
            page.bounds.minX = ReadValue<double>(record + 40);
            page.bounds.minY = ReadValue<double>(record + 48);
            page.bounds.maxX = ReadValue<double>(record + 56);
            page.bounds.maxY = ReadValue<double>(record + 64);
            page.points = pointBase + pointOffset * dataHeader.recordSize;
            page.pointRecordSize = dataHeader.recordSize;
            page.pointFormat = quantizedContract
                ? GpuPagePointFormat::QuantizedUnorm16
                : GpuPagePointFormat::LegacyFloatRgba;
            level.pages.push_back(page);
        }

        totalPageCount_ += level.pages.size();
        totalPointCount_ += level.pointCount;
        pagePointCapacity_ = lod.gpuPagePointCapacity;
        if (pointRecordSize_ != 0 && pointRecordSize_ != lod.gpuPagePointRecordSize)
        {
            throw std::runtime_error("GPU-ready Page point layouts cannot be mixed across LOD levels.");
        }
        pointRecordSize_ = lod.gpuPagePointRecordSize;
        levels_.push_back(std::move(level));
    }

    std::sort(
        levels_.begin(),
        levels_.end(),
        [](const Level& left, const Level& right)
        {
            return left.lodLevel < right.lodLevel;
        });
    available_ = !levels_.empty();
    if (available_)
    {
        std::cout
            << "GPU-ready RAM Page Cache opened: levels=" << levels_.size()
            << ", pages=" << totalPageCount_
            << ", points=" << totalPointCount_
            << ", point_record_bytes=" << pointRecordSize_
            << ", contract="
            << (pointRecordSize_ == sizeof(GpuPagePoint) ? "quantized_v2" : "legacy_v1")
            << '\n';
    }
    return available_;
}

void GpuReadyPageCache::Close()
{
    levels_.clear();
    totalPageCount_ = 0;
    totalPointCount_ = 0;
    pagePointCapacity_ = 0;
    pointRecordSize_ = 0;
    available_ = false;
}

bool GpuReadyPageCache::Available() const
{
    return available_;
}

bool GpuReadyPageCache::Intersects(const Bounds2D& left, const Bounds2D& right)
{
    return
        left.maxX >= right.minX && left.minX <= right.maxX &&
        left.maxY >= right.minY && left.minY <= right.maxY;
}

std::uint64_t GpuReadyPageCache::EstimateVisiblePoints(
    const Level& level,
    const Bounds2D& viewport) const
{
    std::uint64_t count = 0;
    for (const GpuReadyPageView& page : level.pages)
    {
        if (Intersects(page.bounds, viewport))
        {
            count += page.pointCount;
        }
    }
    return count;
}

const GpuReadyPageCache::Level& GpuReadyPageCache::SelectLevel(
    const Bounds2D& viewport,
    const CacheQueryOptions& options) const
{
    const std::uint64_t pointBudget = ComputeViewportPointBudget(options);
    const auto isLocalLevel = [](const Level& level)
    {
        return level.name != "bridge_mean";
    };
    const auto fits = [this, &viewport, pointBudget](const Level& level, double ratio)
    {
        return static_cast<double>(EstimateVisiblePoints(level, viewport)) <=
            static_cast<double>(pointBudget) * ratio;
    };

    if (options.forceExact)
    {
        const auto exact = std::find_if(
            levels_.begin(), levels_.end(), [](const Level& level)
            {
                return level.lodLevel == 0;
            });
        if (exact != levels_.end())
        {
            return *exact;
        }
    }

    if (options.preferGlobalMean)
    {
        const auto globalMean = std::find_if(
            levels_.begin(), levels_.end(), [](const Level& level)
            {
                return level.name == "bridge_mean";
            });
        if (globalMean != levels_.end())
        {
            return *globalMean;
        }
    }

    if (!options.preferredLodName.empty())
    {
        const auto preferred = std::find_if(
            levels_.begin(), levels_.end(), [&options, &isLocalLevel](const Level& level)
            {
                return
                    isLocalLevel(level) &&
                    level.name == options.preferredLodName;
            });
        if (preferred != levels_.end())
        {
            return *preferred;
        }
    }

    const auto current = std::find_if(
        levels_.begin(), levels_.end(), [&options, &isLocalLevel](const Level& level)
        {
            return
                isLocalLevel(level) &&
                (level.name == options.currentLodName ||
                    (level.lodLevel == 0 && options.currentLodName == "lod0_exact"));
        });
    if (current != levels_.end())
    {
        for (auto finer = levels_.begin(); finer != current; ++finer)
        {
            if (isLocalLevel(*finer) && fits(*finer, kLodUpgradeBudgetRatio))
            {
                return *finer;
            }
        }
        if (fits(*current, kLodDowngradeBudgetRatio))
        {
            return *current;
        }
        for (auto coarser = std::next(current); coarser != levels_.end(); ++coarser)
        {
            if (isLocalLevel(*coarser) && fits(*coarser, 1.0))
            {
                return *coarser;
            }
        }
    }
    else
    {
        for (const Level& level : levels_)
        {
            if (isLocalLevel(level) && fits(level, 1.0))
            {
                return level;
            }
        }
    }

    const auto coarsestLocal = std::find_if(
        levels_.rbegin(), levels_.rend(), isLocalLevel);
    if (coarsestLocal != levels_.rend())
    {
        return *coarsestLocal;
    }
    return levels_.back();
}

CacheQueryResult GpuReadyPageCache::QueryViewport(
    const Bounds2D& viewport,
    const CacheQueryOptions& options,
    bool collectVisiblePages) const
{
    if (!available_)
    {
        throw std::runtime_error("GPU-ready Page Cache is not available.");
    }
    const Level& level = SelectLevel(viewport, options);
    CacheQueryResult result;
    result.lodPointsPerPixel = EffectiveLodPointsPerPixel(options);
    result.lodPointBudget = ComputeViewportPointBudget(options);
    result.activeLodLevel = level.lodLevel;
    result.activeLodName = level.lodLevel == 0 ? "lod0_exact" : level.name;
    result.screenSpaceLod = true;
    result.gpuReadyFastPath = true;
    result.gpuDrivenViewport = !collectVisiblePages;
    result.lod0IndexReady = true;
    for (const GpuReadyPageView& page : level.pages)
    {
        if (options.shouldCancel && options.shouldCancel())
        {
            throw CacheQueryCancelled();
        }
        if (!Intersects(page.bounds, viewport))
        {
            continue;
        }
        if (collectVisiblePages)
        {
            result.gpuReadyPages.push_back(page);
        }
        result.estimatedPoints += page.pointCount;
        ++result.hitTileCount;
    }
    result.readPoints = result.estimatedPoints;
    result.estimatedPointsPerPixel = ComputeEstimatedPointsPerPixel(
        options,
        result.estimatedPoints);
    if (level.lodLevel == 0)
    {
        result.lod0PointCount = result.estimatedPoints;
        result.lod0TileCount = result.hitTileCount;
    }
    return result;
}

std::vector<GpuReadyPageView> GpuReadyPageCache::AllPagesCoarseFirst() const
{
    std::vector<GpuReadyPageView> result;
    result.reserve(static_cast<std::size_t>(totalPageCount_));
    for (auto level = levels_.rbegin(); level != levels_.rend(); ++level)
    {
        result.insert(result.end(), level->pages.begin(), level->pages.end());
    }
    return result;
}

std::uint64_t GpuReadyPageCache::TotalPageCount() const
{
    return totalPageCount_;
}

std::uint32_t GpuReadyPageCache::PagePointCapacity() const
{
    return pagePointCapacity_;
}

std::uint32_t GpuReadyPageCache::PointRecordSize() const
{
    return pointRecordSize_;
}

} // namespace gpv
