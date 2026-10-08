// Python 缓存数据结构声明：描述 metadata、tile 索引、CPU 点和 GPU 顶点。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpv
{

constexpr std::size_t MaxCacheAttributes = 8;
constexpr double InteractiveLodPointsPerPixel = 1.0;
constexpr double StableLodPointsPerPixel = 1.25;

struct Bounds2D
{
    double minX = 0.0;
    double minY = 0.0;
    double maxX = 1.0;
    double maxY = 1.0;
};

struct RecordField
{
    std::string name;
    std::string dtype;
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

struct AttributeRange
{
    double minValue = 0.0;
    double maxValue = 1.0;
};

struct CacheLodLevel
{
    int level = 0;
    int aggregateFactor = 0;
    std::string name;
    std::string kind;
    std::string pointPath;
    std::string indexBinaryPath;
    std::string sourceIdPath;
    std::string cellIdPath;
    std::string gpuPagePath;
    std::string gpuPageIndexPath;
    std::uint32_t gpuPageContractVersion = 0;
    std::uint32_t gpuPagePointRecordSize = 0;
    std::uint32_t gpuPageIndexRecordSize = 0;
    std::uint32_t gpuPagePointCapacity = 0;
    std::uint64_t gpuPageCount = 0;
    std::uint64_t gpuPagePointCount = 0;
    std::string attributeSemantics;
    double sampleRatio = 0.0;
    bool mainDisplayChain = false;
    bool recordSubsetOfLod0 = false;
    std::uint32_t tileRows = 0;
    std::uint32_t tileCols = 0;
    std::uint64_t pointCount = 0;
    std::uint64_t tileCount = 0;
};

struct CacheMetadata
{
    std::filesystem::path cacheDirectory;
    Bounds2D xyRange;
    std::uint64_t validPoints = 0;
    std::uint32_t tileRows = 0;
    std::uint32_t tileCols = 0;
    std::size_t recordSize = 0;
    std::string tilesPath;
    std::string tileIndexBinaryPath;
    std::string colormapJsonPath;
    std::string overviewPointsPath;
    std::uint32_t lodContractVersion = 0;
    std::uint32_t gpuReadyPageContractVersion = 0;
    std::vector<std::string> attrFields;
    std::vector<std::string> mainLodLevels;
    std::vector<RecordField> recordFields;
    std::vector<CacheLodLevel> lodLevels;
    std::unordered_map<std::string, AttributeRange> attrRanges;
};

struct TileIndexEntry
{
    std::int64_t tileId = 0;
    std::uint32_t row = 0;
    std::uint32_t col = 0;
    Bounds2D bounds;
    std::uint64_t pointCount = 0;
    std::uint64_t offsetBytes = 0;
    std::uint64_t lengthPoints = 0;
    std::uint32_t metadataIndex = 0;
    std::array<float, MaxCacheAttributes> attrMins{};
    std::array<float, MaxCacheAttributes> attrMaxs{};
    std::array<float, MaxCacheAttributes> attrMeans{};
};

struct Lod0RegionSummary
{
    std::uint64_t tileCount = 0;
    std::uint64_t pointCount = 0;
};

struct CachePoint
{
    std::int64_t tileId = 0;
    std::uint32_t tileIndex = 0;
    double x = 0.0;
    double y = 0.0;
    std::array<float, MaxCacheAttributes> attrs{};
    std::uint64_t sourceId = 0;
    std::uint32_t cellId = 0;
    bool hasSourceId = false;
    bool hasCellId = false;
    bool proxy = false;
};

struct OverviewPoint
{
    double x = 0.0;
    double y = 0.0;
    std::array<float, MaxCacheAttributes> attrs{};
    float count = 0.0f;
};

struct CacheQueryOptions
{
    std::uint32_t framebufferWidth = 0;
    std::uint32_t framebufferHeight = 0;
    double lodPointsPerPixel = StableLodPointsPerPixel;
    bool progressiveLoading = true;
    bool forceExact = false;
    bool preferGlobalMean = false;
    std::string preferredLodName;
    std::string currentLodName;
    std::function<bool()> shouldCancel;
};

struct GpuPoint
{
    float x = 0.0f;
    float y = 0.0f;
    std::uint32_t rgba0 = 0xFFFFFFFFu;
    std::uint32_t rgba1 = 0xFFFFFFFFu;
};

struct GpuPagePoint
{
    std::uint32_t quantizedXY = 0;
    std::uint16_t colorIndices = 0;
    std::uint16_t reserved = 0;
};

static_assert(sizeof(GpuPagePoint) == 8, "Quantized GPU Page point must remain 8 bytes.");

enum class GpuPagePointFormat : std::uint32_t
{
    LegacyFloatRgba = 1,
    QuantizedUnorm16 = 2,
};

struct GpuReadyPageView
{
    std::uint64_t key = 0;
    int lodLevel = 0;
    double originX = 0.0;
    double originY = 0.0;
    Bounds2D bounds;
    const std::byte* points = nullptr;
    std::uint32_t pointCount = 0;
    std::uint32_t pointRecordSize = 0;
    GpuPagePointFormat pointFormat = GpuPagePointFormat::LegacyFloatRgba;
};

struct GpuTilePage
{
    std::uint64_t key = 0;
    std::uint64_t signature = 0;
    int lodLevel = 0;
    double originX = 0.0;
    double originY = 0.0;
    Bounds2D bounds;
    std::vector<GpuPoint> points;
};

struct GpuTileMetadata
{
    float minX = 0.0f;
    float minY = 0.0f;
    float maxX = 0.0f;
    float maxY = 0.0f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float attrMean = 0.0f;
    float pointCountFloat = 0.0f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    std::uint32_t tileIdLow = 0;
    std::uint32_t tileIdHigh = 0;
    std::uint32_t offsetBytesLow = 0;
    std::uint32_t offsetBytesHigh = 0;
    std::uint32_t lengthPoints = 0;
    std::uint32_t row = 0;
    std::uint32_t col = 0;
    std::uint32_t flags = 0;
};

struct CacheQueryResult
{
    std::vector<CachePoint> points;
    std::vector<GpuReadyPageView> gpuReadyPages;
    std::uint64_t readPoints = 0;
    std::uint64_t estimatedPoints = 0;
    std::uint64_t lodPointBudget = 0;
    double lodPointsPerPixel = StableLodPointsPerPixel;
    double estimatedPointsPerPixel = 0.0;
    std::uint64_t lod0PointCount = 0;
    std::uint64_t proxyPointCount = 0;
    std::uint32_t hitTileCount = 0;
    std::uint32_t lod0TileCount = 0;
    std::uint32_t proxyTileCount = 0;
    std::uint32_t deferredTileCount = 0;
    int activeLodLevel = 0;
    std::string activeLodName = "lod0";
    bool lod0IndexReady = true;
    bool lod0IndexPreparing = false;
    bool proxyMode = false;
    bool progressiveMode = false;
    bool screenSpaceLod = false;
    bool gpuReadyFastPath = false;
    bool gpuDrivenViewport = false;
};

struct TileCacheStats
{
    std::size_t cachedTileCount = 0;
    std::uint64_t cachedPointCount = 0;
    std::uint64_t cacheHits = 0;
    std::uint64_t cacheMisses = 0;
    std::uint64_t cacheEvictions = 0;
};

} // namespace gpv
