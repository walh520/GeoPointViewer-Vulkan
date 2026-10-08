// Python 缓存读取器声明：读取 metadata、tile index、二进制 tile 和色标。
#pragma once

#include "SeismicCacheTypes.h"

#include <filesystem>
#include <exception>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace gpv
{

class CacheQueryCancelled final : public std::exception
{
public:
    const char* what() const noexcept override
    {
        return "Cache viewport query cancelled.";
    }
};

class SeismicCacheReader
{
public:
    struct OpenTimingStats
    {
        double metadataMilliseconds = 0.0;
        double colormapMilliseconds = 0.0;
        double lod0IndexStartMilliseconds = 0.0;
        double pointLodIndexMilliseconds = 0.0;
        double overviewMilliseconds = 0.0;
        double totalMilliseconds = 0.0;
    };

    ~SeismicCacheReader();

    void Open(const std::filesystem::path& cacheDirectory);
    CacheQueryResult QueryViewport(
        const Bounds2D& viewport,
        std::size_t activeAttrIndex,
        const CacheQueryOptions& options = CacheQueryOptions{}) const;
    std::vector<CachePoint> QueryExactPoints(const Bounds2D& viewport) const;
    std::optional<CachePoint> ReadLod0PointBySourceId(std::uint64_t sourceId) const;
    std::vector<GpuPoint> BuildGpuPoints(
        const std::vector<CachePoint>& points,
        std::size_t activeAttrIndex,
        double renderOriginX,
        double renderOriginY,
        const std::function<bool()>& shouldCancel = {}) const;
    std::vector<GpuTilePage> BuildGpuTilePages(
        const std::vector<CachePoint>& points,
        const std::string& lodName,
        int lodLevel,
        std::size_t activeAttrIndex,
        const std::function<bool()>& shouldCancel = {}) const;
    std::vector<GpuPoint> BuildOverviewGpuPoints(
        const Bounds2D& viewport,
        std::size_t activeAttrIndex,
        float alpha,
        double renderOriginX,
        double renderOriginY,
        const std::function<bool()>& shouldCancel = {}) const;
    std::vector<GpuTileMetadata> BuildGpuTileMetadata(std::size_t activeAttrIndex) const;
    std::vector<std::uint32_t> BuildGpuProxyMask(const std::vector<CachePoint>& points) const;
    float ComputeOverviewAlpha(const Bounds2D& viewport) const;

    const CacheMetadata& Metadata() const;
    std::size_t OverviewPointCount() const;
    const OpenTimingStats& LastOpenTiming() const;
    const std::array<float, 4>& ColorForValue(float value, std::size_t activeAttrIndex) const;
    std::array<std::uint32_t, 256> PackedColormap() const;
    TileCacheStats GetTileCacheStats() const;
    std::filesystem::path ResolveTilesPath() const;
    void WaitForLod0Index() const;
    Lod0RegionSummary SummarizeLod0Region(const Bounds2D& bounds) const;
    std::vector<TileIndexEntry> CopyLod0TilesIntersecting(const Bounds2D& bounds) const;
    bool IsLod0IndexReady() const;
    void StartLod0IndexLoadAsync() const;

    static std::filesystem::path ResolveDefaultCachePath();
    static bool RunContractCheck(const std::filesystem::path& cacheDirectory);

private:
    static constexpr std::uint64_t MaxLod0PointLoad = 10'000'000;
    static constexpr std::uint32_t MaxLod0TileLoad = 8192;
    static constexpr std::size_t MaxCachedTiles = 1'500'000;
    static constexpr std::uint64_t MaxCachedTilePoints = 16'000'000;

    struct TileCacheKey
    {
        std::uint32_t lodNamespace = 0;
        std::int64_t tileId = 0;

        bool operator==(const TileCacheKey& other) const
        {
            return lodNamespace == other.lodNamespace && tileId == other.tileId;
        }
    };

    struct TileCacheKeyHash
    {
        std::size_t operator()(const TileCacheKey& key) const
        {
            const std::size_t lodHash = std::hash<std::uint32_t>{}(key.lodNamespace);
            const std::size_t tileHash = std::hash<std::int64_t>{}(key.tileId);
            return lodHash ^ (tileHash + 0x9e3779b9u + (lodHash << 6) + (lodHash >> 2));
        }
    };

    struct CachedTile
    {
        std::shared_ptr<const std::vector<CachePoint>> points;
        std::uint64_t pointCount = 0;
        std::list<TileCacheKey>::iterator lruPosition;
    };

    struct LodIndexData
    {
        CacheLodLevel metadata;
        std::uint32_t cacheNamespace = 0;
        std::vector<TileIndexEntry> tiles;
        std::unordered_map<std::int64_t, std::size_t> tileIndexById;
    };

    void LoadMetadata();
    void LoadColormap();
    void LoadTileIndices();
    void LoadOverviewPoints();
    void DrainLod0IndexLoad(bool wait) const;
    LodIndexData LoadLodIndexBinary(const CacheLodLevel& lodLevel) const;
    const LodIndexData* SelectViewportLod(
        const Bounds2D& viewport,
        const CacheQueryOptions& options) const;
    CacheQueryResult QueryLod(
        const LodIndexData& lod,
        const Bounds2D& viewport,
        std::size_t activeAttrIndex,
        const CacheQueryOptions& options) const;
    std::uint64_t EstimateVisiblePoints(
        const LodIndexData& lod,
        const Bounds2D& viewport,
        const CacheQueryOptions& options) const;
    std::vector<CachePoint> ReadTilePoints(
        const std::vector<const TileIndexEntry*>& tiles,
        const Bounds2D& viewport,
        const CacheQueryOptions& options) const;
    std::vector<CachePoint> ReadLodTilePoints(
        const LodIndexData& lod,
        const std::vector<const TileIndexEntry*>& tiles,
        const Bounds2D& viewport,
        const CacheQueryOptions& options) const;
    std::shared_ptr<const std::vector<CachePoint>> GetOrReadTilePoints(
        const TileCacheKey& cacheKey,
        std::ifstream& file,
        const TileIndexEntry& tile,
        const RecordField& xField,
        const RecordField& yField,
        const std::array<const RecordField*, MaxCacheAttributes>& attrFields) const;
    std::vector<CachePoint> ReadRawTilePoints(
        std::ifstream& file,
        const TileIndexEntry& tile,
        const RecordField& xField,
        const RecordField& yField,
        const std::array<const RecordField*, MaxCacheAttributes>& attrFields) const;
    void AttachLodSidecars(
        const LodIndexData& lod,
        const TileIndexEntry& tile,
        std::ifstream* sourceIds,
        std::ifstream* cellIds,
        std::vector<CachePoint>& points) const;
    CachePoint ReadPointRecordAt(
        std::ifstream& file,
        std::uint64_t byteOffset,
        std::int64_t tileId,
        std::uint32_t tileIndex) const;
    void ValidateLodFiles(const LodIndexData& lod) const;
    std::shared_ptr<const std::vector<CachePoint>> FindCachedTilePoints(
        const TileCacheKey& cacheKey) const;
    std::shared_ptr<const std::vector<CachePoint>> StoreTileInCache(
        const TileCacheKey& cacheKey,
        std::vector<CachePoint>&& points) const;
    void PruneTileCacheLocked() const;

    const RecordField& FindRecordField(const std::string& name) const;
    std::uint32_t ColorIndexForValue(float value, std::size_t activeAttrIndex) const;
    float ReadAttributeValue(const char* record, const RecordField& field) const;
    double ReadCoordinateValue(const char* record, const RecordField& field) const;

    CacheMetadata metadata_;
    mutable std::vector<TileIndexEntry> tileIndex_;
    mutable std::unordered_map<std::int64_t, std::size_t> tileIndexById_;
    std::vector<LodIndexData> lodIndices_;
    std::vector<OverviewPoint> overviewPoints_;
    mutable std::future<LodIndexData> lod0IndexFuture_;
    mutable std::mutex indexMutex_;
    mutable std::list<TileCacheKey> tileCacheLru_;
    mutable std::unordered_map<TileCacheKey, CachedTile, TileCacheKeyHash> tileCache_;
    mutable std::mutex tileCacheMutex_;
    mutable TileCacheStats tileCacheStats_;
    mutable std::uint64_t cachedTilePointCount_ = 0;
    std::array<std::array<float, 4>, 256> colormap_{};
    OpenTimingStats lastOpenTiming_;
    mutable bool lod0IndexReady_ = false;
    mutable bool lod0IndexLoading_ = false;
    mutable bool lod0IndexLoadFailed_ = false;
    mutable std::string lod0IndexLoadError_;
    bool opened_ = false;
};

} // namespace gpv
