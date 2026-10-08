// GPU Page 缓存声明：映射二次预处理文件并按视口选择统一 LOD Page。
#pragma once

#include "SeismicCacheTypes.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpv
{

class GpuReadyPageCache
{
public:
    GpuReadyPageCache() = default;
    ~GpuReadyPageCache();

    GpuReadyPageCache(const GpuReadyPageCache&) = delete;
    GpuReadyPageCache& operator=(const GpuReadyPageCache&) = delete;

    bool Open(const std::filesystem::path& cacheDirectory, const CacheMetadata& metadata);
    void Close();
    bool Available() const;
    CacheQueryResult QueryViewport(
        const Bounds2D& viewport,
        const CacheQueryOptions& options,
        bool collectVisiblePages = true) const;
    std::vector<GpuReadyPageView> AllPagesCoarseFirst() const;
    std::uint64_t TotalPageCount() const;
    std::uint32_t PagePointCapacity() const;
    std::uint32_t PointRecordSize() const;

private:
    class MappedFile
    {
    public:
        MappedFile() = default;
        ~MappedFile();
        MappedFile(MappedFile&& other) noexcept;
        MappedFile& operator=(MappedFile&& other) noexcept;
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;

        void Open(const std::filesystem::path& path);
        void Close();
        const std::byte* Data() const;
        std::uint64_t Size() const;

    private:
        void* fileHandle_ = nullptr;
        void* mappingHandle_ = nullptr;
        const std::byte* data_ = nullptr;
        std::uint64_t size_ = 0;
    };

    struct Level
    {
        std::string name;
        int lodLevel = 0;
        std::uint64_t pointCount = 0;
        MappedFile pointFile;
        std::vector<GpuReadyPageView> pages;
    };

    static bool Intersects(const Bounds2D& left, const Bounds2D& right);
    std::uint64_t EstimateVisiblePoints(const Level& level, const Bounds2D& viewport) const;
    const Level& SelectLevel(const Bounds2D& viewport, const CacheQueryOptions& options) const;

    std::vector<Level> levels_;
    Bounds2D worldBounds_;
    std::uint64_t totalPageCount_ = 0;
    std::uint64_t totalPointCount_ = 0;
    std::uint32_t pagePointCapacity_ = 0;
    std::uint32_t pointRecordSize_ = 0;
    bool available_ = false;
};

} // namespace gpv
