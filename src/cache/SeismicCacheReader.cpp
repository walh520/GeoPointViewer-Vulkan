// Python 缓存读取器实现：按 metadata 和 TIDX 二进制索引读取动态 LOD 点数据。
#include "SeismicCacheReader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpv
{
namespace
{

constexpr double kOverviewFullAlphaRatio = 0.35;
constexpr double kOverviewZeroAlphaRatio = 0.08;
constexpr float kOverviewMaxAlpha = 0.55f;
constexpr std::uint32_t kTileIndexVersion = 1;
constexpr std::uint32_t kTileIndexHeaderSize = 56;
constexpr std::size_t kTileIndexFixedRecordSize = 72;
constexpr std::uint64_t kDefaultViewportPointBudget = 2'000'000;
constexpr double kLodUpgradeBudgetRatio = 0.75;
constexpr double kLodDowngradeBudgetRatio = 1.0;
constexpr std::size_t kGpuPageMaxPoints = 65'536;
constexpr std::size_t kGpuPageGridSize = 16;
constexpr std::size_t kGpuPageGridCellCount = kGpuPageGridSize * kGpuPageGridSize;
constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

double EffectiveLodPointsPerPixel(const CacheQueryOptions& options)
{
    return std::clamp(
        options.lodPointsPerPixel,
        InteractiveLodPointsPerPixel,
        StableLodPointsPerPixel);
}

std::uint64_t ComputeViewportPointBudget(
    const CacheQueryOptions& options,
    std::uint64_t maximumBudget)
{
    if (options.framebufferWidth == 0 || options.framebufferHeight == 0)
    {
        return std::min(maximumBudget, kDefaultViewportPointBudget);
    }

    const long double framebufferPixels =
        static_cast<long double>(options.framebufferWidth) *
        static_cast<long double>(options.framebufferHeight);
    const long double requestedBudget =
        framebufferPixels * EffectiveLodPointsPerPixel(options);
    return static_cast<std::uint64_t>(std::ceil(std::min(
        requestedBudget,
        static_cast<long double>(maximumBudget))));
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

void HashBytes(std::uint64_t& hash, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < size; ++index)
    {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

std::uint64_t HashText(const std::string& text)
{
    std::uint64_t hash = kFnvOffsetBasis;
    HashBytes(hash, text.data(), text.size());
    return hash;
}

void MixHash(std::uint64_t& hash, std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
}

void ThrowIfQueryCancelled(const CacheQueryOptions& options)
{
    if (options.shouldCancel && options.shouldCancel())
    {
        throw CacheQueryCancelled();
    }
}

std::uint8_t FloatToUnorm8(float value)
{
    const float clampedValue = std::clamp(value, 0.0f, 1.0f);
    return static_cast<std::uint8_t>(std::lround(clampedValue * 255.0f));
}

std::uint32_t PackRgba8(const std::array<float, 4>& color, float alphaOverride = -1.0f)
{
    const std::uint8_t r = FloatToUnorm8(color[0]);
    const std::uint8_t g = FloatToUnorm8(color[1]);
    const std::uint8_t b = FloatToUnorm8(color[2]);
    const std::uint8_t a = FloatToUnorm8(alphaOverride >= 0.0f ? alphaOverride : color[3]);

    return
        static_cast<std::uint32_t>(r) |
        (static_cast<std::uint32_t>(g) << 8) |
        (static_cast<std::uint32_t>(b) << 16) |
        (static_cast<std::uint32_t>(a) << 24);
}

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

std::size_t FindMatchingClose(const std::string& text, std::size_t openIndex, char openChar, char closeChar)
{
    int depth = 0;
    bool inString = false;
    bool escaped = false;

    for (std::size_t index = openIndex; index < text.size(); ++index)
    {
        const char character = text[index];
        if (inString)
        {
            if (escaped)
            {
                escaped = false;
            }
            else if (character == '\\')
            {
                escaped = true;
            }
            else if (character == '"')
            {
                inString = false;
            }
            continue;
        }

        if (character == '"')
        {
            inString = true;
            continue;
        }

        if (character == openChar)
        {
            ++depth;
        }
        else if (character == closeChar)
        {
            --depth;
            if (depth == 0)
            {
                return index;
            }
        }
    }

    throw std::runtime_error("Invalid JSON structure.");
}

std::string FindJsonBlock(const std::string& text, const std::string& key, char openChar, char closeChar)
{
    const std::string marker = "\"" + key + "\"";
    const std::size_t keyIndex = text.find(marker);
    if (keyIndex == std::string::npos)
    {
        throw std::runtime_error("Missing JSON key: " + key);
    }

    const std::size_t openIndex = text.find(openChar, keyIndex + marker.size());
    if (openIndex == std::string::npos)
    {
        throw std::runtime_error("Missing JSON block for key: " + key);
    }

    const std::size_t closeIndex = FindMatchingClose(text, openIndex, openChar, closeChar);
    return text.substr(openIndex + 1, closeIndex - openIndex - 1);
}

bool TryFindJsonBlock(
    const std::string& text,
    const std::string& key,
    char openChar,
    char closeChar,
    std::string& output)
{
    const std::string marker = "\"" + key + "\"";
    const std::size_t keyIndex = text.find(marker);
    if (keyIndex == std::string::npos)
    {
        return false;
    }

    const std::size_t openIndex = text.find(openChar, keyIndex + marker.size());
    if (openIndex == std::string::npos)
    {
        return false;
    }

    const std::size_t closeIndex = FindMatchingClose(text, openIndex, openChar, closeChar);
    output = text.substr(openIndex + 1, closeIndex - openIndex - 1);
    return true;
}

double ExtractJsonDouble(const std::string& text, const std::string& key)
{
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        throw std::runtime_error("Missing JSON number: " + key);
    }

    return std::stod(match[1].str());
}

bool TryExtractJsonDouble(const std::string& text, const std::string& key, double& output)
{
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return false;
    }

    output = std::stod(match[1].str());
    return true;
}

std::uint64_t ExtractJsonUInt64(const std::string& text, const std::string& key)
{
    const double value = ExtractJsonDouble(text, key);
    if (value < 0.0)
    {
        throw std::runtime_error("JSON number must be non-negative: " + key);
    }

    return static_cast<std::uint64_t>(value);
}

std::uint64_t ExtractJsonUInt64Or(
    const std::string& text,
    const std::string& key,
    std::uint64_t fallback)
{
    double value = 0.0;
    if (!TryExtractJsonDouble(text, key, value))
    {
        return fallback;
    }
    if (value < 0.0)
    {
        throw std::runtime_error("JSON number must be non-negative: " + key);
    }

    return static_cast<std::uint64_t>(value);
}

std::string ExtractJsonString(const std::string& text, const std::string& key)
{
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        throw std::runtime_error("Missing JSON string: " + key);
    }

    return match[1].str();
}

std::string ExtractJsonStringOr(
    const std::string& text,
    const std::string& key,
    const std::string& fallback)
{
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return fallback;
    }

    return match[1].str();
}

std::vector<std::string> ParseJsonStringArray(const std::string& body)
{
    std::vector<std::string> values;
    const std::regex pattern("\"([^\"]*)\"");
    for (std::sregex_iterator it(body.begin(), body.end(), pattern), end; it != end; ++it)
    {
        values.push_back((*it)[1].str());
    }

    return values;
}

bool IsSupportedCacheDirectory(const std::filesystem::path& cacheDirectory)
{
    const std::filesystem::path metadataPath = cacheDirectory / "metadata.json";
    if (!std::filesystem::exists(metadataPath))
    {
        return false;
    }

    try
    {
        const std::string metadataText = ReadTextFile(metadataPath);
        double contractVersion = 0.0;
        if (!TryExtractJsonDouble(metadataText, "lod_contract_version", contractVersion) ||
            contractVersion < 6.0)
        {
            return false;
        }

        return !ExtractJsonStringOr(metadataText, "tile_index_binary_path", "").empty();
    }
    catch (const std::exception&)
    {
        return false;
    }
}

std::vector<std::string> SplitJsonObjects(const std::string& body)
{
    std::vector<std::string> objects;
    std::size_t cursor = 0;

    while (cursor < body.size())
    {
        const std::size_t openIndex = body.find('{', cursor);
        if (openIndex == std::string::npos)
        {
            break;
        }

        const std::size_t closeIndex = FindMatchingClose(body, openIndex, '{', '}');
        objects.push_back(body.substr(openIndex + 1, closeIndex - openIndex - 1));
        cursor = closeIndex + 1;
    }

    return objects;
}

std::uint32_t ClampTileIndex(double value, double minValue, double span, std::uint32_t tileCount)
{
    if (span <= 0.0 || tileCount == 0)
    {
        return 0;
    }

    const double tileWidth = span / static_cast<double>(tileCount);
    const double raw = std::floor((value - minValue) / tileWidth);
    const double clamped = std::clamp(raw, 0.0, static_cast<double>(tileCount - 1));
    return static_cast<std::uint32_t>(clamped);
}

double BoundsWidth(const Bounds2D& bounds)
{
    return bounds.maxX - bounds.minX;
}

double BoundsHeight(const Bounds2D& bounds)
{
    return bounds.maxY - bounds.minY;
}

std::uint32_t Low32(std::uint64_t value)
{
    return static_cast<std::uint32_t>(value & 0xffffffffu);
}

std::uint32_t High32(std::uint64_t value)
{
    return static_cast<std::uint32_t>(value >> 32u);
}

CachePoint BuildProxyPoint(const TileIndexEntry& tile, std::size_t attrCount)
{
    CachePoint point;
    point.tileId = tile.tileId;
    point.tileIndex = tile.metadataIndex;
    point.x = (tile.bounds.minX + tile.bounds.maxX) * 0.5;
    point.y = (tile.bounds.minY + tile.bounds.maxY) * 0.5;
    point.proxy = true;
    for (std::size_t attrIndex = 0; attrIndex < attrCount; ++attrIndex)
    {
        point.attrs[attrIndex] = tile.attrMeans[attrIndex];
    }
    return point;
}

template<typename T>
T ReadBinaryValue(const char* bytes)
{
    T value{};
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

bool ExtractJsonBoolOr(const std::string& text, const std::string& key, bool fallback)
{
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern))
    {
        return fallback;
    }

    return match[1].str() == "true";
}

void AppendFilteredTilePoints(
    const std::vector<CachePoint>& tilePoints,
    const Bounds2D& viewport,
    std::vector<CachePoint>& output)
{
    for (const CachePoint& point : tilePoints)
    {
        if (point.x < viewport.minX || point.x > viewport.maxX ||
            point.y < viewport.minY || point.y > viewport.maxY)
        {
            continue;
        }

        output.push_back(point);
    }
}

} // namespace

SeismicCacheReader::~SeismicCacheReader()
{
    DrainLod0IndexLoad(true);
}

void SeismicCacheReader::Open(const std::filesystem::path& cacheDirectory)
{
    const auto openBegin = std::chrono::steady_clock::now();

    DrainLod0IndexLoad(true);

    metadata_ = {};
    {
        std::lock_guard<std::mutex> lock(indexMutex_);
        tileIndex_.clear();
        tileIndexById_.clear();
        lod0IndexFuture_ = {};
        lod0IndexReady_ = false;
        lod0IndexLoading_ = false;
        lod0IndexLoadFailed_ = false;
        lod0IndexLoadError_.clear();
    }
    lodIndices_.clear();
    overviewPoints_.clear();
    {
        std::lock_guard<std::mutex> lock(tileCacheMutex_);
        tileCacheLru_.clear();
        tileCache_.clear();
        tileCacheStats_ = {};
        cachedTilePointCount_ = 0;
    }
    opened_ = false;
    lastOpenTiming_ = {};

    metadata_.cacheDirectory = cacheDirectory;
    const auto metadataBegin = std::chrono::steady_clock::now();
    LoadMetadata();
    lastOpenTiming_.metadataMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - metadataBegin).count();

    if (metadata_.attrFields.empty())
    {
        throw std::runtime_error("Cache metadata does not contain any attribute fields.");
    }
    if (metadata_.attrFields.size() > MaxCacheAttributes)
    {
        throw std::runtime_error("Too many cache attributes for the current C++ bridge.");
    }

    const auto colormapBegin = std::chrono::steady_clock::now();
    LoadColormap();
    lastOpenTiming_.colormapMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - colormapBegin).count();

    LoadTileIndices();

    const auto overviewBegin = std::chrono::steady_clock::now();
    LoadOverviewPoints();
    lastOpenTiming_.overviewMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - overviewBegin).count();

    opened_ = true;
    lastOpenTiming_.totalMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - openBegin).count();

    std::cout
        << "Cache path: " << metadata_.cacheDirectory.string() << '\n'
        << "Valid points: " << metadata_.validPoints << '\n'
        << "LOD0 tile index: deferred_until_first_global_frame\n"
        << "Record size: " << metadata_.recordSize << " bytes\n"
        << "LOD levels: " << metadata_.lodLevels.size() << '\n'
        << "Startup point LOD indices: " << lodIndices_.size() << '\n'
        << "Overview points: " << overviewPoints_.size() << '\n'
        << "Cache open timing: metadata_ms=" << lastOpenTiming_.metadataMilliseconds
        << ", colormap_ms=" << lastOpenTiming_.colormapMilliseconds
        << ", lod0_index_start_ms=" << lastOpenTiming_.lod0IndexStartMilliseconds
        << ", point_lod_index_ms=" << lastOpenTiming_.pointLodIndexMilliseconds
        << ", overview_ms=" << lastOpenTiming_.overviewMilliseconds
        << ", total_ms=" << lastOpenTiming_.totalMilliseconds
        << '\n';
}

CacheQueryResult SeismicCacheReader::QueryViewport(
    const Bounds2D& viewport,
    std::size_t activeAttrIndex,
    const CacheQueryOptions& options) const
{
    ThrowIfQueryCancelled(options);
    if (!opened_)
    {
        throw std::runtime_error("Seismic cache is not opened.");
    }
    if (activeAttrIndex >= metadata_.attrFields.size())
    {
        throw std::runtime_error("Active attribute index is out of range.");
    }

    const Bounds2D& world = metadata_.xyRange;
    const double x0 = std::max(viewport.minX, world.minX);
    const double x1 = std::min(viewport.maxX, world.maxX);
    const double y0 = std::max(viewport.minY, world.minY);
    const double y1 = std::min(viewport.maxY, world.maxY);

    CacheQueryResult result{};
    result.lodPointsPerPixel = EffectiveLodPointsPerPixel(options);
    result.lodPointBudget = ComputeViewportPointBudget(options, MaxLod0PointLoad);
    if (x0 > x1 || y0 > y1)
    {
        return result;
    }
    const Bounds2D clippedViewport{ x0, y0, x1, y1 };
    if (options.forceExact)
    {
        StartLod0IndexLoadAsync();
    }
    DrainLod0IndexLoad(options.forceExact);
    ThrowIfQueryCancelled(options);
    {
        std::lock_guard<std::mutex> lock(indexMutex_);
        result.lod0IndexReady = lod0IndexReady_;
        result.lod0IndexPreparing = !lod0IndexReady_ && !lod0IndexLoadFailed_;
    }

    if (!options.forceExact)
    {
        if (const LodIndexData* selectedLod = SelectViewportLod(clippedViewport, options))
        {
            CacheQueryResult lodResult = QueryLod(
                *selectedLod,
                clippedViewport,
                activeAttrIndex,
                options);
            lodResult.lod0IndexReady = result.lod0IndexReady;
            lodResult.lod0IndexPreparing = result.lod0IndexPreparing;
            return lodResult;
        }
    }

    if (!result.lod0IndexReady)
    {
        result.activeLodLevel = -3;
        result.activeLodName = "overview_only";
        result.screenSpaceLod = true;
        return result;
    }

    const std::uint32_t row0 = ClampTileIndex(y0, world.minY, BoundsHeight(world), metadata_.tileRows);
    const std::uint32_t row1 = ClampTileIndex(y1, world.minY, BoundsHeight(world), metadata_.tileRows);
    const std::uint32_t col0 = ClampTileIndex(x0, world.minX, BoundsWidth(world), metadata_.tileCols);
    const std::uint32_t col1 = ClampTileIndex(x1, world.minX, BoundsWidth(world), metadata_.tileCols);

    std::vector<const TileIndexEntry*> hitTiles;
    const std::uint64_t candidateTileCount =
        static_cast<std::uint64_t>(row1 - row0 + 1) *
        static_cast<std::uint64_t>(col1 - col0 + 1);
    hitTiles.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(candidateTileCount, 1000000)));

    {
        std::lock_guard<std::mutex> lock(indexMutex_);
        for (std::uint32_t row = row0; row <= row1; ++row)
        {
            ThrowIfQueryCancelled(options);
            for (std::uint32_t col = col0; col <= col1; ++col)
            {
                const std::int64_t tileId =
                    static_cast<std::int64_t>(row) * metadata_.tileCols + static_cast<std::int64_t>(col);
                const auto it = tileIndexById_.find(tileId);
                if (it == tileIndexById_.end())
                {
                    continue;
                }

                const TileIndexEntry& entry = tileIndex_[it->second];
                if (entry.lengthPoints == 0)
                {
                    continue;
                }

                hitTiles.push_back(&entry);
                result.estimatedPoints += entry.lengthPoints;
            }
        }
    }

    result.hitTileCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(hitTiles.size(), std::numeric_limits<std::uint32_t>::max()));

    result.screenSpaceLod =
        !options.forceExact &&
        options.framebufferWidth > 0 &&
        options.framebufferHeight > 0;

    const std::uint64_t pointBudget = ComputeViewportPointBudget(
        options,
        MaxLod0PointLoad);
    if (options.forceExact ||
        (result.estimatedPoints <= pointBudget && hitTiles.size() <= MaxLod0TileLoad))
    {
        result.activeLodLevel = 0;
        result.activeLodName = "lod0_exact";
        result.points = ReadTilePoints(hitTiles, clippedViewport, options);
        result.readPoints = result.estimatedPoints;
        result.estimatedPointsPerPixel = ComputeEstimatedPointsPerPixel(
            options,
            result.estimatedPoints);
        result.lod0PointCount = result.points.size();
        result.lod0TileCount = result.hitTileCount;
        return result;
    }

    result.activeLodLevel = -3;
    result.activeLodName = "overview_only";
    result.screenSpaceLod = true;
    return result;
}

std::vector<CachePoint> SeismicCacheReader::QueryExactPoints(const Bounds2D& viewport) const
{
    CacheQueryOptions options;
    options.forceExact = true;
    options.progressiveLoading = false;

    return QueryViewport(viewport, 0, options).points;
}

std::optional<CachePoint> SeismicCacheReader::ReadLod0PointBySourceId(
    std::uint64_t sourceId) const
{
    if (!opened_ || sourceId >= metadata_.validPoints)
    {
        return std::nullopt;
    }

    const std::uint64_t byteOffset = sourceId * metadata_.recordSize;
    const std::filesystem::path path = ResolveTilesPath();
    if (!std::filesystem::exists(path) ||
        byteOffset + metadata_.recordSize > std::filesystem::file_size(path))
    {
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        return std::nullopt;
    }

    CachePoint point = ReadPointRecordAt(file, byteOffset, 0, 0);
    point.sourceId = sourceId;
    point.hasSourceId = true;
    return point;
}

CachePoint SeismicCacheReader::ReadPointRecordAt(
    std::ifstream& file,
    std::uint64_t byteOffset,
    std::int64_t tileId,
    std::uint32_t tileIndex) const
{
    std::vector<char> record(metadata_.recordSize);
    file.seekg(static_cast<std::streamoff>(byteOffset), std::ios::beg);
    file.read(record.data(), static_cast<std::streamsize>(record.size()));
    if (file.gcount() != static_cast<std::streamsize>(record.size()))
    {
        throw std::runtime_error("Failed to read an LOD0 point record.");
    }

    CachePoint point;
    point.tileId = tileId;
    point.tileIndex = tileIndex;
    point.x = ReadCoordinateValue(record.data(), FindRecordField("x"));
    point.y = ReadCoordinateValue(record.data(), FindRecordField("y"));
    for (std::size_t attrIndex = 0; attrIndex < metadata_.attrFields.size(); ++attrIndex)
    {
        point.attrs[attrIndex] = ReadAttributeValue(
            record.data(),
            FindRecordField(metadata_.attrFields[attrIndex]));
    }

    const std::uint32_t row = ClampTileIndex(
        point.y,
        metadata_.xyRange.minY,
        BoundsHeight(metadata_.xyRange),
        metadata_.tileRows);
    const std::uint32_t col = ClampTileIndex(
        point.x,
        metadata_.xyRange.minX,
        BoundsWidth(metadata_.xyRange),
        metadata_.tileCols);
    point.tileId = static_cast<std::int64_t>(row) * metadata_.tileCols + col;
    return point;
}

const SeismicCacheReader::LodIndexData* SeismicCacheReader::SelectViewportLod(
    const Bounds2D& viewport,
    const CacheQueryOptions& options) const
{
    if (options.forceExact || lodIndices_.empty())
    {
        return nullptr;
    }

    const auto isLocalLod = [](const LodIndexData& lod)
    {
        return lod.metadata.name != "bridge_mean";
    };

    if (options.preferGlobalMean)
    {
        const auto globalMean = std::find_if(
            lodIndices_.begin(),
            lodIndices_.end(),
            [](const LodIndexData& lod)
            {
                return lod.metadata.name == "bridge_mean" ||
                    lod.metadata.kind == "overview_bridge_mean";
            });
        if (globalMean != lodIndices_.end())
        {
            return &*globalMean;
        }
    }

    if (!options.preferredLodName.empty())
    {
        const auto preferred = std::find_if(
            lodIndices_.begin(),
            lodIndices_.end(),
            [&options, &isLocalLod](const LodIndexData& lod)
            {
                return
                    isLocalLod(lod) &&
                    lod.metadata.name == options.preferredLodName;
            });
        if (preferred != lodIndices_.end())
        {
            return &*preferred;
        }
    }

    const std::uint64_t pointBudget = ComputeViewportPointBudget(
        options,
        MaxLod0PointLoad);

    bool lod0Ready = false;
    std::uint64_t lod0Estimate = 0;
    std::uint64_t lod0TileCount = 0;
    {
        std::lock_guard<std::mutex> lock(indexMutex_);
        lod0Ready = lod0IndexReady_;
        if (lod0Ready)
        {
            const std::uint32_t row0 = ClampTileIndex(
                viewport.minY,
                metadata_.xyRange.minY,
                BoundsHeight(metadata_.xyRange),
                metadata_.tileRows);
            const std::uint32_t row1 = ClampTileIndex(
                viewport.maxY,
                metadata_.xyRange.minY,
                BoundsHeight(metadata_.xyRange),
                metadata_.tileRows);
            const std::uint32_t col0 = ClampTileIndex(
                viewport.minX,
                metadata_.xyRange.minX,
                BoundsWidth(metadata_.xyRange),
                metadata_.tileCols);
            const std::uint32_t col1 = ClampTileIndex(
                viewport.maxX,
                metadata_.xyRange.minX,
                BoundsWidth(metadata_.xyRange),
                metadata_.tileCols);
            for (std::uint32_t row = row0; row <= row1; ++row)
            {
                ThrowIfQueryCancelled(options);
                for (std::uint32_t col = col0; col <= col1; ++col)
                {
                    const std::int64_t tileId =
                        static_cast<std::int64_t>(row) * metadata_.tileCols + col;
                    const auto it = tileIndexById_.find(tileId);
                    if (it != tileIndexById_.end())
                    {
                        lod0Estimate += tileIndex_[it->second].lengthPoints;
                        ++lod0TileCount;
                    }
                }
            }
        }
    }

    const auto lod0Fits = [&](double budgetRatio)
    {
        return
            lod0Ready &&
            lod0TileCount <= MaxLod0TileLoad &&
            static_cast<double>(lod0Estimate) <=
                static_cast<double>(pointBudget) * budgetRatio;
    };
    const auto lodFits = [&](const LodIndexData& lod, double budgetRatio)
    {
        return
            static_cast<double>(EstimateVisiblePoints(lod, viewport, options)) <=
            static_cast<double>(pointBudget) * budgetRatio;
    };

    const bool currentIsLod0 =
        options.currentLodName == "lod0" ||
        options.currentLodName == "lod0_exact";
    const auto currentLod = std::find_if(
        lodIndices_.begin(),
        lodIndices_.end(),
        [&options, &isLocalLod](const LodIndexData& lod)
        {
            return
                isLocalLod(lod) &&
                lod.metadata.name == options.currentLodName;
        });

    if (currentIsLod0)
    {
        if (lod0Fits(kLodDowngradeBudgetRatio))
        {
            return nullptr;
        }

        for (const LodIndexData& lod : lodIndices_)
        {
            ThrowIfQueryCancelled(options);
            if (isLocalLod(lod) && lodFits(lod, 1.0))
            {
                return &lod;
            }
        }
    }
    else if (currentLod != lodIndices_.end())
    {
        if (lod0Fits(kLodUpgradeBudgetRatio))
        {
            return nullptr;
        }

        for (auto finerLod = lodIndices_.begin(); finerLod != currentLod; ++finerLod)
        {
            ThrowIfQueryCancelled(options);
            if (isLocalLod(*finerLod) && lodFits(*finerLod, kLodUpgradeBudgetRatio))
            {
                return &*finerLod;
            }
        }

        if (lodFits(*currentLod, kLodDowngradeBudgetRatio))
        {
            return &*currentLod;
        }

        for (auto coarserLod = std::next(currentLod); coarserLod != lodIndices_.end(); ++coarserLod)
        {
            ThrowIfQueryCancelled(options);
            if (isLocalLod(*coarserLod) && lodFits(*coarserLod, 1.0))
            {
                return &*coarserLod;
            }
        }
    }
    else
    {
        if (lod0Fits(1.0))
        {
            return nullptr;
        }

        for (const LodIndexData& lod : lodIndices_)
        {
            ThrowIfQueryCancelled(options);
            if (isLocalLod(lod) && lodFits(lod, 1.0))
            {
                return &lod;
            }
        }
    }

    const auto coarsestLocal = std::find_if(
        lodIndices_.rbegin(),
        lodIndices_.rend(),
        isLocalLod);
    if (coarsestLocal != lodIndices_.rend())
    {
        return &*coarsestLocal;
    }

    return &lodIndices_.back();
}

std::uint64_t SeismicCacheReader::EstimateVisiblePoints(
    const LodIndexData& lod,
    const Bounds2D& viewport,
    const CacheQueryOptions& options) const
{
    const Bounds2D& world = metadata_.xyRange;
    const std::uint32_t row0 = ClampTileIndex(
        viewport.minY,
        world.minY,
        BoundsHeight(world),
        lod.metadata.tileRows);
    const std::uint32_t row1 = ClampTileIndex(
        viewport.maxY,
        world.minY,
        BoundsHeight(world),
        lod.metadata.tileRows);
    const std::uint32_t col0 = ClampTileIndex(
        viewport.minX,
        world.minX,
        BoundsWidth(world),
        lod.metadata.tileCols);
    const std::uint32_t col1 = ClampTileIndex(
        viewport.maxX,
        world.minX,
        BoundsWidth(world),
        lod.metadata.tileCols);

    std::uint64_t estimate = 0;
    for (std::uint32_t row = row0; row <= row1; ++row)
    {
        ThrowIfQueryCancelled(options);
        for (std::uint32_t col = col0; col <= col1; ++col)
        {
            const std::int64_t tileId =
                static_cast<std::int64_t>(row) * lod.metadata.tileCols + col;
            const auto it = lod.tileIndexById.find(tileId);
            if (it != lod.tileIndexById.end())
            {
                estimate += lod.tiles[it->second].lengthPoints;
            }
        }
    }
    return estimate;
}

CacheQueryResult SeismicCacheReader::QueryLod(
    const LodIndexData& lod,
    const Bounds2D& viewport,
    std::size_t activeAttrIndex,
    const CacheQueryOptions& options) const
{
    (void)activeAttrIndex;

    const Bounds2D& world = metadata_.xyRange;
    const std::uint32_t row0 = ClampTileIndex(
        viewport.minY,
        world.minY,
        BoundsHeight(world),
        lod.metadata.tileRows);
    const std::uint32_t row1 = ClampTileIndex(
        viewport.maxY,
        world.minY,
        BoundsHeight(world),
        lod.metadata.tileRows);
    const std::uint32_t col0 = ClampTileIndex(
        viewport.minX,
        world.minX,
        BoundsWidth(world),
        lod.metadata.tileCols);
    const std::uint32_t col1 = ClampTileIndex(
        viewport.maxX,
        world.minX,
        BoundsWidth(world),
        lod.metadata.tileCols);

    std::vector<const TileIndexEntry*> hitTiles;
    const std::uint64_t candidateTileCount =
        static_cast<std::uint64_t>(row1 - row0 + 1) *
        static_cast<std::uint64_t>(col1 - col0 + 1);
    hitTiles.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(candidateTileCount, 1000000)));

    CacheQueryResult result{};
    result.lodPointsPerPixel = EffectiveLodPointsPerPixel(options);
    result.lodPointBudget = ComputeViewportPointBudget(options, MaxLod0PointLoad);
    result.activeLodLevel = lod.metadata.level;
    result.activeLodName = lod.metadata.name;
    result.screenSpaceLod = true;

    for (std::uint32_t row = row0; row <= row1; ++row)
    {
        ThrowIfQueryCancelled(options);
        for (std::uint32_t col = col0; col <= col1; ++col)
        {
            const std::int64_t tileId =
                static_cast<std::int64_t>(row) * lod.metadata.tileCols + static_cast<std::int64_t>(col);
            const auto it = lod.tileIndexById.find(tileId);
            if (it == lod.tileIndexById.end())
            {
                continue;
            }

            const TileIndexEntry& entry = lod.tiles[it->second];
            if (entry.lengthPoints == 0)
            {
                continue;
            }

            hitTiles.push_back(&entry);
            result.estimatedPoints += entry.lengthPoints;
            result.readPoints += entry.lengthPoints;
        }
    }

    result.hitTileCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(hitTiles.size(), std::numeric_limits<std::uint32_t>::max()));
    result.points = ReadLodTilePoints(lod, hitTiles, viewport, options);
    result.estimatedPointsPerPixel = ComputeEstimatedPointsPerPixel(
        options,
        result.estimatedPoints);
    return result;
}

std::vector<GpuPoint> SeismicCacheReader::BuildGpuPoints(
    const std::vector<CachePoint>& points,
    std::size_t activeAttrIndex,
    double renderOriginX,
    double renderOriginY,
    const std::function<bool()>& shouldCancel) const
{
    std::vector<GpuPoint> gpuPoints;
    gpuPoints.reserve(points.size());
    const std::size_t attr0Index = 0;
    const std::size_t attr1Index = std::min<std::size_t>(1, metadata_.attrFields.size() - 1);

    for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
    {
        if ((pointIndex & 4095u) == 0u && shouldCancel && shouldCancel())
        {
            throw CacheQueryCancelled();
        }
        const CachePoint& point = points[pointIndex];
        const auto& color0 = ColorForValue(point.attrs[attr0Index], attr0Index);
        const auto& color1 = ColorForValue(point.attrs[attr1Index], attr1Index);
        gpuPoints.push_back(GpuPoint
        {
            static_cast<float>(point.x - renderOriginX),
            static_cast<float>(point.y - renderOriginY),
            PackRgba8(color0),
            PackRgba8(color1),
        });
    }

    return gpuPoints;
}

std::vector<GpuTilePage> SeismicCacheReader::BuildGpuTilePages(
    const std::vector<CachePoint>& points,
    const std::string& lodName,
    int lodLevel,
    std::size_t activeAttrIndex,
    const std::function<bool()>& shouldCancel) const
{
    std::vector<GpuTilePage> pages;
    if (points.empty())
    {
        return pages;
    }
    if (activeAttrIndex >= metadata_.attrFields.size())
    {
        throw std::runtime_error("Active attribute index is out of range.");
    }

    const double worldWidth = BoundsWidth(metadata_.xyRange);
    const double worldHeight = BoundsHeight(metadata_.xyRange);
    if (worldWidth <= 0.0 || worldHeight <= 0.0)
    {
        throw std::runtime_error("Cache XY range is invalid for GPU page construction.");
    }

    const std::uint64_t lodHash = HashText(lodName);
    const std::size_t attr0Index = 0;
    const std::size_t attr1Index = std::min<std::size_t>(1, metadata_.attrFields.size() - 1);
    const double cellWidth = worldWidth / static_cast<double>(kGpuPageGridSize);
    const double cellHeight = worldHeight / static_cast<double>(kGpuPageGridSize);
    std::array<std::vector<GpuTilePage>, kGpuPageGridCellCount> cellPages;

    for (std::size_t pointIndex = 0; pointIndex < points.size(); ++pointIndex)
    {
        if ((pointIndex & 4095u) == 0u && shouldCancel && shouldCancel())
        {
            throw CacheQueryCancelled();
        }

        const CachePoint& point = points[pointIndex];
        const double normalizedX = std::clamp(
            (point.x - metadata_.xyRange.minX) / worldWidth,
            0.0,
            1.0);
        const double normalizedY = std::clamp(
            (point.y - metadata_.xyRange.minY) / worldHeight,
            0.0,
            1.0);
        const std::size_t column = std::min(
            static_cast<std::size_t>(normalizedX * kGpuPageGridSize),
            kGpuPageGridSize - 1);
        const std::size_t row = std::min(
            static_cast<std::size_t>(normalizedY * kGpuPageGridSize),
            kGpuPageGridSize - 1);
        const std::size_t cellIndex = row * kGpuPageGridSize + column;
        std::vector<GpuTilePage>& chunks = cellPages[cellIndex];

        if (chunks.empty() || chunks.back().points.size() >= kGpuPageMaxPoints)
        {
            const std::uint32_t chunkIndex = static_cast<std::uint32_t>(chunks.size());
            GpuTilePage page;
            page.signature = kFnvOffsetBasis;
            page.key = lodHash;
            page.lodLevel = lodLevel;
            MixHash(page.key, cellIndex);
            MixHash(page.key, chunkIndex);
            page.originX = metadata_.xyRange.minX + static_cast<double>(column) * cellWidth;
            page.originY = metadata_.xyRange.minY + static_cast<double>(row) * cellHeight;
            page.bounds.minX = point.x;
            page.bounds.minY = point.y;
            page.bounds.maxX = point.x;
            page.bounds.maxY = point.y;
            std::uint64_t originXBits = 0;
            std::uint64_t originYBits = 0;
            std::memcpy(&originXBits, &page.originX, sizeof(originXBits));
            std::memcpy(&originYBits, &page.originY, sizeof(originYBits));
            MixHash(page.signature, originXBits);
            MixHash(page.signature, originYBits);
            page.points.reserve(kGpuPageMaxPoints);
            chunks.push_back(std::move(page));
        }

        GpuTilePage& page = chunks.back();
        page.bounds.minX = std::min(page.bounds.minX, point.x);
        page.bounds.minY = std::min(page.bounds.minY, point.y);
        page.bounds.maxX = std::max(page.bounds.maxX, point.x);
        page.bounds.maxY = std::max(page.bounds.maxY, point.y);
        const auto& color0 = ColorForValue(point.attrs[attr0Index], attr0Index);
        const auto& color1 = ColorForValue(point.attrs[attr1Index], attr1Index);
        const GpuPoint gpuPoint
        {
            static_cast<float>(point.x - page.originX),
            static_cast<float>(point.y - page.originY),
            PackRgba8(color0),
            PackRgba8(color1),
        };
        page.points.push_back(gpuPoint);
        std::uint32_t xBits = 0;
        std::uint32_t yBits = 0;
        std::memcpy(&xBits, &gpuPoint.x, sizeof(xBits));
        std::memcpy(&yBits, &gpuPoint.y, sizeof(yBits));
        MixHash(page.signature, static_cast<std::uint64_t>(point.tileId));
        MixHash(
            page.signature,
            static_cast<std::uint64_t>(xBits) |
            (static_cast<std::uint64_t>(yBits) << 32));
        MixHash(page.signature, gpuPoint.rgba0);
        MixHash(page.signature, gpuPoint.rgba1);
    }

    std::size_t pageCount = 0;
    for (const auto& chunks : cellPages)
    {
        pageCount += chunks.size();
    }
    pages.reserve(pageCount);
    for (auto& chunks : cellPages)
    {
        for (GpuTilePage& page : chunks)
        {
            MixHash(page.signature, page.points.size());
            pages.push_back(std::move(page));
        }
    }
    return pages;
}

std::vector<GpuPoint> SeismicCacheReader::BuildOverviewGpuPoints(
    const Bounds2D& viewport,
    std::size_t activeAttrIndex,
    float alpha,
    double renderOriginX,
    double renderOriginY,
    const std::function<bool()>& shouldCancel) const
{
    std::vector<GpuPoint> gpuPoints;
    if (alpha <= 0.0f || overviewPoints_.empty())
    {
        return gpuPoints;
    }
    if (activeAttrIndex >= metadata_.attrFields.size())
    {
        throw std::runtime_error("Active attribute index is out of range.");
    }

    gpuPoints.reserve(overviewPoints_.size());
    const std::size_t attr0Index = 0;
    const std::size_t attr1Index = std::min<std::size_t>(1, metadata_.attrFields.size() - 1);
    for (std::size_t pointIndex = 0; pointIndex < overviewPoints_.size(); ++pointIndex)
    {
        if ((pointIndex & 4095u) == 0u && shouldCancel && shouldCancel())
        {
            throw CacheQueryCancelled();
        }
        const OverviewPoint& point = overviewPoints_[pointIndex];
        if (point.x < viewport.minX || point.x > viewport.maxX ||
            point.y < viewport.minY || point.y > viewport.maxY)
        {
            continue;
        }

        const auto& color0 = ColorForValue(point.attrs[attr0Index], attr0Index);
        const auto& color1 = ColorForValue(point.attrs[attr1Index], attr1Index);
        gpuPoints.push_back(GpuPoint
        {
            static_cast<float>(static_cast<double>(point.x) - renderOriginX),
            static_cast<float>(static_cast<double>(point.y) - renderOriginY),
            PackRgba8(color0, alpha),
            PackRgba8(color1, alpha),
        });
    }

    return gpuPoints;
}

float SeismicCacheReader::ComputeOverviewAlpha(const Bounds2D& viewport) const
{
    if (overviewPoints_.empty())
    {
        return 0.0f;
    }

    const double worldWidth = BoundsWidth(metadata_.xyRange);
    const double worldHeight = BoundsHeight(metadata_.xyRange);
    const double viewportWidth = BoundsWidth(viewport);
    const double viewportHeight = BoundsHeight(viewport);
    if (worldWidth <= 0.0 || worldHeight <= 0.0 ||
        viewportWidth <= 0.0 || viewportHeight <= 0.0)
    {
        return 0.0f;
    }

    const double visibleRatio = std::max(viewportWidth / worldWidth, viewportHeight / worldHeight);
    if (visibleRatio <= kOverviewZeroAlphaRatio)
    {
        return 0.0f;
    }
    if (visibleRatio >= kOverviewFullAlphaRatio)
    {
        return kOverviewMaxAlpha;
    }

    const double fade =
        (visibleRatio - kOverviewZeroAlphaRatio) /
        (kOverviewFullAlphaRatio - kOverviewZeroAlphaRatio);
    return static_cast<float>(fade) * kOverviewMaxAlpha;
}

std::vector<GpuTileMetadata> SeismicCacheReader::BuildGpuTileMetadata(
    std::size_t activeAttrIndex) const
{
    DrainLod0IndexLoad(false);

    if (activeAttrIndex >= metadata_.attrFields.size())
    {
        throw std::runtime_error("Active attribute index is out of range.");
    }

    std::vector<GpuTileMetadata> gpuTiles;
    std::lock_guard<std::mutex> lock(indexMutex_);
    if (!lod0IndexReady_)
    {
        std::cout << "GPU tile metadata skipped: LOD0 index is still loading.\n";
        return gpuTiles;
    }

    gpuTiles.reserve(tileIndex_.size());

    for (const TileIndexEntry& tile : tileIndex_)
    {
        const std::uint64_t tileId = static_cast<std::uint64_t>(tile.tileId);
        const std::uint64_t clampedLength = std::min<std::uint64_t>(
            tile.lengthPoints,
            std::numeric_limits<std::uint32_t>::max());
        const float attrMean = tile.attrMeans[activeAttrIndex];
        const auto& color = ColorForValue(attrMean, activeAttrIndex);

        GpuTileMetadata gpuTile;
        gpuTile.minX = static_cast<float>(tile.bounds.minX);
        gpuTile.minY = static_cast<float>(tile.bounds.minY);
        gpuTile.maxX = static_cast<float>(tile.bounds.maxX);
        gpuTile.maxY = static_cast<float>(tile.bounds.maxY);
        gpuTile.centerX = static_cast<float>((tile.bounds.minX + tile.bounds.maxX) * 0.5);
        gpuTile.centerY = static_cast<float>((tile.bounds.minY + tile.bounds.maxY) * 0.5);
        gpuTile.attrMean = attrMean;
        gpuTile.pointCountFloat = static_cast<float>(tile.lengthPoints);
        gpuTile.r = color[0];
        gpuTile.g = color[1];
        gpuTile.b = color[2];
        gpuTile.a = color[3];
        gpuTile.tileIdLow = Low32(tileId);
        gpuTile.tileIdHigh = High32(tileId);
        gpuTile.offsetBytesLow = Low32(tile.offsetBytes);
        gpuTile.offsetBytesHigh = High32(tile.offsetBytes);
        gpuTile.lengthPoints = static_cast<std::uint32_t>(clampedLength);
        gpuTile.row = tile.row;
        gpuTile.col = tile.col;
        gpuTile.flags = tile.lengthPoints > 0 ? 1u : 0u;
        gpuTiles.push_back(gpuTile);
    }

    return gpuTiles;
}

std::vector<std::uint32_t> SeismicCacheReader::BuildGpuProxyMask(
    const std::vector<CachePoint>& points) const
{
    DrainLod0IndexLoad(false);

    std::lock_guard<std::mutex> lock(indexMutex_);
    std::vector<std::uint32_t> proxyMask(tileIndex_.size(), 0);

    for (const CachePoint& point : points)
    {
        if (!point.proxy)
        {
            continue;
        }

        if (point.tileIndex >= proxyMask.size())
        {
            continue;
        }

        proxyMask[point.tileIndex] = 1;
    }

    return proxyMask;
}

const CacheMetadata& SeismicCacheReader::Metadata() const
{
    return metadata_;
}

std::size_t SeismicCacheReader::OverviewPointCount() const
{
    return overviewPoints_.size();
}

const SeismicCacheReader::OpenTimingStats& SeismicCacheReader::LastOpenTiming() const
{
    return lastOpenTiming_;
}

const std::array<float, 4>& SeismicCacheReader::ColorForValue(
    float value,
    std::size_t activeAttrIndex) const
{
    return colormap_[ColorIndexForValue(value, activeAttrIndex)];
}

std::array<std::uint32_t, 256> SeismicCacheReader::PackedColormap() const
{
    std::array<std::uint32_t, 256> packed{};
    for (std::size_t index = 0; index < colormap_.size(); ++index)
    {
        const auto channel = [](float value)
        {
            return static_cast<std::uint32_t>(std::lround(
                std::clamp(value, 0.0f, 1.0f) * 255.0f));
        };
        packed[index] =
            channel(colormap_[index][0]) |
            (channel(colormap_[index][1]) << 8) |
            (channel(colormap_[index][2]) << 16) |
            (channel(colormap_[index][3]) << 24);
    }
    return packed;
}

TileCacheStats SeismicCacheReader::GetTileCacheStats() const
{
    std::lock_guard<std::mutex> lock(tileCacheMutex_);
    TileCacheStats stats = tileCacheStats_;
    stats.cachedTileCount = tileCache_.size();
    stats.cachedPointCount = cachedTilePointCount_;
    return stats;
}

std::filesystem::path SeismicCacheReader::ResolveTilesPath() const
{
    return metadata_.cacheDirectory / metadata_.tilesPath;
}

void SeismicCacheReader::WaitForLod0Index() const
{
    StartLod0IndexLoadAsync();
    DrainLod0IndexLoad(true);
}

Lod0RegionSummary SeismicCacheReader::SummarizeLod0Region(const Bounds2D& bounds) const
{
    WaitForLod0Index();

    Lod0RegionSummary summary;
    std::lock_guard<std::mutex> lock(indexMutex_);
    for (const TileIndexEntry& tile : tileIndex_)
    {
        if (tile.bounds.maxX < bounds.minX || tile.bounds.minX > bounds.maxX ||
            tile.bounds.maxY < bounds.minY || tile.bounds.minY > bounds.maxY)
        {
            continue;
        }

        ++summary.tileCount;
        summary.pointCount += tile.lengthPoints;
    }
    return summary;
}

std::vector<TileIndexEntry> SeismicCacheReader::CopyLod0TilesIntersecting(
    const Bounds2D& bounds) const
{
    WaitForLod0Index();

    std::vector<TileIndexEntry> matches;
    std::lock_guard<std::mutex> lock(indexMutex_);
    for (const TileIndexEntry& tile : tileIndex_)
    {
        if (tile.bounds.maxX < bounds.minX || tile.bounds.minX > bounds.maxX ||
            tile.bounds.maxY < bounds.minY || tile.bounds.minY > bounds.maxY)
        {
            continue;
        }
        matches.push_back(tile);
    }
    return matches;
}

bool SeismicCacheReader::IsLod0IndexReady() const
{
    DrainLod0IndexLoad(false);

    std::lock_guard<std::mutex> lock(indexMutex_);
    return lod0IndexReady_;
}

std::filesystem::path SeismicCacheReader::ResolveDefaultCachePath()
{
    throw std::runtime_error("Provide --cache <local-cache-directory>. No dataset is bundled.");

}

bool SeismicCacheReader::RunContractCheck(const std::filesystem::path& cacheDirectory)
{
    SeismicCacheReader reader;
    reader.Open(cacheDirectory);
    if (reader.metadata_.gpuReadyPageContractVersion > 0 && reader.lodIndices_.empty())
    {
        reader.metadata_.gpuReadyPageContractVersion = 0;
        reader.LoadTileIndices();
    }
    reader.WaitForLod0Index();

    CacheQueryOptions options;
    options.framebufferWidth = 1;
    options.framebufferHeight = 1;
    options.preferGlobalMean = true;
    CacheQueryResult result = reader.QueryViewport(reader.Metadata().xyRange, 0, options);
    if (result.points.empty())
    {
        std::cout << "Cache check failed: viewport query returned no points.\n";
        return false;
    }
    if (result.activeLodName != "bridge_mean")
    {
        std::cout
            << "Cache check failed: constrained viewport budget selected "
            << result.activeLodName
            << " instead of bridge_mean.\n";
        return false;
    }

    const CachePoint& displayPoint = result.points.front();
    if (!displayPoint.hasSourceId)
    {
        std::cout << "Cache check failed: selected LOD point has no source_id.\n";
        return false;
    }

    const std::optional<CachePoint> original =
        reader.ReadLod0PointBySourceId(displayPoint.sourceId);
    if (!original.has_value())
    {
        std::cout << "Cache check failed: source_id could not be resolved to LOD0.\n";
        return false;
    }
    const double coordinateTolerance = std::max(
        1.0e-9,
        std::max(
            reader.Metadata().xyRange.maxX - reader.Metadata().xyRange.minX,
            reader.Metadata().xyRange.maxY - reader.Metadata().xyRange.minY) * 1.0e-12);
    if (std::abs(displayPoint.x - original->x) > coordinateTolerance ||
        std::abs(displayPoint.y - original->y) > coordinateTolerance)
    {
        std::cout << "Cache check failed: source_id coordinates do not match LOD0.\n";
        return false;
    }
    for (std::size_t attrIndex = 0; attrIndex < reader.Metadata().attrFields.size(); ++attrIndex)
    {
        if (!std::isfinite(original->attrs[attrIndex]))
        {
            std::cout << "Cache check failed: source_id attributes are invalid.\n";
            return false;
        }
    }

    std::cout << "Cache check passed.\n";
    return true;
}

void SeismicCacheReader::LoadMetadata()
{
    const std::string text = ReadTextFile(metadata_.cacheDirectory / "metadata.json");

    const std::string xyRangeBody = FindJsonBlock(text, "xy_range", '{', '}');
    metadata_.xyRange.minX = ExtractJsonDouble(xyRangeBody, "min_x");
    metadata_.xyRange.minY = ExtractJsonDouble(xyRangeBody, "min_y");
    metadata_.xyRange.maxX = ExtractJsonDouble(xyRangeBody, "max_x");
    metadata_.xyRange.maxY = ExtractJsonDouble(xyRangeBody, "max_y");

    metadata_.validPoints = ExtractJsonUInt64(text, "valid_points");
    metadata_.tileRows = static_cast<std::uint32_t>(ExtractJsonUInt64(text, "tile_rows"));
    metadata_.tileCols = static_cast<std::uint32_t>(ExtractJsonUInt64(text, "tile_cols"));
    metadata_.tilesPath = ExtractJsonString(text, "tiles_lod0_path");
    metadata_.tileIndexBinaryPath = ExtractJsonString(text, "tile_index_binary_path");
    metadata_.colormapJsonPath = ExtractJsonString(text, "colormap_json_path");
    metadata_.overviewPointsPath = ExtractJsonStringOr(text, "overview_points_path", "overview_points.bin");
    metadata_.lodContractVersion = static_cast<std::uint32_t>(
        ExtractJsonUInt64Or(text, "lod_contract_version", 0));
    metadata_.gpuReadyPageContractVersion = static_cast<std::uint32_t>(
        ExtractJsonUInt64Or(text, "gpu_ready_page_contract_version", 0));

    if (metadata_.lodContractVersion < 6)
    {
        throw std::runtime_error("Unsupported cache LOD contract. Version 6 or newer is required.");
    }

    const std::string attrFieldsBody = FindJsonBlock(text, "attr_fields", '[', ']');
    metadata_.attrFields = ParseJsonStringArray(attrFieldsBody);

    const std::string mainLodLevelsBody = FindJsonBlock(text, "main_lod_levels", '[', ']');
    metadata_.mainLodLevels = ParseJsonStringArray(mainLodLevelsBody);

    const std::string attrRangesBody = FindJsonBlock(text, "attr_ranges", '{', '}');
    for (const std::string& attr : metadata_.attrFields)
    {
        const std::string attrBody = FindJsonBlock(attrRangesBody, attr, '{', '}');
        metadata_.attrRanges[attr] =
        {
            ExtractJsonDouble(attrBody, "min"),
            ExtractJsonDouble(attrBody, "max"),
        };
    }

    const std::string recordLayoutBody = FindJsonBlock(text, "record_layout", '{', '}');
    metadata_.recordSize = static_cast<std::size_t>(ExtractJsonUInt64(recordLayoutBody, "record_size"));

    const std::string recordFieldsBody = FindJsonBlock(recordLayoutBody, "fields", '[', ']');
    for (const std::string& objectBody : SplitJsonObjects(recordFieldsBody))
    {
        RecordField field;
        field.name = ExtractJsonString(objectBody, "name");
        field.dtype = ExtractJsonString(objectBody, "dtype");
        field.offset = static_cast<std::size_t>(ExtractJsonUInt64(objectBody, "offset"));
        field.bytes = static_cast<std::size_t>(ExtractJsonUInt64(objectBody, "bytes"));
        metadata_.recordFields.push_back(field);
    }

    std::string lodLevelsBody;
    if (TryFindJsonBlock(text, "lod_levels", '[', ']', lodLevelsBody))
    {
        for (const std::string& objectBody : SplitJsonObjects(lodLevelsBody))
        {
            CacheLodLevel lodLevel;
            lodLevel.level = static_cast<int>(ExtractJsonUInt64Or(
                objectBody,
                "level",
                ExtractJsonUInt64Or(objectBody, "lod_level", 0)));
            lodLevel.aggregateFactor = static_cast<int>(ExtractJsonUInt64Or(
                objectBody,
                "aggregate_factor",
                lodLevel.level == 0 ? 0 : 1));
            lodLevel.name = ExtractJsonStringOr(objectBody, "name", "lod" + std::to_string(lodLevel.level));
            lodLevel.kind = ExtractJsonStringOr(objectBody, "kind", lodLevel.level == 0 ? "exact" : "aggregate");
            lodLevel.pointPath = ExtractJsonStringOr(objectBody, "point_path", "");
            lodLevel.indexBinaryPath = ExtractJsonStringOr(objectBody, "index_binary_path", "");
            lodLevel.sourceIdPath = ExtractJsonStringOr(objectBody, "source_id_path", "");
            lodLevel.cellIdPath = ExtractJsonStringOr(objectBody, "cell_id_path", "");
            lodLevel.gpuPagePath = ExtractJsonStringOr(objectBody, "gpu_page_path", "");
            lodLevel.gpuPageIndexPath = ExtractJsonStringOr(
                objectBody,
                "gpu_page_index_path",
                "");
            lodLevel.gpuPageContractVersion = static_cast<std::uint32_t>(ExtractJsonUInt64Or(
                objectBody,
                "gpu_page_contract_version",
                0));
            lodLevel.gpuPagePointRecordSize = static_cast<std::uint32_t>(ExtractJsonUInt64Or(
                objectBody,
                "gpu_page_point_record_size",
                0));
            lodLevel.gpuPageIndexRecordSize = static_cast<std::uint32_t>(ExtractJsonUInt64Or(
                objectBody,
                "gpu_page_index_record_size",
                0));
            lodLevel.gpuPagePointCapacity = static_cast<std::uint32_t>(ExtractJsonUInt64Or(
                objectBody,
                "gpu_page_point_capacity",
                0));
            lodLevel.gpuPageCount = ExtractJsonUInt64Or(objectBody, "gpu_page_count", 0);
            lodLevel.gpuPagePointCount = ExtractJsonUInt64Or(
                objectBody,
                "gpu_page_point_count",
                0);
            lodLevel.attributeSemantics = ExtractJsonStringOr(
                objectBody,
                "attribute_semantics",
                "original_point_attributes");
            lodLevel.mainDisplayChain = ExtractJsonBoolOr(objectBody, "main_display_chain", false);
            lodLevel.recordSubsetOfLod0 = ExtractJsonBoolOr(
                objectBody,
                "record_subset_of_lod0",
                false);
            TryExtractJsonDouble(objectBody, "sample_ratio", lodLevel.sampleRatio);
            lodLevel.tileRows = static_cast<std::uint32_t>(ExtractJsonUInt64Or(
                objectBody,
                "tile_rows",
                metadata_.tileRows));
            lodLevel.tileCols = static_cast<std::uint32_t>(ExtractJsonUInt64Or(
                objectBody,
                "tile_cols",
                metadata_.tileCols));
            lodLevel.pointCount = ExtractJsonUInt64Or(objectBody, "point_count", 0);
            lodLevel.tileCount = ExtractJsonUInt64Or(objectBody, "tile_count", 0);
            metadata_.lodLevels.push_back(lodLevel);
        }
    }

    if (metadata_.lodLevels.empty())
    {
        throw std::runtime_error("Cache metadata does not contain the version 6 LOD chain.");
    }

    if (metadata_.mainLodLevels.empty())
    {
        throw std::runtime_error("Cache metadata does not contain main_lod_levels.");
    }

    (void)FindRecordField("x");
    (void)FindRecordField("y");
    for (const std::string& attr : metadata_.attrFields)
    {
        (void)FindRecordField(attr);
    }
}

void SeismicCacheReader::LoadColormap()
{
    const std::string text = ReadTextFile(metadata_.cacheDirectory / metadata_.colormapJsonPath);
    std::vector<int> values;
    values.reserve(1024);

    std::string current;
    for (char character : text)
    {
        if (std::isdigit(static_cast<unsigned char>(character)) != 0)
        {
            current.push_back(character);
        }
        else if (!current.empty())
        {
            values.push_back(std::stoi(current));
            current.clear();
        }
    }
    if (!current.empty())
    {
        values.push_back(std::stoi(current));
    }

    if (values.size() < 1024)
    {
        throw std::runtime_error("Colormap JSON does not contain 256 RGBA colors.");
    }

    for (std::size_t index = 0; index < 256; ++index)
    {
        colormap_[index] =
        {
            static_cast<float>(values[index * 4 + 0]) / 255.0f,
            static_cast<float>(values[index * 4 + 1]) / 255.0f,
            static_cast<float>(values[index * 4 + 2]) / 255.0f,
            static_cast<float>(values[index * 4 + 3]) / 255.0f,
        };
    }
}

void SeismicCacheReader::LoadTileIndices()
{
    const auto lodBegin = std::chrono::steady_clock::now();
    if (metadata_.gpuReadyPageContractVersion > 0)
    {
        lastOpenTiming_.pointLodIndexMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - lodBegin).count();
        std::cout << "CPU display LOD indices skipped: GPU-ready Page Cache is available.\n";
        return;
    }
    for (const CacheLodLevel& lodLevel : metadata_.lodLevels)
    {
        if (lodLevel.level <= 0 ||
            !lodLevel.mainDisplayChain ||
            lodLevel.kind == "overview_grid" ||
            lodLevel.pointPath.empty() ||
            lodLevel.indexBinaryPath.empty())
        {
            continue;
        }

        if (!std::filesystem::exists(metadata_.cacheDirectory / lodLevel.pointPath) ||
            !std::filesystem::exists(metadata_.cacheDirectory / lodLevel.indexBinaryPath))
        {
            throw std::runtime_error("LOD files declared by metadata are missing: " + lodLevel.name);
        }

        LodIndexData lod = LoadLodIndexBinary(lodLevel);
        ValidateLodFiles(lod);
        lodIndices_.push_back(std::move(lod));
    }

    std::sort(
        lodIndices_.begin(),
        lodIndices_.end(),
        [this](const LodIndexData& left, const LodIndexData& right)
        {
            const auto leftIt = std::find(
                metadata_.mainLodLevels.begin(),
                metadata_.mainLodLevels.end(),
                left.metadata.name);
            const auto rightIt = std::find(
                metadata_.mainLodLevels.begin(),
                metadata_.mainLodLevels.end(),
                right.metadata.name);
            return leftIt < rightIt;
        });

    for (std::size_t lodIndex = 0; lodIndex < lodIndices_.size(); ++lodIndex)
    {
        lodIndices_[lodIndex].cacheNamespace = static_cast<std::uint32_t>(lodIndex + 1);
    }

    lastOpenTiming_.pointLodIndexMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - lodBegin).count();
}

void SeismicCacheReader::StartLod0IndexLoadAsync() const
{
    if (!opened_)
    {
        throw std::runtime_error("Cannot start the LOD0 index before the cache is opened.");
    }

    const auto lod0It = std::find_if(
        metadata_.lodLevels.begin(),
        metadata_.lodLevels.end(),
        [](const CacheLodLevel& level)
        {
            return level.name == "lod0" && level.kind == "exact";
        });
    if (lod0It == metadata_.lodLevels.end())
    {
        throw std::runtime_error("Cache metadata does not declare the lod0 exact level.");
    }
    const CacheLodLevel lod0 = *lod0It;

    std::lock_guard<std::mutex> lock(indexMutex_);
    if (lod0IndexReady_ || lod0IndexLoading_ || lod0IndexFuture_.valid())
    {
        return;
    }

    lod0IndexLoading_ = true;
    lod0IndexReady_ = false;
    lod0IndexLoadFailed_ = false;
    lod0IndexLoadError_.clear();
    lod0IndexFuture_ = std::async(
        std::launch::async,
        [this, lod0]()
        {
            const auto begin = std::chrono::steady_clock::now();
            LodIndexData result = LoadLodIndexBinary(lod0);
            ValidateLodFiles(result);
            const auto end = std::chrono::steady_clock::now();
            const double milliseconds =
                std::chrono::duration<double, std::milli>(end - begin).count();

            std::cout
                << "LOD0 index prepared in background: tiles=" << result.tiles.size()
                << ", load_ms=" << milliseconds
                << '\n';
            return result;
        });

    std::cout << "LOD0 index loading started in background.\n";
}

void SeismicCacheReader::DrainLod0IndexLoad(bool wait) const
{
    std::future<LodIndexData> readyFuture;
    {
        std::lock_guard<std::mutex> lock(indexMutex_);
        if (!lod0IndexFuture_.valid())
        {
            return;
        }
        if (!wait &&
            lod0IndexFuture_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        {
            return;
        }

        readyFuture = std::move(lod0IndexFuture_);
        lod0IndexLoading_ = false;
    }

    try
    {
        LodIndexData lod0Index = readyFuture.get();
        std::lock_guard<std::mutex> lock(indexMutex_);
        tileIndex_ = std::move(lod0Index.tiles);
        tileIndexById_ = std::move(lod0Index.tileIndexById);
        lod0IndexReady_ = true;
        lod0IndexLoadFailed_ = false;
        lod0IndexLoadError_.clear();

        std::cout
            << "LOD0 index applied: tiles=" << tileIndex_.size()
            << '\n';
    }
    catch (const std::exception& error)
    {
        std::lock_guard<std::mutex> lock(indexMutex_);
        lod0IndexReady_ = false;
        lod0IndexLoadFailed_ = true;
        lod0IndexLoadError_ = error.what();

        std::cerr
            << "LOD0 index loading failed: " << lod0IndexLoadError_
            << '\n';
    }
}

void SeismicCacheReader::LoadOverviewPoints()
{
    overviewPoints_.clear();

    const std::filesystem::path path = metadata_.cacheDirectory / metadata_.overviewPointsPath;
    if (!std::filesystem::exists(path))
    {
        std::cout
            << "Overview skipped: path=" << metadata_.overviewPointsPath
            << ", reason=missing_file\n";
        return;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open overview point file: " + path.string());
    }

    char magic[8]{};
    std::uint32_t version = 0;
    std::uint32_t attrCount = 0;
    std::uint64_t pointCount = 0;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&attrCount), sizeof(attrCount));
    file.read(reinterpret_cast<char*>(&pointCount), sizeof(pointCount));

    if (!file)
    {
        throw std::runtime_error("Failed to read overview point header.");
    }
    const bool isVersion1 = std::strncmp(magic, "GPVOVR1", 7) == 0 && version == 1;
    const bool isVersion2 = std::strncmp(magic, "GPVOVR2", 7) == 0 && version == 2;
    if (!isVersion1 && !isVersion2)
    {
        throw std::runtime_error("Unsupported overview point file format.");
    }
    if (attrCount == 0 || attrCount > MaxCacheAttributes ||
        attrCount != metadata_.attrFields.size())
    {
        throw std::runtime_error("Overview point attribute count does not match metadata.");
    }
    if (pointCount > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::runtime_error("Overview point count exceeds the addressable range.");
    }
    const std::uint64_t coordinateBytes = isVersion2 ? sizeof(double) * 2 : sizeof(float) * 2;
    const std::uint64_t recordBytes =
        coordinateBytes + static_cast<std::uint64_t>(attrCount) * sizeof(float) + sizeof(float);
    const std::uint64_t headerBytes = sizeof(magic) + sizeof(version) + sizeof(attrCount) + sizeof(pointCount);
    if (pointCount > (std::numeric_limits<std::uint64_t>::max() - headerBytes) / recordBytes)
    {
        throw std::runtime_error("Overview point file size overflows the supported range.");
    }
    const std::uint64_t expectedBytes = headerBytes + pointCount * recordBytes;
    const std::uint64_t actualBytes = std::filesystem::file_size(path);
    if (actualBytes != expectedBytes)
    {
        throw std::runtime_error("Overview point file size does not match its header.");
    }

    overviewPoints_.reserve(static_cast<std::size_t>(pointCount));

    for (std::uint64_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        OverviewPoint point;
        if (isVersion2)
        {
            file.read(reinterpret_cast<char*>(&point.x), sizeof(point.x));
            file.read(reinterpret_cast<char*>(&point.y), sizeof(point.y));
        }
        else
        {
            float x = 0.0f;
            float y = 0.0f;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            file.read(reinterpret_cast<char*>(&y), sizeof(y));
            point.x = static_cast<double>(x);
            point.y = static_cast<double>(y);
        }
        for (std::uint32_t attrIndex = 0; attrIndex < attrCount; ++attrIndex)
        {
            file.read(reinterpret_cast<char*>(&point.attrs[attrIndex]), sizeof(float));
        }
        file.read(reinterpret_cast<char*>(&point.count), sizeof(point.count));

        if (!file)
        {
            throw std::runtime_error("Failed to read overview point data.");
        }
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.count) || point.count <= 0.0f)
        {
            throw std::runtime_error("Overview point data contains invalid values.");
        }
        for (std::uint32_t attrIndex = 0; attrIndex < attrCount; ++attrIndex)
        {
            if (!std::isfinite(point.attrs[attrIndex]))
            {
                throw std::runtime_error("Overview point attributes contain invalid values.");
            }
        }

        overviewPoints_.push_back(point);
    }
}

SeismicCacheReader::LodIndexData SeismicCacheReader::LoadLodIndexBinary(
    const CacheLodLevel& lodLevel) const
{
    const std::filesystem::path path = metadata_.cacheDirectory / lodLevel.indexBinaryPath;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open TIDX file: " + lodLevel.indexBinaryPath);
    }

    std::array<char, kTileIndexHeaderSize> header{};
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (file.gcount() != static_cast<std::streamsize>(header.size()))
    {
        throw std::runtime_error("TIDX header is truncated: " + lodLevel.indexBinaryPath);
    }

    if (std::memcmp(header.data(), "TIDX", 4) != 0)
    {
        throw std::runtime_error("TIDX magic mismatch: " + lodLevel.indexBinaryPath);
    }

    const std::uint32_t version = ReadBinaryValue<std::uint32_t>(header.data() + 4);
    const std::uint32_t headerSize = ReadBinaryValue<std::uint32_t>(header.data() + 8);
    const std::uint64_t recordCount = ReadBinaryValue<std::uint64_t>(header.data() + 12);
    const std::uint32_t attrCount = ReadBinaryValue<std::uint32_t>(header.data() + 20);
    const std::uint32_t recordSize = ReadBinaryValue<std::uint32_t>(header.data() + 24);
    const std::uint32_t tileRows = ReadBinaryValue<std::uint32_t>(header.data() + 32);
    const std::uint32_t tileCols = ReadBinaryValue<std::uint32_t>(header.data() + 36);
    const std::uint32_t expectedRecordSize = static_cast<std::uint32_t>(
        kTileIndexFixedRecordSize + metadata_.attrFields.size() * 3 * sizeof(float));

    if (version != kTileIndexVersion || headerSize != kTileIndexHeaderSize)
    {
        throw std::runtime_error("Unsupported TIDX contract: " + lodLevel.indexBinaryPath);
    }
    if (attrCount != metadata_.attrFields.size() || recordSize != expectedRecordSize)
    {
        throw std::runtime_error("TIDX attribute or record size mismatch: " + lodLevel.indexBinaryPath);
    }
    if ((tileRows != 0 && tileRows != lodLevel.tileRows) ||
        (tileCols != 0 && tileCols != lodLevel.tileCols))
    {
        throw std::runtime_error("TIDX tile grid mismatch: " + lodLevel.indexBinaryPath);
    }

    const std::uint64_t expectedBytes =
        static_cast<std::uint64_t>(headerSize) + recordCount * recordSize;
    if (std::filesystem::file_size(path) != expectedBytes)
    {
        throw std::runtime_error("TIDX file length mismatch: " + lodLevel.indexBinaryPath);
    }

    LodIndexData result;
    result.metadata = lodLevel;
    result.tiles.reserve(static_cast<std::size_t>(recordCount));
    std::vector<char> record(recordSize);

    for (std::uint64_t recordIndex = 0; recordIndex < recordCount; ++recordIndex)
    {
        file.read(record.data(), static_cast<std::streamsize>(record.size()));
        if (file.gcount() != static_cast<std::streamsize>(record.size()))
        {
            throw std::runtime_error("TIDX record is truncated: " + lodLevel.indexBinaryPath);
        }

        TileIndexEntry entry;
        entry.tileId = ReadBinaryValue<std::int64_t>(record.data());
        entry.row = ReadBinaryValue<std::uint32_t>(record.data() + 8);
        entry.col = ReadBinaryValue<std::uint32_t>(record.data() + 12);
        entry.bounds.minX = ReadBinaryValue<double>(record.data() + 16);
        entry.bounds.maxX = ReadBinaryValue<double>(record.data() + 24);
        entry.bounds.minY = ReadBinaryValue<double>(record.data() + 32);
        entry.bounds.maxY = ReadBinaryValue<double>(record.data() + 40);
        entry.pointCount = ReadBinaryValue<std::uint64_t>(record.data() + 48);
        entry.offsetBytes = ReadBinaryValue<std::uint64_t>(record.data() + 56);
        entry.lengthPoints = ReadBinaryValue<std::uint64_t>(record.data() + 64);
        entry.metadataIndex = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                result.tiles.size(),
                std::numeric_limits<std::uint32_t>::max()));

        for (std::size_t attrIndex = 0; attrIndex < metadata_.attrFields.size(); ++attrIndex)
        {
            const std::size_t attrOffset = kTileIndexFixedRecordSize + attrIndex * 3 * sizeof(float);
            entry.attrMins[attrIndex] = ReadBinaryValue<float>(record.data() + attrOffset);
            entry.attrMaxs[attrIndex] = ReadBinaryValue<float>(record.data() + attrOffset + 4);
            entry.attrMeans[attrIndex] = ReadBinaryValue<float>(record.data() + attrOffset + 8);
        }

        if (!result.tileIndexById.emplace(entry.tileId, result.tiles.size()).second)
        {
            throw std::runtime_error("TIDX contains duplicate tile_id: " + lodLevel.indexBinaryPath);
        }
        result.tiles.push_back(entry);
    }

    return result;
}

void SeismicCacheReader::ValidateLodFiles(const LodIndexData& lod) const
{
    const std::filesystem::path pointPath = metadata_.cacheDirectory / lod.metadata.pointPath;
    if (!std::filesystem::exists(pointPath))
    {
        throw std::runtime_error("LOD point file is missing: " + lod.metadata.pointPath);
    }

    std::uint64_t indexedPoints = 0;
    std::uint64_t requiredBytes = 0;
    std::uint64_t expectedOffset = 0;
    for (const TileIndexEntry& tile : lod.tiles)
    {
        if (tile.offsetBytes != expectedOffset)
        {
            throw std::runtime_error("TIDX offsets are not contiguous: " + lod.metadata.name);
        }
        if (tile.row >= lod.metadata.tileRows || tile.col >= lod.metadata.tileCols ||
            tile.tileId != static_cast<std::int64_t>(tile.row) * lod.metadata.tileCols + tile.col)
        {
            throw std::runtime_error("TIDX tile coordinates are invalid: " + lod.metadata.name);
        }
        indexedPoints += tile.lengthPoints;
        requiredBytes = std::max(
            requiredBytes,
            tile.offsetBytes + tile.lengthPoints * metadata_.recordSize);
        expectedOffset += tile.lengthPoints * metadata_.recordSize;
    }

    const std::uint64_t pointFileBytes = std::filesystem::file_size(pointPath);
    if (requiredBytes > pointFileBytes ||
        pointFileBytes != indexedPoints * metadata_.recordSize)
    {
        throw std::runtime_error("LOD point file size does not match its TIDX index: " + lod.metadata.name);
    }
    if (lod.metadata.pointCount != 0 && indexedPoints != lod.metadata.pointCount)
    {
        throw std::runtime_error("LOD metadata point_count does not match TIDX: " + lod.metadata.name);
    }

    if (!lod.metadata.sourceIdPath.empty())
    {
        const std::filesystem::path sourceIdPath =
            metadata_.cacheDirectory / lod.metadata.sourceIdPath;
        if (!std::filesystem::exists(sourceIdPath) ||
            std::filesystem::file_size(sourceIdPath) != indexedPoints * sizeof(std::uint64_t))
        {
            throw std::runtime_error("LOD source_id sidecar size mismatch: " + lod.metadata.name);
        }
    }

    if (!lod.metadata.cellIdPath.empty())
    {
        const std::filesystem::path cellIdPath =
            metadata_.cacheDirectory / lod.metadata.cellIdPath;
        if (!std::filesystem::exists(cellIdPath) ||
            std::filesystem::file_size(cellIdPath) != indexedPoints * sizeof(std::uint32_t))
        {
            throw std::runtime_error("LOD cell_id sidecar size mismatch: " + lod.metadata.name);
        }
    }
}

std::vector<CachePoint> SeismicCacheReader::ReadTilePoints(
    const std::vector<const TileIndexEntry*>& tiles,
    const Bounds2D& viewport,
    const CacheQueryOptions& options) const
{
    const RecordField& xField = FindRecordField("x");
    const RecordField& yField = FindRecordField("y");
    std::array<const RecordField*, MaxCacheAttributes> attrFields{};
    for (std::size_t attrIndex = 0; attrIndex < metadata_.attrFields.size(); ++attrIndex)
    {
        attrFields[attrIndex] = &FindRecordField(metadata_.attrFields[attrIndex]);
    }

    std::ifstream file(ResolveTilesPath(), std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open tiles_lod0.bin.");
    }

    std::vector<CachePoint> points;
    const std::uint64_t reserveCount = std::min<std::uint64_t>(
        MaxLod0PointLoad,
        std::accumulate(
            tiles.begin(),
            tiles.end(),
            std::uint64_t{ 0 },
            [](std::uint64_t sum, const TileIndexEntry* entry)
            {
                return sum + entry->lengthPoints;
            }));
    points.reserve(static_cast<std::size_t>(reserveCount));

    for (const TileIndexEntry* tile : tiles)
    {
        ThrowIfQueryCancelled(options);
        const std::shared_ptr<const std::vector<CachePoint>> tilePoints =
            GetOrReadTilePoints(
                TileCacheKey{ 0, tile->tileId },
                file,
                *tile,
                xField,
                yField,
                attrFields);
        for (const CachePoint& point : *tilePoints)
        {
            if (point.x < viewport.minX || point.x > viewport.maxX ||
                point.y < viewport.minY || point.y > viewport.maxY)
            {
                continue;
            }

            points.push_back(point);
        }
    }

    return points;
}

std::vector<CachePoint> SeismicCacheReader::ReadLodTilePoints(
    const LodIndexData& lod,
    const std::vector<const TileIndexEntry*>& tiles,
    const Bounds2D& viewport,
    const CacheQueryOptions& options) const
{
    const RecordField& xField = FindRecordField("x");
    const RecordField& yField = FindRecordField("y");
    std::array<const RecordField*, MaxCacheAttributes> attrFields{};
    for (std::size_t attrIndex = 0; attrIndex < metadata_.attrFields.size(); ++attrIndex)
    {
        attrFields[attrIndex] = &FindRecordField(metadata_.attrFields[attrIndex]);
    }

    std::ifstream file(metadata_.cacheDirectory / lod.metadata.pointPath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open LOD point file: " + lod.metadata.pointPath);
    }

    std::ifstream sourceIds;
    if (!lod.metadata.sourceIdPath.empty())
    {
        sourceIds.open(metadata_.cacheDirectory / lod.metadata.sourceIdPath, std::ios::binary);
    }
    std::ifstream cellIds;
    if (!lod.metadata.cellIdPath.empty())
    {
        cellIds.open(metadata_.cacheDirectory / lod.metadata.cellIdPath, std::ios::binary);
    }

    std::vector<CachePoint> points;
    const std::uint64_t reserveCount = std::accumulate(
        tiles.begin(),
        tiles.end(),
        std::uint64_t{ 0 },
        [](std::uint64_t sum, const TileIndexEntry* entry)
        {
            return sum + entry->lengthPoints;
        });
    points.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(
        reserveCount,
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))));

    for (const TileIndexEntry* tile : tiles)
    {
        ThrowIfQueryCancelled(options);
        const TileCacheKey cacheKey{ lod.cacheNamespace, tile->tileId };
        std::shared_ptr<const std::vector<CachePoint>> tilePoints =
            FindCachedTilePoints(cacheKey);
        if (tilePoints == nullptr)
        {
            std::vector<CachePoint> loadedPoints =
                ReadRawTilePoints(file, *tile, xField, yField, attrFields);
            AttachLodSidecars(
                lod,
                *tile,
                sourceIds.is_open() ? &sourceIds : nullptr,
                cellIds.is_open() ? &cellIds : nullptr,
                loadedPoints);
            tilePoints = StoreTileInCache(cacheKey, std::move(loadedPoints));
        }
        AppendFilteredTilePoints(*tilePoints, viewport, points);
    }

    return points;
}

std::shared_ptr<const std::vector<CachePoint>> SeismicCacheReader::GetOrReadTilePoints(
    const TileCacheKey& cacheKey,
    std::ifstream& file,
    const TileIndexEntry& tile,
    const RecordField& xField,
    const RecordField& yField,
    const std::array<const RecordField*, MaxCacheAttributes>& attrFields) const
{
    const std::shared_ptr<const std::vector<CachePoint>> cachedPoints =
        FindCachedTilePoints(cacheKey);
    if (cachedPoints != nullptr)
    {
        return cachedPoints;
    }

    std::vector<CachePoint> points = ReadRawTilePoints(file, tile, xField, yField, attrFields);
    return StoreTileInCache(cacheKey, std::move(points));
}

std::vector<CachePoint> SeismicCacheReader::ReadRawTilePoints(
    std::ifstream& file,
    const TileIndexEntry& tile,
    const RecordField& xField,
    const RecordField& yField,
    const std::array<const RecordField*, MaxCacheAttributes>& attrFields) const
{
    const std::uint64_t byteCount = tile.lengthPoints * metadata_.recordSize;
    std::vector<char> buffer(static_cast<std::size_t>(byteCount));

    file.seekg(static_cast<std::streamoff>(tile.offsetBytes), std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    if (file.gcount() != static_cast<std::streamsize>(buffer.size()))
    {
        throw std::runtime_error("Failed to read the requested tile bytes.");
    }

    std::vector<CachePoint> points;
    points.reserve(static_cast<std::size_t>(tile.lengthPoints));

    for (std::uint64_t pointIndex = 0; pointIndex < tile.lengthPoints; ++pointIndex)
    {
        const char* record = buffer.data() + pointIndex * metadata_.recordSize;

        CachePoint point;
        point.tileId = tile.tileId;
        point.tileIndex = tile.metadataIndex;
        point.sourceId = tile.offsetBytes / metadata_.recordSize + pointIndex;
        point.hasSourceId = true;
        point.x = ReadCoordinateValue(record, xField);
        point.y = ReadCoordinateValue(record, yField);
        for (std::size_t attrIndex = 0; attrIndex < metadata_.attrFields.size(); ++attrIndex)
        {
            point.attrs[attrIndex] = ReadAttributeValue(record, *attrFields[attrIndex]);
        }
        points.push_back(point);
    }

    return points;
}

void SeismicCacheReader::AttachLodSidecars(
    const LodIndexData& lod,
    const TileIndexEntry& tile,
    std::ifstream* sourceIds,
    std::ifstream* cellIds,
    std::vector<CachePoint>& points) const
{
    const std::uint64_t firstPoint = tile.offsetBytes / metadata_.recordSize;

    if (!lod.metadata.sourceIdPath.empty())
    {
        if (sourceIds == nullptr)
        {
            throw std::runtime_error("Failed to open LOD source_id sidecar: " + lod.metadata.name);
        }
        sourceIds->seekg(
            static_cast<std::streamoff>(firstPoint * sizeof(std::uint64_t)),
            std::ios::beg);
        for (CachePoint& point : points)
        {
            sourceIds->read(
                reinterpret_cast<char*>(&point.sourceId),
                sizeof(point.sourceId));
            if (!*sourceIds)
            {
                throw std::runtime_error("Failed to read LOD source_id sidecar: " + lod.metadata.name);
            }
            if (point.sourceId >= metadata_.validPoints)
            {
                throw std::runtime_error("LOD source_id is outside the LOD0 record range: " + lod.metadata.name);
            }
            point.hasSourceId = true;
        }
    }

    if (!lod.metadata.cellIdPath.empty())
    {
        if (cellIds == nullptr)
        {
            throw std::runtime_error("Failed to open LOD cell_id sidecar: " + lod.metadata.name);
        }
        cellIds->seekg(
            static_cast<std::streamoff>(firstPoint * sizeof(std::uint32_t)),
            std::ios::beg);
        for (CachePoint& point : points)
        {
            cellIds->read(
                reinterpret_cast<char*>(&point.cellId),
                sizeof(point.cellId));
            if (!*cellIds)
            {
                throw std::runtime_error("Failed to read LOD cell_id sidecar: " + lod.metadata.name);
            }
            point.hasCellId = true;
        }
    }
}

std::shared_ptr<const std::vector<CachePoint>> SeismicCacheReader::FindCachedTilePoints(
    const TileCacheKey& cacheKey) const
{
    std::lock_guard<std::mutex> lock(tileCacheMutex_);

    const auto cacheIt = tileCache_.find(cacheKey);
    if (cacheIt == tileCache_.end())
    {
        ++tileCacheStats_.cacheMisses;
        return nullptr;
    }

    tileCacheLru_.erase(cacheIt->second.lruPosition);
    tileCacheLru_.push_front(cacheKey);
    cacheIt->second.lruPosition = tileCacheLru_.begin();
    ++tileCacheStats_.cacheHits;
    return cacheIt->second.points;
}

std::shared_ptr<const std::vector<CachePoint>> SeismicCacheReader::StoreTileInCache(
    const TileCacheKey& cacheKey,
    std::vector<CachePoint>&& points) const
{
    auto sharedPoints = std::make_shared<std::vector<CachePoint>>(std::move(points));
    const std::uint64_t pointCount = static_cast<std::uint64_t>(sharedPoints->size());

    std::lock_guard<std::mutex> lock(tileCacheMutex_);

    const auto existing = tileCache_.find(cacheKey);
    if (existing != tileCache_.end())
    {
        cachedTilePointCount_ -= existing->second.pointCount;
        tileCacheLru_.erase(existing->second.lruPosition);
        tileCache_.erase(existing);
    }

    tileCacheLru_.push_front(cacheKey);
    tileCache_.emplace(
        cacheKey,
        CachedTile
        {
            sharedPoints,
            pointCount,
            tileCacheLru_.begin(),
        });

    cachedTilePointCount_ += pointCount;
    PruneTileCacheLocked();

    return sharedPoints;
}

void SeismicCacheReader::PruneTileCacheLocked() const
{
    while (!tileCacheLru_.empty() &&
        (tileCache_.size() > MaxCachedTiles || cachedTilePointCount_ > MaxCachedTilePoints))
    {
        const TileCacheKey evictedKey = tileCacheLru_.back();
        const auto evicted = tileCache_.find(evictedKey);

        tileCacheLru_.pop_back();

        if (evicted != tileCache_.end())
        {
            cachedTilePointCount_ -= evicted->second.pointCount;
            tileCache_.erase(evicted);
            ++tileCacheStats_.cacheEvictions;
        }
    }
}

const RecordField& SeismicCacheReader::FindRecordField(const std::string& name) const
{
    const auto it = std::find_if(
        metadata_.recordFields.begin(),
        metadata_.recordFields.end(),
        [&name](const RecordField& field)
        {
            return field.name == name;
        });

    if (it == metadata_.recordFields.end())
    {
        throw std::runtime_error("Record layout is missing field: " + name);
    }

    return *it;
}

std::uint32_t SeismicCacheReader::ColorIndexForValue(float value, std::size_t activeAttrIndex) const
{
    const std::string& attr = metadata_.attrFields[activeAttrIndex];
    const auto rangeIt = metadata_.attrRanges.find(attr);
    if (rangeIt == metadata_.attrRanges.end())
    {
        return 0;
    }

    const AttributeRange& range = rangeIt->second;
    if (range.maxValue <= range.minValue)
    {
        return 0;
    }

    const double normalized = (static_cast<double>(value) - range.minValue) /
        (range.maxValue - range.minValue);
    const double clamped = std::clamp(normalized, 0.0, 1.0);
    return static_cast<std::uint32_t>(std::lround(clamped * 255.0));
}

float SeismicCacheReader::ReadAttributeValue(const char* record, const RecordField& field) const
{
    if (field.dtype == "float32")
    {
        float value = 0.0f;
        std::memcpy(&value, record + field.offset, sizeof(value));
        return value;
    }

    if (field.dtype == "float64")
    {
        double value = 0.0;
        std::memcpy(&value, record + field.offset, sizeof(value));
        return static_cast<float>(value);
    }

    throw std::runtime_error("Unsupported attribute dtype: " + field.dtype);
}

double SeismicCacheReader::ReadCoordinateValue(const char* record, const RecordField& field) const
{
    if (field.dtype == "float64")
    {
        double value = 0.0;
        std::memcpy(&value, record + field.offset, sizeof(value));
        return value;
    }

    if (field.dtype == "float32")
    {
        float value = 0.0f;
        std::memcpy(&value, record + field.offset, sizeof(value));
        return static_cast<double>(value);
    }

    throw std::runtime_error("Unsupported coordinate dtype: " + field.dtype);
}

} // namespace gpv
