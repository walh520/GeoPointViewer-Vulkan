// 应用主控声明：负责 GLFW 窗口、二维视口交互和正式 cache 数据刷新。
#pragma once

#include "AppConfig.h"
#include "../cache/GpuReadyPageCache.h"
#include "../cache/SeismicCacheReader.h"
#include "../cache/Viewport2D.h"
#include "../interaction/BoxSelector.h"
#include "../performance/PerformanceRecorder.h"
#include "../profile/ProfileAnalysisService.h"
#include "../vulkan/VulkanContext.h"

#include <chrono>
#include <atomic>
#include <future>
#include <functional>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace gpv
{

class App
{
public:
    App() = default;
    explicit App(const AppConfig& config);
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    void Run(std::uint32_t maxFrames = 0);

private:
    static constexpr std::uint32_t WindowWidth = 1920;
    static constexpr std::uint32_t WindowHeight = 1080;

    enum class LodBudgetMode
    {
        Interactive,
        Stable,
    };

    struct PreparedVisibleUpload
    {
        std::vector<GpuPoint> gpuPoints;
        std::vector<GpuTilePage> tilePages;
        std::vector<GpuReadyPageView> gpuReadyPages;
        std::vector<std::uint32_t> gpuProxyMask;
        double renderOriginX = 0.0;
        double renderOriginY = 0.0;
        double buildMilliseconds = 0.0;
        std::size_t overviewPointCount = 0;
        std::size_t cpuVisiblePointCount = 0;
        float overviewAlpha = 0.0f;
        bool useGpuProxy = false;
        bool gpuReadyFastPath = false;
    };

    void Initialize();
    void InitializeWindow();
    void MainLoop(std::uint32_t maxFrames);
    void ProcessInput();
    void ResetInputLatches();
    void RefreshVisiblePoints(bool rebuildQuery);
    CacheQueryOptions BuildCacheQueryOptions(
        LodBudgetMode mode = LodBudgetMode::Stable) const;
    CacheQueryResult QueryVisibleData(
        const Bounds2D& viewport,
        std::size_t attrIndex,
        const CacheQueryOptions& options) const;
    bool CanUseGpuDrivenViewport() const;
    std::uint64_t ApplyGpuDrivenViewport(
        const Bounds2D& viewport,
        const CacheQueryOptions& options,
        const char* reason);
    void WarmGpuReadyResidency();
    void RebaseRenderOriginToViewport();
    void ShowPendingOverviewUnderlay(const char* reason);
    std::uint64_t RequestVisiblePointsAsync(
        const char* reason,
        LodBudgetMode mode = LodBudgetMode::Stable);
    std::uint64_t QueueVisiblePointsAsync(
        const Bounds2D& viewport,
        const CacheQueryOptions& options,
        const char* reason);
    void ScheduleWheelViewportQuery(const char* reason);
    void PollWheelViewportQuery();
    void ScheduleInteractivePanQuery();
    void PollInteractivePanQuery();
    void StartNextVisiblePointsRequest();
    void TrySchedulePendingLod0Refresh();
    void TryScheduleGuardBandRefresh();
    void PollVisiblePointsRequest();
    void PollBackgroundIndexPreparation();
    void RetireVisiblePoints(std::vector<CachePoint>&& points);
    void WaitForRetiredVisiblePoints();
    PreparedVisibleUpload PrepareVisibleUpload(
        const CacheQueryResult& query,
        const Bounds2D& renderViewport,
        std::size_t attrIndex,
        const std::function<bool()>& shouldCancel = {}) const;
    void ApplyVisibleQueryResult(
        const char* reason,
        CacheQueryResult&& query,
        PreparedVisibleUpload&& upload);
    void SubmitAsyncSmokeTestIfNeeded();
    void ResetToWorld();
    void SetActiveAttribute(std::size_t attrIndex);
    void SetActiveAttributeByName(const char* name, std::size_t fallbackIndex);
    void CycleAttribute();
    void EnterProfileMode();
    void QueueProfileViewportPreparation(const char* reason);
    void PollProfileAnalysis();
    void AddProfileVertex(double cursorX, double cursorY);
    void UndoProfileVertex();
    void CommitProfileAnalysis();
    void ClearProfileDraft();
    void UpdateProfileWindowTitle(const char* state);
    std::vector<glm::vec2> BuildProfileOverlayNdc() const;
    void StartBoxSelection();
    void StopBoxSelection();
    void UpdateBoxSelection(double cursorX, double cursorY);
    void CommitBoxSelectionZoom(double cursorX, double cursorY);
    void UpdateBoxSelectionFlash();
    std::optional<CachePoint> FindExactPointAtCursor(
        double cursorX,
        double cursorY,
        double radiusPixels) const;
    void PrintPointInfo(const char* prefix, const CachePoint& point) const;
    void PickPointAtCursor(double cursorX, double cursorY) const;
    void UpdateHoverAtCursor(double cursorX, double cursorY);
    void UpdateStatusPanel();
    void UpdateBenchmarkBeforeFrame();
    void UpdateBenchmarkAfterFrame();
    void FinishBenchmark();
    void PrintFirstGlobalFrameTiming();
    void PrintBackgroundReadyTiming();
    void Shutdown();

    static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void WindowFocusCallback(GLFWwindow* window, int focused);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void CursorPositionCallback(GLFWwindow* window, double cursorX, double cursorY);
    static void ScrollCallback(GLFWwindow* window, double xOffset, double yOffset);

    struct AsyncVisibleQuery
    {
        CacheQueryResult query;
        PreparedVisibleUpload upload;
        std::uint64_t requestId = 0;
        Bounds2D viewport;
        Bounds2D renderViewport;
        CacheQueryOptions options;
        std::size_t attrIndex = 0;
        std::string reason;
        double submitTimeSeconds = 0.0;
        double queryMilliseconds = 0.0;
        bool cancelled = false;
    };

    void StageInteractivePanResult(AsyncVisibleQuery&& result);
    bool CommitStagedInteractivePanResult();

    struct BenchmarkAction
    {
        std::string name;
        double value0 = 0.0;
        double value1 = 0.0;
        double value2 = 0.0;
        bool zoom = true;
    };

    GLFWwindow* window_ = nullptr;
    AppConfig config_;
    VulkanContext vulkan_;
    SeismicCacheReader cache_;
    GpuReadyPageCache gpuReadyPageCache_;
    PerformanceRecorder performanceRecorder_;
    ProfileAnalysisService profileAnalysis_;
    Viewport2D viewport_;
    BoxSelectionState boxSelection_;
    std::vector<CachePoint> visiblePoints_;
    std::size_t activeAttrIndex_ = 0;

    bool framebufferResized_ = false;
    bool initialized_ = false;
    bool canvasDragging_ = false;
    bool boxSelecting_ = false;
    bool leftMouseWasDown_ = false;
    bool leftClickCandidate_ = false;
    bool leftDragMoved_ = false;
    bool keyEscWasDown_ = false;
    bool key1WasDown_ = false;
    bool key2WasDown_ = false;
    bool keyCWasDown_ = false;
    bool keyFWasDown_ = false;
    bool keyRWasDown_ = false;
    bool keyPlusWasDown_ = false;
    bool keyMinusWasDown_ = false;
    bool keyEnterWasDown_ = false;
    bool keyBackspaceWasDown_ = false;

    double renderOriginX_ = 0.0;
    double renderOriginY_ = 0.0;
    double boxSelectionStartX_ = 0.0;
    double boxSelectionStartY_ = 0.0;
    double leftMousePressX_ = 0.0;
    double leftMousePressY_ = 0.0;
    double lastCursorX_ = 0.0;
    double lastCursorY_ = 0.0;
    double lastHoverPrintTime_ = 0.0;
    double lastHoverHitTime_ = 0.0;
    double lastStatusPrintTime_ = 0.0;
    double currentFps_ = 0.0;
    std::uint32_t statusFrameCounter_ = 0;
    double accumulatedFrameMilliseconds_ = 0.0;
    double maxFrameMilliseconds_ = 0.0;
    double accumulatedFenceWaitMilliseconds_ = 0.0;
    double accumulatedVulkanUploadMilliseconds_ = 0.0;
    double accumulatedAcquireMilliseconds_ = 0.0;
    double accumulatedRecordMilliseconds_ = 0.0;
    double accumulatedSubmitMilliseconds_ = 0.0;
    double accumulatedPresentMilliseconds_ = 0.0;
    double accumulatedPointBufferResizeMilliseconds_ = 0.0;
    double accumulatedGpuCullMilliseconds_ = 0.0;
    double accumulatedGpuDrawMilliseconds_ = 0.0;
    std::uint32_t accumulatedPointBufferResizeCount_ = 0;
    std::uint32_t accumulatedGpuTimingFrameCount_ = 0;
    std::uint32_t accumulatedPipelineStatisticsFrameCount_ = 0;
    std::uint32_t accumulatedVisiblePageFrameCount_ = 0;
    std::uint64_t accumulatedVulkanUploadBytes_ = 0;
    std::uint64_t accumulatedInputAssemblyVertices_ = 0;
    std::uint64_t accumulatedVertexShaderInvocations_ = 0;
    std::uint64_t accumulatedComputeShaderInvocations_ = 0;
    std::uint64_t accumulatedVisiblePageCount_ = 0;
    std::vector<double> cpuFrameSamples_;
    std::vector<double> gpuFrameSamples_;
    std::vector<double> appWorkSamples_;
    double peakProcessMemoryMegabytes_ = 0.0;
    double peakGpuMemoryUsageMegabytes_ = -1.0;
    std::uint64_t lastProcessCpuTime100Nanoseconds_ = 0;
    std::uint64_t lastProcessIoReadBytes_ = 0;
    std::uint64_t lastProcessIoWriteBytes_ = 0;
    bool processResourceSampleInitialized_ = false;
    double lastQueryMilliseconds_ = 0.0;
    double lastQueryTotalMilliseconds_ = 0.0;
    double lastLodPointsPerPixel_ = StableLodPointsPerPixel;
    double lastEstimatedPointsPerPixel_ = 0.0;
    std::uint64_t lastLodPointBudget_ = 0;
    double lastCpuBuildUploadRequestMilliseconds_ = 0.0;
    double lastMainApplyMilliseconds_ = 0.0;
    std::size_t lastCpuUploadPoints_ = 0;
    std::uint64_t lastRenderedPointCount_ = 0;
    double lastPointUploadMegabytes_ = 0.0;
    std::string lastQueryReason_ = "initial";
    std::uint64_t lastEstimatedPoints_ = 0;
    std::uint64_t lastLod0PointCount_ = 0;
    std::uint64_t lastProxyPointCount_ = 0;
    std::uint32_t lastHitTileCount_ = 0;
    std::uint32_t lastLod0TileCount_ = 0;
    std::uint32_t lastProxyTileCount_ = 0;
    std::uint32_t lastDeferredTileCount_ = 0;
    std::size_t lastOverviewPointCount_ = 0;
    float lastOverviewAlpha_ = 0.0f;
    int lastActiveLodLevel_ = 0;
    std::string lastActiveLodName_ = "lod0";
    bool lastProxyMode_ = false;
    bool lastScreenSpaceLod_ = false;
    bool lastGpuProxyDraw_ = false;
    bool lastLod0IndexReady_ = true;
    bool lastLod0IndexPreparing_ = false;
    bool pendingTargetUnderlayActive_ = false;
    bool gpuTileMetadataUploaded_ = false;
    bool startupRefinementPending_ = false;
    bool firstFrameRendered_ = false;
    bool firstGlobalFrameTimingPrinted_ = false;
    bool backgroundReadyTimingPrinted_ = false;
    std::chrono::steady_clock::time_point startupBegin_{};
    double startupCacheOpenMilliseconds_ = 0.0;
    double startupWindowInitMilliseconds_ = 0.0;
    double startupVulkanInitMilliseconds_ = 0.0;
    double startupFirstQueryMilliseconds_ = 0.0;
    double startupFirstUploadRequestMilliseconds_ = 0.0;
    double startupInitializeMilliseconds_ = 0.0;
    double startupFirstGlobalFrameMilliseconds_ = 0.0;
    double startupLod0IndexReadyMilliseconds_ = 0.0;
    double startupGpuTileMetadataUploadMilliseconds_ = 0.0;
    std::atomic<std::uint64_t> desiredQueryRequestId_{ 0 };
    std::atomic<bool> gpuReadyFullResidencyReady_{ false };
    std::uint64_t runningQueryRequestId_ = 0;
    std::uint64_t lastAppliedQueryRequestId_ = 0;
    std::uint64_t queryRequestCount_ = 0;
    std::uint64_t querySupersededCount_ = 0;
    std::uint64_t queryCancelledCount_ = 0;
    std::uint64_t queryDiscardedCount_ = 0;
    std::uint64_t interactiveQueryCoalescedCount_ = 0;
    std::uint64_t interactiveQueryStagedCount_ = 0;
    std::uint64_t interactiveQueryCommittedCount_ = 0;
    double queryCancellationTotalMilliseconds_ = 0.0;
    double queryCancellationMaxMilliseconds_ = 0.0;
    Bounds2D desiredQueryViewport_;
    Bounds2D desiredRenderViewport_;
    CacheQueryOptions desiredQueryOptions_;
    std::size_t desiredQueryAttrIndex_ = 0;
    std::string desiredQueryReason_;
    std::string runningQueryReason_;
    std::future<AsyncVisibleQuery> visibleQueryFuture_;
    std::optional<AsyncVisibleQuery> stagedInteractivePanResult_;
    std::vector<std::future<void>> retiredVisiblePointTasks_;
    bool visibleQueryRequested_ = false;
    bool visibleQueryRunning_ = false;
    bool interactivePanQueryPending_ = false;
    bool lodQualityRefinementPending_ = false;
    double lodQualityRefinementDeadlineSeconds_ = 0.0;
    bool guardBandRefreshPending_ = false;
    bool hasGuardBandCoverage_ = false;
    Bounds2D guardBandCoverage_;
    bool lod0RefreshPending_ = false;
    bool asyncQueryEnabled_ = false;
    bool asyncSmokeTestSubmitted_ = false;
    bool gpuReadyRuntimeSetupPending_ = false;
    bool gpuReadyBackgroundResidencyEnabled_ = false;
    bool gpuReadyResidencyCompletePrinted_ = false;
    bool lastGpuDrivenViewport_ = false;
    bool benchmarkStarted_ = false;
    bool benchmarkActionPending_ = false;
    bool benchmarkCompleted_ = false;
    std::size_t benchmarkActionIndex_ = 0;
    std::uint64_t benchmarkActionRequestId_ = 0;
    std::chrono::steady_clock::time_point benchmarkActionBegin_{};
    std::vector<BenchmarkAction> benchmarkActions_;
    std::vector<double> benchmarkResponseSamples_;
    std::vector<GpuReadyPageView> gpuReadyBackgroundPages_;
    std::size_t gpuReadyBackgroundCursor_ = 0;
    std::int64_t hoveredTileId_ = std::numeric_limits<std::int64_t>::min();
    double hoveredX_ = 0.0;
    double hoveredY_ = 0.0;
    bool hoverValid_ = false;
    ProfileModeState profileModeState_ = ProfileModeState::Performance;
    std::vector<ProfileWorldPoint> profilePolyline_;
    ProfileWorldPoint profileCursorWorld_;
    bool profileCursorValid_ = false;
    bool profilePolylineCommitted_ = false;
    bool profileCommitEventPending_ = false;
    bool profileCommitDeferred_ = false;
    std::uint64_t profilePrepareRequestId_ = 0;
    std::uint64_t profileAnalysisRequestId_ = 0;
    bool profileViewportPreparationPending_ = false;
    double profileViewportPreparationDeadlineSeconds_ = 0.0;
    std::string profileViewportPreparationReason_;
    double profileCorridorWidth_ = 0.0;
    double lastProfileTotalMilliseconds_ = 0.0;
    double lastProfileIoMilliseconds_ = 0.0;
    double lastProfileFilterMilliseconds_ = 0.0;
    double lastProfileReadMegabytes_ = 0.0;
    std::uint64_t lastProfileTestedPoints_ = 0;
    std::uint64_t lastProfileMatchedPoints_ = 0;
    std::uint64_t lastProfileCandidateTiles_ = 0;
    std::filesystem::path lastProfileOutputPath_;
};

} // namespace gpv
