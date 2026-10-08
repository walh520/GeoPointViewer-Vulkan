// 应用主控实现：管理二维视口交互并按 viewport 从 Python cache 刷新点数据。
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "Psapi.lib")
#endif

#include "App.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpv
{
namespace
{

constexpr double kClickMoveThresholdPixels = 4.0;
constexpr double kPickRadiusPixels = 6.0;
constexpr double kHoverRadiusPixels = 8.0;
constexpr double kHoverPrintIntervalSeconds = 0.15;
constexpr double kHoverStaleSeconds = 1.2;
constexpr double kStatusPrintIntervalSeconds = 1.0;
constexpr double kBoxMinPixels = 6.0;
constexpr double kWheelQueryDebounceSeconds = 0.09;
constexpr double kGuardBandLinearScale = 1.5;
constexpr double kGlobalTrendMinimumCoverage = 0.95;
constexpr double kProfileHalfWidthPixels = 10.0;
constexpr std::size_t kMaximumProfileVertices = 64;
constexpr const char* kWindowTitle = "GeoPointViewer - Vulkan Cache Preview";

double Square(double value)
{
    return value * value;
}

double ElapsedMilliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool IsShiftDown(GLFWwindow* window)
{
    return glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
}

bool IsHighFrequencyInteractionReason(const char* reason)
{
    return
        std::strcmp(reason, "pan_interactive") == 0 ||
        std::strcmp(reason, "wheel_zoom_in") == 0 ||
        std::strcmp(reason, "wheel_zoom_out") == 0;
}

const char* ProfileModeStateName(ProfileModeState state)
{
    switch (state)
    {
    case ProfileModeState::Performance: return "performance";
    case ProfileModeState::Loading: return "profile_loading";
    case ProfileModeState::Ready: return "profile_ready";
    case ProfileModeState::Analyzing: return "profile_analyzing";
    default: return "unknown";
    }
}

float ClampOverviewAlphaForLod(float alpha, const std::string& activeLodName)
{
    if (activeLodName == "overview_only")
    {
        return std::max(alpha, 0.55f);
    }
    if (activeLodName == "bridge_mean")
    {
        return 0.0f;
    }
    if (activeLodName == "bridge_real")
    {
        return std::max(alpha, 0.275f);
    }
    if (activeLodName != "lod0_exact" && activeLodName != "lod0")
    {
        return std::min(std::max(alpha, 0.08f), 0.12f);
    }

    return 0.0f;
}

double BoundsWidth(const Bounds2D& bounds)
{
    return bounds.maxX - bounds.minX;
}

double BoundsHeight(const Bounds2D& bounds)
{
    return bounds.maxY - bounds.minY;
}

bool IsGlobalOverviewViewport(const Bounds2D& viewport, const Bounds2D& worldBounds)
{
    const double worldWidth = BoundsWidth(worldBounds);
    const double worldHeight = BoundsHeight(worldBounds);
    if (worldWidth <= 0.0 || worldHeight <= 0.0)
    {
        return false;
    }

    const double intersectionWidth = std::max(
        0.0,
        std::min(viewport.maxX, worldBounds.maxX) -
        std::max(viewport.minX, worldBounds.minX));
    const double intersectionHeight = std::max(
        0.0,
        std::min(viewport.maxY, worldBounds.maxY) -
        std::max(viewport.minY, worldBounds.minY));

    return
        intersectionWidth / worldWidth >= kGlobalTrendMinimumCoverage &&
        intersectionHeight / worldHeight >= kGlobalTrendMinimumCoverage;
}

Bounds2D ExpandBoundsClamped(
    const Bounds2D& bounds,
    const Bounds2D& limit,
    double linearScale)
{
    const double centerX = (bounds.minX + bounds.maxX) * 0.5;
    const double centerY = (bounds.minY + bounds.maxY) * 0.5;
    const double halfWidth = BoundsWidth(bounds) * linearScale * 0.5;
    const double halfHeight = BoundsHeight(bounds) * linearScale * 0.5;

    return Bounds2D
    {
        std::max(limit.minX, centerX - halfWidth),
        std::max(limit.minY, centerY - halfHeight),
        std::min(limit.maxX, centerX + halfWidth),
        std::min(limit.maxY, centerY + halfHeight),
    };
}

bool BoundsContains(const Bounds2D& outer, const Bounds2D& inner)
{
    return
        inner.minX >= outer.minX &&
        inner.minY >= outer.minY &&
        inner.maxX <= outer.maxX &&
        inner.maxY <= outer.maxY;
}

bool BoundsNearlyEqual(const Bounds2D& left, const Bounds2D& right)
{
    const double epsilon = 1e-9;
    return
        std::abs(left.minX - right.minX) <= epsilon &&
        std::abs(left.minY - right.minY) <= epsilon &&
        std::abs(left.maxX - right.maxX) <= epsilon &&
        std::abs(left.maxY - right.maxY) <= epsilon;
}

double BytesToMegabytes(std::uint64_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double GetProcessWorkingSetMegabytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(
        GetCurrentProcess(),
        reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
        sizeof(counters)) != FALSE)
    {
        return BytesToMegabytes(static_cast<std::uint64_t>(counters.WorkingSetSize));
    }
#endif

    return 0.0;
}

#ifdef _WIN32
std::uint64_t FileTimeValue(const FILETIME& fileTime)
{
    ULARGE_INTEGER value{};
    value.LowPart = fileTime.dwLowDateTime;
    value.HighPart = fileTime.dwHighDateTime;
    return value.QuadPart;
}
#endif

std::vector<CachePoint> FilterLod0Points(const std::vector<CachePoint>& points)
{
    std::vector<CachePoint> lod0Points;
    lod0Points.reserve(points.size());

    for (const CachePoint& point : points)
    {
        if (!point.proxy)
        {
            lod0Points.push_back(point);
        }
    }

    return lod0Points;
}

std::string RenderTierLabel(bool proxyMode, const std::string& activeLodName, int activeLodLevel)
{
    if (proxyMode)
    {
        return "proxy";
    }
    if (activeLodLevel == 0)
    {
        return "lod0_exact";
    }

    return activeLodName;
}

} // namespace

App::App(const AppConfig& config)
    : config_(config)
{
}

App::~App()
{
    Shutdown();
}

void App::Run(std::uint32_t maxFrames)
{
    Initialize();
    MainLoop(maxFrames);
    Shutdown();
}

void App::Initialize()
{
    if (initialized_)
    {
        return;
    }

    startupBegin_ = std::chrono::steady_clock::now();

    if (config_.performanceCsvEnabled)
    {
        const std::filesystem::path performancePath =
            config_.performanceCsvPath.empty()
            ? PerformanceRecorder::CreateDefaultPath()
            : config_.performanceCsvPath;
        performanceRecorder_.Open(performancePath);
        std::cout
            << "Performance CSV enabled: path="
            << std::filesystem::absolute(performanceRecorder_.Path()).string()
            << '\n';
    }

    if (config_.benchmark)
    {
        benchmarkActions_ =
        {
            { "benchmark_zoom_01", 0.8, 0.18, 0.24, true },
            { "benchmark_zoom_02", 0.8, 0.73, 0.31, true },
            { "benchmark_zoom_03", 1.25, 0.42, 0.76, true },
            { "benchmark_zoom_04", 0.8, 0.81, 0.66, true },
            { "benchmark_zoom_05", 1.25, 0.27, 0.52, true },
            { "benchmark_zoom_06", 0.8, 0.61, 0.18, true },
            { "benchmark_zoom_07", 1.25, 0.35, 0.39, true },
            { "benchmark_zoom_08", 0.8, 0.69, 0.84, true },
            { "benchmark_zoom_09", 1.25, 0.14, 0.71, true },
            { "benchmark_zoom_10", 1.25, 0.56, 0.47, true },
            { "benchmark_drag_01", 140.0, 80.0, 0.0, false },
            { "benchmark_drag_02", -210.0, 45.0, 0.0, false },
            { "benchmark_drag_03", 95.0, -130.0, 0.0, false },
            { "benchmark_drag_04", 180.0, 105.0, 0.0, false },
            { "benchmark_drag_05", -160.0, -90.0, 0.0, false },
        };
    }

    auto cacheOpenFuture = std::async(
        std::launch::async,
        [this]()
        {
            const auto begin = std::chrono::steady_clock::now();
            cache_.Open(config_.cacheDirectory);
            gpuReadyPageCache_.Open(config_.cacheDirectory, cache_.Metadata());
            return ElapsedMilliseconds(begin, std::chrono::steady_clock::now());
        });

    const auto windowInitBegin = std::chrono::steady_clock::now();
    InitializeWindow();
    startupWindowInitMilliseconds_ =
        ElapsedMilliseconds(windowInitBegin, std::chrono::steady_clock::now());

    const auto vulkanInitBegin = std::chrono::steady_clock::now();
    vulkan_.Initialize(window_, WindowWidth, WindowHeight, "GeoPointViewer", config_);
    startupVulkanInitMilliseconds_ =
        ElapsedMilliseconds(vulkanInitBegin, std::chrono::steady_clock::now());

    startupCacheOpenMilliseconds_ = cacheOpenFuture.get();
    viewport_.SetWorldBounds(cache_.Metadata().xyRange);
    if (config_.startInProfileMode)
    {
        profileAnalysis_.Start(cache_);
    }

    const auto elevationIt = std::find(
        cache_.Metadata().attrFields.begin(),
        cache_.Metadata().attrFields.end(),
        "elevation");
    activeAttrIndex_ = elevationIt == cache_.Metadata().attrFields.end()
        ? 0
        : static_cast<std::size_t>(std::distance(cache_.Metadata().attrFields.begin(), elevationIt));

    vulkan_.SetPointAttributeIndex(static_cast<std::uint32_t>(activeAttrIndex_));
    if (gpuReadyPageCache_.Available())
    {
        gpuReadyBackgroundPages_ = gpuReadyPageCache_.AllPagesCoarseFirst();
        gpuReadyRuntimeSetupPending_ = true;
    }
    else
    {
        vulkan_.UploadGpuPageColormap(cache_.PackedColormap());
    }
    std::cout
        << "Startup rendering uses overview before deferred LOD0 index preparation."
        << " Cache and Vulkan initialization overlapped.\n";

    const auto firstQueryBegin = std::chrono::steady_clock::now();
    if (cache_.OverviewPointCount() > 0)
    {
        lastActiveLodLevel_ = -1;
        lastActiveLodName_ = "overview_only";
        lastProxyMode_ = false;
        lastScreenSpaceLod_ = true;
        lastLod0PointCount_ = 0;
        lastProxyPointCount_ = 0;
        lastLod0IndexReady_ = cache_.IsLod0IndexReady();
        lastLod0IndexPreparing_ = !lastLod0IndexReady_;
        ShowPendingOverviewUnderlay("startup_low_latency");
        lastRenderedPointCount_ = lastOverviewPointCount_;
        startupRefinementPending_ = true;
    }
    else
    {
        RefreshVisiblePoints(true);
    }
    startupFirstQueryMilliseconds_ =
        ElapsedMilliseconds(firstQueryBegin, std::chrono::steady_clock::now());
    startupFirstUploadRequestMilliseconds_ = lastCpuBuildUploadRequestMilliseconds_;

    desiredQueryViewport_ = viewport_.Bounds();
    desiredQueryOptions_ = BuildCacheQueryOptions();
    desiredQueryAttrIndex_ = activeAttrIndex_;
    asyncQueryEnabled_ = true;
    initialized_ = true;
    if (config_.startInProfileMode)
    {
        EnterProfileMode();
    }
    startupInitializeMilliseconds_ =
        ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now());

    std::cout
        << "GLFW + Vulkan initialization completed.\n"
        << "Active attribute: " << cache_.Metadata().attrFields[activeAttrIndex_] << '\n'
        << "Renderer session: mode="
        << (config_.startInProfileMode ? "profile" : "performance")
        << ", runtime_switching=disabled.\n";
    if (config_.startInProfileMode)
    {
        std::cout
            << "Profile mode: viewport_pipeline=performance_shared, "
            << "analysis_source=lod0_exact, analysis_priority=background.\n";
    }
    else
    {
        std::cout
            << "Performance mode: frame_limit=off, gpu_viewport=full_catalog_when_resident.\n";
    }
}

void App::InitializeWindow()
{
    if (glfwInit() != GLFW_TRUE)
    {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    if (config_.startHidden)
    {
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }

    window_ = glfwCreateWindow(
        static_cast<int>(WindowWidth),
        static_cast<int>(WindowHeight),
        kWindowTitle,
        nullptr,
        nullptr);

    if (window_ == nullptr)
    {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window.");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, FramebufferResizeCallback);
    glfwSetWindowFocusCallback(window_, WindowFocusCallback);
    glfwSetMouseButtonCallback(window_, MouseButtonCallback);
    glfwSetCursorPosCallback(window_, CursorPositionCallback);
    glfwSetScrollCallback(window_, ScrollCallback);
}

void App::MainLoop(std::uint32_t maxFrames)
{
    std::uint32_t frameCount = 0;

    while (!glfwWindowShouldClose(window_))
    {
        const auto frameBegin = std::chrono::steady_clock::now();

        glfwPollEvents();
        ProcessInput();
        PollWheelViewportQuery();
        PollInteractivePanQuery();
        if (firstFrameRendered_ && gpuReadyRuntimeSetupPending_)
        {
            gpuReadyRuntimeSetupPending_ = false;
            gpuReadyFullResidencyReady_.store(false, std::memory_order_release);
            gpuReadyBackgroundResidencyEnabled_ = vulkan_.ConfigureGpuReadyPagePool(
                gpuReadyPageCache_.TotalPageCount(),
                gpuReadyPageCache_.PagePointCapacity(),
                gpuReadyPageCache_.PointRecordSize());
            if (gpuReadyBackgroundResidencyEnabled_)
            {
                vulkan_.RegisterGpuReadyPageCatalog(gpuReadyBackgroundPages_);
                std::cout
                    << "GPU-ready full-layer metadata registered after first present: pages="
                    << gpuReadyBackgroundPages_.size()
                    << '\n';
            }
            vulkan_.UploadGpuPageColormap(cache_.PackedColormap());
        }
        if (
            firstFrameRendered_ &&
            startupRefinementPending_ &&
            !gpuReadyRuntimeSetupPending_)
        {
            startupRefinementPending_ = false;
            RequestVisiblePointsAsync("startup_refine", LodBudgetMode::Interactive);
        }
        SubmitAsyncSmokeTestIfNeeded();
        PollVisiblePointsRequest();
        PollBackgroundIndexPreparation();
        if (config_.startInProfileMode)
        {
            profileAnalysis_.SetRenderBusy(
                !firstFrameRendered_ ||
                visibleQueryRunning_ ||
                visibleQueryRequested_ ||
                interactivePanQueryPending_ ||
                lodQualityRefinementPending_ ||
                canvasDragging_ ||
                boxSelecting_);
            PollProfileAnalysis();
        }
        WarmGpuReadyResidency();
        if (firstFrameRendered_)
        {
            vulkan_.WarmDeferredPipelines();
        }
        UpdateBenchmarkBeforeFrame();
        UpdateBoxSelectionFlash();
        const Bounds2D frameViewport = viewport_.Bounds();
        const std::vector<glm::vec2> profileOverlay = BuildProfileOverlayNdc();
        vulkan_.DrawFrame(
            framebufferResized_,
            viewport_.BuildViewProjection(renderOriginX_, renderOriginY_),
            frameViewport,
            renderOriginX_,
            renderOriginY_,
            boxSelection_,
            profileOverlay);
        firstFrameRendered_ = true;
        if (!firstGlobalFrameTimingPrinted_)
        {
            startupFirstGlobalFrameMilliseconds_ =
                ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now());
            PrintFirstGlobalFrameTiming();
            firstGlobalFrameTimingPrinted_ = true;
            cache_.StartLod0IndexLoadAsync();
        }
        const VulkanContext::FrameTimingStats& frameTiming = vulkan_.LastFrameTiming();
        accumulatedFrameMilliseconds_ += frameTiming.frameMilliseconds;
        maxFrameMilliseconds_ = std::max(maxFrameMilliseconds_, frameTiming.frameMilliseconds);
        accumulatedFenceWaitMilliseconds_ += frameTiming.fenceWaitMilliseconds;
        accumulatedVulkanUploadMilliseconds_ += frameTiming.uploadMilliseconds;
        accumulatedAcquireMilliseconds_ += frameTiming.acquireMilliseconds;
        accumulatedRecordMilliseconds_ += frameTiming.recordMilliseconds;
        accumulatedSubmitMilliseconds_ += frameTiming.submitMilliseconds;
        accumulatedPresentMilliseconds_ += frameTiming.presentMilliseconds;
        accumulatedPointBufferResizeMilliseconds_ += frameTiming.pointBufferResizeMilliseconds;
        accumulatedPointBufferResizeCount_ += frameTiming.pointBufferResizeCount;
        accumulatedVulkanUploadBytes_ += frameTiming.uploadBytes;
        cpuFrameSamples_.push_back(frameTiming.frameMilliseconds);
        if (frameTiming.gpuTimestampsValid)
        {
            gpuFrameSamples_.push_back(frameTiming.gpuFrameMilliseconds);
            accumulatedGpuCullMilliseconds_ += frameTiming.gpuCullMilliseconds;
            accumulatedGpuDrawMilliseconds_ += frameTiming.gpuDrawMilliseconds;
            ++accumulatedGpuTimingFrameCount_;
        }
        if (frameTiming.pipelineStatisticsValid)
        {
            accumulatedInputAssemblyVertices_ += frameTiming.inputAssemblyVertices;
            accumulatedVertexShaderInvocations_ += frameTiming.vertexShaderInvocations;
            accumulatedComputeShaderInvocations_ += frameTiming.computeShaderInvocations;
            ++accumulatedPipelineStatisticsFrameCount_;
        }
        if (frameTiming.visiblePageCountValid)
        {
            accumulatedVisiblePageCount_ += frameTiming.visiblePageCount;
            ++accumulatedVisiblePageFrameCount_;
        }
        ++statusFrameCounter_;
        UpdateBenchmarkAfterFrame();
        appWorkSamples_.push_back(ElapsedMilliseconds(
            frameBegin,
            std::chrono::steady_clock::now()));
        UpdateStatusPanel();

        if (!config_.benchmark &&
            maxFrames > 0 &&
            ++frameCount >= maxFrames &&
            !visibleQueryRunning_ &&
            !visibleQueryRequested_)
        {
            break;
        }

    }

    vulkan_.WaitIdle();
}

void App::ProcessInput()
{
    const bool keyEscDown = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    if (keyEscDown && !keyEscWasDown_)
    {
        if (config_.startInProfileMode)
        {
            if (!profilePolyline_.empty() ||
                profileModeState_ == ProfileModeState::Analyzing)
            {
                ClearProfileDraft();
            }
        }
        else if (boxSelecting_ || boxSelection_.active)
        {
            StopBoxSelection();
        }
        else
        {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
    }
    keyEscWasDown_ = keyEscDown;

    const bool key1Down = glfwGetKey(window_, GLFW_KEY_1) == GLFW_PRESS;
    const bool key2Down = glfwGetKey(window_, GLFW_KEY_2) == GLFW_PRESS;
    const bool keyCDown = glfwGetKey(window_, GLFW_KEY_C) == GLFW_PRESS;
    const bool keyFDown = glfwGetKey(window_, GLFW_KEY_F) == GLFW_PRESS;
    const bool keyRDown = glfwGetKey(window_, GLFW_KEY_R) == GLFW_PRESS;
    const bool keyEnterDown =
        glfwGetKey(window_, GLFW_KEY_ENTER) == GLFW_PRESS ||
        glfwGetKey(window_, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
    const bool keyBackspaceDown =
        glfwGetKey(window_, GLFW_KEY_BACKSPACE) == GLFW_PRESS;
    const bool keyPlusDown =
        glfwGetKey(window_, GLFW_KEY_EQUAL) == GLFW_PRESS ||
        glfwGetKey(window_, GLFW_KEY_KP_ADD) == GLFW_PRESS;
    const bool keyMinusDown =
        glfwGetKey(window_, GLFW_KEY_MINUS) == GLFW_PRESS ||
        glfwGetKey(window_, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS;

    if (key1Down && !key1WasDown_)
    {
        SetActiveAttributeByName("fold", 0);
    }
    if (key2Down && !key2WasDown_)
    {
        SetActiveAttributeByName("elevation", std::min<std::size_t>(1, cache_.Metadata().attrFields.size() - 1));
    }
    if (keyCDown && !keyCWasDown_)
    {
        CycleAttribute();
    }
    if (profileModeState_ != ProfileModeState::Performance &&
        keyEnterDown && !keyEnterWasDown_)
    {
        CommitProfileAnalysis();
    }
    if (profileModeState_ != ProfileModeState::Performance &&
        keyBackspaceDown && !keyBackspaceWasDown_)
    {
        UndoProfileVertex();
    }
    if ((keyFDown && !keyFWasDown_) || (keyRDown && !keyRWasDown_))
    {
        ResetToWorld();
    }
    if (keyPlusDown && !keyPlusWasDown_)
    {
        const Bounds2D bounds = viewport_.Bounds();
        viewport_.ZoomAt(
            (bounds.minX + bounds.maxX) * 0.5,
            (bounds.minY + bounds.maxY) * 0.5,
            0.5);
        RequestVisiblePointsAsync("keyboard_zoom_in");
        QueueProfileViewportPreparation("keyboard_zoom_in");
    }
    if (keyMinusDown && !keyMinusWasDown_)
    {
        const Bounds2D bounds = viewport_.Bounds();
        viewport_.ZoomAt(
            (bounds.minX + bounds.maxX) * 0.5,
            (bounds.minY + bounds.maxY) * 0.5,
            2.0);
        RequestVisiblePointsAsync("keyboard_zoom_out");
        QueueProfileViewportPreparation("keyboard_zoom_out");
    }

    key1WasDown_ = key1Down;
    key2WasDown_ = key2Down;
    keyCWasDown_ = keyCDown;
    keyFWasDown_ = keyFDown;
    keyRWasDown_ = keyRDown;
    keyPlusWasDown_ = keyPlusDown;
    keyMinusWasDown_ = keyMinusDown;
    keyEnterWasDown_ = keyEnterDown;
    keyBackspaceWasDown_ = keyBackspaceDown;

    const bool leftMouseDown =
        glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (profileModeState_ != ProfileModeState::Performance)
    {
        if (leftMouseDown && !leftMouseWasDown_)
        {
            glfwGetCursorPos(window_, &leftMousePressX_, &leftMousePressY_);
            lastCursorX_ = leftMousePressX_;
            lastCursorY_ = leftMousePressY_;
            canvasDragging_ = true;
            leftClickCandidate_ = true;
            leftDragMoved_ = false;
        }
        else if (!leftMouseDown && leftMouseWasDown_)
        {
            interactivePanQueryPending_ = false;
            if (leftClickCandidate_ && !leftDragMoved_)
            {
                double cursorX = 0.0;
                double cursorY = 0.0;
                glfwGetCursorPos(window_, &cursorX, &cursorY);
                AddProfileVertex(cursorX, cursorY);
            }
            else if (canvasDragging_ && leftDragMoved_)
            {
                canvasDragging_ = false;
                if (!CommitStagedInteractivePanResult())
                {
                    RequestVisiblePointsAsync("pan_release");
                }
                QueueProfileViewportPreparation("pan_release");
            }

            canvasDragging_ = false;
            leftClickCandidate_ = false;
            leftDragMoved_ = false;
        }
        if (profileCommitEventPending_)
        {
            profileCommitEventPending_ = false;
            CommitProfileAnalysis();
        }
        leftMouseWasDown_ = leftMouseDown;
        return;
    }

    const bool shiftDown = IsShiftDown(window_);

    if (leftMouseDown && !leftMouseWasDown_ && shiftDown)
    {
        canvasDragging_ = false;
        leftClickCandidate_ = false;
        leftDragMoved_ = false;
        StartBoxSelection();
    }
    else if (leftMouseDown && !leftMouseWasDown_ && !shiftDown)
    {
        glfwGetCursorPos(window_, &leftMousePressX_, &leftMousePressY_);
        lastCursorX_ = leftMousePressX_;
        lastCursorY_ = leftMousePressY_;
        canvasDragging_ = true;
        leftClickCandidate_ = true;
        leftDragMoved_ = false;
    }
    else if (!leftMouseDown && leftMouseWasDown_)
    {
        interactivePanQueryPending_ = false;
        if (boxSelecting_)
        {
            double cursorX = 0.0;
            double cursorY = 0.0;
            glfwGetCursorPos(window_, &cursorX, &cursorY);
            CommitBoxSelectionZoom(cursorX, cursorY);
        }
        else if (leftClickCandidate_ && !leftDragMoved_)
        {
            double cursorX = 0.0;
            double cursorY = 0.0;
            glfwGetCursorPos(window_, &cursorX, &cursorY);
            PickPointAtCursor(cursorX, cursorY);
        }
        else if (canvasDragging_ && leftDragMoved_)
        {
            canvasDragging_ = false;
            if (!CommitStagedInteractivePanResult())
            {
                RequestVisiblePointsAsync("pan_release");
            }
        }

        canvasDragging_ = false;
        leftClickCandidate_ = false;
        leftDragMoved_ = false;
    }

    leftMouseWasDown_ = leftMouseDown;
    profileCommitEventPending_ = false;
}

void App::ResetInputLatches()
{
    keyEscWasDown_ = false;
    key1WasDown_ = false;
    key2WasDown_ = false;
    keyCWasDown_ = false;
    keyFWasDown_ = false;
    keyRWasDown_ = false;
    keyPlusWasDown_ = false;
    keyMinusWasDown_ = false;
    keyEnterWasDown_ = false;
    keyBackspaceWasDown_ = false;
    leftMouseWasDown_ = false;
    leftClickCandidate_ = false;
    leftDragMoved_ = false;
    canvasDragging_ = false;
}

void App::RefreshVisiblePoints(bool rebuildQuery)
{
    if (rebuildQuery)
    {
        const auto queryBegin = std::chrono::steady_clock::now();
        CacheQueryResult query = QueryVisibleData(
            viewport_.Bounds(),
            activeAttrIndex_,
            BuildCacheQueryOptions());
        const auto queryEnd = std::chrono::steady_clock::now();
        lastQueryMilliseconds_ =
            std::chrono::duration<double, std::milli>(queryEnd - queryBegin).count();
        lastQueryTotalMilliseconds_ = lastQueryMilliseconds_;
        lastQueryReason_ = "sync";
        PreparedVisibleUpload upload = PrepareVisibleUpload(
            query,
            viewport_.Bounds(),
            activeAttrIndex_);
        ApplyVisibleQueryResult("sync", std::move(query), std::move(upload));
        return;
    }

    const bool useGpuProxy =
        lastScreenSpaceLod_ &&
        lastProxyPointCount_ > 0;
    const std::vector<CachePoint> cpuVisiblePoints = useGpuProxy
        ? FilterLod0Points(visiblePoints_)
        : visiblePoints_;
    lastOverviewAlpha_ = ClampOverviewAlphaForLod(
        cache_.ComputeOverviewAlpha(viewport_.Bounds()),
        lastActiveLodName_);
    std::vector<GpuPoint> gpuPoints =
        cache_.BuildOverviewGpuPoints(
            viewport_.Bounds(),
            activeAttrIndex_,
            lastOverviewAlpha_,
            renderOriginX_,
            renderOriginY_);
    lastOverviewPointCount_ = gpuPoints.size();

    if (useGpuProxy)
    {
        const std::vector<GpuPoint> lodGpuPoints =
            cache_.BuildGpuPoints(
                cpuVisiblePoints,
                activeAttrIndex_,
                renderOriginX_,
                renderOriginY_);
        gpuPoints.insert(gpuPoints.end(), lodGpuPoints.begin(), lodGpuPoints.end());
        vulkan_.UploadGpuProxyMask(cache_.BuildGpuProxyMask(visiblePoints_));
        vulkan_.ClearActiveTilePages();
    }
    else
    {
        const std::vector<GpuTilePage> tilePages = cache_.BuildGpuTilePages(
            cpuVisiblePoints,
            lastActiveLodName_,
            lastActiveLodLevel_,
            activeAttrIndex_);
        vulkan_.UpdateTilePagePool(tilePages);
    }
    vulkan_.SetGpuProxyDrawEnabled(useGpuProxy);
    lastGpuProxyDraw_ = useGpuProxy;
    vulkan_.UploadPointData(std::move(gpuPoints));
}

CacheQueryOptions App::BuildCacheQueryOptions(LodBudgetMode mode) const
{
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    if (window_ != nullptr)
    {
        glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
    }

    if (framebufferWidth <= 0 || framebufferHeight <= 0)
    {
        framebufferWidth = static_cast<int>(WindowWidth);
        framebufferHeight = static_cast<int>(WindowHeight);
    }

    CacheQueryOptions options;
    options.framebufferWidth = static_cast<std::uint32_t>(framebufferWidth);
    options.framebufferHeight = static_cast<std::uint32_t>(framebufferHeight);
    options.lodPointsPerPixel = mode == LodBudgetMode::Interactive
        ? InteractiveLodPointsPerPixel
        : StableLodPointsPerPixel;
    options.progressiveLoading = true;
    options.preferGlobalMean = IsGlobalOverviewViewport(
        viewport_.Bounds(),
        cache_.Metadata().xyRange);
    if (
        lastActiveLodName_ != "overview_only" &&
        lastActiveLodName_ != "overview_fallback")
    {
        options.currentLodName = lastActiveLodName_;
    }
    return options;
}

CacheQueryResult App::QueryVisibleData(
    const Bounds2D& viewport,
    std::size_t attrIndex,
    const CacheQueryOptions& options) const
{
    if (gpuReadyPageCache_.Available() && gpuReadyBackgroundResidencyEnabled_)
    {
        CacheQueryResult result = gpuReadyPageCache_.QueryViewport(
            viewport,
            options,
            !CanUseGpuDrivenViewport());
        result.lod0IndexReady = cache_.IsLod0IndexReady();
        result.lod0IndexPreparing = !result.lod0IndexReady;
        return result;
    }
    return cache_.QueryViewport(viewport, attrIndex, options);
}

bool App::CanUseGpuDrivenViewport() const
{
    return
        gpuReadyPageCache_.Available() &&
        gpuReadyBackgroundResidencyEnabled_ &&
        gpuReadyFullResidencyReady_.load(std::memory_order_acquire);
}

std::uint64_t App::ApplyGpuDrivenViewport(
    const Bounds2D& viewport,
    const CacheQueryOptions& options,
    const char* reason)
{
    const auto queryBegin = std::chrono::steady_clock::now();
    ++queryRequestCount_;
    desiredQueryViewport_ = viewport;
    desiredRenderViewport_ = viewport_.Bounds();
    desiredQueryOptions_ = options;
    desiredQueryAttrIndex_ = activeAttrIndex_;
    desiredQueryReason_ = reason;
    const std::uint64_t requestId =
        desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel) + 1;
    visibleQueryRequested_ = false;
    guardBandRefreshPending_ = false;
    hasGuardBandCoverage_ = false;

    CacheQueryResult query = gpuReadyPageCache_.QueryViewport(viewport, options, false);
    query.lod0IndexReady = cache_.IsLod0IndexReady();
    query.lod0IndexPreparing = !query.lod0IndexReady;
    const auto queryEnd = std::chrono::steady_clock::now();
    PreparedVisibleUpload upload = PrepareVisibleUpload(
        query,
        desiredRenderViewport_,
        activeAttrIndex_);
    const auto preparedEnd = std::chrono::steady_clock::now();

    if (std::strcmp(reason, "pan_interactive") == 0 && canvasDragging_)
    {
        AsyncVisibleQuery staged;
        staged.query = std::move(query);
        staged.upload = std::move(upload);
        staged.requestId = requestId;
        staged.viewport = viewport;
        staged.renderViewport = desiredRenderViewport_;
        staged.options = options;
        staged.attrIndex = activeAttrIndex_;
        staged.reason = reason;
        staged.submitTimeSeconds = glfwGetTime();
        staged.queryMilliseconds = ElapsedMilliseconds(queryBegin, queryEnd);
        StageInteractivePanResult(std::move(staged));
        return requestId;
    }

    lastQueryMilliseconds_ = ElapsedMilliseconds(queryBegin, queryEnd);
    lastQueryTotalMilliseconds_ = ElapsedMilliseconds(queryBegin, preparedEnd);
    lastQueryReason_ = reason;
    ApplyVisibleQueryResult(reason, std::move(query), std::move(upload));
    lastAppliedQueryRequestId_ = requestId;

    if (!IsHighFrequencyInteractionReason(reason))
    {
        std::cout
            << "GPU-driven query applied: id=" << requestId
            << ", reason=" << reason
            << ", query_ms=" << lastQueryMilliseconds_
            << ", total_ms=" << lastQueryTotalMilliseconds_
            << ", cpu_visible_page_list=no"
            << '\n';
    }
    return requestId;
}

void App::WarmGpuReadyResidency()
{
    if (!firstFrameRendered_ || !gpuReadyBackgroundResidencyEnabled_)
    {
        return;
    }

    if (!gpuReadyFullResidencyReady_.load(std::memory_order_acquire) &&
        vulkan_.GpuReadyFullResidencyReady())
    {
        gpuReadyFullResidencyReady_.store(true, std::memory_order_release);
        gpuReadyBackgroundCursor_ = gpuReadyBackgroundPages_.size();
        guardBandRefreshPending_ = false;
        hasGuardBandCoverage_ = false;
        std::cout
            << "GPU-driven viewport enabled: residency=full, "
            << "cpu_visible_page_list=no, culling=gpu_full_catalog.\n";
        CacheQueryOptions stableOptions = BuildCacheQueryOptions();
        stableOptions.currentLodName.clear();
        ApplyGpuDrivenViewport(
            viewport_.Bounds(),
            stableOptions,
            "gpu_full_residency_ready");
    }

    if (gpuReadyFullResidencyReady_.load(std::memory_order_acquire) ||
        visibleQueryRunning_ ||
        visibleQueryRequested_ ||
        gpuReadyBackgroundCursor_ >= gpuReadyBackgroundPages_.size())
    {
        return;
    }


    const bool fullResidency = vulkan_.GpuReadyFullResidencyEnabled();
    const VulkanContext::DeviceRuntimeStrategy& deviceStrategy =
        vulkan_.GetDeviceRuntimeStrategy();
    const std::size_t capacityPages = vulkan_.GpuReadyResidencyCapacityPages();
    const std::size_t targetPages = fullResidency
        ? gpuReadyBackgroundPages_.size()
        : std::min(
            gpuReadyBackgroundPages_.size(),
            static_cast<std::size_t>(
                static_cast<double>(capacityPages) *
                deviceStrategy.backgroundResidencyTargetRatio));
    const std::size_t residentPages = vulkan_.ResidentTilePageCount();
    if (residentPages >= targetPages)
    {
        gpuReadyBackgroundCursor_ = gpuReadyBackgroundPages_.size();
        if (!gpuReadyResidencyCompletePrinted_)
        {
            gpuReadyResidencyCompletePrinted_ = true;
            std::cout
                << "GPU-ready background residency target reached: mode="
                << (fullResidency ? "full" : "adaptive")
                << ", resident_pages=" << residentPages
                << ", capacity_pages=" << capacityPages
                << '\n';
        }
        return;
    }

    const VulkanContext::TilePageTransferStats transferStats =
        vulkan_.GetTilePageTransferStats();
    if (transferStats.busyStagingSlots >= 2)
    {
        return;
    }
    const VulkanContext::GpuMemoryBudgetStats memoryStats =
        vulkan_.GetGpuMemoryBudgetStats();
    if (memoryStats.available && memoryStats.deviceLocalBudgetBytes > 0 &&
        static_cast<double>(memoryStats.deviceLocalUsageBytes) /
            static_cast<double>(memoryStats.deviceLocalBudgetBytes) >
            deviceStrategy.memoryPressureStopRatio)
    {
        return;
    }

    const std::size_t basePreloadPages =
        std::max<std::size_t>(2, deviceStrategy.backgroundPreloadPages);
    const std::size_t minimumPreloadPages =
        std::max<std::size_t>(2, basePreloadPages / 4);
    const std::size_t maximumPreloadPages =
        std::min<std::size_t>(128, basePreloadPages * 2);
    std::size_t preloadPages = basePreloadPages;
    const VulkanContext::FrameTimingStats& frameTiming = vulkan_.LastFrameTiming();
    if (frameTiming.frameMilliseconds > 8.0)
    {
        preloadPages = minimumPreloadPages;
    }
    else if (frameTiming.frameMilliseconds > 5.0 || transferStats.busyStagingSlots > 0)
    {
        preloadPages = std::max(minimumPreloadPages, basePreloadPages / 2);
    }
    else if (frameTiming.frameMilliseconds > 0.0 && frameTiming.frameMilliseconds < 2.5)
    {
        preloadPages = maximumPreloadPages;
    }
    preloadPages = std::min(preloadPages, targetPages - residentPages);

    const std::size_t end = std::min(
        gpuReadyBackgroundCursor_ + preloadPages,
        gpuReadyBackgroundPages_.size());
    std::vector<GpuReadyPageView> batch(
        gpuReadyBackgroundPages_.begin() + gpuReadyBackgroundCursor_,
        gpuReadyBackgroundPages_.begin() + end);
    const std::optional<std::size_t> uploadedPages =
        vulkan_.PreloadGpuReadyPages(batch);
    if (!uploadedPages.has_value())
    {
        return;
    }
    gpuReadyBackgroundCursor_ = end;

    if (gpuReadyBackgroundCursor_ == gpuReadyBackgroundPages_.size() &&
        !gpuReadyResidencyCompletePrinted_)
    {
        gpuReadyResidencyCompletePrinted_ = true;
        std::cout
            << "GPU-ready background residency completed: pages="
            << gpuReadyBackgroundPages_.size()
            << ", resident_pages=" << vulkan_.ResidentTilePageCount()
            << '\n';
    }
}

void App::RebaseRenderOriginToViewport()
{
    const Bounds2D& bounds = viewport_.Bounds();
    renderOriginX_ = bounds.minX + (bounds.maxX - bounds.minX) * 0.5;
    renderOriginY_ = bounds.minY + (bounds.maxY - bounds.minY) * 0.5;
}

void App::ShowPendingOverviewUnderlay(const char* reason)
{
    const auto fallbackBegin = std::chrono::steady_clock::now();
    RebaseRenderOriginToViewport();
    std::vector<GpuPoint> overviewPoints =
        cache_.BuildOverviewGpuPoints(
            viewport_.Bounds(),
            activeAttrIndex_,
            1.0f,
            renderOriginX_,
            renderOriginY_);
    const std::size_t overviewPointCount = overviewPoints.size();

    vulkan_.UploadPointData(std::move(overviewPoints));

    lastOverviewPointCount_ = overviewPointCount;
    lastOverviewAlpha_ = 1.0f;
    pendingTargetUnderlayActive_ = true;
    lastCpuUploadPoints_ = overviewPointCount;
    lastPointUploadMegabytes_ =
        BytesToMegabytes(static_cast<std::uint64_t>(overviewPointCount * sizeof(GpuPoint)));
    lastCpuBuildUploadRequestMilliseconds_ =
        ElapsedMilliseconds(fallbackBegin, std::chrono::steady_clock::now());
    lastQueryReason_ = std::string(reason) + "_pending";

    std::cout
        << "Pending overview underlay displayed: reason=" << reason
        << ", points=" << overviewPointCount
        << ", retained_lod=" << lastActiveLodName_
        << ", target_query_pending=yes"
        << '\n';
}

std::uint64_t App::RequestVisiblePointsAsync(
    const char* reason,
    LodBudgetMode mode)
{
    if (std::strcmp(reason, "pan_interactive") != 0)
    {
        interactivePanQueryPending_ = false;
    }
    const bool wheelInteraction =
        std::strcmp(reason, "wheel_zoom_in") == 0 ||
        std::strcmp(reason, "wheel_zoom_out") == 0;
    if (!wheelInteraction && std::strcmp(reason, "wheel_lod_refine") != 0)
    {
        lodQualityRefinementPending_ = false;
    }

    const CacheQueryOptions options = BuildCacheQueryOptions(mode);

    if (CanUseGpuDrivenViewport())
    {
        return ApplyGpuDrivenViewport(viewport_.Bounds(), options, reason);
    }

    if (!asyncQueryEnabled_)
    {
        RefreshVisiblePoints(true);
        return 0;
    }

    if (
        std::strcmp(reason, "wheel_zoom_out") == 0 ||
        std::strcmp(reason, "keyboard_zoom_out") == 0 ||
        std::strcmp(reason, "fit_world") == 0)
    {
        if (!pendingTargetUnderlayActive_)
        {
            ShowPendingOverviewUnderlay(reason);
        }
    }

    const bool panInteraction =
        std::strcmp(reason, "pan_interactive") == 0 ||
        std::strcmp(reason, "pan_release") == 0;
    if (
        panInteraction &&
        hasGuardBandCoverage_ &&
        BoundsContains(guardBandCoverage_, viewport_.Bounds()))
    {
        guardBandRefreshPending_ = std::strcmp(reason, "pan_release") == 0;
        if (!IsHighFrequencyInteractionReason(reason))
        {
            std::cout
                << "Viewport reused guard band: reason=" << reason
                << ", foreground_query=no\n";
        }
        return lastAppliedQueryRequestId_;
    }

    hasGuardBandCoverage_ = false;
    guardBandRefreshPending_ = false;
    return QueueVisiblePointsAsync(viewport_.Bounds(), options, reason);
}

std::uint64_t App::QueueVisiblePointsAsync(
    const Bounds2D& viewport,
    const CacheQueryOptions& options,
    const char* reason)
{
    if (CanUseGpuDrivenViewport())
    {
        return ApplyGpuDrivenViewport(viewport, options, reason);
    }

    const bool supersedesExisting = visibleQueryRunning_ || visibleQueryRequested_;

    desiredQueryViewport_ = viewport;
    desiredRenderViewport_ = viewport_.Bounds();
    desiredQueryOptions_ = options;
    desiredQueryAttrIndex_ = activeAttrIndex_;
    desiredQueryReason_ = reason;
    const std::uint64_t requestId =
        desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel) + 1;
    visibleQueryRequested_ = true;
    ++queryRequestCount_;
    if (supersedesExisting)
    {
        ++querySupersededCount_;
    }

    if (!IsHighFrequencyInteractionReason(reason))
    {
        std::cout
            << "Async query requested: id=" << requestId
            << ", reason=" << desiredQueryReason_
            << ", running=" << (visibleQueryRunning_ ? "yes" : "no")
            << '\n';
    }

    StartNextVisiblePointsRequest();
    return requestId;
}

void App::ScheduleWheelViewportQuery(const char* reason)
{
    lodQualityRefinementPending_ = true;
    lodQualityRefinementDeadlineSeconds_ =
        glfwGetTime() + kWheelQueryDebounceSeconds;
    RequestVisiblePointsAsync(reason, LodBudgetMode::Interactive);
}

void App::PollWheelViewportQuery()
{
    const double now = glfwGetTime();
    if (lodQualityRefinementPending_ && now >= lodQualityRefinementDeadlineSeconds_)
    {
        lodQualityRefinementPending_ = false;
        if (CanUseGpuDrivenViewport())
        {
            CacheQueryOptions stableOptions = BuildCacheQueryOptions();
            stableOptions.currentLodName.clear();
            ApplyGpuDrivenViewport(
                viewport_.Bounds(),
                stableOptions,
                "wheel_lod_refine");
            return;
        }
        RequestVisiblePointsAsync("wheel_lod_refine", LodBudgetMode::Stable);
    }
}

void App::ScheduleInteractivePanQuery()
{
    if (interactivePanQueryPending_)
    {
        ++interactiveQueryCoalescedCount_;
    }
    interactivePanQueryPending_ = true;
}

void App::PollInteractivePanQuery()
{
    if (!interactivePanQueryPending_)
    {
        return;
    }

    interactivePanQueryPending_ = false;
    RequestVisiblePointsAsync("pan_interactive", LodBudgetMode::Stable);
}

void App::StartNextVisiblePointsRequest()
{
    if (!asyncQueryEnabled_ || visibleQueryRunning_ || !visibleQueryRequested_)
    {
        return;
    }

    const std::uint64_t requestId = desiredQueryRequestId_.load(std::memory_order_acquire);
    const Bounds2D viewport = desiredQueryViewport_;
    const Bounds2D renderViewport = desiredRenderViewport_;
    const CacheQueryOptions options = desiredQueryOptions_;
    const std::size_t attrIndex = desiredQueryAttrIndex_;
    const std::string reason = desiredQueryReason_;
    const double submitTimeSeconds = glfwGetTime();

    visibleQueryRequested_ = false;
    visibleQueryRunning_ = true;
    runningQueryRequestId_ = requestId;
    runningQueryReason_ = reason;

    CacheQueryOptions cancellableOptions = options;
    cancellableOptions.shouldCancel = [this, requestId]()
    {
        return desiredQueryRequestId_.load(std::memory_order_acquire) != requestId;
    };

    visibleQueryFuture_ = std::async(
        std::launch::async,
        [this, requestId, viewport, renderViewport, cancellableOptions, attrIndex, reason, submitTimeSeconds]()
        {
            const auto begin = std::chrono::steady_clock::now();
            AsyncVisibleQuery result;
            try
            {
                result.query = QueryVisibleData(viewport, attrIndex, cancellableOptions);
                const auto queryEnd = std::chrono::steady_clock::now();
                result.queryMilliseconds =
                    std::chrono::duration<double, std::milli>(queryEnd - begin).count();
                result.upload = PrepareVisibleUpload(
                    result.query,
                    renderViewport,
                    attrIndex,
                    cancellableOptions.shouldCancel);
            }
            catch (const CacheQueryCancelled&)
            {
                result.cancelled = true;
            }
            const auto end = std::chrono::steady_clock::now();

            result.requestId = requestId;
            result.viewport = viewport;
            result.renderViewport = renderViewport;
            result.options = cancellableOptions;
            result.attrIndex = attrIndex;
            result.reason = reason;
            result.submitTimeSeconds = submitTimeSeconds;
            if (result.cancelled)
            {
                result.queryMilliseconds =
                    std::chrono::duration<double, std::milli>(end - begin).count();
            }
            return result;
        });

    if (!IsHighFrequencyInteractionReason(reason.c_str()))
    {
        std::cout
            << "Async query started: id=" << requestId
            << ", reason=" << reason
            << '\n';
    }
}

void App::TrySchedulePendingLod0Refresh()
{
    if (!lod0RefreshPending_ || visibleQueryRunning_ || visibleQueryRequested_)
    {
        return;
    }

    if (lastLod0IndexReady_)
    {
        lod0RefreshPending_ = false;
        std::cout << "LOD0 refresh skipped: latest query already used the prepared index.\n";
        return;
    }

    lod0RefreshPending_ = false;
    RequestVisiblePointsAsync("lod0_index_ready");
}

void App::TryScheduleGuardBandRefresh()
{
    if (CanUseGpuDrivenViewport())
    {
        guardBandRefreshPending_ = false;
        hasGuardBandCoverage_ = false;
        return;
    }

    if (
        (config_.benchmark && benchmarkStarted_ && !benchmarkCompleted_) ||
        !guardBandRefreshPending_ ||
        visibleQueryRunning_ ||
        visibleQueryRequested_ ||
        interactivePanQueryPending_ ||
        canvasDragging_ ||
        lod0RefreshPending_)
    {
        return;
    }

    if (lastActiveLodName_ == "overview_only" || lastActiveLodName_ == "overview_fallback")
    {
        guardBandRefreshPending_ = false;
        return;
    }

    const Bounds2D visibleViewport = viewport_.Bounds();
    const Bounds2D expandedViewport = ExpandBoundsClamped(
        visibleViewport,
        cache_.Metadata().xyRange,
        kGuardBandLinearScale);
    if (BoundsNearlyEqual(visibleViewport, expandedViewport))
    {
        guardBandCoverage_ = visibleViewport;
        hasGuardBandCoverage_ = true;
        guardBandRefreshPending_ = false;
        return;
    }

    CacheQueryOptions options = BuildCacheQueryOptions();
    if (lastActiveLodName_ == "lod0_exact" || lastActiveLodName_ == "lod0")
    {
        options.forceExact = true;
    }
    else
    {
        options.preferredLodName = lastActiveLodName_;
    }

    guardBandRefreshPending_ = false;
    QueueVisiblePointsAsync(expandedViewport, options, "guard_band_1_5x");
}

void App::PollVisiblePointsRequest()
{
    if (!visibleQueryRunning_ || !visibleQueryFuture_.valid())
    {
        StartNextVisiblePointsRequest();
        TrySchedulePendingLod0Refresh();
        TryScheduleGuardBandRefresh();
        return;
    }

    if (visibleQueryFuture_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
    {
        return;
    }

    AsyncVisibleQuery result;
    try
    {
        result = visibleQueryFuture_.get();
    }
    catch (const std::exception& error)
    {
        visibleQueryRunning_ = false;
        std::cerr << "Async query failed: " << error.what() << '\n';
        StartNextVisiblePointsRequest();
        TrySchedulePendingLod0Refresh();
        TryScheduleGuardBandRefresh();
        return;
    }

    visibleQueryRunning_ = false;

    if (result.cancelled)
    {
        ++queryCancelledCount_;
        queryCancellationTotalMilliseconds_ += result.queryMilliseconds;
        queryCancellationMaxMilliseconds_ = std::max(
            queryCancellationMaxMilliseconds_,
            result.queryMilliseconds);
        if (!IsHighFrequencyInteractionReason(result.reason.c_str()))
        {
            std::cout
                << "Async query cancelled: id=" << result.requestId
                << ", reason=" << result.reason
                << ", elapsed_ms=" << result.queryMilliseconds
                << '\n';
        }
        StartNextVisiblePointsRequest();
        TrySchedulePendingLod0Refresh();
        TryScheduleGuardBandRefresh();
        return;
    }

    const std::uint64_t latestRequestId =
        desiredQueryRequestId_.load(std::memory_order_acquire);
    if (result.requestId == latestRequestId)
    {
        if (result.reason == "pan_interactive" && canvasDragging_)
        {
            StageInteractivePanResult(std::move(result));
            StartNextVisiblePointsRequest();
            TrySchedulePendingLod0Refresh();
            TryScheduleGuardBandRefresh();
            return;
        }

        const double totalMilliseconds =
            (glfwGetTime() - result.submitTimeSeconds) * 1000.0;

        lastQueryMilliseconds_ = result.queryMilliseconds;
        lastQueryTotalMilliseconds_ = totalMilliseconds;
        lastQueryReason_ = result.reason;
        ApplyVisibleQueryResult(
            result.reason.c_str(),
            std::move(result.query),
            std::move(result.upload));
        lastAppliedQueryRequestId_ = result.requestId;
        if (result.reason == "guard_band_1_5x")
        {
            guardBandCoverage_ = result.viewport;
            hasGuardBandCoverage_ = true;
            guardBandRefreshPending_ = false;
        }
        else
        {
            hasGuardBandCoverage_ = false;
            guardBandRefreshPending_ =
                lastActiveLodName_ != "overview_only" &&
                lastActiveLodName_ != "overview_fallback";
        }
        if (!IsHighFrequencyInteractionReason(result.reason.c_str()))
        {
            std::cout
                << "Async query applied: id=" << result.requestId
                << ", reason=" << result.reason
                << ", query_ms=" << result.queryMilliseconds
                << ", total_ms=" << totalMilliseconds
                << '\n';
        }
    }
    else
    {
        ++queryDiscardedCount_;
        if (!IsHighFrequencyInteractionReason(result.reason.c_str()))
        {
            std::cout
                << "Async query discarded: id=" << result.requestId
                << ", latest_id=" << latestRequestId
                << ", reason=" << result.reason
                << '\n';
        }
    }

    StartNextVisiblePointsRequest();
    TrySchedulePendingLod0Refresh();
    TryScheduleGuardBandRefresh();
}

void App::StageInteractivePanResult(AsyncVisibleQuery&& result)
{
    if (
        result.attrIndex != activeAttrIndex_ ||
        !BoundsNearlyEqual(result.viewport, viewport_.Bounds()))
    {
        return;
    }

    lastQueryMilliseconds_ = result.queryMilliseconds;
    lastQueryTotalMilliseconds_ = result.queryMilliseconds;
    lastQueryReason_ = "pan_interactive_staged";
    stagedInteractivePanResult_ = std::move(result);
    ++interactiveQueryStagedCount_;
}

bool App::CommitStagedInteractivePanResult()
{
    if (!stagedInteractivePanResult_.has_value())
    {
        return false;
    }

    AsyncVisibleQuery result = std::move(*stagedInteractivePanResult_);
    stagedInteractivePanResult_.reset();
    if (
        result.attrIndex != activeAttrIndex_ ||
        !BoundsNearlyEqual(result.viewport, viewport_.Bounds()))
    {
        return false;
    }

    visibleQueryRequested_ = false;
    const std::uint64_t releaseRequestId =
        desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel) + 1;
    desiredQueryViewport_ = viewport_.Bounds();
    desiredRenderViewport_ = viewport_.Bounds();
    desiredQueryOptions_ = result.options;
    desiredQueryAttrIndex_ = activeAttrIndex_;
    desiredQueryReason_ = "pan_release_staged";

    lastQueryMilliseconds_ = result.queryMilliseconds;
    lastQueryTotalMilliseconds_ = result.queryMilliseconds;
    lastQueryReason_ = "pan_release_staged";
    ApplyVisibleQueryResult(
        "pan_release_staged",
        std::move(result.query),
        std::move(result.upload));
    lastAppliedQueryRequestId_ = releaseRequestId;
    ++interactiveQueryCommittedCount_;

    std::cout
        << "Staged pan viewport committed: request_id=" << releaseRequestId
        << ", query_ms=" << lastQueryMilliseconds_
        << ", display_during_drag=no\n";
    return true;
}

void App::PollBackgroundIndexPreparation()
{
    if (firstFrameRendered_ && !backgroundReadyTimingPrinted_ && cache_.IsLod0IndexReady())
    {
        startupLod0IndexReadyMilliseconds_ =
            ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now());
        if (gpuReadyPageCache_.Available())
        {
            startupGpuTileMetadataUploadMilliseconds_ = 0.0;
            lastLod0IndexReady_ = true;
            lastLod0IndexPreparing_ = false;
            std::cout << "LOD0 exact lookup ready: GPU tile metadata upload skipped.\n";
            PrintBackgroundReadyTiming();
            backgroundReadyTimingPrinted_ = true;
            return;
        }
        const auto gpuMetadataBegin = std::chrono::steady_clock::now();
        vulkan_.UploadTileMetadata(cache_.BuildGpuTileMetadata(activeAttrIndex_));
        startupGpuTileMetadataUploadMilliseconds_ =
            ElapsedMilliseconds(gpuMetadataBegin, std::chrono::steady_clock::now());
        gpuTileMetadataUploaded_ = true;
        std::cout << "LOD0 index ready: GPU tile metadata uploaded.\n";
        PrintBackgroundReadyTiming();
        backgroundReadyTimingPrinted_ = true;

        if (!lastLod0IndexReady_)
        {
            lod0RefreshPending_ = true;
            if (visibleQueryRunning_ || visibleQueryRequested_)
            {
                std::cout << "LOD0 refresh deferred: user query has priority.\n";
            }
            TrySchedulePendingLod0Refresh();
        }
        else
        {
            std::cout << "LOD0 refresh skipped: current result already uses the prepared index.\n";
        }
        return;
    }

}

App::PreparedVisibleUpload App::PrepareVisibleUpload(
    const CacheQueryResult& query,
    const Bounds2D& renderViewport,
    std::size_t attrIndex,
    const std::function<bool()>& shouldCancel) const
{
    const auto buildBegin = std::chrono::steady_clock::now();
    PreparedVisibleUpload upload;
    upload.renderOriginX =
        renderViewport.minX + (renderViewport.maxX - renderViewport.minX) * 0.5;
    upload.renderOriginY =
        renderViewport.minY + (renderViewport.maxY - renderViewport.minY) * 0.5;
    upload.useGpuProxy =
        query.screenSpaceLod &&
        query.proxyPointCount > 0;

    if (query.gpuReadyFastPath && query.gpuDrivenViewport)
    {
        upload.gpuReadyFastPath = true;
        upload.cpuVisiblePointCount = 0;
        upload.overviewAlpha = 0.0f;
        upload.overviewPointCount = 0;
        upload.buildMilliseconds = ElapsedMilliseconds(
            buildBegin,
            std::chrono::steady_clock::now());
        return upload;
    }

    std::vector<CachePoint> filteredPoints;
    const std::vector<CachePoint>* cpuVisiblePoints = &query.points;
    if (upload.useGpuProxy)
    {
        filteredPoints = FilterLod0Points(query.points);
        cpuVisiblePoints = &filteredPoints;
    }
    upload.cpuVisiblePointCount = cpuVisiblePoints->size();
    upload.overviewAlpha = ClampOverviewAlphaForLod(
        cache_.ComputeOverviewAlpha(renderViewport),
        query.activeLodName);
    upload.gpuPoints = cache_.BuildOverviewGpuPoints(
        renderViewport,
        attrIndex,
        upload.overviewAlpha,
        upload.renderOriginX,
        upload.renderOriginY,
        shouldCancel);
    upload.overviewPointCount = upload.gpuPoints.size();

    if (query.gpuReadyFastPath)
    {
        upload.gpuReadyFastPath = true;
        upload.gpuReadyPages = query.gpuReadyPages;
        upload.cpuVisiblePointCount = static_cast<std::size_t>(std::min<std::uint64_t>(
            query.estimatedPoints,
            std::numeric_limits<std::size_t>::max()));
        upload.buildMilliseconds = ElapsedMilliseconds(
            buildBegin,
            std::chrono::steady_clock::now());
        return upload;
    }

    if (upload.useGpuProxy)
    {
        std::vector<GpuPoint> lodGpuPoints = cache_.BuildGpuPoints(
            *cpuVisiblePoints,
            attrIndex,
            upload.renderOriginX,
            upload.renderOriginY,
            shouldCancel);
        upload.gpuPoints.reserve(upload.gpuPoints.size() + lodGpuPoints.size());
        upload.gpuPoints.insert(
            upload.gpuPoints.end(),
            std::make_move_iterator(lodGpuPoints.begin()),
            std::make_move_iterator(lodGpuPoints.end()));
        upload.gpuProxyMask = cache_.BuildGpuProxyMask(query.points);
    }
    else
    {
        upload.tilePages = cache_.BuildGpuTilePages(
            *cpuVisiblePoints,
            query.activeLodName,
            query.activeLodLevel,
            attrIndex,
            shouldCancel);
    }
    upload.buildMilliseconds = ElapsedMilliseconds(
        buildBegin,
        std::chrono::steady_clock::now());
    return upload;
}

void App::RetireVisiblePoints(std::vector<CachePoint>&& points)
{
    for (auto it = retiredVisiblePointTasks_.begin(); it != retiredVisiblePointTasks_.end();)
    {
        if (it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            it->get();
            it = retiredVisiblePointTasks_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (points.empty())
    {
        return;
    }

    retiredVisiblePointTasks_.push_back(std::async(
        std::launch::async,
        [retiredPoints = std::move(points)]() mutable
        {
            std::vector<CachePoint>().swap(retiredPoints);
        }));
}

void App::WaitForRetiredVisiblePoints()
{
    for (std::future<void>& task : retiredVisiblePointTasks_)
    {
        task.wait();
        task.get();
    }
    retiredVisiblePointTasks_.clear();
}

void App::ApplyVisibleQueryResult(
    const char* reason,
    CacheQueryResult&& query,
    PreparedVisibleUpload&& upload)
{
    const auto applyBegin = std::chrono::steady_clock::now();
    const bool wasGpuDrivenViewport = lastGpuDrivenViewport_;

    std::vector<CachePoint> retiredPoints = std::move(visiblePoints_);
    visiblePoints_ = std::move(query.points);
    RetireVisiblePoints(std::move(retiredPoints));
    renderOriginX_ = upload.renderOriginX;
    renderOriginY_ = upload.renderOriginY;
    lastEstimatedPoints_ = query.estimatedPoints;
    lastLodPointsPerPixel_ = query.lodPointsPerPixel;
    lastEstimatedPointsPerPixel_ = query.estimatedPointsPerPixel;
    lastLodPointBudget_ = query.lodPointBudget;
    lastLod0PointCount_ = query.lod0PointCount;
    lastProxyPointCount_ = query.proxyPointCount;
    lastHitTileCount_ = query.hitTileCount;
    lastLod0TileCount_ = query.lod0TileCount;
    lastProxyTileCount_ = query.proxyTileCount;
    lastDeferredTileCount_ = query.deferredTileCount;
    lastActiveLodLevel_ = query.activeLodLevel;
    lastActiveLodName_ = query.activeLodName;
    lastProxyMode_ = query.proxyMode;
    lastScreenSpaceLod_ = query.screenSpaceLod;
    lastLod0IndexReady_ = query.lod0IndexReady;
    lastLod0IndexPreparing_ = query.lod0IndexPreparing;
    lastGpuDrivenViewport_ = query.gpuDrivenViewport;
    pendingTargetUnderlayActive_ = false;
    if (query.gpuDrivenViewport)
    {
        hasGuardBandCoverage_ = false;
        guardBandRefreshPending_ = false;
    }
    else if (std::strcmp(reason, "guard_band_1_5x") != 0)
    {
        hasGuardBandCoverage_ = false;
        guardBandRefreshPending_ =
            lastActiveLodName_ != "overview_only" &&
            lastActiveLodName_ != "overview_fallback";
    }

    lastOverviewAlpha_ = upload.overviewAlpha;
    lastOverviewPointCount_ = upload.overviewPointCount;
    lastGpuProxyDraw_ = upload.useGpuProxy;
    const std::size_t cpuVisiblePointCount = upload.cpuVisiblePointCount;
    std::size_t gpuPointCount = upload.gpuPoints.size();
    std::uint64_t gpuPointBytes =
        static_cast<std::uint64_t>(upload.gpuPoints.size()) * sizeof(GpuPoint);
    for (const GpuTilePage& page : upload.tilePages)
    {
        gpuPointCount += page.points.size();
        gpuPointBytes +=
            static_cast<std::uint64_t>(page.points.size()) * sizeof(GpuPoint);
    }
    for (const GpuReadyPageView& page : upload.gpuReadyPages)
    {
        gpuPointCount += page.pointCount;
        gpuPointBytes +=
            static_cast<std::uint64_t>(page.pointCount) * page.pointRecordSize;
    }
    const std::uint64_t renderedPointCount = query.gpuDrivenViewport
        ? query.estimatedPoints
        : static_cast<std::uint64_t>(gpuPointCount);
    const double buildMilliseconds = upload.buildMilliseconds;

    if (upload.gpuReadyFastPath)
    {
        if (query.gpuDrivenViewport)
        {
            vulkan_.ActivateGpuReadyViewport(query.activeLodLevel);
        }
        else
        {
            vulkan_.ActivateGpuReadyPages(upload.gpuReadyPages);
        }
    }
    else if (upload.useGpuProxy)
    {
        vulkan_.UploadGpuProxyMask(std::move(upload.gpuProxyMask));
        vulkan_.ClearActiveTilePages();
    }
    else if (!upload.tilePages.empty())
    {
        vulkan_.UpdateTilePagePool(upload.tilePages);
    }
    else
    {
        vulkan_.ClearActiveTilePages();
    }
    vulkan_.SetGpuProxyDrawEnabled(upload.useGpuProxy);
    if (!query.gpuDrivenViewport || !wasGpuDrivenViewport)
    {
        vulkan_.UploadPointData(std::move(upload.gpuPoints));
    }

    if (!upload.tilePages.empty())
    {
        retiredVisiblePointTasks_.push_back(std::async(
            std::launch::async,
            [retiredPages = std::move(upload.tilePages)]() mutable
            {
                std::vector<GpuTilePage>().swap(retiredPages);
            }));
    }

    const double mainApplyMilliseconds = ElapsedMilliseconds(
        applyBegin,
        std::chrono::steady_clock::now());
    lastMainApplyMilliseconds_ = mainApplyMilliseconds;
    lastCpuBuildUploadRequestMilliseconds_ = buildMilliseconds;
    lastCpuUploadPoints_ = gpuPointCount;
    lastRenderedPointCount_ = renderedPointCount;
    lastPointUploadMegabytes_ = BytesToMegabytes(gpuPointBytes);
    const std::string renderTier =
        RenderTierLabel(lastProxyMode_, lastActiveLodName_, lastActiveLodLevel_);
    const bool completeResult =
        lastDeferredTileCount_ == 0 &&
        !(lastLod0PointCount_ > 0 && lastProxyPointCount_ > 0) &&
        !lodQualityRefinementPending_;

    std::cout
        << "Viewport refresh: reason=" << reason
        << ", points=" << lastRenderedPointCount_
        << ", hit_tiles=" << lastHitTileCount_
        << ", estimated_points=" << lastEstimatedPoints_
        << ", lod_points_per_pixel=" << lastLodPointsPerPixel_
        << ", estimated_points_per_pixel=" << lastEstimatedPointsPerPixel_
        << ", lod_point_budget=" << lastLodPointBudget_
        << ", lod_budget_satisfied="
        << (lastEstimatedPoints_ <= lastLodPointBudget_ ? "yes" : "coarsest_fallback")
        << ", lod0_points=" << lastLod0PointCount_
        << ", proxy_points=" << lastProxyPointCount_
        << ", lod0_tiles=" << lastLod0TileCount_
        << ", proxy_tiles=" << lastProxyTileCount_
        << ", deferred_tiles=" << lastDeferredTileCount_
        << ", cpu_upload_points=" << cpuVisiblePointCount
        << ", overview_points=" << lastOverviewPointCount_
        << ", overview_alpha=" << lastOverviewAlpha_
        << ", proxy_mask_tiles=" << (lastGpuProxyDraw_ ? lastProxyTileCount_ : 0)
        << ", mode=" << (lastProxyMode_ ? "proxy" : lastActiveLodName_)
        << ", render_tier=" << renderTier
        << ", complete=" << (completeResult ? "yes" : "no")
        << ", lod0_index_ready=" << (lastLod0IndexReady_ ? "yes" : "no")
        << ", lod0_index_preparing=" << (lastLod0IndexPreparing_ ? "yes" : "no")
        << ", lod_level=" << lastActiveLodLevel_
        << ", screen_lod=" << (lastScreenSpaceLod_ ? "yes" : "no")
        << ", gpu_proxy=" << (lastGpuProxyDraw_ ? "yes" : "no")
        << ", gpu_viewport=" << (lastGpuDrivenViewport_ ? "full_catalog" : "cpu_list")
        << ", attr=" << cache_.Metadata().attrFields[activeAttrIndex_]
        << ", cpu_build_upload_request_ms=" << buildMilliseconds
        << ", main_apply_ms=" << mainApplyMilliseconds
        << ", point_upload_mb=" << lastPointUploadMegabytes_
        << '\n';
}

void App::SubmitAsyncSmokeTestIfNeeded()
{
    if (!config_.asyncSmokeTest || asyncSmokeTestSubmitted_ || !asyncQueryEnabled_)
    {
        return;
    }

    const Bounds2D bounds = viewport_.Bounds();
    viewport_.ZoomAt(
        (bounds.minX + bounds.maxX) * 0.5,
        (bounds.minY + bounds.maxY) * 0.5,
        0.5);

    asyncSmokeTestSubmitted_ = true;
    RequestVisiblePointsAsync("async_smoke_test");
}

void App::ResetToWorld()
{
    viewport_.FitWorld();
    StopBoxSelection();
    RequestVisiblePointsAsync("fit_world");
    QueueProfileViewportPreparation("fit_world");
    std::cout << "Viewport reset to full extent.\n";
}

void App::SetActiveAttribute(std::size_t attrIndex)
{
    if (attrIndex >= cache_.Metadata().attrFields.size())
    {
        return;
    }

    activeAttrIndex_ = attrIndex;
    vulkan_.SetPointAttributeIndex(static_cast<std::uint32_t>(activeAttrIndex_));
    if (gpuReadyPageCache_.Available())
    {
        std::cout << "Active attribute: " << cache_.Metadata().attrFields[activeAttrIndex_] << '\n';
        return;
    }
    if (cache_.IsLod0IndexReady())
    {
        vulkan_.UploadTileMetadata(cache_.BuildGpuTileMetadata(activeAttrIndex_));
        gpuTileMetadataUploaded_ = true;
    }
    else
    {
        gpuTileMetadataUploaded_ = false;
        std::cout << "Attribute changed before LOD0 index was ready; GPU tile metadata upload deferred.\n";
    }
    if (asyncQueryEnabled_)
    {
        RequestVisiblePointsAsync("attribute_change");
    }
    else
    {
        RefreshVisiblePoints(false);
    }
    std::cout << "Active attribute: " << cache_.Metadata().attrFields[activeAttrIndex_] << '\n';
}

void App::SetActiveAttributeByName(const char* name, std::size_t fallbackIndex)
{
    const auto& attrs = cache_.Metadata().attrFields;
    const auto it = std::find(attrs.begin(), attrs.end(), name);
    if (it != attrs.end())
    {
        SetActiveAttribute(static_cast<std::size_t>(std::distance(attrs.begin(), it)));
        return;
    }

    SetActiveAttribute(std::min(fallbackIndex, attrs.size() - 1));
}

void App::CycleAttribute()
{
    SetActiveAttribute((activeAttrIndex_ + 1) % cache_.Metadata().attrFields.size());
}

void App::EnterProfileMode()
{
    if (profileModeState_ != ProfileModeState::Performance)
    {
        return;
    }

    StopBoxSelection();
    canvasDragging_ = false;
    leftClickCandidate_ = false;
    interactivePanQueryPending_ = false;
    guardBandRefreshPending_ = false;
    visibleQueryRequested_ = false;
    desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel);
    profilePolyline_.clear();
    profilePolylineCommitted_ = false;
    profileCursorValid_ = false;
    profileCommitEventPending_ = false;
    profileCommitDeferred_ = false;
    profileModeState_ = ProfileModeState::Loading;
    profileViewportPreparationPending_ = false;
    profilePrepareRequestId_ = profileAnalysis_.PrepareViewport(viewport_.Bounds());
    std::cout
        << "Profile mode: loading, request_id=" << profilePrepareRequestId_
        << ", display=held_performance_frame"
        << ", source=lod0_exact"
        << '\n'
        << std::flush;
    UpdateProfileWindowTitle("Loading");
}

void App::QueueProfileViewportPreparation(const char* reason)
{
    if (profileModeState_ == ProfileModeState::Performance ||
        profileModeState_ == ProfileModeState::Analyzing ||
        !profilePolyline_.empty())
    {
        return;
    }

    profileAnalysis_.Cancel();
    profileModeState_ = ProfileModeState::Loading;
    profilePrepareRequestId_ = 0;
    profileViewportPreparationPending_ = true;
    profileViewportPreparationDeadlineSeconds_ = glfwGetTime() + 0.15;
    profileViewportPreparationReason_ = reason;
    std::cout
        << "Profile viewport reload scheduled: delay_ms=150"
        << ", reason=" << reason
        << '\n';
}

void App::PollProfileAnalysis()
{
    if (profileViewportPreparationPending_ &&
        glfwGetTime() >= profileViewportPreparationDeadlineSeconds_ &&
        !lodQualityRefinementPending_)
    {
        profileViewportPreparationPending_ = false;
        profilePrepareRequestId_ = profileAnalysis_.PrepareViewport(viewport_.Bounds());
        std::cout
            << "Profile viewport reload started: request_id=" << profilePrepareRequestId_
            << ", reason=" << profileViewportPreparationReason_
            << '\n';
    }

    const std::optional<ProfileResult> completed = profileAnalysis_.PollCompleted();
    if (!completed.has_value())
    {
        return;
    }

    const ProfileResult& result = completed.value();
    if (result.state == ProfileTaskState::Cancelled)
    {
        return;
    }
    if (result.state == ProfileTaskState::Failed)
    {
        std::cerr
            << "Profile analysis failed: request_id=" << result.requestId
            << ", error=" << result.error
            << '\n';
        if (result.requestId == profilePrepareRequestId_)
        {
            profileModeState_ = ProfileModeState::Ready;
            profileCommitDeferred_ = false;
            UpdateProfileWindowTitle("Failed");
        }
        else if (profileModeState_ != ProfileModeState::Performance)
        {
            profileModeState_ = ProfileModeState::Ready;
            UpdateProfileWindowTitle("Ready");
        }
        return;
    }

    if (result.state == ProfileTaskState::Prepared &&
        result.requestId == profilePrepareRequestId_ &&
        profileModeState_ != ProfileModeState::Performance)
    {
        profileModeState_ = ProfileModeState::Ready;
        std::cout
            << "Profile mode: ready, request_id=" << result.requestId
            << ", viewport_tiles=" << result.candidateTileCount
            << ", viewport_lod0_points=" << result.candidatePointCount
            << ", prepare_ms=" << result.totalMilliseconds
            << ", controls=left_add/right_or_enter_commit/backspace_undo/esc_clear/qt_mode_button_restart"
            << '\n'
            << std::flush;
        UpdateProfileWindowTitle("Ready");
        if (profileCommitDeferred_)
        {
            profileCommitDeferred_ = false;
            std::cout << "Profile deferred commit resumed after viewport preparation.\n"
                << std::flush;
            CommitProfileAnalysis();
        }
        return;
    }

    if (result.state == ProfileTaskState::Complete &&
        result.requestId == profileAnalysisRequestId_)
    {
        profileModeState_ = ProfileModeState::Ready;
        lastProfileTotalMilliseconds_ = result.totalMilliseconds;
        lastProfileIoMilliseconds_ = result.ioMilliseconds;
        lastProfileFilterMilliseconds_ = result.filterMilliseconds;
        lastProfileReadMegabytes_ = result.readMegabytes;
        lastProfileTestedPoints_ = result.testedPointCount;
        lastProfileMatchedPoints_ = result.matchedPointCount;
        lastProfileCandidateTiles_ = result.candidateTileCount;
        lastProfileOutputPath_ = std::filesystem::absolute(result.outputPath);
        const std::filesystem::path profileSamplePath =
            std::filesystem::absolute(result.samplePath);
        std::cout
            << "Profile result: id=" << result.requestId
            << ", path=\"" << lastProfileOutputPath_.string() << "\""
            << ", samples_path=\"" << profileSamplePath.string() << "\""
            << ", length=" << result.totalLength
            << ", corridor_width=" << result.corridorWidth
            << ", candidate_tiles=" << result.candidateTileCount
            << ", tested_points=" << result.testedPointCount
            << ", matched_points=" << result.matchedPointCount
            << ", sample_points=" << result.samplePointCount
            << ", bins=" << result.bins.size()
            << ", index_ms=" << result.indexMilliseconds
            << ", io_ms=" << result.ioMilliseconds
            << ", filter_ms=" << result.filterMilliseconds
            << ", write_ms=" << result.writeMilliseconds
            << ", total_ms=" << result.totalMilliseconds
            << ", read_mb=" << result.readMegabytes
            << '\n'
            << std::flush;
        UpdateProfileWindowTitle("Complete");
        if (performanceRecorder_.IsOpen())
        {
            PerformanceReportRow row{};
            row.rowType = "profile_analysis";
            row.name = "lod0_corridor";
            row.elapsedSeconds =
                ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now()) / 1000.0;
            row.renderTier = lastActiveLodName_;
            row.queryReason = "profile_commit";
            row.appMode = ProfileModeStateName(profileModeState_);
            row.profileState = ProfileTaskStateName(result.state);
            row.profileCorridorWidth = result.corridorWidth;
            row.profileTotalMilliseconds = result.totalMilliseconds;
            row.profileIoMilliseconds = result.ioMilliseconds;
            row.profileFilterMilliseconds = result.filterMilliseconds;
            row.profileReadMegabytes = result.readMegabytes;
            row.profileTestedPoints = result.testedPointCount;
            row.profileMatchedPoints = result.matchedPointCount;
            row.profileCandidateTiles = result.candidateTileCount;
            row.complete = true;
            performanceRecorder_.Write(row);
        }
    }
}

void App::AddProfileVertex(double cursorX, double cursorY)
{
    if (profileModeState_ == ProfileModeState::Loading)
    {
        std::cout << "Profile input ignored: viewport preparation is still running.\n";
        return;
    }
    if (profilePolylineCommitted_ || profileModeState_ == ProfileModeState::Analyzing)
    {
        profileAnalysis_.Cancel();
        profilePolyline_.clear();
        profilePolylineCommitted_ = false;
        profileModeState_ = ProfileModeState::Ready;
    }
    if (profilePolyline_.size() >= kMaximumProfileVertices)
    {
        std::cout << "Profile input ignored: maximum vertex count reached.\n";
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetWindowSize(window_, &width, &height);
    const Bounds2D world = viewport_.ScreenToWorld(cursorX, cursorY, width, height);
    profilePolyline_.push_back(ProfileWorldPoint{ world.minX, world.minY });
    profileCursorWorld_ = profilePolyline_.back();
    profileCursorValid_ = true;
    std::cout
        << "Profile vertex added: index=" << profilePolyline_.size() - 1
        << ", x=" << world.minX
        << ", y=" << world.minY
        << '\n'
        << std::flush;
    UpdateProfileWindowTitle("Ready");
}

void App::UndoProfileVertex()
{
    if (profilePolylineCommitted_ || profileModeState_ == ProfileModeState::Analyzing)
    {
        ClearProfileDraft();
        return;
    }
    if (!profilePolyline_.empty())
    {
        profilePolyline_.pop_back();
        std::cout << "Profile vertex removed: remaining=" << profilePolyline_.size() << '\n';
    }
}

void App::CommitProfileAnalysis()
{
    if (profileModeState_ == ProfileModeState::Loading)
    {
        profileCommitDeferred_ = true;
        std::cout
            << "Profile analysis deferred: viewport preparation is still running; commit is queued.\n"
            << std::flush;
        UpdateProfileWindowTitle("Commit Queued");
        return;
    }
    if (profileModeState_ == ProfileModeState::Analyzing)
    {
        std::cout << "Profile analysis is already running.\n" << std::flush;
        return;
    }
    if (profilePolyline_.size() < 2)
    {
        std::cout << "Profile analysis requires at least two vertices.\n" << std::flush;
        UpdateProfileWindowTitle("Need 2+ Vertices");
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    const Bounds2D bounds = viewport_.Bounds();
    const double worldPerPixelX = width > 0
        ? (bounds.maxX - bounds.minX) / static_cast<double>(width)
        : 0.0;
    const double worldPerPixelY = height > 0
        ? (bounds.maxY - bounds.minY) / static_cast<double>(height)
        : 0.0;
    profileCorridorWidth_ =
        std::max(worldPerPixelX, worldPerPixelY) * kProfileHalfWidthPixels * 2.0;
    const double halfWidth = profileCorridorWidth_ * 0.5;
    if (halfWidth <= 0.0)
    {
        std::cout << "Profile analysis rejected: invalid viewport scale.\n";
        return;
    }

    profileAnalysisRequestId_ = profileAnalysis_.Analyze(
        viewport_.Bounds(),
        profilePolyline_,
        halfWidth,
        2048);
    profilePolylineCommitted_ = true;
    profileModeState_ = ProfileModeState::Analyzing;
    profileCommitDeferred_ = false;
    std::cout
        << "Profile analysis requested: id=" << profileAnalysisRequestId_
        << ", vertices=" << profilePolyline_.size()
        << ", corridor_width=" << profileCorridorWidth_
        << ", bins=2048"
        << ", source=lod0_exact"
        << '\n'
        << std::flush;
    UpdateProfileWindowTitle("Analyzing");
}

void App::ClearProfileDraft()
{
    if (profileModeState_ == ProfileModeState::Analyzing)
    {
        profileAnalysis_.Cancel();
    }
    profilePolyline_.clear();
    profilePolylineCommitted_ = false;
    profileCommitEventPending_ = false;
    profileCommitDeferred_ = false;
    if (profileModeState_ != ProfileModeState::Loading &&
        profileModeState_ != ProfileModeState::Performance)
    {
        profileModeState_ = ProfileModeState::Ready;
    }
    std::cout << "Profile draft cleared.\n" << std::flush;
    UpdateProfileWindowTitle("Ready");
}

void App::UpdateProfileWindowTitle(const char* state)
{
    (void)state;
}

std::vector<glm::vec2> App::BuildProfileOverlayNdc() const
{
    std::vector<glm::vec2> vertices;
    if (profileModeState_ == ProfileModeState::Performance || profilePolyline_.empty())
    {
        return vertices;
    }

    const glm::mat4 viewProjection =
        viewport_.BuildViewProjection(renderOriginX_, renderOriginY_);
    const auto project = [this, &viewProjection](const ProfileWorldPoint& point)
    {
        const glm::vec4 clip = viewProjection * glm::vec4(
            static_cast<float>(point.x - renderOriginX_),
            static_cast<float>(point.y - renderOriginY_),
            0.0f,
            1.0f);
        return glm::vec2(clip.x / clip.w, clip.y / clip.w);
    };

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
    const Bounds2D bounds = viewport_.Bounds();
    const double worldPerPixelX = framebufferWidth > 0
        ? (bounds.maxX - bounds.minX) / static_cast<double>(framebufferWidth)
        : 0.0;
    const double worldPerPixelY = framebufferHeight > 0
        ? (bounds.maxY - bounds.minY) / static_cast<double>(framebufferHeight)
        : 0.0;
    const double halfWidth =
        std::max(worldPerPixelX, worldPerPixelY) * kProfileHalfWidthPixels;

    const auto appendSegment = [&vertices, &project, halfWidth](
        const ProfileWorldPoint& begin,
        const ProfileWorldPoint& end)
    {
        vertices.push_back(project(begin));
        vertices.push_back(project(end));

        const double deltaX = end.x - begin.x;
        const double deltaY = end.y - begin.y;
        const double length = std::hypot(deltaX, deltaY);
        if (length <= 0.0 || halfWidth <= 0.0)
        {
            return;
        }
        const double offsetX = -deltaY / length * halfWidth;
        const double offsetY = deltaX / length * halfWidth;
        vertices.push_back(project(ProfileWorldPoint{ begin.x + offsetX, begin.y + offsetY }));
        vertices.push_back(project(ProfileWorldPoint{ end.x + offsetX, end.y + offsetY }));
        vertices.push_back(project(ProfileWorldPoint{ begin.x - offsetX, begin.y - offsetY }));
        vertices.push_back(project(ProfileWorldPoint{ end.x - offsetX, end.y - offsetY }));
    };

    vertices.reserve(profilePolyline_.size() * 6);
    for (std::size_t index = 1; index < profilePolyline_.size(); ++index)
    {
        appendSegment(profilePolyline_[index - 1], profilePolyline_[index]);
    }
    if (!profilePolylineCommitted_ && profileCursorValid_)
    {
        appendSegment(profilePolyline_.back(), profileCursorWorld_);
    }
    return vertices;
}

void App::StartBoxSelection()
{
    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(window_, &cursorX, &cursorY);

    boxSelecting_ = true;
    boxSelection_.active = true;
    boxSelectionStartX_ = cursorX;
    boxSelectionStartY_ = cursorY;
    UpdateBoxSelection(cursorX, cursorY);
}

void App::StopBoxSelection()
{
    boxSelecting_ = false;
    boxSelection_.active = false;
}

void App::UpdateBoxSelection(double cursorX, double cursorY)
{
    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);

    if (windowWidth <= 0 || windowHeight <= 0)
    {
        boxSelection_.active = false;
        return;
    }

    const double minX = std::min(boxSelectionStartX_, cursorX);
    const double maxX = std::max(boxSelectionStartX_, cursorX);
    const double minY = std::min(boxSelectionStartY_, cursorY);
    const double maxY = std::max(boxSelectionStartY_, cursorY);

    boxSelection_.active = true;
    boxSelection_.minNdcX = static_cast<float>(minX / windowWidth * 2.0 - 1.0);
    boxSelection_.maxNdcX = static_cast<float>(maxX / windowWidth * 2.0 - 1.0);
    boxSelection_.minNdcY = static_cast<float>(minY / windowHeight * 2.0 - 1.0);
    boxSelection_.maxNdcY = static_cast<float>(maxY / windowHeight * 2.0 - 1.0);
}

void App::CommitBoxSelectionZoom(double cursorX, double cursorY)
{
    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);

    const double widthPixels = std::abs(cursorX - boxSelectionStartX_);
    const double heightPixels = std::abs(cursorY - boxSelectionStartY_);
    if (windowWidth > 0 && windowHeight > 0 &&
        widthPixels >= kBoxMinPixels && heightPixels >= kBoxMinPixels)
    {
        const Bounds2D start = viewport_.ScreenToWorld(
            boxSelectionStartX_,
            boxSelectionStartY_,
            windowWidth,
            windowHeight);
        const Bounds2D end = viewport_.ScreenToWorld(cursorX, cursorY, windowWidth, windowHeight);
        viewport_.ZoomToBox(Bounds2D
        {
            std::min(start.minX, end.minX),
            std::min(start.minY, end.minY),
            std::max(start.minX, end.minX),
            std::max(start.minY, end.minY),
        });
        RequestVisiblePointsAsync("box_zoom");
    }

    StopBoxSelection();
}

void App::UpdateBoxSelectionFlash()
{
    if (!boxSelection_.active)
    {
        boxSelection_.flashIntensity = 1.0f;
        return;
    }

    const double time = glfwGetTime();
    boxSelection_.flashIntensity = static_cast<float>(0.5 + 0.5 * std::sin(time * 8.0));
}

std::optional<CachePoint> App::FindExactPointAtCursor(
    double cursorX,
    double cursorY,
    double radiusPixels) const
{
    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);

    if (windowWidth <= 0 || windowHeight <= 0)
    {
        return std::nullopt;
    }
    if (!cache_.IsLod0IndexReady())
    {
        return std::nullopt;
    }

    const Bounds2D bounds = viewport_.Bounds();
    const double width = BoundsWidth(bounds);
    const double height = BoundsHeight(bounds);
    if (width <= 0.0 || height <= 0.0)
    {
        return std::nullopt;
    }

    const Bounds2D cursorWorld = viewport_.ScreenToWorld(cursorX, cursorY, windowWidth, windowHeight);
    const double worldRadiusX = radiusPixels / static_cast<double>(windowWidth) * width;
    const double worldRadiusY = radiusPixels / static_cast<double>(windowHeight) * height;
    const Bounds2D searchBox
    {
        cursorWorld.minX - worldRadiusX,
        cursorWorld.minY - worldRadiusY,
        cursorWorld.minX + worldRadiusX,
        cursorWorld.minY + worldRadiusY,
    };

    double nearestDistanceSquared = Square(radiusPixels);
    const CachePoint* renderedCandidate = nullptr;

    for (const CachePoint& point : visiblePoints_)
    {
        const double screenX = (point.x - bounds.minX) / width * windowWidth;
        const double screenY = (bounds.maxY - point.y) / height * windowHeight;
        const double distanceSquared = Square(screenX - cursorX) + Square(screenY - cursorY);

        if (distanceSquared <= nearestDistanceSquared)
        {
            nearestDistanceSquared = distanceSquared;
            renderedCandidate = &point;
        }
    }

    if (renderedCandidate != nullptr && renderedCandidate->hasSourceId)
    {
        const std::optional<CachePoint> original =
            cache_.ReadLod0PointBySourceId(renderedCandidate->sourceId);
        if (original.has_value())
        {
            return original;
        }
    }

    const std::vector<CachePoint> candidates = cache_.QueryExactPoints(searchBox);
    std::optional<CachePoint> pickedPoint;
    nearestDistanceSquared = Square(radiusPixels);

    for (const CachePoint& point : candidates)
    {
        const double screenX = (point.x - bounds.minX) / width * windowWidth;
        const double screenY = (bounds.maxY - point.y) / height * windowHeight;
        const double distanceSquared = Square(screenX - cursorX) + Square(screenY - cursorY);

        if (distanceSquared <= nearestDistanceSquared)
        {
            nearestDistanceSquared = distanceSquared;
            pickedPoint = point;
        }
    }

    return pickedPoint;
}

void App::PrintPointInfo(const char* prefix, const CachePoint& point) const
{
    std::cout
        << prefix
        << ": tile_id=" << point.tileId
        << ", x=" << point.x
        << ", y=" << point.y;

    if (point.hasSourceId)
    {
        std::cout << ", source_id=" << point.sourceId;
    }
    if (point.hasCellId)
    {
        std::cout << ", cell_id=" << point.cellId;
    }

    for (std::size_t index = 0; index < cache_.Metadata().attrFields.size(); ++index)
    {
        std::cout
            << ", " << cache_.Metadata().attrFields[index]
            << "=" << point.attrs[index];
    }

    std::cout << '\n';
}

void App::PickPointAtCursor(double cursorX, double cursorY) const
{
    const std::optional<CachePoint> pickedPoint =
        FindExactPointAtCursor(cursorX, cursorY, kPickRadiusPixels);
    if (!pickedPoint.has_value())
    {
        std::cout << "Invalid operation.\n";
        return;
    }

    PrintPointInfo("Point", *pickedPoint);
}

void App::UpdateHoverAtCursor(double cursorX, double cursorY)
{
    const double now = glfwGetTime();
    if (now - lastHoverPrintTime_ < kHoverPrintIntervalSeconds)
    {
        return;
    }
    lastHoverPrintTime_ = now;

    const std::optional<CachePoint> hoveredPoint =
        FindExactPointAtCursor(cursorX, cursorY, kHoverRadiusPixels);
    if (!hoveredPoint.has_value())
    {
        if (hoverValid_ && now - lastHoverHitTime_ > kHoverStaleSeconds)
        {
            std::cout << "Hover: none\n";
            hoverValid_ = false;
            hoveredTileId_ = std::numeric_limits<std::int64_t>::min();
        }
        return;
    }

    if (hoverValid_ &&
        hoveredTileId_ == hoveredPoint->tileId &&
        hoveredX_ == hoveredPoint->x &&
        hoveredY_ == hoveredPoint->y)
    {
        lastHoverHitTime_ = now;
        return;
    }

    hoverValid_ = true;
    lastHoverHitTime_ = now;
    hoveredTileId_ = hoveredPoint->tileId;
    hoveredX_ = hoveredPoint->x;
    hoveredY_ = hoveredPoint->y;
    PrintPointInfo("Hover point", *hoveredPoint);
}

void App::UpdateBenchmarkBeforeFrame()
{
    if (!config_.benchmark || benchmarkCompleted_ || benchmarkActionPending_)
    {
        return;
    }

    if (!benchmarkStarted_)
    {
        const VulkanContext::TilePageTransferStats transferStats =
            vulkan_.GetTilePageTransferStats();
        const bool residencyReady =
            !gpuReadyBackgroundResidencyEnabled_ ||
            (gpuReadyBackgroundCursor_ >= gpuReadyBackgroundPages_.size() &&
             transferStats.completedValue >= transferStats.submittedValue);
        const bool queryIdle =
            !visibleQueryRunning_ &&
            !visibleQueryRequested_ &&
            !interactivePanQueryPending_ &&
            !lodQualityRefinementPending_;
        if (!firstFrameRendered_ || !residencyReady || !queryIdle)
        {
            return;
        }

        benchmarkStarted_ = true;
        guardBandRefreshPending_ = false;
        hasGuardBandCoverage_ = false;
        std::cout
            << "Performance benchmark started: zoom_actions=10, drag_actions=5"
            << ", gpu_residency_ready=" << (residencyReady ? "yes" : "no")
            << '\n';
    }

    if (benchmarkActionIndex_ >= benchmarkActions_.size())
    {
        FinishBenchmark();
        return;
    }

    const BenchmarkAction& action = benchmarkActions_[benchmarkActionIndex_];
    if (action.zoom)
    {
        const Bounds2D bounds = viewport_.Bounds();
        viewport_.ZoomAt(
            bounds.minX + BoundsWidth(bounds) * action.value1,
            bounds.minY + BoundsHeight(bounds) * action.value2,
            action.value0);
    }
    else
    {
        int framebufferWidth = 0;
        int framebufferHeight = 0;
        glfwGetFramebufferSize(window_, &framebufferWidth, &framebufferHeight);
        viewport_.PanPixels(
            action.value0,
            action.value1,
            std::max(framebufferWidth, 1),
            std::max(framebufferHeight, 1));
    }

    guardBandRefreshPending_ = false;
    hasGuardBandCoverage_ = false;
    benchmarkActionBegin_ = std::chrono::steady_clock::now();
    benchmarkActionRequestId_ = QueueVisiblePointsAsync(
        viewport_.Bounds(),
        BuildCacheQueryOptions(),
        action.name.c_str());
    benchmarkActionPending_ = true;
}

void App::UpdateBenchmarkAfterFrame()
{
    if (!config_.benchmark || !benchmarkActionPending_ || benchmarkCompleted_)
    {
        return;
    }
    if (lastAppliedQueryRequestId_ < benchmarkActionRequestId_)
    {
        return;
    }

    const BenchmarkAction& action = benchmarkActions_[benchmarkActionIndex_];
    const double responseMilliseconds = ElapsedMilliseconds(
        benchmarkActionBegin_,
        std::chrono::steady_clock::now());
    benchmarkResponseSamples_.push_back(responseMilliseconds);

    std::cout
        << "Benchmark action completed: index=" << benchmarkActionIndex_ + 1
        << ", name=" << action.name
        << ", response_ms=" << responseMilliseconds
        << ", render_tier="
        << RenderTierLabel(lastProxyMode_, lastActiveLodName_, lastActiveLodLevel_)
        << ", rendered_points=" << lastRenderedPointCount_
        << '\n';

    if (performanceRecorder_.IsOpen())
    {
        PerformanceReportRow row{};
        row.rowType = "benchmark_action";
        row.name = action.name;
        row.elapsedSeconds =
            ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now()) / 1000.0;
        row.responseMilliseconds = responseMilliseconds;
        row.response = PerformanceRecorder::ComputeDistribution({ responseMilliseconds });
        row.renderTier = RenderTierLabel(
            lastProxyMode_,
            lastActiveLodName_,
            lastActiveLodLevel_);
        row.queryReason = lastQueryReason_;
        row.queryMilliseconds = lastQueryMilliseconds_;
        row.queryTotalMilliseconds = lastQueryTotalMilliseconds_;
        row.lodPointsPerPixel = lastLodPointsPerPixel_;
        row.estimatedPointsPerPixel = lastEstimatedPointsPerPixel_;
        row.lodPointBudget = lastLodPointBudget_;
        row.cpuBuildMilliseconds = lastCpuBuildUploadRequestMilliseconds_;
        row.mainApplyMilliseconds = lastMainApplyMilliseconds_;
        row.renderedPoints = lastRenderedPointCount_;
        row.hitTiles = lastHitTileCount_;
        row.activePages = static_cast<std::uint32_t>(vulkan_.ActiveTilePageCount());
        row.residentPages = static_cast<std::uint32_t>(vulkan_.ResidentTilePageCount());
        row.processMemoryMegabytes = GetProcessWorkingSetMegabytes();
        const VulkanContext::GpuMemoryBudgetStats memoryStats =
            vulkan_.GetGpuMemoryBudgetStats();
        if (memoryStats.available)
        {
            row.gpuMemoryUsageMegabytes = BytesToMegabytes(
                memoryStats.deviceLocalUsageBytes);
            row.gpuMemoryBudgetMegabytes = BytesToMegabytes(
                memoryStats.deviceLocalBudgetBytes);
        }
        row.gpuBufferUsedMegabytes = BytesToMegabytes(vulkan_.PointBufferUsedBytes());
        row.gpuBufferCapacityMegabytes = BytesToMegabytes(
            vulkan_.PointBufferCapacityBytes());
        row.complete = true;
        performanceRecorder_.Write(row);
    }

    ++benchmarkActionIndex_;
    benchmarkActionPending_ = false;
    benchmarkActionRequestId_ = 0;
    guardBandRefreshPending_ = false;
}

void App::FinishBenchmark()
{
    if (benchmarkCompleted_)
    {
        return;
    }

    benchmarkCompleted_ = true;
    const DistributionStats responseStats =
        PerformanceRecorder::ComputeDistribution(benchmarkResponseSamples_);
    std::cout << std::fixed << std::setprecision(2)
        << "Performance benchmark completed: actions=" << responseStats.sampleCount
        << ", response_avg_ms=" << responseStats.average
        << ", response_p50_ms=" << responseStats.p50
        << ", response_p95_ms=" << responseStats.p95
        << ", response_p99_ms=" << responseStats.p99
        << ", response_max_ms=" << responseStats.maximum
        << '\n';

    if (performanceRecorder_.IsOpen())
    {
        PerformanceReportRow row{};
        row.rowType = "benchmark_summary";
        row.name = "10_zoom_5_drag";
        row.elapsedSeconds =
            ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now()) / 1000.0;
        row.response = responseStats;
        row.renderTier = RenderTierLabel(
            lastProxyMode_,
            lastActiveLodName_,
            lastActiveLodLevel_);
        row.lodPointsPerPixel = lastLodPointsPerPixel_;
        row.estimatedPointsPerPixel = lastEstimatedPointsPerPixel_;
        row.lodPointBudget = lastLodPointBudget_;
        row.renderedPoints = lastRenderedPointCount_;
        row.complete = responseStats.sampleCount == benchmarkActions_.size();
        performanceRecorder_.Write(row);
    }

    glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

void App::UpdateStatusPanel()
{
    const double now = glfwGetTime();
    bool firstStatusLine = false;
    if (lastStatusPrintTime_ <= 0.0)
    {
        firstStatusLine = true;
    }

    const double elapsed = now - lastStatusPrintTime_;
    if (!firstStatusLine && elapsed < kStatusPrintIntervalSeconds)
    {
        return;
    }

    currentFps_ = firstStatusLine
        ? 0.0
        : static_cast<double>(statusFrameCounter_) / elapsed;
    const std::uint32_t framesInWindow = std::max<std::uint32_t>(statusFrameCounter_, 1);
    const double averageFrameMilliseconds =
        accumulatedFrameMilliseconds_ / static_cast<double>(framesInWindow);
    const double averageFenceWaitMilliseconds =
        accumulatedFenceWaitMilliseconds_ / static_cast<double>(framesInWindow);
    const double averageVulkanUploadMilliseconds =
        accumulatedVulkanUploadMilliseconds_ / static_cast<double>(framesInWindow);
    const double averageAcquireMilliseconds =
        accumulatedAcquireMilliseconds_ / static_cast<double>(framesInWindow);
    const double averageRecordMilliseconds =
        accumulatedRecordMilliseconds_ / static_cast<double>(framesInWindow);
    const double averageSubmitMilliseconds =
        accumulatedSubmitMilliseconds_ / static_cast<double>(framesInWindow);
    const double averagePresentMilliseconds =
        accumulatedPresentMilliseconds_ / static_cast<double>(framesInWindow);
    const double pointBufferResizeMilliseconds = accumulatedPointBufferResizeMilliseconds_;
    const std::uint32_t pointBufferResizeCount = accumulatedPointBufferResizeCount_;
    const double frameMaxMilliseconds = maxFrameMilliseconds_;
    const double vulkanUploadMegabytes = BytesToMegabytes(accumulatedVulkanUploadBytes_);
    const DistributionStats cpuFrameStats =
        PerformanceRecorder::ComputeDistribution(cpuFrameSamples_);
    const DistributionStats gpuFrameStats =
        PerformanceRecorder::ComputeDistribution(gpuFrameSamples_);
    const DistributionStats appWorkStats =
        PerformanceRecorder::ComputeDistribution(appWorkSamples_);
    const double averageGpuCullMilliseconds = accumulatedGpuTimingFrameCount_ == 0
        ? 0.0
        : accumulatedGpuCullMilliseconds_ /
            static_cast<double>(accumulatedGpuTimingFrameCount_);
    const double averageGpuDrawMilliseconds = accumulatedGpuTimingFrameCount_ == 0
        ? 0.0
        : accumulatedGpuDrawMilliseconds_ /
            static_cast<double>(accumulatedGpuTimingFrameCount_);
    const std::uint64_t averageInputAssemblyVertices =
        accumulatedPipelineStatisticsFrameCount_ == 0
        ? 0
        : accumulatedInputAssemblyVertices_ / accumulatedPipelineStatisticsFrameCount_;
    const std::uint64_t averageVertexShaderInvocations =
        accumulatedPipelineStatisticsFrameCount_ == 0
        ? 0
        : accumulatedVertexShaderInvocations_ / accumulatedPipelineStatisticsFrameCount_;
    const std::uint64_t averageComputeShaderInvocations =
        accumulatedPipelineStatisticsFrameCount_ == 0
        ? 0
        : accumulatedComputeShaderInvocations_ / accumulatedPipelineStatisticsFrameCount_;
    const std::uint32_t averageVisiblePageCount =
        accumulatedVisiblePageFrameCount_ == 0
        ? 0
        : static_cast<std::uint32_t>(
            accumulatedVisiblePageCount_ / accumulatedVisiblePageFrameCount_);

    statusFrameCounter_ = 0;
    accumulatedFrameMilliseconds_ = 0.0;
    maxFrameMilliseconds_ = 0.0;
    accumulatedFenceWaitMilliseconds_ = 0.0;
    accumulatedVulkanUploadMilliseconds_ = 0.0;
    accumulatedAcquireMilliseconds_ = 0.0;
    accumulatedRecordMilliseconds_ = 0.0;
    accumulatedSubmitMilliseconds_ = 0.0;
    accumulatedPresentMilliseconds_ = 0.0;
    accumulatedPointBufferResizeMilliseconds_ = 0.0;
    accumulatedPointBufferResizeCount_ = 0;
    accumulatedVulkanUploadBytes_ = 0;
    accumulatedGpuCullMilliseconds_ = 0.0;
    accumulatedGpuDrawMilliseconds_ = 0.0;
    accumulatedGpuTimingFrameCount_ = 0;
    accumulatedPipelineStatisticsFrameCount_ = 0;
    accumulatedVisiblePageFrameCount_ = 0;
    accumulatedInputAssemblyVertices_ = 0;
    accumulatedVertexShaderInvocations_ = 0;
    accumulatedComputeShaderInvocations_ = 0;
    accumulatedVisiblePageCount_ = 0;
    cpuFrameSamples_.clear();
    gpuFrameSamples_.clear();
    appWorkSamples_.clear();
    lastStatusPrintTime_ = now;

    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window_, &windowWidth, &windowHeight);

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(window_, &cursorX, &cursorY);

    Bounds2D worldPoint{};
    if (windowWidth > 0 && windowHeight > 0)
    {
        worldPoint = viewport_.ScreenToWorld(cursorX, cursorY, windowWidth, windowHeight);
    }

    const TileCacheStats cacheStats = cache_.GetTileCacheStats();
    const double processMemoryMb = GetProcessWorkingSetMegabytes();
    double processCpuPercent = 0.0;
    double processIoReadMegabytesPerSecond = 0.0;
    double processIoWriteMegabytesPerSecond = 0.0;
#ifdef _WIN32
    FILETIME creationTime{};
    FILETIME exitTime{};
    FILETIME kernelTime{};
    FILETIME userTime{};
    IO_COUNTERS ioCounters{};
    const bool processTimesReady = GetProcessTimes(
        GetCurrentProcess(),
        &creationTime,
        &exitTime,
        &kernelTime,
        &userTime) != FALSE;
    const bool processIoReady = GetProcessIoCounters(
        GetCurrentProcess(),
        &ioCounters) != FALSE;
    const std::uint64_t processCpuTime100Nanoseconds = processTimesReady
        ? FileTimeValue(kernelTime) + FileTimeValue(userTime)
        : 0;
    const std::uint64_t processIoReadBytes = processIoReady
        ? ioCounters.ReadTransferCount
        : 0;
    const std::uint64_t processIoWriteBytes = processIoReady
        ? ioCounters.WriteTransferCount
        : 0;
    if (processResourceSampleInitialized_ &&
        processTimesReady &&
        processIoReady &&
        elapsed > 0.0)
    {
        const DWORD processorCount = std::max<DWORD>(
            GetActiveProcessorCount(ALL_PROCESSOR_GROUPS),
            1);
        const std::uint64_t cpuDelta =
            processCpuTime100Nanoseconds - lastProcessCpuTime100Nanoseconds_;
        processCpuPercent =
            static_cast<double>(cpuDelta) /
            (elapsed * 10'000'000.0 * static_cast<double>(processorCount)) *
            100.0;
        processIoReadMegabytesPerSecond = BytesToMegabytes(
            processIoReadBytes - lastProcessIoReadBytes_) / elapsed;
        processIoWriteMegabytesPerSecond = BytesToMegabytes(
            processIoWriteBytes - lastProcessIoWriteBytes_) / elapsed;
    }
    lastProcessCpuTime100Nanoseconds_ = processCpuTime100Nanoseconds;
    lastProcessIoReadBytes_ = processIoReadBytes;
    lastProcessIoWriteBytes_ = processIoWriteBytes;
    processResourceSampleInitialized_ = processTimesReady && processIoReady;
#endif
    const double tileCacheMemoryMb = BytesToMegabytes(
        cacheStats.cachedPointCount * static_cast<std::uint64_t>(sizeof(CachePoint)));
    const double gpuUsedMb = BytesToMegabytes(vulkan_.PointBufferUsedBytes());
    const double gpuCapacityMb = BytesToMegabytes(vulkan_.PointBufferCapacityBytes());
    const VulkanContext::TilePageTransferStats transferStats =
        vulkan_.GetTilePageTransferStats();
    const VulkanContext::GpuMemoryBudgetStats gpuMemoryStats =
        vulkan_.GetGpuMemoryBudgetStats();
    const VulkanContext::GpuDeviceInfo gpuDeviceInfo = vulkan_.GetGpuDeviceInfo();
    const double gpuMemoryUsageMb = gpuMemoryStats.available
        ? BytesToMegabytes(gpuMemoryStats.deviceLocalUsageBytes)
        : -1.0;
    const double gpuMemoryBudgetMb = gpuMemoryStats.available
        ? BytesToMegabytes(gpuMemoryStats.deviceLocalBudgetBytes)
        : -1.0;
    peakProcessMemoryMegabytes_ = std::max(
        peakProcessMemoryMegabytes_,
        processMemoryMb);
    if (gpuMemoryStats.available)
    {
        peakGpuMemoryUsageMegabytes_ = std::max(
            peakGpuMemoryUsageMegabytes_,
            gpuMemoryUsageMb);
    }

    const std::ios::fmtflags oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();
    const std::string renderTier =
        RenderTierLabel(lastProxyMode_, lastActiveLodName_, lastActiveLodLevel_);
    const bool completeResult =
        lastDeferredTileCount_ == 0 &&
        !(lastLod0PointCount_ > 0 && lastProxyPointCount_ > 0) &&
        (!lastLod0IndexPreparing_ || gpuReadyPageCache_.Available()) &&
        !pendingTargetUnderlayActive_ &&
        !interactivePanQueryPending_ &&
        !lodQualityRefinementPending_;
    const bool targetSwitchPending =
        lodQualityRefinementPending_ ||
        interactivePanQueryPending_ ||
        visibleQueryRequested_ ||
        (visibleQueryRunning_ && runningQueryReason_ != "guard_band_1_5x");
    const double queryCancellationAverageMilliseconds = queryCancelledCount_ > 0
        ? queryCancellationTotalMilliseconds_ / static_cast<double>(queryCancelledCount_)
        : 0.0;
    std::cout << std::fixed << std::setprecision(2);
    std::cout
        << "Status: fps=" << currentFps_
        << ", frame_ms=" << averageFrameMilliseconds
        << ", frame_max_ms=" << frameMaxMilliseconds
        << ", frame_p50_ms=" << cpuFrameStats.p50
        << ", frame_p95_ms=" << cpuFrameStats.p95
        << ", frame_p99_ms=" << cpuFrameStats.p99
        << ", app_work_ms=" << appWorkStats.average
        << ", app_work_p50_ms=" << appWorkStats.p50
        << ", app_work_p95_ms=" << appWorkStats.p95
        << ", app_work_p99_ms=" << appWorkStats.p99
        << ", app_work_max_ms=" << appWorkStats.maximum
        << ", gpu_frame_ms=" << gpuFrameStats.average
        << ", gpu_frame_p50_ms=" << gpuFrameStats.p50
        << ", gpu_frame_p95_ms=" << gpuFrameStats.p95
        << ", gpu_frame_p99_ms=" << gpuFrameStats.p99
        << ", gpu_frame_max_ms=" << gpuFrameStats.maximum
        << ", gpu_cull_ms=" << averageGpuCullMilliseconds
        << ", gpu_draw_ms=" << averageGpuDrawMilliseconds
        << ", fence_wait_ms=" << averageFenceWaitMilliseconds
        << ", acquire_ms=" << averageAcquireMilliseconds
        << ", record_ms=" << averageRecordMilliseconds
        << ", submit_ms=" << averageSubmitMilliseconds
        << ", present_ms=" << averagePresentMilliseconds
        << ", vulkan_upload_ms=" << averageVulkanUploadMilliseconds
        << ", point_buffer_resize_ms=" << pointBufferResizeMilliseconds
        << ", point_buffer_resize_count=" << pointBufferResizeCount
        << ", vulkan_upload_mb=" << vulkanUploadMegabytes
        << ", query_ms=" << lastQueryMilliseconds_
        << ", query_total_ms=" << lastQueryTotalMilliseconds_
        << ", query_reason=" << lastQueryReason_
        << ", query_requests=" << queryRequestCount_
        << ", query_superseded=" << querySupersededCount_
        << ", query_cancelled=" << queryCancelledCount_
        << ", query_discarded=" << queryDiscardedCount_
        << ", query_cancel_avg_ms=" << queryCancellationAverageMilliseconds
        << ", query_cancel_max_ms=" << queryCancellationMaxMilliseconds_
        << ", interactive_coalesced=" << interactiveQueryCoalescedCount_
        << ", interactive_staged=" << interactiveQueryStagedCount_
        << ", interactive_committed=" << interactiveQueryCommittedCount_
        << ", pan_preview_ready="
        << (stagedInteractivePanResult_.has_value() ? "yes" : "no")
        << ", lod_points_per_pixel=" << lastLodPointsPerPixel_
        << ", estimated_points_per_pixel=" << lastEstimatedPointsPerPixel_
        << ", lod_point_budget=" << lastLodPointBudget_
        << ", lod_budget_satisfied="
        << (lastEstimatedPoints_ <= lastLodPointBudget_ ? "yes" : "coarsest_fallback")
        << ", cpu_build_upload_request_ms=" << lastCpuBuildUploadRequestMilliseconds_
        << ", main_apply_ms=" << lastMainApplyMilliseconds_
        << ", cpu_upload_points=" << lastCpuUploadPoints_
        << ", point_upload_mb=" << lastPointUploadMegabytes_
        << ", rendered_points=" << lastRenderedPointCount_
        << ", hit_tiles=" << lastHitTileCount_
        << ", estimated_points=" << lastEstimatedPoints_
        << ", lod0_points=" << lastLod0PointCount_
        << ", proxy_points=" << lastProxyPointCount_
        << ", lod0_tiles=" << lastLod0TileCount_
        << ", proxy_tiles=" << lastProxyTileCount_
        << ", deferred_tiles=" << lastDeferredTileCount_
        << ", overview_points=" << lastOverviewPointCount_
        << ", overview_alpha=" << lastOverviewAlpha_
        << ", mode=" << (lastProxyMode_ ? "proxy" : lastActiveLodName_)
        << ", render_tier=" << renderTier
        << ", complete=" << (completeResult ? "yes" : "no")
        << ", lod0_index_ready=" << (lastLod0IndexReady_ ? "yes" : "no")
        << ", lod0_index_preparing=" << (lastLod0IndexPreparing_ ? "yes" : "no")
        << ", lod_level=" << lastActiveLodLevel_
        << ", screen_lod=" << (lastScreenSpaceLod_ ? "yes" : "no")
        << ", gpu_proxy=" << (lastGpuProxyDraw_ ? "yes" : "no")
        << ", gpu_tile_metadata_uploaded=" << (gpuTileMetadataUploaded_ ? "yes" : "no")
        << ", attr=" << cache_.Metadata().attrFields[activeAttrIndex_]
        << ", app_mode=" << ProfileModeStateName(profileModeState_)
        << ", profile_task_state=" << ProfileTaskStateName(profileAnalysis_.State())
        << ", profile_vertices=" << profilePolyline_.size()
        << ", profile_corridor_width=" << profileCorridorWidth_
        << ", profile_total_ms=" << lastProfileTotalMilliseconds_
        << ", profile_io_ms=" << lastProfileIoMilliseconds_
        << ", profile_filter_ms=" << lastProfileFilterMilliseconds_
        << ", profile_read_mb=" << lastProfileReadMegabytes_
        << ", profile_tested_points=" << lastProfileTestedPoints_
        << ", profile_matched_points=" << lastProfileMatchedPoints_
        << ", profile_candidate_tiles=" << lastProfileCandidateTiles_
        << ", render_origin=(" << renderOriginX_ << "," << renderOriginY_ << ")"
        << ", mouse_world=(" << worldPoint.minX << "," << worldPoint.minY << ")"
        << ", process_mb=" << processMemoryMb
        << ", process_peak_mb=" << peakProcessMemoryMegabytes_
        << ", process_cpu_percent=" << processCpuPercent
        << ", process_io_read_mbps=" << processIoReadMegabytesPerSecond
        << ", process_io_write_mbps=" << processIoWriteMegabytesPerSecond
        << ", gpu_buffer_mb=" << gpuUsedMb << "/" << gpuCapacityMb
        << ", gpu_memory_budget_available=" << (gpuMemoryStats.available ? "yes" : "no")
        << ", gpu_memory_usage_mb=" << gpuMemoryUsageMb
        << ", gpu_memory_budget_mb=" << gpuMemoryBudgetMb
        << ", gpu_memory_peak_mb=" << peakGpuMemoryUsageMegabytes_
        << ", gpu_name=\"" << gpuDeviceInfo.name << "\""
        << ", gpu_type=" << gpuDeviceInfo.type
        << ", gpu_selection_score=" << gpuDeviceInfo.selectionScore
        << ", gpu_local_heap_mb=" << BytesToMegabytes(gpuDeviceInfo.deviceLocalHeapBytes)
        << ", gpu_strategy=" << vulkan_.GetDeviceRuntimeStrategy().name
        << ", gpu_preload_pages="
        << vulkan_.GetDeviceRuntimeStrategy().backgroundPreloadPages
        << ", vulkan_api="
        << VK_VERSION_MAJOR(gpuDeviceInfo.apiVersion) << "."
        << VK_VERSION_MINOR(gpuDeviceInfo.apiVersion) << "."
        << VK_VERSION_PATCH(gpuDeviceInfo.apiVersion)
        << ", gpu_page_active=" << vulkan_.ActiveTilePageCount()
        << ", gpu_page_resident=" << vulkan_.ResidentTilePageCount()
        << ", gpu_page_slots=" << vulkan_.TilePageSlotCount()
        << ", gpu_page_culling="
        << (vulkan_.ActiveTilePageCount() > 0 ? "on" : "off")
        << ", gpu_viewport=" << (lastGpuDrivenViewport_ ? "full_catalog" : "cpu_list")
        << ", frame_limit=off"
        << ", gpu_page_draw_api=vkCmdDrawIndirectCount"
        << ", cpu_page_draw_calls=0"
        << ", gpu_visible_pages=" << averageVisiblePageCount
        << ", ia_vertices=" << averageInputAssemblyVertices
        << ", vs_invocations=" << averageVertexShaderInvocations
        << ", compute_invocations=" << averageComputeShaderInvocations
        << ", transfer_queue=" << (transferStats.dedicatedQueue ? "dedicated" : "shared")
        << ", staging_ring=2"
        << ", staging_busy=" << transferStats.busyStagingSlots
        << ", transfer_timeline=" << transferStats.completedValue
        << "/" << transferStats.submittedValue
        << ", active_page_ready=" << transferStats.activeReadyValue
        << ", transfer_submits=" << transferStats.submitCount
        << ", transfer_mb=" << BytesToMegabytes(transferStats.submittedBytes)
        << ", transfer_gpu_ms=" << transferStats.gpuMilliseconds
        << ", transfer_last_gpu_ms=" << transferStats.lastBatchGpuMilliseconds
        << ", transfer_gbps=" << transferStats.gpuThroughputGigabytesPerSecond
        << ", transfer_gpu_timestamps="
        << (transferStats.gpuTimestampsAvailable ? "yes" : "no")
        << ", transfer_cpu_wait_ms=" << transferStats.cpuWaitMilliseconds
        << ", transfer_deferred=" << transferStats.deferredCount
        << ", gpu_ready_fast_path="
        << (gpuReadyBackgroundResidencyEnabled_ ? "yes" : "no")
        << ", gpu_full_residency="
        << (vulkan_.GpuReadyFullResidencyEnabled() ? "yes" : "no")
        << ", gpu_residency_progress=" << gpuReadyBackgroundCursor_
        << "/" << gpuReadyBackgroundPages_.size()
        << ", tile_cache_tiles=" << cacheStats.cachedTileCount
        << ", tile_cache_mb=" << tileCacheMemoryMb
        << ", cache_hits=" << cacheStats.cacheHits
        << ", cache_misses=" << cacheStats.cacheMisses
        << ", cache_evictions=" << cacheStats.cacheEvictions
        << ", viewport_query_running=" << (visibleQueryRunning_ ? "yes" : "no")
        << ", viewport_query_requested=" << (visibleQueryRequested_ ? "yes" : "no")
        << ", pan_query_pending=" << (interactivePanQueryPending_ ? "yes" : "no")
        << ", target_switch_pending=" << (targetSwitchPending ? "yes" : "no")
        << ", lod_quality_refine_pending="
        << (lodQualityRefinementPending_ ? "yes" : "no")
        << ", retained_lod_during_pending="
        << (targetSwitchPending ? lastActiveLodName_ : "none")
        << ", guard_band_ready=" << (hasGuardBandCoverage_ ? "yes" : "no")
        << ", guard_band_pending=" << (guardBandRefreshPending_ ? "yes" : "no")
        << ", guard_band_loading="
        << (visibleQueryRunning_ && runningQueryReason_ == "guard_band_1_5x" ? "yes" : "no")
        << ", viewport_query_id=" << desiredQueryRequestId_.load(std::memory_order_acquire)
        << '\n';

    if (performanceRecorder_.IsOpen())
    {
        PerformanceReportRow row{};
        row.rowType = "status_window";
        row.name = "one_second";
        row.elapsedSeconds =
            ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now()) / 1000.0;
        row.renderTier = renderTier;
        row.queryReason = lastQueryReason_;
        row.gpuName = gpuDeviceInfo.name;
        row.gpuType = gpuDeviceInfo.type;
        row.appMode = ProfileModeStateName(profileModeState_);
        row.profileState = ProfileTaskStateName(profileAnalysis_.State());
        row.gpuVendorId = gpuDeviceInfo.vendorId;
        row.gpuDeviceId = gpuDeviceInfo.deviceId;
        row.gpuDriverVersion = gpuDeviceInfo.driverVersion;
        row.vulkanApiVersion = gpuDeviceInfo.apiVersion;
        row.fps = currentFps_;
        row.appWork = appWorkStats;
        row.cpuFrame = cpuFrameStats;
        row.gpuFrame = gpuFrameStats;
        row.gpuCullAverageMilliseconds = averageGpuCullMilliseconds;
        row.gpuDrawAverageMilliseconds = averageGpuDrawMilliseconds;
        row.fenceWaitAverageMilliseconds = averageFenceWaitMilliseconds;
        row.acquireAverageMilliseconds = averageAcquireMilliseconds;
        row.recordAverageMilliseconds = averageRecordMilliseconds;
        row.submitAverageMilliseconds = averageSubmitMilliseconds;
        row.presentAverageMilliseconds = averagePresentMilliseconds;
        row.vulkanUploadAverageMilliseconds = averageVulkanUploadMilliseconds;
        row.queryMilliseconds = lastQueryMilliseconds_;
        row.queryTotalMilliseconds = lastQueryTotalMilliseconds_;
        row.lodPointsPerPixel = lastLodPointsPerPixel_;
        row.estimatedPointsPerPixel = lastEstimatedPointsPerPixel_;
        row.lodPointBudget = lastLodPointBudget_;
        row.cpuBuildMilliseconds = lastCpuBuildUploadRequestMilliseconds_;
        row.mainApplyMilliseconds = lastMainApplyMilliseconds_;
        row.processMemoryMegabytes = processMemoryMb;
        row.processCpuPercent = processCpuPercent;
        row.processIoReadMegabytesPerSecond = processIoReadMegabytesPerSecond;
        row.processIoWriteMegabytesPerSecond = processIoWriteMegabytesPerSecond;
        row.peakProcessMemoryMegabytes = peakProcessMemoryMegabytes_;
        row.gpuMemoryUsageMegabytes = gpuMemoryUsageMb;
        row.gpuMemoryBudgetMegabytes = gpuMemoryBudgetMb;
        row.peakGpuMemoryUsageMegabytes = peakGpuMemoryUsageMegabytes_;
        row.gpuBufferUsedMegabytes = gpuUsedMb;
        row.gpuBufferCapacityMegabytes = gpuCapacityMb;
        row.pointUploadMegabytes = lastPointUploadMegabytes_;
        row.vulkanUploadMegabytes = vulkanUploadMegabytes;
        row.tileCacheMegabytes = tileCacheMemoryMb;
        row.transferGpuMilliseconds = transferStats.gpuMilliseconds;
        row.transferThroughputGigabytesPerSecond =
            transferStats.gpuThroughputGigabytesPerSecond;
        row.transferCpuWaitMilliseconds = transferStats.cpuWaitMilliseconds;
        row.profileCorridorWidth = profileCorridorWidth_;
        row.profileTotalMilliseconds = lastProfileTotalMilliseconds_;
        row.profileIoMilliseconds = lastProfileIoMilliseconds_;
        row.profileFilterMilliseconds = lastProfileFilterMilliseconds_;
        row.profileReadMegabytes = lastProfileReadMegabytes_;
        row.profileTestedPoints = lastProfileTestedPoints_;
        row.profileMatchedPoints = lastProfileMatchedPoints_;
        row.profileCandidateTiles = lastProfileCandidateTiles_;
        row.renderedPoints = lastRenderedPointCount_;
        row.inputAssemblyVertices = averageInputAssemblyVertices;
        row.vertexShaderInvocations = averageVertexShaderInvocations;
        row.computeShaderInvocations = averageComputeShaderInvocations;
        row.transferBytes = transferStats.submittedBytes;
        row.lod0Points = lastLod0PointCount_;
        row.proxyPoints = lastProxyPointCount_;
        row.overviewPoints = lastOverviewPointCount_;
        row.cacheHits = cacheStats.cacheHits;
        row.cacheMisses = cacheStats.cacheMisses;
        row.cacheEvictions = cacheStats.cacheEvictions;
        row.gpuResidencyProgress = gpuReadyBackgroundCursor_;
        row.gpuResidencyTotal = gpuReadyBackgroundPages_.size();
        row.hitTiles = lastHitTileCount_;
        row.visiblePages = averageVisiblePageCount;
        row.activePages = static_cast<std::uint32_t>(vulkan_.ActiveTilePageCount());
        row.residentPages = static_cast<std::uint32_t>(vulkan_.ResidentTilePageCount());
        row.complete = completeResult;
        performanceRecorder_.Write(row);
    }
    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);
}

void App::PrintFirstGlobalFrameTiming()
{
    const std::ios::fmtflags oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();
    const SeismicCacheReader::OpenTimingStats& openTiming = cache_.LastOpenTiming();

    std::cout << std::fixed << std::setprecision(2);
    std::cout
        << "Startup timing: phase=first_global_frame"
        << ", cache_open_ms=" << startupCacheOpenMilliseconds_
        << ", cache_metadata_ms=" << openTiming.metadataMilliseconds
        << ", cache_colormap_ms=" << openTiming.colormapMilliseconds
        << ", cache_lod0_index_start_ms=" << openTiming.lod0IndexStartMilliseconds
        << ", cache_point_lod_index_ms=" << openTiming.pointLodIndexMilliseconds
        << ", cache_overview_ms=" << openTiming.overviewMilliseconds
        << ", window_init_ms=" << startupWindowInitMilliseconds_
        << ", vulkan_init_ms=" << startupVulkanInitMilliseconds_
        << ", first_query_total_ms=" << startupFirstQueryMilliseconds_
        << ", first_query_ms=" << lastQueryMilliseconds_
        << ", first_upload_request_ms=" << startupFirstUploadRequestMilliseconds_
        << ", initialize_ms=" << startupInitializeMilliseconds_
        << ", first_global_frame_ms=" << startupFirstGlobalFrameMilliseconds_
        << ", render_tier=" << RenderTierLabel(lastProxyMode_, lastActiveLodName_, lastActiveLodLevel_)
        << ", overview_points=" << lastOverviewPointCount_
        << ", rendered_points=" << lastRenderedPointCount_
        << ", lod0_index_ready=" << (lastLod0IndexReady_ ? "yes" : "no")
        << ", gpu_tile_metadata_uploaded=" << (gpuTileMetadataUploaded_ ? "yes" : "no")
        << '\n'
        << std::flush;

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    if (performanceRecorder_.IsOpen())
    {
        PerformanceReportRow row{};
        row.rowType = "startup";
        row.name = "first_global_frame";
        row.elapsedSeconds = startupFirstGlobalFrameMilliseconds_ / 1000.0;
        row.responseMilliseconds = startupFirstGlobalFrameMilliseconds_;
        row.response = PerformanceRecorder::ComputeDistribution(
            { startupFirstGlobalFrameMilliseconds_ });
        row.renderTier = RenderTierLabel(
            lastProxyMode_,
            lastActiveLodName_,
            lastActiveLodLevel_);
        row.queryReason = lastQueryReason_;
        const VulkanContext::GpuDeviceInfo gpuDeviceInfo = vulkan_.GetGpuDeviceInfo();
        row.gpuName = gpuDeviceInfo.name;
        row.gpuType = gpuDeviceInfo.type;
        row.gpuVendorId = gpuDeviceInfo.vendorId;
        row.gpuDeviceId = gpuDeviceInfo.deviceId;
        row.gpuDriverVersion = gpuDeviceInfo.driverVersion;
        row.vulkanApiVersion = gpuDeviceInfo.apiVersion;
        row.queryMilliseconds = lastQueryMilliseconds_;
        row.queryTotalMilliseconds = startupFirstQueryMilliseconds_;
        row.cpuBuildMilliseconds = startupFirstUploadRequestMilliseconds_;
        row.cacheOpenMilliseconds = startupCacheOpenMilliseconds_;
        row.windowInitMilliseconds = startupWindowInitMilliseconds_;
        row.vulkanInitMilliseconds = startupVulkanInitMilliseconds_;
        row.initializeMilliseconds = startupInitializeMilliseconds_;
        row.lodPointsPerPixel = lastLodPointsPerPixel_;
        row.estimatedPointsPerPixel = lastEstimatedPointsPerPixel_;
        row.lodPointBudget = lastLodPointBudget_;
        row.renderedPoints = lastRenderedPointCount_;
        row.lod0Points = lastLod0PointCount_;
        row.proxyPoints = lastProxyPointCount_;
        row.overviewPoints = lastOverviewPointCount_;
        row.hitTiles = lastHitTileCount_;
        row.processMemoryMegabytes = GetProcessWorkingSetMegabytes();
        peakProcessMemoryMegabytes_ = std::max(
            peakProcessMemoryMegabytes_,
            row.processMemoryMegabytes);
        row.peakProcessMemoryMegabytes = peakProcessMemoryMegabytes_;
        const VulkanContext::GpuMemoryBudgetStats memoryStats =
            vulkan_.GetGpuMemoryBudgetStats();
        if (memoryStats.available)
        {
            row.gpuMemoryUsageMegabytes = BytesToMegabytes(
                memoryStats.deviceLocalUsageBytes);
            row.gpuMemoryBudgetMegabytes = BytesToMegabytes(
                memoryStats.deviceLocalBudgetBytes);
            peakGpuMemoryUsageMegabytes_ = std::max(
                peakGpuMemoryUsageMegabytes_,
                row.gpuMemoryUsageMegabytes);
            row.peakGpuMemoryUsageMegabytes = peakGpuMemoryUsageMegabytes_;
        }
        row.gpuBufferUsedMegabytes = BytesToMegabytes(vulkan_.PointBufferUsedBytes());
        row.gpuBufferCapacityMegabytes = BytesToMegabytes(
            vulkan_.PointBufferCapacityBytes());
        row.complete = firstFrameRendered_;
        performanceRecorder_.Write(row);
    }
}

void App::PrintBackgroundReadyTiming()
{
    if (backgroundReadyTimingPrinted_)
    {
        return;
    }

    const std::ios::fmtflags oldFlags = std::cout.flags();
    const std::streamsize oldPrecision = std::cout.precision();

    std::cout << std::fixed << std::setprecision(2);
    std::cout
        << "Startup timing: phase=background_ready"
        << ", lod0_index_ready_ms=" << startupLod0IndexReadyMilliseconds_
        << ", gpu_tile_metadata_upload_ms=" << startupGpuTileMetadataUploadMilliseconds_
        << ", background_ready_ms=" << ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now())
        << ", gpu_tile_metadata_uploaded=" << (gpuTileMetadataUploaded_ ? "yes" : "no")
        << '\n';

    std::cout.flags(oldFlags);
    std::cout.precision(oldPrecision);

    if (performanceRecorder_.IsOpen())
    {
        PerformanceReportRow row{};
        row.rowType = "startup";
        row.name = "background_ready";
        row.elapsedSeconds =
            ElapsedMilliseconds(startupBegin_, std::chrono::steady_clock::now()) / 1000.0;
        row.responseMilliseconds = row.elapsedSeconds * 1000.0;
        row.response = PerformanceRecorder::ComputeDistribution(
            { row.responseMilliseconds });
        row.renderTier = RenderTierLabel(
            lastProxyMode_,
            lastActiveLodName_,
            lastActiveLodLevel_);
        row.lodPointsPerPixel = lastLodPointsPerPixel_;
        row.estimatedPointsPerPixel = lastEstimatedPointsPerPixel_;
        row.lodPointBudget = lastLodPointBudget_;
        row.renderedPoints = lastRenderedPointCount_;
        row.lod0IndexReadyMilliseconds = startupLod0IndexReadyMilliseconds_;
        row.gpuMetadataUploadMilliseconds = startupGpuTileMetadataUploadMilliseconds_;
        row.processMemoryMegabytes = GetProcessWorkingSetMegabytes();
        row.gpuBufferUsedMegabytes = BytesToMegabytes(vulkan_.PointBufferUsedBytes());
        row.gpuBufferCapacityMegabytes = BytesToMegabytes(
            vulkan_.PointBufferCapacityBytes());
        row.complete = true;
        performanceRecorder_.Write(row);
    }
}

void App::Shutdown()
{
    if (!initialized_ && window_ == nullptr)
    {
        return;
    }

    interactivePanQueryPending_ = false;
    desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel);
    profileAnalysis_.Stop();

    if (visibleQueryFuture_.valid())
    {
        try
        {
            visibleQueryFuture_.wait();
            (void)visibleQueryFuture_.get();
        }
        catch (const std::exception& error)
        {
            std::cerr << "Async query shutdown wait failed: " << error.what() << '\n';
        }
    }

    visibleQueryRunning_ = false;
    visibleQueryRequested_ = false;
    WaitForRetiredVisiblePoints();

    vulkan_.Shutdown();
    gpuReadyBackgroundPages_.clear();
    gpuReadyPageCache_.Close();
    performanceRecorder_.Close();

    if (window_ != nullptr)
    {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
    initialized_ = false;
}

void App::FramebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    (void)width;
    (void)height;

    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app != nullptr)
    {
        app->framebufferResized_ = true;
    }
}

void App::WindowFocusCallback(GLFWwindow* window, int focused)
{
    if (focused == GLFW_TRUE)
    {
        return;
    }

    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app != nullptr)
    {
        app->ResetInputLatches();
    }
}

void App::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    (void)mods;

    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app == nullptr)
    {
        return;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT &&
        action == GLFW_PRESS &&
        app->profileModeState_ != ProfileModeState::Performance)
    {
        app->profileCommitEventPending_ = true;
    }
}

void App::CursorPositionCallback(GLFWwindow* window, double cursorX, double cursorY)
{
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app == nullptr)
    {
        return;
    }

    if (app->profileModeState_ != ProfileModeState::Performance)
    {
        if (app->canvasDragging_)
        {
            const double dragDistanceSquared =
                Square(cursorX - app->leftMousePressX_) +
                Square(cursorY - app->leftMousePressY_);
            if (app->leftDragMoved_ ||
                dragDistanceSquared > Square(kClickMoveThresholdPixels))
            {
                if (!app->leftDragMoved_)
                {
                    app->guardBandRefreshPending_ = false;
                    app->visibleQueryRequested_ = false;
                    app->stagedInteractivePanResult_.reset();
                    app->desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel);
                }
                app->leftDragMoved_ = true;

                int framebufferWidth = 0;
                int framebufferHeight = 0;
                glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
                app->viewport_.PanPixels(
                    cursorX - app->lastCursorX_,
                    cursorY - app->lastCursorY_,
                    framebufferWidth,
                    framebufferHeight);
                app->lastCursorX_ = cursorX;
                app->lastCursorY_ = cursorY;
                app->ScheduleInteractivePanQuery();
            }
        }

        int width = 0;
        int height = 0;
        glfwGetWindowSize(window, &width, &height);
        const Bounds2D world = app->viewport_.ScreenToWorld(
            cursorX,
            cursorY,
            width,
            height);
        app->profileCursorWorld_ = ProfileWorldPoint{ world.minX, world.minY };
        app->profileCursorValid_ = true;
        return;
    }

    if (app->boxSelecting_)
    {
        app->UpdateBoxSelection(cursorX, cursorY);
        return;
    }

    if (!app->canvasDragging_)
    {
        app->UpdateHoverAtCursor(cursorX, cursorY);
        return;
    }

    const double dragDistanceSquared =
        Square(cursorX - app->leftMousePressX_) +
        Square(cursorY - app->leftMousePressY_);

    if (!app->leftDragMoved_ &&
        dragDistanceSquared <= Square(kClickMoveThresholdPixels))
    {
        return;
    }

    if (!app->leftDragMoved_)
    {
        app->guardBandRefreshPending_ = false;
        app->visibleQueryRequested_ = false;
        app->stagedInteractivePanResult_.reset();
        app->desiredQueryRequestId_.fetch_add(1, std::memory_order_acq_rel);
    }
    app->leftDragMoved_ = true;

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

    app->viewport_.PanPixels(
        cursorX - app->lastCursorX_,
        cursorY - app->lastCursorY_,
        framebufferWidth,
        framebufferHeight);

    app->lastCursorX_ = cursorX;
    app->lastCursorY_ = cursorY;
    app->ScheduleInteractivePanQuery();
}

void App::ScrollCallback(GLFWwindow* window, double xOffset, double yOffset)
{
    (void)xOffset;

    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (app == nullptr)
    {
        return;
    }

    int windowWidth = 0;
    int windowHeight = 0;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(window, &cursorX, &cursorY);

    const Bounds2D worldPoint = app->viewport_.ScreenToWorld(cursorX, cursorY, windowWidth, windowHeight);
    app->viewport_.ZoomAt(
        worldPoint.minX,
        worldPoint.minY,
        yOffset > 0.0 ? 0.8 : 1.25);
    app->ScheduleWheelViewportQuery(yOffset > 0.0 ? "wheel_zoom_in" : "wheel_zoom_out");
    app->QueueProfileViewportPreparation(
        yOffset > 0.0 ? "wheel_zoom_in" : "wheel_zoom_out");
}

} // namespace gpv
