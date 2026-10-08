// Vulkan 上下文声明：封装 instance、device、swapchain、动态点缓冲和同步对象。
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "VulkanBuffer.h"
#include "../cache/SeismicCacheTypes.h"
#include "../core/AppConfig.h"
#include "../interaction/BoxSelector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpv
{

class VulkanContext
{
public:
    struct FrameTimingStats
    {
        double frameMilliseconds = 0.0;
        double fenceWaitMilliseconds = 0.0;
        double uploadMilliseconds = 0.0;
        double acquireMilliseconds = 0.0;
        double recordMilliseconds = 0.0;
        double submitMilliseconds = 0.0;
        double presentMilliseconds = 0.0;
        double pointBufferResizeMilliseconds = 0.0;
        double gpuFrameMilliseconds = 0.0;
        double gpuCullMilliseconds = 0.0;
        double gpuDrawMilliseconds = 0.0;
        std::uint32_t pointBufferResizeCount = 0;
        std::uint64_t uploadBytes = 0;
        std::uint64_t inputAssemblyVertices = 0;
        std::uint64_t vertexShaderInvocations = 0;
        std::uint64_t computeShaderInvocations = 0;
        std::uint32_t visiblePageCount = 0;
        bool gpuTimestampsValid = false;
        bool pipelineStatisticsValid = false;
        bool visiblePageCountValid = false;
    };

    struct TilePageTransferStats
    {
        std::uint64_t submittedValue = 0;
        std::uint64_t completedValue = 0;
        std::uint64_t activeReadyValue = 0;
        std::uint64_t submittedBytes = 0;
        std::uint64_t submitCount = 0;
        std::uint64_t deferredCount = 0;
        std::uint64_t gpuMeasuredBytes = 0;
        double cpuWaitMilliseconds = 0.0;
        double gpuMilliseconds = 0.0;
        double lastBatchGpuMilliseconds = 0.0;
        double gpuThroughputGigabytesPerSecond = 0.0;
        std::uint32_t busyStagingSlots = 0;
        bool dedicatedQueue = false;
        bool gpuTimestampsAvailable = false;
    };

    struct GpuMemoryBudgetStats
    {
        std::uint64_t deviceLocalUsageBytes = 0;
        std::uint64_t deviceLocalBudgetBytes = 0;
        std::uint64_t deviceLocalHeapBytes = 0;
        bool available = false;
    };

    struct GpuDeviceInfo
    {
        std::string name;
        std::string type;
        std::uint32_t vendorId = 0;
        std::uint32_t deviceId = 0;
        std::uint32_t driverVersion = 0;
        std::uint32_t apiVersion = 0;
        std::uint64_t deviceLocalHeapBytes = 0;
        std::int64_t selectionScore = 0;
        bool dedicatedTransferQueue = false;
        bool memoryBudgetAvailable = false;
    };

    struct DeviceRuntimeStrategy
    {
        std::string name = "balanced";
        double pagePoolBudgetRatio = 0.70;
        double backgroundResidencyTargetRatio = 0.85;
        double memoryPressureStopRatio = 0.82;
        std::uint32_t backgroundPreloadPages = 16;
    };

    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    void Initialize(
        GLFWwindow* window,
        std::uint32_t width,
        std::uint32_t height,
        const char* appName,
        const AppConfig& config);
    void UploadPointData(std::vector<GpuPoint>&& points);
    void UpdateTilePagePool(const std::vector<GpuTilePage>& pages);
    bool ConfigureGpuReadyPagePool(
        std::uint64_t totalPageCount,
        std::uint32_t pagePointCapacity,
        std::uint32_t pointRecordSize);
    void UploadGpuPageColormap(const std::array<std::uint32_t, 256>& colors);
    void RegisterGpuReadyPageCatalog(const std::vector<GpuReadyPageView>& pages);
    void ActivateGpuReadyPages(const std::vector<GpuReadyPageView>& pages);
    void ActivateGpuReadyViewport(int lodLevel);
    std::optional<std::size_t> PreloadGpuReadyPages(
        const std::vector<GpuReadyPageView>& pages);
    bool GpuReadyFullResidencyEnabled() const;
    bool GpuReadyFullResidencyReady() const;
    std::uint32_t GpuReadyResidencyCapacityPages() const;
    void ClearActiveTilePages();
    void UploadTileMetadata(const std::vector<GpuTileMetadata>& tiles);
    void UploadGpuProxyMask(std::vector<std::uint32_t>&& proxyMask);
    void SetGpuProxyDrawEnabled(bool enabled);
    void SetPointAttributeIndex(std::uint32_t attrIndex);
    void DrawFrame(
        bool& framebufferResized,
        const glm::mat4& viewProjection,
        const Bounds2D& viewportBounds,
        double renderOriginX,
        double renderOriginY,
        const BoxSelectionState& boxSelection,
        const std::vector<glm::vec2>& profileOverlayNdc);
    std::uint64_t PointBufferUsedBytes() const;
    std::uint64_t PointBufferCapacityBytes() const;
    std::size_t ActiveTilePageCount() const;
    std::size_t ResidentTilePageCount() const;
    std::uint32_t TilePageSlotCount() const;
    TilePageTransferStats GetTilePageTransferStats() const;
    GpuMemoryBudgetStats GetGpuMemoryBudgetStats() const;
    GpuDeviceInfo GetGpuDeviceInfo() const;
    const DeviceRuntimeStrategy& GetDeviceRuntimeStrategy() const;
    const FrameTimingStats& LastFrameTiming() const;
    bool WarmDeferredPipelines();
    void WaitIdle() const;
    void Shutdown();

private:
    static constexpr int MaxFramesInFlight = 2;
    static constexpr int TilePageStagingRingSize = 2;

    void CreateInstance(const char* appName);
    void SetupDebugMessenger();
    void CreateSurface();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSwapChain();
    void CreateImageViews();
    void CreateRenderPass();
    void CreateGpuProxyDescriptorSetLayout();
    void CreateGpuPageDescriptorSetLayout();
    void CreateGraphicsPipeline(
        bool createProxy,
        bool createGpuPage,
        bool createOverlay);
    void CreatePipelineCache();
    void SavePipelineCache() const;
    void CreateFramebuffers();
    void CreateCommandPool();
    void CreatePerformanceQueryResources();
    void DestroyPerformanceQueryResources();
    void CreateTransferTimelineSemaphore();
    void CreateTransferUploadResources();
    void DestroyTransferUploadResources();
    void CreateSelectionOverlayBuffer();
    void CreateTilePagePool();
    void AllocateTilePagePool(
        VkDeviceSize capacityBytes,
        bool retainAllPages,
        std::uint32_t pointRecordSize);
    void CreateCommandBuffers();
    void CreateSyncObjects();
    void CreateRenderFinishedSemaphores();

    void DestroyGraphicsPipeline();
    void DestroyGpuProxyResources();
    void CreateGpuPageResources();
    void DestroyGpuPageResources();
    void DestroyRenderFinishedSemaphores();
    void CleanupSwapChain();
    void RecreateSwapChain();
    void EnsurePointBufferCapacity(VkDeviceSize requiredBytes, const char* reason, FrameTimingStats* timing);
    void AllocatePointBuffers(VkDeviceSize capacityBytes);
    std::uint64_t ApplyPendingPointUpload(std::uint32_t frameIndex, FrameTimingStats& timing);
    std::uint64_t ApplyPendingGpuProxyMaskUpload(std::uint32_t frameIndex);
    std::uint64_t ApplyPendingGpuPageMetadataUpload(std::uint32_t frameIndex);
    void RecordCommandBuffer(
        VkCommandBuffer commandBuffer,
        std::uint32_t imageIndex,
        const glm::mat4& viewProjection,
        const Bounds2D& viewportBounds,
        double renderOriginX,
        double renderOriginY,
        const BoxSelectionState& boxSelection,
        const std::vector<glm::vec2>& profileOverlayNdc);
    void CreateOrUpdateGpuProxyDescriptors();
    void CreateOrUpdateGpuPageDescriptors();
    void RecordGpuProxyCull(
        VkCommandBuffer commandBuffer,
        const Bounds2D& viewportBounds,
        std::uint32_t frameIndex);
    void RecordGpuPageCull(
        VkCommandBuffer commandBuffer,
        const Bounds2D& viewportBounds,
        double renderOriginX,
        double renderOriginY,
        std::uint32_t frameIndex);
    void RecordGpuPageVisibleCountReadback(
        VkCommandBuffer commandBuffer,
        std::uint32_t frameIndex);
    void CollectFrameGpuPerformance(std::uint32_t frameIndex, FrameTimingStats& timing);
    void CollectCompletedTilePageTransferTimings();
    void CollectTilePageTransferTiming(std::uint32_t slotIndex);
    void RebuildActiveGpuPageMetadata();

    struct ResidentTilePage
    {
        std::uint64_t signature = 0;
        std::uint32_t slot = 0;
        std::uint32_t pointCount = 0;
        int lodLevel = 0;
        std::uint64_t readyTimelineValue = 0;
        std::uint64_t lastUsedSerial = 0;
    };

    struct ActiveTilePage
    {
        std::uint64_t key = 0;
        std::uint32_t slot = 0;
        std::uint32_t pointCount = 0;
        double originX = 0.0;
        double originY = 0.0;
        Bounds2D bounds;
        GpuPagePointFormat pointFormat = GpuPagePointFormat::LegacyFloatRgba;
    };

    struct GpuPageDrawMetadata
    {
        glm::vec4 localBounds;
        glm::vec4 originSplit;
        std::array<std::uint32_t, 4> draw{};
        std::array<std::uint32_t, 4> format{};
    };

    static_assert(sizeof(GpuPageDrawMetadata) == 64);
    static_assert(offsetof(GpuPageDrawMetadata, localBounds) == 0);
    static_assert(offsetof(GpuPageDrawMetadata, originSplit) == 16);
    static_assert(offsetof(GpuPageDrawMetadata, draw) == 32);
    static_assert(offsetof(GpuPageDrawMetadata, format) == 48);

    struct TilePageUpload
    {
        std::uint32_t slot = 0;
        const void* points = nullptr;
        std::uint32_t pointCount = 0;
        std::uint32_t pointRecordSize = 0;
    };

    struct TilePageStagingSlot
    {
        VulkanBuffer buffer;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        std::uint64_t completionValue = 0;
        std::uint64_t timingBytes = 0;
        bool timingPending = false;
    };

    std::optional<std::uint64_t> UploadTilePageCopies(
        const std::vector<TilePageUpload>& uploads,
        bool waitForAvailableSlot);
    std::optional<std::size_t> EnsureGpuReadyPagesResident(
        const std::vector<GpuReadyPageView>& pages,
        bool waitForAvailableSlot);
    std::optional<std::uint32_t> AcquireTilePageStagingSlot(bool waitForAvailableSlot);
    std::uint64_t CompletedTilePageTransferValue() const;
    std::uint32_t AllocateTilePageSlot();

    GLFWwindow* window_ = nullptr;
    std::uint32_t initialWidth_ = 0;
    std::uint32_t initialHeight_ = 0;
    GpuPreference gpuPreference_ = GpuPreference::Auto;
    std::string gpuNameFilter_;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue transferQueue_ = VK_NULL_HANDLE;
    std::uint32_t graphicsQueueFamilyIndex_ = 0;
    std::uint32_t presentQueueFamilyIndex_ = 0;
    std::uint32_t transferQueueFamilyIndex_ = 0;
    bool dedicatedTransferQueue_ = false;

    VkSwapchainKHR swapChain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages_;
    VkFormat swapChainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent_{};
    std::vector<VkImageView> swapChainImageViews_;
    std::vector<VkFramebuffer> swapChainFramebuffers_;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;
    std::filesystem::path pipelineCachePath_;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout gpuProxyDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool gpuProxyDescriptorPool_ = VK_NULL_HANDLE;
    VkPipelineLayout gpuProxyGraphicsPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gpuProxyGraphicsPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout gpuProxyComputePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gpuProxyComputePipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout gpuPageDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool gpuPageDescriptorPool_ = VK_NULL_HANDLE;
    VkPipelineLayout gpuPageGraphicsPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gpuPageGraphicsPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout gpuPageComputePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline gpuPageComputePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout selectionOverlayPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline selectionOverlayPipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandPool transferCommandPool_ = VK_NULL_HANDLE;
    VkSemaphore tilePageTransferTimelineSemaphore_ = VK_NULL_HANDLE;
    VkQueryPool graphicsTimestampQueryPool_ = VK_NULL_HANDLE;
    VkQueryPool pipelineStatisticsQueryPool_ = VK_NULL_HANDLE;
    VkQueryPool transferTimestampQueryPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> pointVertexBuffers_;
    VulkanBuffer tilePagePoolBuffer_;
    VulkanBuffer gpuPageColormapBuffer_;
    std::array<TilePageStagingSlot, TilePageStagingRingSize> tilePageStagingRing_;
    VulkanBuffer gpuTileMetadataBuffer_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuVisibleTileIdsBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuProxyIndirectBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuProxyMaskBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuPageMetadataBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuPageIndirectBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuPageIndirectCountBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> gpuPageVisibleCountReadbackBuffers_;
    std::array<VulkanBuffer, MaxFramesInFlight> selectionOverlayVertexBuffers_;
    std::array<VkDescriptorSet, MaxFramesInFlight> gpuProxyDescriptorSets_{};
    std::array<VkDescriptorSet, MaxFramesInFlight> gpuPageDescriptorSets_{};
    std::array<VkDeviceSize, MaxFramesInFlight> pointVertexBufferCapacityBytes_{};
    std::array<std::uint32_t, MaxFramesInFlight> pointVertexCapacities_{};
    std::array<bool, MaxFramesInFlight> pointVertexBufferDirty_{};
    std::array<bool, MaxFramesInFlight> gpuProxyMaskDirty_{};
    std::array<bool, MaxFramesInFlight> gpuPageMetadataDirty_{};
    std::vector<GpuPoint> pendingPointData_;
    std::vector<GpuPageDrawMetadata> pendingGpuPageMetadata_;
    std::vector<GpuReadyPageView> gpuReadyPageCatalog_;
    std::unordered_map<std::uint64_t, std::size_t> gpuReadyPageCatalogByKey_;
    std::unordered_map<int, std::uint32_t> gpuReadyPageCountByLod_;
    std::unordered_map<int, std::uint64_t> gpuReadyPointCountByLod_;
    std::unordered_map<std::uint64_t, ResidentTilePage> residentTilePages_;
    std::vector<ActiveTilePage> activeTilePages_;
    VkDeviceSize tilePagePoolCapacityBytes_ = 0;
    std::uint32_t tilePageSlotCount_ = 0;
    std::uint32_t tilePagePointRecordSize_ = sizeof(GpuPoint);
    std::uint32_t nextTilePageSlot_ = 0;
    std::uint64_t activeTilePagePointCount_ = 0;
    std::uint32_t activeGpuPageDrawCount_ = 0;
    std::uint32_t gpuPageDrawCapacity_ = 0;
    std::uint32_t gpuPageMetadataCapacity_ = 0;
    std::uint64_t activeTilePageReadyValue_ = 0;
    std::uint64_t tilePageTransferNextValue_ = 0;
    std::uint64_t gpuPageMetadataCompletedTransferValue_ = 0;
    std::uint64_t tilePageTransferSubmittedBytes_ = 0;
    std::uint64_t tilePageTransferSubmitCount_ = 0;
    std::uint64_t tilePageTransferDeferredCount_ = 0;
    std::uint64_t tilePageTransferGpuMeasuredBytes_ = 0;
    double tilePageTransferCpuWaitMilliseconds_ = 0.0;
    double tilePageTransferGpuMilliseconds_ = 0.0;
    double tilePageTransferLastBatchGpuMilliseconds_ = 0.0;
    std::uint32_t nextTilePageStagingSlot_ = 0;
    int activeTilePageLodLevel_ = -1;
    bool retainAllTilePages_ = false;
    bool retainGpuReadyWorkingSet_ = false;
    bool gpuDrivenViewportActive_ = false;
    std::uint64_t gpuReadyResidencySerial_ = 0;
    std::vector<std::uint32_t> pendingTilePageSlots_;
    std::vector<std::uint32_t> reusableTilePageSlots_;
    std::vector<std::uint32_t> pendingGpuProxyMask_;
    std::uint32_t pointVertexCount_ = 0;
    std::uint32_t activePointAttributeIndex_ = 0;
    std::uint32_t gpuTileCount_ = 0;
    VkDeviceSize gpuTileMetadataBytes_ = 0;
    bool gpuProxyDrawEnabled_ = false;
    std::uint32_t selectionOverlayVertexCount_ = 0;

    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    std::uint32_t currentFrame_ = 0;
    FrameTimingStats lastFrameTiming_;
    std::array<bool, MaxFramesInFlight> frameGpuQueriesPending_{};
    std::array<bool, MaxFramesInFlight> frameVisibleCountPending_{};
    std::array<bool, MaxFramesInFlight> frameHadActivePages_{};
    float timestampPeriodNanoseconds_ = 0.0f;
    std::uint32_t graphicsTimestampValidBits_ = 0;
    std::uint32_t transferTimestampValidBits_ = 0;
    bool graphicsTimestampsAvailable_ = false;
    bool transferTimestampsAvailable_ = false;
    bool pipelineStatisticsAvailable_ = false;
    bool hostQueryResetAvailable_ = false;
    bool memoryBudgetExtensionEnabled_ = false;
    std::uint64_t selectedDeviceLocalHeapBytes_ = 0;
    std::int64_t selectedDeviceScore_ = 0;
    DeviceRuntimeStrategy deviceRuntimeStrategy_;
    bool performanceQueryResourcesCreated_ = false;
    std::uint32_t deferredPipelineStage_ = 0;

    bool initialized_ = false;
};

} // namespace gpv
