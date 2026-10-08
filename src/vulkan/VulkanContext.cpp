// Vulkan 上下文实现：从旧 Vulkan_study 项目拆出的 GLFW + Vulkan 基础流程。
#include "VulkanContext.h"

#include "VulkanShader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace gpv
{
namespace
{

const std::vector<const char*> kValidationLayers =
{
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char*> kDeviceExtensions =
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

constexpr VkDeviceSize kInitialPointBufferBytes = 16ull * 1024ull * 1024ull;
constexpr VkDeviceSize kPointBufferAlignmentBytes = 4ull * 1024ull * 1024ull;
constexpr std::uint32_t kGpuProxyCullGroupSize = 256;
constexpr std::uint32_t kGpuPageCullGroupSize = 256;
constexpr std::uint32_t kTilePagePointCapacity = 65'536;
constexpr VkDeviceSize kInitialTilePagePoolBytes = 64ull * 1024ull * 1024ull;
constexpr VkDeviceSize kTilePageStagingBytes = 64ull * 1024ull * 1024ull;
constexpr VkDeviceSize kMaximumShaderAddressablePagePoolBytes = 3ull * 1024ull * 1024ull * 1024ull;
constexpr std::uint32_t kTimestampQueriesPerFrame = 6;
constexpr std::uint32_t kTransferTimestampQueriesPerSlot = 2;

enum FrameTimestampQuery : std::uint32_t
{
    FrameBeginTimestamp = 0,
    CullBeginTimestamp = 1,
    CullEndTimestamp = 2,
    DrawBeginTimestamp = 3,
    DrawEndTimestamp = 4,
    FrameEndTimestamp = 5,
};

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    return ((value + alignment - 1) / alignment) * alignment;
}

std::uint64_t TimestampDelta(
    std::uint64_t begin,
    std::uint64_t end,
    std::uint32_t validBits)
{
    if (validBits == 0)
    {
        return 0;
    }
    if (validBits >= 64)
    {
        return end - begin;
    }

    const std::uint64_t mask = (std::uint64_t{ 1 } << validBits) - 1;
    return (end - begin) & mask;
}

double TimestampMilliseconds(
    std::uint64_t begin,
    std::uint64_t end,
    std::uint32_t validBits,
    float timestampPeriodNanoseconds)
{
    return
        static_cast<double>(TimestampDelta(begin, end, validBits)) *
        static_cast<double>(timestampPeriodNanoseconds) /
        1'000'000.0;
}

struct QueueFamilyIndices
{
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;
    std::optional<std::uint32_t> transferFamily;
    bool dedicatedTransfer = false;

    bool IsComplete() const
    {
        return
            graphicsFamily.has_value() &&
            presentFamily.has_value() &&
            transferFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct CameraPushConstants
{
    glm::mat4 viewProjection;
    glm::vec4 selectionRect;
    glm::vec4 selectionParams;
    glm::vec4 renderOrigin;
};

static_assert(sizeof(CameraPushConstants) == 112, "Camera push constant layout must match HLSL.");

struct TileCullPushConstants
{
    glm::vec4 viewportBounds;
    std::uint32_t tileCount = 0;
    std::uint32_t maxVisibleTiles = 0;
    std::uint32_t padding0 = 0;
    std::uint32_t padding1 = 0;
};

struct PageCullPushConstants
{
    glm::vec4 viewportBounds;
    glm::vec4 renderOriginSplit;
    std::uint32_t pageCount = 0;
    std::uint32_t maxDrawCount = 0;
    std::int32_t activeLodLevel = -1;
    std::uint32_t padding1 = 0;
};

static_assert(sizeof(PageCullPushConstants) == 48, "Page cull push constant layout must match HLSL.");

struct OverlayVertex
{
    float x = 0.0f;
    float y = 0.0f;
};

constexpr std::size_t kMaximumOverlayVertices = 512;

VkVertexInputBindingDescription GetPointBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(GpuPoint);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3> GetPointAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(GpuPoint, x);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R8G8B8A8_UNORM;
    attributeDescriptions[1].offset = offsetof(GpuPoint, rgba0);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
    attributeDescriptions[2].offset = offsetof(GpuPoint, rgba1);

    return attributeDescriptions;
}

VkVertexInputBindingDescription GetOverlayBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(OverlayVertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 1> GetOverlayAttributeDescriptions()
{
    std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(OverlayVertex, x);

    return attributeDescriptions;
}

std::array<OverlayVertex, 8> CreateSelectionOverlayVertices(const BoxSelectionState& boxSelection)
{
    const float minX = boxSelection.minNdcX;
    const float minY = boxSelection.minNdcY;
    const float maxX = boxSelection.maxNdcX;
    const float maxY = boxSelection.maxNdcY;

    return
    {
        OverlayVertex{ minX, minY },
        OverlayVertex{ maxX, minY },
        OverlayVertex{ maxX, minY },
        OverlayVertex{ maxX, maxY },
        OverlayVertex{ maxX, maxY },
        OverlayVertex{ minX, maxY },
        OverlayVertex{ minX, maxY },
        OverlayVertex{ minX, minY },
    };
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData)
{
    (void)messageSeverity;
    (void)messageType;
    (void)userData;

    std::cerr << "Vulkan validation: " << callbackData->pMessage << '\n';
    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* createInfo,
    const VkAllocationCallbacks* allocator,
    VkDebugUtilsMessengerEXT* debugMessenger)
{
    auto function = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

    if (function == nullptr)
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    return function(instance, createInfo, allocator, debugMessenger);
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* allocator)
{
    auto function = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    if (function != nullptr)
    {
        function(instance, debugMessenger, allocator);
    }
}

void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;
}

bool CheckValidationLayerSupport()
{
    std::uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : kValidationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
        {
            if (std::strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char*> GetRequiredExtensions()
{
    std::uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (kEnableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;

    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (std::uint32_t i = 0; i < queueFamilies.size(); ++i)
    {
        const bool graphicsAndCompute =
            (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 &&
            (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (graphicsAndCompute && !indices.graphicsFamily.has_value())
        {
            indices.graphicsFamily = i;
        }

        const bool transfer = (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        const bool dedicatedTransfer =
            transfer && (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0;
        if (transfer &&
            (!indices.transferFamily.has_value() ||
             (dedicatedTransfer && !indices.dedicatedTransfer)))
        {
            indices.transferFamily = i;
            indices.dedicatedTransfer = dedicatedTransfer;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport && !indices.presentFamily.has_value())
        {
            indices.presentFamily = i;
        }
    }

    if (!indices.transferFamily.has_value() && indices.graphicsFamily.has_value())
    {
        indices.transferFamily = indices.graphicsFamily;
        indices.dedicatedTransfer = false;
    }

    return indices;
}

std::vector<OverlayVertex> CreateOverlayVertices(
    const BoxSelectionState& boxSelection,
    const std::vector<glm::vec2>& profileOverlayNdc)
{
    std::vector<OverlayVertex> vertices;
    vertices.reserve(std::min<std::size_t>(
        kMaximumOverlayVertices,
        profileOverlayNdc.size() + (boxSelection.active ? 8 : 0)));
    if (boxSelection.active)
    {
        const auto selectionVertices = CreateSelectionOverlayVertices(boxSelection);
        vertices.insert(vertices.end(), selectionVertices.begin(), selectionVertices.end());
    }
    for (const glm::vec2& point : profileOverlayNdc)
    {
        if (vertices.size() >= kMaximumOverlayVertices)
        {
            break;
        }
        vertices.push_back(OverlayVertex{ point.x, point.y });
    }
    if ((vertices.size() & 1u) != 0)
    {
        vertices.pop_back();
    }
    return vertices;
}

glm::vec4 SplitWorldOrigin(double x, double y)
{
    const float highX = static_cast<float>(x);
    const float highY = static_cast<float>(y);
    const float lowX = static_cast<float>(x - static_cast<double>(highX));
    const float lowY = static_cast<float>(y - static_cast<double>(highY));
    return glm::vec4(highX, highY, lowX, lowY);
}

float ConservativeMinimum(double value)
{
    return std::nextafter(
        static_cast<float>(value),
        -std::numeric_limits<float>::infinity());
}

float ConservativeMaximum(double value)
{
    return std::nextafter(
        static_cast<float>(value),
        std::numeric_limits<float>::infinity());
}

bool SupportsRequiredVulkan12Features(VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_2)
    {
        return false;
    }

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &vulkan12Features;
    vkGetPhysicalDeviceFeatures2(device, &features);
    return
        features.features.drawIndirectFirstInstance == VK_TRUE &&
        vulkan12Features.timelineSemaphore == VK_TRUE &&
        vulkan12Features.drawIndirectCount == VK_TRUE;
}

bool CheckDeviceExtensionSupport(VkPhysicalDevice device)
{
    std::uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

bool IsDeviceSuitable(
    VkPhysicalDevice device,
    VkSurfaceKHR surface,
    QueueFamilyIndices* queueFamilyIndices)
{
    QueueFamilyIndices indices = FindQueueFamilies(device, surface);
    bool extensionsSupported = CheckDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    const bool suitable =
        indices.IsComplete() &&
        extensionsSupported &&
        swapChainAdequate &&
        SupportsRequiredVulkan12Features(device);
    if (suitable && queueFamilyIndices != nullptr)
    {
        *queueFamilyIndices = indices;
    }
    return suitable;
}

bool HasDeviceExtension(VkPhysicalDevice device, const char* extensionName)
{
    std::uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(
        device,
        nullptr,
        &extensionCount,
        availableExtensions.data());
    return std::any_of(
        availableExtensions.begin(),
        availableExtensions.end(),
        [extensionName](const VkExtensionProperties& extension)
        {
            return std::strcmp(extension.extensionName, extensionName) == 0;
        });
}

std::uint32_t QueueTimestampValidBits(
    VkPhysicalDevice device,
    std::uint32_t queueFamilyIndex)
{
    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device,
        &queueFamilyCount,
        queueFamilies.data());
    if (queueFamilyIndex >= queueFamilies.size())
    {
        return 0;
    }
    return queueFamilies[queueFamilyIndex].timestampValidBits;
}

std::string ToLower(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

const char* DeviceTypeName(VkPhysicalDeviceType type)
{
    switch (type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "cpu";
    default:
        return "other";
    }
}

const char* GpuPreferenceName(GpuPreference preference)
{
    switch (preference)
    {
    case GpuPreference::Integrated:
        return "integrated";
    case GpuPreference::Discrete:
        return "discrete";
    default:
        return "auto";
    }
}

bool MatchesGpuPreference(const VkPhysicalDeviceProperties& properties, GpuPreference preference)
{
    if (preference == GpuPreference::Auto)
    {
        return true;
    }
    if (preference == GpuPreference::Integrated)
    {
        return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }
    if (preference == GpuPreference::Discrete)
    {
        return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    }

    return true;
}

bool MatchesGpuNameFilter(const VkPhysicalDeviceProperties& properties, const std::string& nameFilter)
{
    if (nameFilter.empty())
    {
        return true;
    }

    return ToLower(properties.deviceName).find(ToLower(nameFilter)) != std::string::npos;
}

VkDeviceSize LargestDeviceLocalHeap(VkPhysicalDevice device)
{
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties);

    VkDeviceSize largestHeap = 0;
    for (std::uint32_t index = 0; index < memoryProperties.memoryHeapCount; ++index)
    {
        const VkMemoryHeap& heap = memoryProperties.memoryHeaps[index];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
        {
            largestHeap = std::max(largestHeap, heap.size);
        }
    }
    return largestHeap;
}

std::int64_t ScorePhysicalDevice(
    VkPhysicalDevice device,
    const VkPhysicalDeviceProperties& properties,
    const QueueFamilyIndices& queueFamilies)
{
    std::int64_t score = 0;
    switch (properties.deviceType)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score += 1'000'000'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score += 200'000'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score += 100'000'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        score -= 1'000'000'000;
        break;
    default:
        break;
    }

    constexpr std::uint64_t gibibyte = 1024ull * 1024ull * 1024ull;
    const std::uint64_t localMemoryGiB = LargestDeviceLocalHeap(device) / gibibyte;
    score += static_cast<std::int64_t>(std::min<std::uint64_t>(localMemoryGiB, 64)) * 1'000'000;
    if (properties.vendorID == 0x10DE)
    {
        score += 20'000'000;
    }
    const std::string normalizedName = ToLower(properties.deviceName);
    if (normalizedName.find("microsoft direct3d12") != std::string::npos ||
        normalizedName.find("swiftshader") != std::string::npos ||
        normalizedName.find("llvmpipe") != std::string::npos)
    {
        score -= 100'000'000;
    }
    if (queueFamilies.dedicatedTransfer)
    {
        score += 5'000'000;
    }
    if (HasDeviceExtension(device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
    {
        score += 2'000'000;
    }
    score += static_cast<std::int64_t>(
        std::min<std::uint64_t>(properties.limits.maxStorageBufferRange / (256ull * 1024ull * 1024ull), 16)) *
        250'000;
    return score;
}

VulkanContext::DeviceRuntimeStrategy BuildDeviceRuntimeStrategy(
    const VkPhysicalDeviceProperties& properties,
    VkDeviceSize localHeapBytes)
{
    VulkanContext::DeviceRuntimeStrategy strategy;
    const double localHeapGiB = static_cast<double>(localHeapBytes) /
        static_cast<double>(1024ull * 1024ull * 1024ull);

    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
        if (localHeapGiB >= 10.0)
        {
            strategy.name = "discrete_high";
            strategy.pagePoolBudgetRatio = 0.82;
            strategy.backgroundResidencyTargetRatio = 0.95;
            strategy.memoryPressureStopRatio = 0.90;
            strategy.backgroundPreloadPages = 64;
        }
        else if (localHeapGiB >= 6.0)
        {
            strategy.name = "discrete_balanced";
            strategy.pagePoolBudgetRatio = 0.76;
            strategy.backgroundResidencyTargetRatio = 0.90;
            strategy.memoryPressureStopRatio = 0.86;
            strategy.backgroundPreloadPages = 32;
        }
        else
        {
            strategy.name = "discrete_compact";
            strategy.pagePoolBudgetRatio = 0.66;
            strategy.backgroundResidencyTargetRatio = 0.82;
            strategy.memoryPressureStopRatio = 0.80;
            strategy.backgroundPreloadPages = 16;
        }
    }
    else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
        strategy.name = "integrated_shared";
        strategy.pagePoolBudgetRatio = 0.45;
        strategy.backgroundResidencyTargetRatio = 0.70;
        strategy.memoryPressureStopRatio = 0.72;
        strategy.backgroundPreloadPages = 8;
    }
    else
    {
        strategy.name = "compatibility";
        strategy.pagePoolBudgetRatio = 0.50;
        strategy.backgroundResidencyTargetRatio = 0.75;
        strategy.memoryPressureStopRatio = 0.75;
        strategy.backgroundPreloadPages = 8;
    }
    return strategy;
}

VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    const auto immediate = std::find(
        availablePresentModes.begin(),
        availablePresentModes.end(),
        VK_PRESENT_MODE_IMMEDIATE_KHR);
    if (immediate != availablePresentModes.end())
    {
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    const auto mailbox = std::find(
        availablePresentModes.begin(),
        availablePresentModes.end(),
        VK_PRESENT_MODE_MAILBOX_KHR);
    if (mailbox != availablePresentModes.end())
    {
        return VK_PRESENT_MODE_MAILBOX_KHR;
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

const char* PresentModeName(VkPresentModeKHR presentMode)
{
    switch (presentMode)
    {
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return "mailbox";
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return "immediate";
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        return "fifo_relaxed";
    case VK_PRESENT_MODE_FIFO_KHR:
    default:
        return "fifo";
    }
}

VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent =
    {
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
    };

    actualExtent.width = std::clamp(
        actualExtent.width,
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(
        actualExtent.height,
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height);

    return actualExtent;
}

} // namespace

VulkanContext::~VulkanContext()
{
    Shutdown();
}

void VulkanContext::CreatePipelineCache()
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);

    std::filesystem::path cacheRoot;
#ifdef _WIN32
    char* configuredCacheDirectory = nullptr;
    std::size_t configuredCacheDirectoryLength = 0;
    if (_dupenv_s(
            &configuredCacheDirectory,
            &configuredCacheDirectoryLength,
            "GPV_PIPELINE_CACHE_DIR") == 0 &&
        configuredCacheDirectory != nullptr &&
        configuredCacheDirectory[0] != '\0')
    {
        cacheRoot = configuredCacheDirectory;
    }
    std::free(configuredCacheDirectory);

    char* localAppData = nullptr;
    std::size_t localAppDataLength = 0;
    if (cacheRoot.empty() &&
        _dupenv_s(&localAppData, &localAppDataLength, "LOCALAPPDATA") == 0 &&
        localAppData != nullptr)
    {
        cacheRoot = std::filesystem::path(localAppData) / "GeoPointViewer" / "pipeline_cache";
    }
    std::free(localAppData);
#endif
    if (cacheRoot.empty())
    {
        cacheRoot = std::filesystem::path("cache") / "pipeline_cache";
    }

    std::error_code directoryError;
    std::filesystem::create_directories(cacheRoot, directoryError);
    if (!directoryError)
    {
        pipelineCachePath_ = cacheRoot /
            (std::to_string(properties.vendorID) + "_" +
             std::to_string(properties.deviceID) + "_" +
             std::to_string(properties.driverVersion) + ".bin");
    }
    else
    {
        pipelineCachePath_.clear();
        std::cerr
            << "Pipeline cache directory unavailable; using memory-only cache: path="
            << cacheRoot.string()
            << ", error=" << directoryError.message()
            << '\n';
    }

    std::vector<char> initialData;
    std::ifstream input;
    if (!pipelineCachePath_.empty())
    {
        input.open(pipelineCachePath_, std::ios::binary | std::ios::ate);
    }
    if (input)
    {
        const std::streamsize size = input.tellg();
        if (size > 0 && size <= 64 * 1024 * 1024)
        {
            initialData.resize(static_cast<std::size_t>(size));
            input.seekg(0, std::ios::beg);
            input.read(initialData.data(), size);
            if (!input)
            {
                initialData.clear();
            }
        }
    }

    VkPipelineCacheCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    createInfo.initialDataSize = initialData.size();
    createInfo.pInitialData = initialData.empty() ? nullptr : initialData.data();
    VkResult result = vkCreatePipelineCache(device_, &createInfo, nullptr, &pipelineCache_);
    if (result != VK_SUCCESS && !initialData.empty())
    {
        initialData.clear();
        createInfo.initialDataSize = 0;
        createInfo.pInitialData = nullptr;
        result = vkCreatePipelineCache(device_, &createInfo, nullptr, &pipelineCache_);
    }
    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan pipeline cache.");
    }

    std::cout
        << "Pipeline cache initialized: state="
        << (initialData.empty() ? "cold" : "warm")
        << ", bytes=" << initialData.size()
        << ", path="
        << (pipelineCachePath_.empty() ? "<memory-only>" : pipelineCachePath_.string())
        << '\n';
}

void VulkanContext::SavePipelineCache() const
{
    if (device_ == VK_NULL_HANDLE || pipelineCache_ == VK_NULL_HANDLE || pipelineCachePath_.empty())
    {
        return;
    }

    std::size_t size = 0;
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, nullptr) != VK_SUCCESS || size == 0)
    {
        return;
    }
    std::vector<std::byte> data(size);
    if (vkGetPipelineCacheData(device_, pipelineCache_, &size, data.data()) != VK_SUCCESS)
    {
        return;
    }
    data.resize(size);

    const std::filesystem::path temporaryPath = pipelineCachePath_.string() + ".tmp";
    std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        return;
    }
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    output.close();
    if (!output)
    {
        return;
    }

    std::error_code renameError;
    std::filesystem::remove(pipelineCachePath_, renameError);
    renameError.clear();
    std::filesystem::rename(temporaryPath, pipelineCachePath_, renameError);
    if (!renameError)
    {
        std::cout << "Pipeline cache saved: bytes=" << data.size() << '\n';
    }
}

void VulkanContext::Initialize(
    GLFWwindow* window,
    std::uint32_t width,
    std::uint32_t height,
    const char* appName,
    const AppConfig& config)
{
    if (initialized_)
    {
        return;
    }

    window_ = window;
    initialWidth_ = width;
    initialHeight_ = height;
    gpuPreference_ = config.gpuPreference;
    gpuNameFilter_ = config.gpuNameFilter;

    const auto initializationBegin = Clock::now();
    auto phaseBegin = initializationBegin;
    CreateInstance(appName);
    SetupDebugMessenger();
    CreateSurface();
    const double instanceSurfaceMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    PickPhysicalDevice();
    const double physicalDeviceMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    CreateLogicalDevice();
    const double logicalDeviceMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    CreatePipelineCache();
    const double pipelineCacheMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    const double swapchainMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    CreateGraphicsPipeline(false, false, false);
    const double basePipelineMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    CreateFramebuffers();
    CreateCommandPool();
    CreateCommandBuffers();
    CreateSyncObjects();
    const double frameResourcesMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    phaseBegin = Clock::now();
    EnsurePointBufferCapacity(kInitialPointBufferBytes, "startup_preallocate", nullptr);
    const double pointBufferMilliseconds = ElapsedMilliseconds(phaseBegin, Clock::now());

    initialized_ = true;

    std::cout
        << "Vulkan initialization timing: instance_surface_ms="
        << instanceSurfaceMilliseconds
        << ", physical_device_ms=" << physicalDeviceMilliseconds
        << ", logical_device_ms=" << logicalDeviceMilliseconds
        << ", pipeline_cache_ms=" << pipelineCacheMilliseconds
        << ", swapchain_ms=" << swapchainMilliseconds
        << ", base_pipeline_ms=" << basePipelineMilliseconds
        << ", frame_resources_ms=" << frameResourcesMilliseconds
        << ", point_buffer_ms=" << pointBufferMilliseconds
        << ", total_ms=" << ElapsedMilliseconds(initializationBegin, Clock::now())
        << ", deferred_heavy_resources=yes"
        << '\n';
}

void VulkanContext::UploadPointData(std::vector<GpuPoint>&& points)
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }

    pointVertexCount_ = static_cast<std::uint32_t>(
        std::min<std::size_t>(points.size(), std::numeric_limits<std::uint32_t>::max()));

    if (points.size() > pointVertexCount_)
    {
        points.resize(pointVertexCount_);
    }
    pendingPointData_ = std::move(points);
    pointVertexBufferDirty_.fill(true);
}

void VulkanContext::SetPointAttributeIndex(std::uint32_t attrIndex)
{
    activePointAttributeIndex_ = std::min<std::uint32_t>(attrIndex, 1u);
}

void VulkanContext::UpdateTilePagePool(const std::vector<GpuTilePage>& pages)
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }
    if (pages.empty())
    {
        ClearActiveTilePages();
        return;
    }

    CreateTransferUploadResources();
    if (gpuPageGraphicsPipeline_ == VK_NULL_HANDLE || gpuPageComputePipeline_ == VK_NULL_HANDLE)
    {
        CreateGraphicsPipeline(false, true, false);
    }
    if (tilePagePoolBuffer_.Handle() == VK_NULL_HANDLE)
    {
        CreateTilePagePool();
    }
    const std::size_t requiredSlots = pages.size();
    if (requiredSlots > tilePageSlotCount_)
    {
        const VkDeviceSize slotBytes =
            static_cast<VkDeviceSize>(kTilePagePointCapacity) * sizeof(GpuPoint);
        const std::uint64_t grownSlots = std::max<std::uint64_t>(
            requiredSlots,
            std::max<std::uint64_t>(1, tilePageSlotCount_) * 3 / 2);
        const VkDeviceSize newCapacityBytes = grownSlots * slotBytes;
        vkDeviceWaitIdle(device_);
        const std::uint32_t oldSlotCount = tilePageSlotCount_;
        AllocateTilePagePool(newCapacityBytes, false, sizeof(GpuPoint));
        std::cout
            << "GPU tile page pool resized: old_slots=" << oldSlotCount
            << ", new_slots=" << tilePageSlotCount_
            << ", required_slots=" << requiredSlots
            << ", growth_policy=1.5x"
            << '\n';
    }

    const int currentLodLevel = pages.front().lodLevel;
    std::size_t retiredPages = 0;
    for (auto resident = residentTilePages_.begin(); resident != residentTilePages_.end();)
    {
        const int residentLevel = resident->second.lodLevel;
        if (residentLevel < currentLodLevel - 1 || residentLevel > currentLodLevel + 1)
        {
            pendingTilePageSlots_.push_back(resident->second.slot);
            resident = residentTilePages_.erase(resident);
            ++retiredPages;
        }
        else
        {
            ++resident;
        }
    }

    std::size_t newPageCount = 0;
    for (const GpuTilePage& page : pages)
    {
        if (page.lodLevel != currentLodLevel)
        {
            throw std::runtime_error("A GPU tile page update cannot mix LOD levels.");
        }
        if (page.points.size() > kTilePagePointCapacity)
        {
            throw std::runtime_error("GPU tile page exceeds its point capacity.");
        }

        const auto resident = residentTilePages_.find(page.key);
        if (resident != residentTilePages_.end() && resident->second.signature != page.signature)
        {
            pendingTilePageSlots_.push_back(resident->second.slot);
            residentTilePages_.erase(resident);
            ++retiredPages;
            ++newPageCount;
        }
        else if (resident == residentTilePages_.end())
        {
            ++newPageCount;
        }
    }

    bool resetPool = false;
    bool waitedForReclaim = false;
    std::size_t reclaimedPages = 0;
    const auto immediatelyAvailableSlots = [this]()
    {
        return
            static_cast<std::size_t>(tilePageSlotCount_ - nextTilePageSlot_) +
            reusableTilePageSlots_.size();
    };

    if (newPageCount > immediatelyAvailableSlots() && !pendingTilePageSlots_.empty())
    {
        vkDeviceWaitIdle(device_);
        waitedForReclaim = true;
        reclaimedPages = pendingTilePageSlots_.size();
        reusableTilePageSlots_.insert(
            reusableTilePageSlots_.end(),
            pendingTilePageSlots_.begin(),
            pendingTilePageSlots_.end());
        pendingTilePageSlots_.clear();
    }

    if (newPageCount > immediatelyAvailableSlots())
    {
        if (!waitedForReclaim)
        {
            vkDeviceWaitIdle(device_);
        }
        residentTilePages_.clear();
        pendingTilePageSlots_.clear();
        reusableTilePageSlots_.clear();
        nextTilePageSlot_ = 0;
        newPageCount = pages.size();
        resetPool = true;
    }

    std::size_t reusedPages = 0;
    std::size_t uploadedPages = 0;
    std::uint64_t uploadedBytes = 0;
    std::vector<ActiveTilePage> nextActiveTilePages;
    nextActiveTilePages.reserve(pages.size());
    std::vector<TilePageUpload> pageUploads;
    pageUploads.reserve(newPageCount);
    std::vector<std::uint64_t> newlyUploadedKeys;
    newlyUploadedKeys.reserve(newPageCount);
    std::uint64_t nextActiveTilePagePointCount = 0;

    for (const GpuTilePage& page : pages)
    {
        auto resident = residentTilePages_.find(page.key);
        if (resident == residentTilePages_.end())
        {
            std::uint32_t slot = 0;
            if (!reusableTilePageSlots_.empty())
            {
                slot = reusableTilePageSlots_.back();
                reusableTilePageSlots_.pop_back();
            }
            else
            {
                slot = nextTilePageSlot_++;
            }
            const VkDeviceSize uploadBytes =
                static_cast<VkDeviceSize>(page.points.size()) * sizeof(GpuPoint);
            pageUploads.push_back(TilePageUpload
            {
                slot,
                page.points.data(),
                static_cast<std::uint32_t>(page.points.size()),
                sizeof(GpuPoint),
            });

            ResidentTilePage replacement;
            replacement.signature = page.signature;
            replacement.slot = slot;
            replacement.pointCount = static_cast<std::uint32_t>(page.points.size());
            replacement.lodLevel = page.lodLevel;
            resident = residentTilePages_.insert_or_assign(page.key, replacement).first;
            newlyUploadedKeys.push_back(page.key);
            ++uploadedPages;
            uploadedBytes += uploadBytes;
        }
        else
        {
            ++reusedPages;
        }

        nextActiveTilePages.push_back(ActiveTilePage
        {
            page.key,
            resident->second.slot,
            resident->second.pointCount,
            page.originX,
            page.originY,
            page.bounds,
            GpuPagePointFormat::LegacyFloatRgba,
        });
        nextActiveTilePagePointCount += resident->second.pointCount;
    }

    const std::optional<std::uint64_t> readyTimelineValue =
        UploadTilePageCopies(pageUploads, true);
    if (!readyTimelineValue.has_value())
    {
        throw std::runtime_error("Failed to upload visible GPU tile Pages.");
    }
    for (const std::uint64_t key : newlyUploadedKeys)
    {
        residentTilePages_.at(key).readyTimelineValue = readyTimelineValue.value();
    }

    activeTilePages_.swap(nextActiveTilePages);
    activeTilePagePointCount_ = nextActiveTilePagePointCount;
    activeTilePageReadyValue_ = 0;
    for (const GpuTilePage& page : pages)
    {
        activeTilePageReadyValue_ = std::max(
            activeTilePageReadyValue_,
            residentTilePages_.at(page.key).readyTimelineValue);
    }
    activeTilePageLodLevel_ = currentLodLevel;
    RebuildActiveGpuPageMetadata();

    std::cout
        << "GPU tile page pool updated: active_pages=" << activeTilePages_.size()
        << ", resident_pages=" << residentTilePages_.size()
        << ", reused_pages=" << reusedPages
        << ", uploaded_pages=" << uploadedPages
        << ", retired_pages=" << retiredPages
        << ", reclaimed_pages=" << reclaimedPages
        << ", uploaded_mb=" << static_cast<double>(uploadedBytes) / (1024.0 * 1024.0)
        << ", current_lod=" << currentLodLevel
        << ", retained_lod_min=" << currentLodLevel - 1
        << ", retained_lod_max=" << currentLodLevel + 1
        << ", reset=" << (resetPool ? "yes" : "no")
        << '\n';
}

bool VulkanContext::ConfigureGpuReadyPagePool(
    std::uint64_t totalPageCount,
    std::uint32_t pagePointCapacity,
    std::uint32_t pointRecordSize)
{
    if (device_ == VK_NULL_HANDLE || totalPageCount == 0)
    {
        return false;
    }
    CreateTransferUploadResources();
    if (gpuPageGraphicsPipeline_ == VK_NULL_HANDLE || gpuPageComputePipeline_ == VK_NULL_HANDLE)
    {
        CreateGraphicsPipeline(false, true, false);
    }
    if (pagePointCapacity != kTilePagePointCapacity ||
        (pointRecordSize != sizeof(GpuPoint) && pointRecordSize != sizeof(GpuPagePoint)) ||
        totalPageCount > std::numeric_limits<std::uint32_t>::max())
    {
        std::cout << "GPU-ready full residency disabled: incompatible page capacity.\n";
        return false;
    }

    const VkDeviceSize slotBytes =
        static_cast<VkDeviceSize>(kTilePagePointCapacity) * pointRecordSize;
    const VkDeviceSize requiredBytes = static_cast<VkDeviceSize>(totalPageCount) * slotBytes;
    gpuPageMetadataCapacity_ = static_cast<std::uint32_t>(totalPageCount);

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);
    VkDeviceSize largestDeviceLocalHeap = 0;
    for (std::uint32_t heapIndex = 0; heapIndex < memoryProperties.memoryHeapCount; ++heapIndex)
    {
        const VkMemoryHeap& heap = memoryProperties.memoryHeaps[heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
        {
            largestDeviceLocalHeap = std::max(largestDeviceLocalHeap, heap.size);
        }
    }
    VkDeviceSize safeBudget = std::min(
        kMaximumShaderAddressablePagePoolBytes,
        static_cast<VkDeviceSize>(
            static_cast<double>(largestDeviceLocalHeap) *
            deviceRuntimeStrategy_.pagePoolBudgetRatio));
    const GpuMemoryBudgetStats budgetStats = GetGpuMemoryBudgetStats();
    if (budgetStats.available && budgetStats.deviceLocalBudgetBytes > 0)
    {
        const VkDeviceSize nonPoolUsage = budgetStats.deviceLocalUsageBytes > tilePagePoolCapacityBytes_
            ? budgetStats.deviceLocalUsageBytes - tilePagePoolCapacityBytes_
            : 0;
        const VkDeviceSize targetTotalUsage = static_cast<VkDeviceSize>(
            static_cast<double>(budgetStats.deviceLocalBudgetBytes) *
            deviceRuntimeStrategy_.pagePoolBudgetRatio);
        const VkDeviceSize liveBudget = targetTotalUsage > nonPoolUsage
            ? targetTotalUsage - nonPoolUsage
            : 0;
        safeBudget = std::min(safeBudget, liveBudget);
    }

    VkDeviceSize allocationBytes = std::min(requiredBytes, safeBudget);
    allocationBytes = (allocationBytes / slotBytes) * slotBytes;
    const VkDeviceSize minimumWorkingSetBytes = std::min(
        requiredBytes,
        static_cast<VkDeviceSize>(128ull * 1024ull * 1024ull));
    if (allocationBytes < slotBytes || allocationBytes < minimumWorkingSetBytes)
    {
        std::cout << "GPU-ready residency disabled: insufficient safe device-local memory.\n";
        return false;
    }

    bool allocated = false;
    while (allocationBytes >= minimumWorkingSetBytes)
    {
        try
        {
            vkDeviceWaitIdle(device_);
            const bool fullResidency = allocationBytes >= requiredBytes;
            AllocateTilePagePool(allocationBytes, fullResidency, pointRecordSize);
            allocated = true;
            break;
        }
        catch (const std::exception& exception)
        {
            std::cout
                << "GPU-ready residency allocation retry: requested_mb="
                << static_cast<double>(allocationBytes) / (1024.0 * 1024.0)
                << ", reason=" << exception.what()
                << '\n';
            allocationBytes = ((allocationBytes / 2) / slotBytes) * slotBytes;
        }
    }
    if (!allocated)
    {
        CreateTilePagePool();
        return false;
    }

    retainGpuReadyWorkingSet_ = true;

    std::cout
        << "GPU-ready residency enabled: mode="
        << (retainAllTilePages_ ? "full" : "adaptive")
        << ", total_pages=" << totalPageCount
        << ", resident_capacity_pages=" << tilePageSlotCount_
        << ", required_mb=" << static_cast<double>(requiredBytes) / (1024.0 * 1024.0)
        << ", capacity_mb=" << static_cast<double>(tilePagePoolCapacityBytes_) / (1024.0 * 1024.0)
        << ", memory=device_local"
        << ", point_record_bytes=" << pointRecordSize
        << ", strategy=" << deviceRuntimeStrategy_.name
        << ", budget_ratio=" << deviceRuntimeStrategy_.pagePoolBudgetRatio
        << '\n';
    return true;
}

void VulkanContext::UploadGpuPageColormap(const std::array<std::uint32_t, 256>& colors)
{
    if (gpuPageColormapBuffer_.Handle() != VK_NULL_HANDLE)
    {
        gpuPageColormapBuffer_.Upload(colors.data(), sizeof(colors));
    }
}

void VulkanContext::RegisterGpuReadyPageCatalog(const std::vector<GpuReadyPageView>& pages)
{
    if (pages.size() > gpuPageMetadataCapacity_)
    {
        throw std::runtime_error("GPU-ready Page catalog exceeds the metadata capacity.");
    }

    gpuReadyPageCatalog_ = pages;
    gpuReadyPageCatalogByKey_.clear();
    gpuReadyPageCatalogByKey_.reserve(pages.size());
    gpuReadyPageCountByLod_.clear();
    gpuReadyPointCountByLod_.clear();
    for (std::size_t index = 0; index < pages.size(); ++index)
    {
        if (!gpuReadyPageCatalogByKey_.emplace(pages[index].key, index).second)
        {
            throw std::runtime_error("GPU-ready Page catalog contains duplicate keys.");
        }
        ++gpuReadyPageCountByLod_[pages[index].lodLevel];
        gpuReadyPointCountByLod_[pages[index].lodLevel] += pages[index].pointCount;
    }
    RebuildActiveGpuPageMetadata();
}

std::uint32_t VulkanContext::AllocateTilePageSlot()
{
    if (!reusableTilePageSlots_.empty())
    {
        const std::uint32_t slot = reusableTilePageSlots_.back();
        reusableTilePageSlots_.pop_back();
        return slot;
    }
    if (nextTilePageSlot_ >= tilePageSlotCount_)
    {
        throw std::runtime_error("GPU tile page pool has no free slots.");
    }
    return nextTilePageSlot_++;
}

std::uint64_t VulkanContext::CompletedTilePageTransferValue() const
{
    if (device_ == VK_NULL_HANDLE || tilePageTransferTimelineSemaphore_ == VK_NULL_HANDLE)
    {
        return 0;
    }

    std::uint64_t completedValue = 0;
    if (vkGetSemaphoreCounterValue(
            device_,
            tilePageTransferTimelineSemaphore_,
            &completedValue) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to read the Vulkan transfer timeline value.");
    }
    return completedValue;
}

void VulkanContext::CollectTilePageTransferTiming(std::uint32_t slotIndex)
{
    if (slotIndex >= tilePageStagingRing_.size())
    {
        return;
    }

    TilePageStagingSlot& slot = tilePageStagingRing_[slotIndex];
    if (!slot.timingPending || transferTimestampQueryPool_ == VK_NULL_HANDLE)
    {
        return;
    }

    std::array<std::uint64_t, kTransferTimestampQueriesPerSlot> timestamps{};
    const std::uint32_t firstQuery = slotIndex * kTransferTimestampQueriesPerSlot;
    const VkResult result = vkGetQueryPoolResults(
        device_,
        transferTimestampQueryPool_,
        firstQuery,
        kTransferTimestampQueriesPerSlot,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result == VK_SUCCESS)
    {
        tilePageTransferLastBatchGpuMilliseconds_ = TimestampMilliseconds(
            timestamps[0],
            timestamps[1],
            transferTimestampValidBits_,
            timestampPeriodNanoseconds_);
        tilePageTransferGpuMilliseconds_ += tilePageTransferLastBatchGpuMilliseconds_;
        tilePageTransferGpuMeasuredBytes_ += slot.timingBytes;
    }
    slot.timingPending = false;
    slot.timingBytes = 0;
}

void VulkanContext::CollectCompletedTilePageTransferTimings()
{
    if (transferTimestampQueryPool_ == VK_NULL_HANDLE ||
        tilePageTransferTimelineSemaphore_ == VK_NULL_HANDLE)
    {
        return;
    }

    const std::uint64_t completedValue = CompletedTilePageTransferValue();
    for (std::uint32_t slotIndex = 0;
         slotIndex < tilePageStagingRing_.size();
         ++slotIndex)
    {
        const TilePageStagingSlot& slot = tilePageStagingRing_[slotIndex];
        if (slot.timingPending && slot.completionValue <= completedValue)
        {
            CollectTilePageTransferTiming(slotIndex);
        }
    }
}

std::optional<std::uint32_t> VulkanContext::AcquireTilePageStagingSlot(
    bool waitForAvailableSlot)
{
    const std::uint64_t completedValue = CompletedTilePageTransferValue();
    for (std::uint32_t offset = 0; offset < TilePageStagingRingSize; ++offset)
    {
        const std::uint32_t slotIndex =
            (nextTilePageStagingSlot_ + offset) % TilePageStagingRingSize;
        if (tilePageStagingRing_[slotIndex].completionValue <= completedValue)
        {
            CollectTilePageTransferTiming(slotIndex);
            nextTilePageStagingSlot_ = (slotIndex + 1) % TilePageStagingRingSize;
            return slotIndex;
        }
    }

    if (!waitForAvailableSlot)
    {
        ++tilePageTransferDeferredCount_;
        return std::nullopt;
    }

    const std::uint32_t slotIndex = nextTilePageStagingSlot_;
    const std::uint64_t waitValue = tilePageStagingRing_[slotIndex].completionValue;
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &tilePageTransferTimelineSemaphore_;
    waitInfo.pValues = &waitValue;

    const auto waitBegin = Clock::now();
    const VkResult waitResult = vkWaitSemaphores(device_, &waitInfo, UINT64_MAX);
    tilePageTransferCpuWaitMilliseconds_ +=
        ElapsedMilliseconds(waitBegin, Clock::now());
    if (waitResult != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to wait for a Vulkan staging ring slot.");
    }

    CollectTilePageTransferTiming(slotIndex);

    nextTilePageStagingSlot_ = (slotIndex + 1) % TilePageStagingRingSize;
    return slotIndex;
}

std::optional<std::uint64_t> VulkanContext::UploadTilePageCopies(
    const std::vector<TilePageUpload>& uploads,
    bool waitForAvailableSlot)
{
    if (uploads.empty())
    {
        return std::uint64_t{ 0 };
    }

    VkDeviceSize totalBytes = 0;
    for (const TilePageUpload& upload : uploads)
    {
        if (upload.pointRecordSize != tilePagePointRecordSize_)
        {
            throw std::runtime_error("GPU Page upload point format does not match the page pool.");
        }
        totalBytes += static_cast<VkDeviceSize>(upload.pointCount) * upload.pointRecordSize;
    }
    if (!waitForAvailableSlot && totalBytes > kTilePageStagingBytes)
    {
        ++tilePageTransferDeferredCount_;
        return std::nullopt;
    }

    const VkDeviceSize pageSlotBytes =
        static_cast<VkDeviceSize>(kTilePagePointCapacity) * tilePagePointRecordSize_;
    std::size_t firstUpload = 0;
    std::uint64_t highestTimelineValue = 0;
    while (firstUpload < uploads.size())
    {
        const std::optional<std::uint32_t> stagingSlotIndex =
            AcquireTilePageStagingSlot(waitForAvailableSlot);
        if (!stagingSlotIndex.has_value())
        {
            return std::nullopt;
        }

        TilePageStagingSlot& stagingSlot = tilePageStagingRing_[stagingSlotIndex.value()];
        std::vector<VkBufferCopy> regions;
        VkDeviceSize stagingOffset = 0;
        std::size_t uploadIndex = firstUpload;
        for (; uploadIndex < uploads.size(); ++uploadIndex)
        {
            const TilePageUpload& upload = uploads[uploadIndex];
            const VkDeviceSize bytes =
                static_cast<VkDeviceSize>(upload.pointCount) * upload.pointRecordSize;
            if (bytes == 0)
            {
                continue;
            }
            if (!regions.empty() && stagingOffset + bytes > stagingSlot.buffer.Size())
            {
                break;
            }
            if (bytes > stagingSlot.buffer.Size())
            {
                throw std::runtime_error("A GPU tile page is larger than the staging ring slot.");
            }

            stagingSlot.buffer.UploadAt(upload.points, bytes, stagingOffset);
            VkBufferCopy region{};
            region.srcOffset = stagingOffset;
            region.dstOffset = static_cast<VkDeviceSize>(upload.slot) * pageSlotBytes;
            region.size = bytes;
            regions.push_back(region);
            stagingOffset += bytes;
        }

        firstUpload = uploadIndex;
        if (regions.empty())
        {
            continue;
        }

        if (vkResetCommandBuffer(stagingSlot.commandBuffer, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to reset a Vulkan transfer command buffer.");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(stagingSlot.commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin a Vulkan transfer command buffer.");
        }
        const std::uint32_t transferQueryBase =
            stagingSlotIndex.value() * kTransferTimestampQueriesPerSlot;
        if (transferTimestampQueryPool_ != VK_NULL_HANDLE)
        {
            vkResetQueryPool(
                device_,
                transferTimestampQueryPool_,
                transferQueryBase,
                kTransferTimestampQueriesPerSlot);
            vkCmdWriteTimestamp(
                stagingSlot.commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                transferTimestampQueryPool_,
                transferQueryBase);
        }
        vkCmdCopyBuffer(
            stagingSlot.commandBuffer,
            stagingSlot.buffer.Handle(),
            tilePagePoolBuffer_.Handle(),
            static_cast<std::uint32_t>(regions.size()),
            regions.data());
        if (transferTimestampQueryPool_ != VK_NULL_HANDLE)
        {
            vkCmdWriteTimestamp(
                stagingSlot.commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                transferTimestampQueryPool_,
                transferQueryBase + 1);
        }
        if (vkEndCommandBuffer(stagingSlot.commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to record a Vulkan transfer command buffer.");
        }

        const std::uint64_t signalValue = ++tilePageTransferNextValue_;
        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.signalSemaphoreValueCount = 1;
        timelineInfo.pSignalSemaphoreValues = &signalValue;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = &timelineInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &stagingSlot.commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &tilePageTransferTimelineSemaphore_;
        if (vkQueueSubmit(transferQueue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit GPU-ready Pages to the transfer queue.");
        }

        stagingSlot.completionValue = signalValue;
        stagingSlot.timingBytes = static_cast<std::uint64_t>(stagingOffset);
        stagingSlot.timingPending = transferTimestampQueryPool_ != VK_NULL_HANDLE;
        highestTimelineValue = signalValue;
        tilePageTransferSubmittedBytes_ += static_cast<std::uint64_t>(stagingOffset);
        ++tilePageTransferSubmitCount_;
    }

    return highestTimelineValue;
}

std::optional<std::size_t> VulkanContext::EnsureGpuReadyPagesResident(
    const std::vector<GpuReadyPageView>& pages,
    bool waitForAvailableSlot)
{
    const std::uint64_t useSerial = waitForAvailableSlot
        ? ++gpuReadyResidencySerial_
        : gpuReadyResidencySerial_;
    std::unordered_set<std::uint64_t> requestedKeys;
    requestedKeys.reserve(pages.size());
    std::size_t missingPageCount = 0;
    for (const GpuReadyPageView& page : pages)
    {
        requestedKeys.insert(page.key);
        const auto resident = residentTilePages_.find(page.key);
        if (resident == residentTilePages_.end())
        {
            ++missingPageCount;
        }
        else if (waitForAvailableSlot)
        {
            resident->second.lastUsedSerial = useSerial;
        }
    }

    const auto availableSlotCount = [this]()
    {
        return
            static_cast<std::size_t>(tilePageSlotCount_ - nextTilePageSlot_) +
            reusableTilePageSlots_.size();
    };
    if (missingPageCount > availableSlotCount())
    {
        if (!waitForAvailableSlot || !retainGpuReadyWorkingSet_)
        {
            ++tilePageTransferDeferredCount_;
            return std::nullopt;
        }

        std::vector<std::pair<std::uint64_t, ResidentTilePage>> candidates;
        candidates.reserve(residentTilePages_.size());
        for (const auto& [key, resident] : residentTilePages_)
        {
            if (requestedKeys.find(key) == requestedKeys.end())
            {
                candidates.emplace_back(key, resident);
            }
        }
        std::sort(
            candidates.begin(),
            candidates.end(),
            [this](const auto& left, const auto& right)
            {
                const bool leftNearActive =
                    std::abs(left.second.lodLevel - activeTilePageLodLevel_) <= 1;
                const bool rightNearActive =
                    std::abs(right.second.lodLevel - activeTilePageLodLevel_) <= 1;
                if (leftNearActive != rightNearActive)
                {
                    return !leftNearActive;
                }
                return left.second.lastUsedSerial < right.second.lastUsedSerial;
            });

        const std::size_t reclaimCount = missingPageCount - availableSlotCount();
        if (candidates.size() < reclaimCount)
        {
            throw std::runtime_error("GPU-ready working set cannot retain the visible viewport.");
        }
        vkDeviceWaitIdle(device_);
        for (std::size_t index = 0; index < reclaimCount; ++index)
        {
            reusableTilePageSlots_.push_back(candidates[index].second.slot);
            residentTilePages_.erase(candidates[index].first);
        }
        RebuildActiveGpuPageMetadata();
    }

    std::vector<TilePageUpload> uploads;
    uploads.reserve(pages.size());
    struct PendingResident
    {
        std::uint64_t key = 0;
        ResidentTilePage resident;
    };
    std::vector<PendingResident> pendingResidents;
    pendingResidents.reserve(pages.size());

    for (const GpuReadyPageView& page : pages)
    {
        if (residentTilePages_.find(page.key) != residentTilePages_.end())
        {
            continue;
        }
        if (page.points == nullptr ||
            page.pointCount > kTilePagePointCapacity ||
            page.pointRecordSize != tilePagePointRecordSize_)
        {
            throw std::runtime_error("GPU-ready Page view is invalid.");
        }
        const std::uint32_t slot = AllocateTilePageSlot();
        uploads.push_back(TilePageUpload
        {
            slot,
            page.points,
            page.pointCount,
            page.pointRecordSize,
        });
        pendingResidents.push_back(PendingResident
        {
            page.key,
            ResidentTilePage
            {
                page.key,
                slot,
                page.pointCount,
                page.lodLevel,
                0,
                waitForAvailableSlot ? useSerial : 0,
            },
        });
    }

    const std::optional<std::uint64_t> readyTimelineValue =
        UploadTilePageCopies(uploads, waitForAvailableSlot);
    if (!readyTimelineValue.has_value())
    {
        for (const PendingResident& pending : pendingResidents)
        {
            reusableTilePageSlots_.push_back(pending.resident.slot);
        }
        return std::nullopt;
    }

    for (PendingResident& pending : pendingResidents)
    {
        pending.resident.readyTimelineValue = readyTimelineValue.value();
        residentTilePages_.insert_or_assign(pending.key, pending.resident);
    }
    if (!pendingResidents.empty())
    {
        RebuildActiveGpuPageMetadata();
    }
    return uploads.size();
}

void VulkanContext::ActivateGpuReadyPages(const std::vector<GpuReadyPageView>& pages)
{
    if (pages.empty())
    {
        ClearActiveTilePages();
        return;
    }
    if (pages.size() > tilePageSlotCount_)
    {
        throw std::runtime_error("Visible GPU-ready Pages exceed the GPU page pool capacity.");
    }

    if (!retainGpuReadyWorkingSet_)
    {
        vkDeviceWaitIdle(device_);
        residentTilePages_.clear();
        reusableTilePageSlots_.clear();
        pendingTilePageSlots_.clear();
        nextTilePageSlot_ = 0;
    }
    const std::optional<std::size_t> uploadedPages =
        EnsureGpuReadyPagesResident(pages, true);
    if (!uploadedPages.has_value())
    {
        throw std::runtime_error("Failed to reserve a staging ring slot for visible GPU Pages.");
    }

    std::vector<ActiveTilePage> activePages;
    activePages.reserve(pages.size());
    std::uint64_t activePointCount = 0;
    std::uint64_t activeReadyValue = 0;
    for (const GpuReadyPageView& page : pages)
    {
        const ResidentTilePage& resident = residentTilePages_.at(page.key);
        activePages.push_back(ActiveTilePage
        {
            page.key,
            resident.slot,
            resident.pointCount,
            page.originX,
            page.originY,
            page.bounds,
            page.pointFormat,
        });
        activePointCount += resident.pointCount;
        activeReadyValue = std::max(activeReadyValue, resident.readyTimelineValue);
    }
    activeTilePages_.swap(activePages);
    gpuDrivenViewportActive_ = false;
    activeTilePagePointCount_ = activePointCount;
    activeTilePageReadyValue_ = activeReadyValue;
    activeTilePageLodLevel_ = pages.front().lodLevel;
    RebuildActiveGpuPageMetadata();

    std::cout
        << "GPU-ready viewport activated: lod=" << activeTilePageLodLevel_
        << ", active_pages=" << activeTilePages_.size()
        << ", uploaded_pages=" << uploadedPages.value()
        << ", resident_pages=" << residentTilePages_.size()
        << ", ready_timeline=" << activeTilePageReadyValue_
        << '\n';
}

void VulkanContext::ActivateGpuReadyViewport(int lodLevel)
{
    if (!GpuReadyFullResidencyReady())
    {
        throw std::runtime_error("GPU-driven viewport requires completed full Page residency.");
    }
    const auto pageCount = gpuReadyPageCountByLod_.find(lodLevel);
    const auto pointCount = gpuReadyPointCountByLod_.find(lodLevel);
    if (pageCount == gpuReadyPageCountByLod_.end() ||
        pointCount == gpuReadyPointCountByLod_.end())
    {
        throw std::runtime_error("GPU-driven viewport selected an unknown LOD level.");
    }

    activeTilePages_.clear();
    activeTilePagePointCount_ = pointCount->second;
    activeTilePageReadyValue_ = CompletedTilePageTransferValue();
    activeTilePageLodLevel_ = lodLevel;
    gpuDrivenViewportActive_ = true;

    std::cout
        << "GPU-driven viewport activated: lod=" << lodLevel
        << ", catalog_pages=" << activeGpuPageDrawCount_
        << ", lod_pages=" << pageCount->second
        << ", cpu_visible_page_list=no"
        << '\n';
}

std::optional<std::size_t> VulkanContext::PreloadGpuReadyPages(
    const std::vector<GpuReadyPageView>& pages)
{
    if (!retainGpuReadyWorkingSet_)
    {
        return std::size_t{ 0 };
    }
    return EnsureGpuReadyPagesResident(pages, false);
}

bool VulkanContext::GpuReadyFullResidencyEnabled() const
{
    return retainAllTilePages_;
}

bool VulkanContext::GpuReadyFullResidencyReady() const
{
    if (!retainAllTilePages_ || gpuReadyPageCatalog_.empty() ||
        residentTilePages_.size() != gpuReadyPageCatalog_.size())
    {
        return false;
    }

    std::uint64_t requiredTimelineValue = 0;
    for (const auto& [key, resident] : residentTilePages_)
    {
        (void)key;
        requiredTimelineValue = std::max(requiredTimelineValue, resident.readyTimelineValue);
    }
    return CompletedTilePageTransferValue() >= requiredTimelineValue;
}

std::uint32_t VulkanContext::GpuReadyResidencyCapacityPages() const
{
    return tilePageSlotCount_;
}

void VulkanContext::ClearActiveTilePages()
{
    activeTilePages_.clear();
    gpuDrivenViewportActive_ = false;
    activeTilePagePointCount_ = 0;
    activeTilePageReadyValue_ = 0;
    activeTilePageLodLevel_ = -1;
    RebuildActiveGpuPageMetadata();
}

void VulkanContext::UploadTileMetadata(const std::vector<GpuTileMetadata>& tiles)
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }

    CreateGraphicsPipeline(true, false, false);

    WaitIdle();
    DestroyGpuProxyResources();

    gpuTileCount_ = static_cast<std::uint32_t>(
        std::min<std::size_t>(tiles.size(), std::numeric_limits<std::uint32_t>::max()));
    gpuProxyDrawEnabled_ = false;

    if (gpuTileCount_ == 0)
    {
        return;
    }

    gpuTileMetadataBytes_ = sizeof(tiles[0]) * static_cast<VkDeviceSize>(gpuTileCount_);
    gpuTileMetadataBuffer_.Create(
        physicalDevice_,
        device_,
        gpuTileMetadataBytes_,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    gpuTileMetadataBuffer_.Upload(tiles.data(), gpuTileMetadataBytes_);

    const VkDeviceSize tileIdBufferBytes =
        sizeof(std::uint32_t) * static_cast<VkDeviceSize>(gpuTileCount_);
    const std::vector<std::uint32_t> emptyProxyMask(gpuTileCount_, 0);

    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        gpuVisibleTileIdsBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            tileIdBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        gpuProxyIndirectBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        gpuProxyMaskBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            tileIdBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        gpuProxyMaskBuffers_[frameIndex].Upload(emptyProxyMask.data(), tileIdBufferBytes);
    }

    pendingGpuProxyMask_ = emptyProxyMask;
    gpuProxyMaskDirty_.fill(false);

    CreateOrUpdateGpuProxyDescriptors();

    std::cout
        << "GPU tile metadata uploaded: tiles=" << gpuTileCount_
        << ", metadata_mb=" << static_cast<double>(gpuTileMetadataBytes_) / (1024.0 * 1024.0)
        << '\n';
}

void VulkanContext::UploadGpuProxyMask(std::vector<std::uint32_t>&& proxyMask)
{
    if (device_ == VK_NULL_HANDLE || gpuTileCount_ == 0)
    {
        return;
    }

    proxyMask.resize(gpuTileCount_, 0);
    pendingGpuProxyMask_ = std::move(proxyMask);
    gpuProxyMaskDirty_.fill(true);
}

void VulkanContext::SetGpuProxyDrawEnabled(bool enabled)
{
    const bool descriptorSetsReady = std::all_of(
        gpuProxyDescriptorSets_.begin(),
        gpuProxyDescriptorSets_.end(),
        [](VkDescriptorSet descriptorSet)
        {
            return descriptorSet != VK_NULL_HANDLE;
        });

    gpuProxyDrawEnabled_ =
        enabled &&
        gpuTileCount_ > 0 &&
        descriptorSetsReady &&
        gpuProxyGraphicsPipeline_ != VK_NULL_HANDLE &&
        gpuProxyComputePipeline_ != VK_NULL_HANDLE;
}

void VulkanContext::DrawFrame(
    bool& framebufferResized,
    const glm::mat4& viewProjection,
    const Bounds2D& viewportBounds,
    double renderOriginX,
    double renderOriginY,
    const BoxSelectionState& boxSelection,
    const std::vector<glm::vec2>& profileOverlayNdc)
{
    if ((boxSelection.active || !profileOverlayNdc.empty()) &&
        selectionOverlayPipeline_ == VK_NULL_HANDLE)
    {
        CreateGraphicsPipeline(false, false, true);
    }

    FrameTimingStats timing{};
    const auto frameBegin = Clock::now();

    const auto waitBegin = Clock::now();
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    const auto waitEnd = Clock::now();
    timing.fenceWaitMilliseconds = ElapsedMilliseconds(waitBegin, waitEnd);
    CollectFrameGpuPerformance(currentFrame_, timing);
    CollectCompletedTilePageTransferTimings();
    const std::uint64_t completedTransferValue = CompletedTilePageTransferValue();
    if (completedTransferValue > gpuPageMetadataCompletedTransferValue_ &&
        !gpuReadyPageCatalog_.empty())
    {
        gpuPageMetadataCompletedTransferValue_ = completedTransferValue;
        RebuildActiveGpuPageMetadata();
    }

    const auto uploadBegin = Clock::now();
    timing.uploadBytes += ApplyPendingPointUpload(currentFrame_, timing);
    timing.uploadBytes += ApplyPendingGpuProxyMaskUpload(currentFrame_);
    timing.uploadBytes += ApplyPendingGpuPageMetadataUpload(currentFrame_);
    const auto uploadEnd = Clock::now();
    timing.uploadMilliseconds = ElapsedMilliseconds(uploadBegin, uploadEnd);

    std::uint32_t imageIndex = 0;
    const auto acquireBegin = Clock::now();
    VkResult result = vkAcquireNextImageKHR(
        device_,
        swapChain_,
        UINT64_MAX,
        imageAvailableSemaphores_[currentFrame_],
        VK_NULL_HANDLE,
        &imageIndex);
    const auto acquireEnd = Clock::now();
    timing.acquireMilliseconds = ElapsedMilliseconds(acquireBegin, acquireEnd);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        timing.frameMilliseconds = ElapsedMilliseconds(frameBegin, Clock::now());
        lastFrameTiming_ = timing;
        RecreateSwapChain();
        return;
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("Failed to acquire swapchain image.");
    }

    const auto recordBegin = Clock::now();
    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
    vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
    RecordCommandBuffer(
        commandBuffers_[currentFrame_],
        imageIndex,
        viewProjection,
        viewportBounds,
        renderOriginX,
        renderOriginY,
        boxSelection,
        profileOverlayNdc);
    const auto recordEnd = Clock::now();
    timing.recordMilliseconds = ElapsedMilliseconds(recordBegin, recordEnd);

    std::array<VkSemaphore, 2> waitSemaphores =
    {
        imageAvailableSemaphores_[currentFrame_],
        VK_NULL_HANDLE,
    };
    std::array<VkPipelineStageFlags, 2> waitStages =
    {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
    };
    std::array<std::uint64_t, 2> waitValues = { 0, 0 };
    std::uint32_t waitSemaphoreCount = 1;
    if (activeTilePageReadyValue_ > 0)
    {
        if (tilePageTransferTimelineSemaphore_ == VK_NULL_HANDLE)
        {
            throw std::runtime_error("GPU Page rendering requires the transfer timeline semaphore.");
        }
        waitSemaphores[1] = tilePageTransferTimelineSemaphore_;
        waitValues[1] = activeTilePageReadyValue_;
        waitSemaphoreCount = 2;
    }
    VkSemaphore signalSemaphores[] = { renderFinishedSemaphores_[imageIndex] };
    const std::uint64_t signalValues[] = { 0 };

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = waitSemaphoreCount;
    timelineInfo.pWaitSemaphoreValues = waitValues.data();
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = signalValues;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = waitSemaphoreCount > 1 ? &timelineInfo : nullptr;
    submitInfo.waitSemaphoreCount = waitSemaphoreCount;
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    const auto submitBegin = Clock::now();
    if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to submit Vulkan draw command.");
    }
    const auto submitEnd = Clock::now();
    timing.submitMilliseconds = ElapsedMilliseconds(submitBegin, submitEnd);
    frameGpuQueriesPending_[currentFrame_] =
        graphicsTimestampQueryPool_ != VK_NULL_HANDLE ||
        pipelineStatisticsQueryPool_ != VK_NULL_HANDLE;
    frameVisibleCountPending_[currentFrame_] = true;
    frameHadActivePages_[currentFrame_] = activeGpuPageDrawCount_ > 0;

    VkSwapchainKHR swapChains[] = { swapChain_ };

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    const auto presentBegin = Clock::now();
    result = vkQueuePresentKHR(presentQueue_, &presentInfo);
    const auto presentEnd = Clock::now();
    timing.presentMilliseconds = ElapsedMilliseconds(presentBegin, presentEnd);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
    {
        framebufferResized = false;
        timing.frameMilliseconds = ElapsedMilliseconds(frameBegin, Clock::now());
        lastFrameTiming_ = timing;
        RecreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to present swapchain image.");
    }

    timing.frameMilliseconds = ElapsedMilliseconds(frameBegin, Clock::now());
    lastFrameTiming_ = timing;
    currentFrame_ = (currentFrame_ + 1) % MaxFramesInFlight;
}

void VulkanContext::EnsurePointBufferCapacity(
    VkDeviceSize requiredBytes,
    const char* reason,
    FrameTimingStats* timing)
{
    const VkDeviceSize currentCapacity = pointVertexBufferCapacityBytes_[0];
    if (requiredBytes <= currentCapacity &&
        std::all_of(
            pointVertexBufferCapacityBytes_.begin(),
            pointVertexBufferCapacityBytes_.end(),
            [currentCapacity](VkDeviceSize capacity)
            {
                return capacity == currentCapacity;
            }))
    {
        return;
    }

    const auto resizeBegin = Clock::now();
    const bool hadExistingBuffers = std::any_of(
        pointVertexBufferCapacityBytes_.begin(),
        pointVertexBufferCapacityBytes_.end(),
        [](VkDeviceSize capacity)
        {
            return capacity > 0;
        });

    if (hadExistingBuffers)
    {
        vkDeviceWaitIdle(device_);
    }

    VkDeviceSize grownCapacity = kInitialPointBufferBytes;
    if (currentCapacity > 0)
    {
        grownCapacity = currentCapacity + currentCapacity / 2;
    }

    const VkDeviceSize newCapacity = AlignUp(
        std::max({ requiredBytes, grownCapacity, kInitialPointBufferBytes }),
        kPointBufferAlignmentBytes);

    AllocatePointBuffers(newCapacity);

    const auto resizeEnd = Clock::now();
    const double resizeMilliseconds = ElapsedMilliseconds(resizeBegin, resizeEnd);
    if (timing != nullptr)
    {
        timing->pointBufferResizeMilliseconds += resizeMilliseconds;
        ++timing->pointBufferResizeCount;
    }

    std::cout
        << "Point buffer resized: old_capacity_mb="
        << static_cast<double>(currentCapacity * MaxFramesInFlight) / (1024.0 * 1024.0)
        << ", new_capacity_mb="
        << static_cast<double>(newCapacity * MaxFramesInFlight) / (1024.0 * 1024.0)
        << ", required_mb="
        << static_cast<double>(requiredBytes) / (1024.0 * 1024.0)
        << ", growth_policy=1.5x"
        << ", reason=" << reason
        << ", frames=" << MaxFramesInFlight
        << '\n';
}

void VulkanContext::AllocatePointBuffers(VkDeviceSize capacityBytes)
{
    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        pointVertexBuffers_[frameIndex].Destroy();
        pointVertexBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            capacityBytes,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        pointVertexBufferCapacityBytes_[frameIndex] = capacityBytes;
        pointVertexCapacities_[frameIndex] = static_cast<std::uint32_t>(
            std::min<VkDeviceSize>(
                capacityBytes / sizeof(GpuPoint),
                std::numeric_limits<std::uint32_t>::max()));
    }
}

void VulkanContext::CreateTilePagePool()
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    const bool discreteGpu = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    retainGpuReadyWorkingSet_ = false;
    gpuPageMetadataCapacity_ = 0;
    AllocateTilePagePool(kInitialTilePagePoolBytes, false, sizeof(GpuPoint));

    std::cout
        << "GPU tile page pool created: capacity_mb="
        << static_cast<double>(tilePagePoolCapacityBytes_) / (1024.0 * 1024.0)
        << ", slots=" << tilePageSlotCount_
        << ", points_per_page=" << kTilePagePointCapacity
        << ", device_class=" << (discreteGpu ? "discrete" : "integrated")
        << '\n';
}

void VulkanContext::AllocateTilePagePool(
    VkDeviceSize capacityBytes,
    bool retainAllPages,
    std::uint32_t pointRecordSize)
{
    if (pointRecordSize != sizeof(GpuPoint) && pointRecordSize != sizeof(GpuPagePoint))
    {
        throw std::runtime_error("GPU tile page point record size is unsupported.");
    }
    const VkDeviceSize slotBytes =
        static_cast<VkDeviceSize>(kTilePagePointCapacity) * pointRecordSize;
    const std::uint64_t slotCount64 = capacityBytes / slotBytes;
    if (slotCount64 == 0 || slotCount64 > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::runtime_error("GPU tile page pool capacity is invalid.");
    }

    DestroyGpuPageResources();
    tilePagePoolBuffer_.Destroy();
    tilePageSlotCount_ = static_cast<std::uint32_t>(slotCount64);
    tilePagePoolCapacityBytes_ = static_cast<VkDeviceSize>(tilePageSlotCount_) * slotBytes;
    tilePagePointRecordSize_ = pointRecordSize;

    const std::array<std::uint32_t, 2> queueFamilyIndices =
    {
        graphicsQueueFamilyIndex_,
        transferQueueFamilyIndex_,
    };
    const bool concurrentQueueAccess =
        graphicsQueueFamilyIndex_ != transferQueueFamilyIndex_;
    tilePagePoolBuffer_.Create(
        physicalDevice_,
        device_,
        tilePagePoolCapacityBytes_,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        concurrentQueueAccess ? queueFamilyIndices.data() : nullptr,
        concurrentQueueAccess ? static_cast<std::uint32_t>(queueFamilyIndices.size()) : 0);

    residentTilePages_.clear();
    activeTilePages_.clear();
    pendingTilePageSlots_.clear();
    reusableTilePageSlots_.clear();
    nextTilePageSlot_ = 0;
    activeTilePagePointCount_ = 0;
    activeTilePageReadyValue_ = 0;
    activeTilePageLodLevel_ = -1;
    gpuDrivenViewportActive_ = false;
    retainAllTilePages_ = retainAllPages;
    CreateGpuPageResources();
}

void VulkanContext::CreateGpuPageResources()
{
    if (tilePageSlotCount_ == 0)
    {
        return;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    gpuPageDrawCapacity_ = std::min(
        tilePageSlotCount_,
        properties.limits.maxDrawIndirectCount);
    if (gpuPageDrawCapacity_ == 0)
    {
        throw std::runtime_error("The Vulkan device exposes no indirect-count draw capacity.");
    }
    const std::uint32_t metadataCapacity = std::max(
        gpuPageMetadataCapacity_,
        gpuPageDrawCapacity_);
    const VkDeviceSize metadataBytes =
        sizeof(GpuPageDrawMetadata) * static_cast<VkDeviceSize>(metadataCapacity);
    const VkDeviceSize commandBytes =
        sizeof(VkDrawIndirectCommand) * static_cast<VkDeviceSize>(gpuPageDrawCapacity_);

    gpuPageColormapBuffer_.Create(
        physicalDevice_,
        device_,
        sizeof(std::uint32_t) * 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::array<std::uint32_t, 256> defaultColors{};
    defaultColors.fill(0xFFFFFFFFu);
    gpuPageColormapBuffer_.Upload(defaultColors.data(), sizeof(defaultColors));

    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        gpuPageMetadataBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            metadataBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        gpuPageIndirectBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            commandBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        gpuPageIndirectCountBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        gpuPageVisibleCountReadbackBuffers_[frameIndex].Create(
            physicalDevice_,
            device_,
            sizeof(std::uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        const std::uint32_t zero = 0;
        gpuPageVisibleCountReadbackBuffers_[frameIndex].Upload(&zero, sizeof(zero));
    }

    gpuPageMetadataDirty_.fill(true);
    CreateOrUpdateGpuPageDescriptors();

    std::cout
        << "GPU Page culling resources created: max_pages=" << gpuPageDrawCapacity_
        << ", metadata_pages=" << metadataCapacity
        << ", device_max_draw_indirect_count="
        << properties.limits.maxDrawIndirectCount
        << ", indirect_command_mb="
        << static_cast<double>(commandBytes * MaxFramesInFlight) / (1024.0 * 1024.0)
        << ", frames=" << MaxFramesInFlight
        << '\n';
}

void VulkanContext::DestroyGpuPageResources()
{
    gpuPageColormapBuffer_.Destroy();
    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        gpuPageMetadataBuffers_[frameIndex].Destroy();
        gpuPageIndirectBuffers_[frameIndex].Destroy();
        gpuPageIndirectCountBuffers_[frameIndex].Destroy();
        gpuPageVisibleCountReadbackBuffers_[frameIndex].Destroy();
    }
    if (gpuPageDescriptorPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gpuPageDescriptorPool_, nullptr);
        gpuPageDescriptorPool_ = VK_NULL_HANDLE;
    }
    gpuPageDescriptorSets_.fill(VK_NULL_HANDLE);
    gpuPageMetadataDirty_.fill(false);
    pendingGpuPageMetadata_.clear();
    activeGpuPageDrawCount_ = 0;
    gpuPageDrawCapacity_ = 0;
}

void VulkanContext::RebuildActiveGpuPageMetadata()
{
    const std::size_t metadataCount = gpuReadyPageCatalog_.empty()
        ? activeTilePages_.size()
        : gpuReadyPageCatalog_.size();
    if (metadataCount > gpuPageMetadataCapacity_ && !gpuReadyPageCatalog_.empty())
    {
        throw std::runtime_error("GPU-ready Page metadata exceeds its registered capacity.");
    }

    pendingGpuPageMetadata_.clear();
    pendingGpuPageMetadata_.reserve(metadataCount);

    if (!gpuReadyPageCatalog_.empty())
    {
        std::unordered_set<std::uint64_t> activeKeys;
        activeKeys.reserve(activeTilePages_.size());
        for (const ActiveTilePage& activePage : activeTilePages_)
        {
            activeKeys.insert(activePage.key);
        }
        for (const GpuReadyPageView& page : gpuReadyPageCatalog_)
        {
            GpuPageDrawMetadata metadata{};
            metadata.localBounds = glm::vec4(
                ConservativeMinimum(page.bounds.minX - page.originX),
                ConservativeMinimum(page.bounds.minY - page.originY),
                ConservativeMaximum(page.bounds.maxX - page.originX),
                ConservativeMaximum(page.bounds.maxY - page.originY));
            metadata.originSplit = SplitWorldOrigin(page.originX, page.originY);
            metadata.format =
            {
                static_cast<std::uint32_t>(page.pointFormat),
                page.pointRecordSize,
                static_cast<std::uint32_t>(std::max(page.lodLevel, 0)),
                0,
            };

            const auto resident = residentTilePages_.find(page.key);
            if (resident != residentTilePages_.end())
            {
                const std::uint64_t firstByteOffset64 =
                    static_cast<std::uint64_t>(resident->second.slot) *
                    kTilePagePointCapacity *
                    tilePagePointRecordSize_;
                if (firstByteOffset64 > std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::runtime_error("GPU Page byte offset exceeds the shader address limit.");
                }
                const bool readyForDraw =
                    resident->second.readyTimelineValue <= gpuPageMetadataCompletedTransferValue_ ||
                    activeKeys.find(page.key) != activeKeys.end();
                metadata.draw =
                {
                    static_cast<std::uint32_t>(firstByteOffset64),
                    readyForDraw ? resident->second.pointCount : 0,
                    0,
                    0,
                };
                metadata.format[3] = readyForDraw ? 1u : 0u;
            }
            pendingGpuPageMetadata_.push_back(metadata);
        }
    }
    else
    {
        for (const ActiveTilePage& page : activeTilePages_)
        {
            const std::uint64_t firstByteOffset64 =
                static_cast<std::uint64_t>(page.slot) *
                kTilePagePointCapacity *
                tilePagePointRecordSize_;
            if (firstByteOffset64 > std::numeric_limits<std::uint32_t>::max())
            {
                throw std::runtime_error("GPU Page byte offset exceeds the shader address limit.");
            }

            GpuPageDrawMetadata metadata{};
            metadata.localBounds = glm::vec4(
                ConservativeMinimum(page.bounds.minX - page.originX),
                ConservativeMinimum(page.bounds.minY - page.originY),
                ConservativeMaximum(page.bounds.maxX - page.originX),
                ConservativeMaximum(page.bounds.maxY - page.originY));
            metadata.originSplit = SplitWorldOrigin(page.originX, page.originY);
            metadata.draw =
            {
                static_cast<std::uint32_t>(firstByteOffset64),
                page.pointCount,
                0,
                0,
            };
            metadata.format =
            {
                static_cast<std::uint32_t>(page.pointFormat),
                tilePagePointRecordSize_,
                static_cast<std::uint32_t>(std::max(activeTilePageLodLevel_, 0)),
                1,
            };
            pendingGpuPageMetadata_.push_back(metadata);
        }
    }

    activeGpuPageDrawCount_ = static_cast<std::uint32_t>(pendingGpuPageMetadata_.size());
    gpuPageMetadataDirty_.fill(true);
}

std::uint64_t VulkanContext::ApplyPendingPointUpload(
    std::uint32_t frameIndex,
    FrameTimingStats& timing)
{
    if (frameIndex >= pointVertexBuffers_.size() || !pointVertexBufferDirty_[frameIndex])
    {
        return 0;
    }

    pointVertexBufferDirty_[frameIndex] = false;

    if (pointVertexCount_ == 0)
    {
        return 0;
    }

    const VkDeviceSize bufferSize =
        sizeof(GpuPoint) * static_cast<VkDeviceSize>(pointVertexCount_);
    EnsurePointBufferCapacity(bufferSize, "point_upload", &timing);

    pointVertexBuffers_[frameIndex].Upload(pendingPointData_.data(), bufferSize);
    return static_cast<std::uint64_t>(bufferSize);
}

std::uint64_t VulkanContext::ApplyPendingGpuProxyMaskUpload(std::uint32_t frameIndex)
{
    if (frameIndex >= gpuProxyMaskBuffers_.size() ||
        !gpuProxyMaskDirty_[frameIndex] ||
        gpuTileCount_ == 0 ||
        gpuProxyMaskBuffers_[frameIndex].Handle() == VK_NULL_HANDLE)
    {
        return 0;
    }

    gpuProxyMaskDirty_[frameIndex] = false;

    if (pendingGpuProxyMask_.size() != gpuTileCount_)
    {
        pendingGpuProxyMask_.assign(gpuTileCount_, 0);
    }

    const VkDeviceSize bufferSize =
        sizeof(std::uint32_t) * static_cast<VkDeviceSize>(gpuTileCount_);
    gpuProxyMaskBuffers_[frameIndex].Upload(pendingGpuProxyMask_.data(), bufferSize);
    return static_cast<std::uint64_t>(bufferSize);
}

std::uint64_t VulkanContext::ApplyPendingGpuPageMetadataUpload(std::uint32_t frameIndex)
{
    if (frameIndex >= gpuPageMetadataBuffers_.size() ||
        !gpuPageMetadataDirty_[frameIndex])
    {
        return 0;
    }

    gpuPageMetadataDirty_[frameIndex] = false;
    if (activeGpuPageDrawCount_ == 0)
    {
        return 0;
    }

    const VkDeviceSize uploadBytes =
        sizeof(GpuPageDrawMetadata) * static_cast<VkDeviceSize>(activeGpuPageDrawCount_);
    if (uploadBytes > gpuPageMetadataBuffers_[frameIndex].Size())
    {
        throw std::runtime_error("GPU Page metadata exceeds its frame buffer capacity.");
    }
    gpuPageMetadataBuffers_[frameIndex].Upload(pendingGpuPageMetadata_.data(), uploadBytes);
    return static_cast<std::uint64_t>(uploadBytes);
}

std::uint64_t VulkanContext::PointBufferUsedBytes() const
{
    return
        static_cast<std::uint64_t>(pointVertexCount_) * sizeof(GpuPoint) +
        activeTilePagePointCount_ * tilePagePointRecordSize_;
}

std::uint64_t VulkanContext::PointBufferCapacityBytes() const
{
    VkDeviceSize totalCapacity = 0;
    for (VkDeviceSize capacity : pointVertexBufferCapacityBytes_)
    {
        totalCapacity += capacity;
    }

    totalCapacity += tilePagePoolCapacityBytes_;

    return static_cast<std::uint64_t>(totalCapacity);
}

std::size_t VulkanContext::ActiveTilePageCount() const
{
    if (gpuDrivenViewportActive_)
    {
        const auto count = gpuReadyPageCountByLod_.find(activeTilePageLodLevel_);
        return count != gpuReadyPageCountByLod_.end() ? count->second : 0;
    }
    return activeTilePages_.size();
}

std::size_t VulkanContext::ResidentTilePageCount() const
{
    return residentTilePages_.size();
}

std::uint32_t VulkanContext::TilePageSlotCount() const
{
    return tilePageSlotCount_;
}

VulkanContext::TilePageTransferStats VulkanContext::GetTilePageTransferStats() const
{
    TilePageTransferStats stats{};
    stats.submittedValue = tilePageTransferNextValue_;
    stats.completedValue = CompletedTilePageTransferValue();
    stats.activeReadyValue = activeTilePageReadyValue_;
    stats.submittedBytes = tilePageTransferSubmittedBytes_;
    stats.submitCount = tilePageTransferSubmitCount_;
    stats.deferredCount = tilePageTransferDeferredCount_;
    stats.gpuMeasuredBytes = tilePageTransferGpuMeasuredBytes_;
    stats.cpuWaitMilliseconds = tilePageTransferCpuWaitMilliseconds_;
    stats.gpuMilliseconds = tilePageTransferGpuMilliseconds_;
    stats.lastBatchGpuMilliseconds = tilePageTransferLastBatchGpuMilliseconds_;
    stats.gpuTimestampsAvailable = transferTimestampsAvailable_;
    if (tilePageTransferGpuMilliseconds_ > 0.0)
    {
        stats.gpuThroughputGigabytesPerSecond =
            static_cast<double>(tilePageTransferGpuMeasuredBytes_) /
            (tilePageTransferGpuMilliseconds_ * 1'000'000.0);
    }
    stats.dedicatedQueue = dedicatedTransferQueue_;
    for (const TilePageStagingSlot& slot : tilePageStagingRing_)
    {
        if (slot.completionValue > stats.completedValue)
        {
            ++stats.busyStagingSlots;
        }
    }
    return stats;
}

VulkanContext::GpuMemoryBudgetStats VulkanContext::GetGpuMemoryBudgetStats() const
{
    GpuMemoryBudgetStats stats{};
    if (physicalDevice_ == VK_NULL_HANDLE)
    {
        return stats;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProperties{};
    budgetProperties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 memoryProperties{};
    memoryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memoryProperties.pNext = memoryBudgetExtensionEnabled_ ? &budgetProperties : nullptr;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice_, &memoryProperties);

    for (std::uint32_t heapIndex = 0;
         heapIndex < memoryProperties.memoryProperties.memoryHeapCount;
         ++heapIndex)
    {
        const VkMemoryHeap& heap = memoryProperties.memoryProperties.memoryHeaps[heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0)
        {
            continue;
        }

        stats.deviceLocalHeapBytes += heap.size;
        if (memoryBudgetExtensionEnabled_)
        {
            stats.deviceLocalUsageBytes += budgetProperties.heapUsage[heapIndex];
            stats.deviceLocalBudgetBytes += budgetProperties.heapBudget[heapIndex];
        }
    }
    stats.available = memoryBudgetExtensionEnabled_;
    return stats;
}

VulkanContext::GpuDeviceInfo VulkanContext::GetGpuDeviceInfo() const
{
    GpuDeviceInfo info{};
    if (physicalDevice_ == VK_NULL_HANDLE)
    {
        return info;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    info.name = properties.deviceName;
    info.type = DeviceTypeName(properties.deviceType);
    info.vendorId = properties.vendorID;
    info.deviceId = properties.deviceID;
    info.driverVersion = properties.driverVersion;
    info.apiVersion = properties.apiVersion;
    info.deviceLocalHeapBytes = selectedDeviceLocalHeapBytes_;
    info.selectionScore = selectedDeviceScore_;
    info.dedicatedTransferQueue = dedicatedTransferQueue_;
    info.memoryBudgetAvailable = memoryBudgetExtensionEnabled_;
    return info;
}

const VulkanContext::DeviceRuntimeStrategy& VulkanContext::GetDeviceRuntimeStrategy() const
{
    return deviceRuntimeStrategy_;
}

const VulkanContext::FrameTimingStats& VulkanContext::LastFrameTiming() const
{
    return lastFrameTiming_;
}

bool VulkanContext::WarmDeferredPipelines()
{
    if (!initialized_ || deferredPipelineStage_ >= 2)
    {
        return initialized_ && deferredPipelineStage_ >= 2;
    }

    CreatePerformanceQueryResources();

    if (deferredPipelineStage_ == 0)
    {
        CreateGraphicsPipeline(true, false, false);
        std::cout << "Deferred Vulkan pipelines warmed: group=tile_proxy.\n";
    }
    else
    {
        CreateGraphicsPipeline(false, false, true);
        std::cout << "Deferred Vulkan pipelines warmed: group=selection_overlay.\n";
    }
    ++deferredPipelineStage_;
    return deferredPipelineStage_ >= 2;
}

void VulkanContext::WaitIdle() const
{
    if (device_ != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device_);
    }
}

void VulkanContext::Shutdown()
{
    if (!initialized_ && instance_ == VK_NULL_HANDLE)
    {
        return;
    }

    WaitIdle();
    DestroyRenderFinishedSemaphores();
    CleanupSwapChain();
    for (VulkanBuffer& selectionOverlayVertexBuffer : selectionOverlayVertexBuffers_)
    {
        selectionOverlayVertexBuffer.Destroy();
    }
    DestroyGpuProxyResources();
    DestroyGpuPageResources();
    for (VulkanBuffer& pointVertexBuffer : pointVertexBuffers_)
    {
        pointVertexBuffer.Destroy();
    }
    tilePagePoolBuffer_.Destroy();
    DestroyTransferUploadResources();
    DestroyPerformanceQueryResources();
    residentTilePages_.clear();
    activeTilePages_.clear();
    gpuReadyPageCatalog_.clear();
    gpuReadyPageCatalogByKey_.clear();
    gpuReadyPageCountByLod_.clear();
    gpuReadyPointCountByLod_.clear();
    tilePagePoolCapacityBytes_ = 0;
    tilePageSlotCount_ = 0;
    gpuPageMetadataCapacity_ = 0;
    nextTilePageSlot_ = 0;
    activeTilePagePointCount_ = 0;
    activeTilePageReadyValue_ = 0;
    retainAllTilePages_ = false;
    retainGpuReadyWorkingSet_ = false;
    gpuDrivenViewportActive_ = false;
    gpuReadyResidencySerial_ = 0;
    activeTilePageLodLevel_ = -1;
    pendingTilePageSlots_.clear();
    reusableTilePageSlots_.clear();
    pointVertexCount_ = 0;
    pointVertexCapacities_.fill(0);
    pointVertexBufferCapacityBytes_.fill(0);
    pointVertexBufferDirty_.fill(false);
    pendingPointData_.clear();
    DestroyGraphicsPipeline();
    SavePipelineCache();
    if (pipelineCache_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
        pipelineCache_ = VK_NULL_HANDLE;
    }

    if (gpuProxyDescriptorSetLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, gpuProxyDescriptorSetLayout_, nullptr);
        gpuProxyDescriptorSetLayout_ = VK_NULL_HANDLE;
    }

    if (gpuPageDescriptorSetLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, gpuPageDescriptorSetLayout_, nullptr);
        gpuPageDescriptorSetLayout_ = VK_NULL_HANDLE;
    }

    for (std::size_t i = 0; i < imageAvailableSemaphores_.size(); ++i)
    {
        vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
        vkDestroyFence(device_, inFlightFences_[i], nullptr);
    }

    imageAvailableSemaphores_.clear();
    inFlightFences_.clear();

    if (commandPool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
    }

    if (renderPass_ != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        renderPass_ = VK_NULL_HANDLE;
    }

    if (device_ != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (kEnableValidationLayers && debugMessenger_ != VK_NULL_HANDLE)
    {
        DestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
        debugMessenger_ = VK_NULL_HANDLE;
    }

    if (surface_ != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    physicalDevice_ = VK_NULL_HANDLE;
    graphicsQueue_ = VK_NULL_HANDLE;
    presentQueue_ = VK_NULL_HANDLE;
    transferQueue_ = VK_NULL_HANDLE;
    graphicsQueueFamilyIndex_ = 0;
    presentQueueFamilyIndex_ = 0;
    transferQueueFamilyIndex_ = 0;
    dedicatedTransferQueue_ = false;
    selectedDeviceLocalHeapBytes_ = 0;
    selectedDeviceScore_ = 0;
    deviceRuntimeStrategy_ = DeviceRuntimeStrategy{};
    currentFrame_ = 0;
    deferredPipelineStage_ = 0;
    initialized_ = false;
}

void VulkanContext::CreateInstance(const char* appName)
{
    if (kEnableValidationLayers && !CheckValidationLayerSupport())
    {
        throw std::runtime_error("Requested Vulkan validation layers are not available.");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = appName;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "GeoPointViewer";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    auto extensions = GetRequiredExtensions();

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (kEnableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<std::uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
        PopulateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = &debugCreateInfo;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

void VulkanContext::SetupDebugMessenger()
{
    if (!kEnableValidationLayers)
    {
        return;
    }

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    PopulateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan debug messenger.");
    }
}

void VulkanContext::CreateSurface()
{
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create GLFW Vulkan surface.");
    }
}

void VulkanContext::PickPhysicalDevice()
{
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
        throw std::runtime_error("Failed to find a GPU with Vulkan support.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    std::int64_t bestScore = std::numeric_limits<std::int64_t>::min();
    QueueFamilyIndices selectedQueueFamilies;

    for (const auto& device : devices)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);

        QueueFamilyIndices candidateQueueFamilies;
        const bool suitable = IsDeviceSuitable(
            device,
            surface_,
            &candidateQueueFamilies);
        const bool preferenceMatches = MatchesGpuPreference(properties, gpuPreference_);
        const bool nameMatches = MatchesGpuNameFilter(properties, gpuNameFilter_);
        const VkDeviceSize localHeapBytes = LargestDeviceLocalHeap(device);
        const std::int64_t score = suitable
            ? ScorePhysicalDevice(device, properties, candidateQueueFamilies)
            : std::numeric_limits<std::int64_t>::min();

        std::cout
            << "Vulkan device found: name=\"" << properties.deviceName
            << "\", type=" << DeviceTypeName(properties.deviceType)
            << ", suitable=" << (suitable ? "yes" : "no")
            << ", local_heap_mb="
            << static_cast<double>(localHeapBytes) / (1024.0 * 1024.0)
            << ", dedicated_transfer="
            << (candidateQueueFamilies.dedicatedTransfer ? "yes" : "no")
            << ", memory_budget="
            << (HasDeviceExtension(device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) ? "yes" : "no")
            << ", score=" << (suitable ? score : 0)
            << '\n';

        if (!suitable || !preferenceMatches || !nameMatches)
        {
            continue;
        }

        if (score > bestScore)
        {
            physicalDevice_ = device;
            selectedQueueFamilies = candidateQueueFamilies;
            selectedDeviceLocalHeapBytes_ = localHeapBytes;
            selectedDeviceScore_ = score;
            bestScore = score;
        }
    }

    if (physicalDevice_ != VK_NULL_HANDLE)
    {
        graphicsQueueFamilyIndex_ = selectedQueueFamilies.graphicsFamily.value();
        presentQueueFamilyIndex_ = selectedQueueFamilies.presentFamily.value();
        transferQueueFamilyIndex_ = selectedQueueFamilies.transferFamily.value();
        dedicatedTransferQueue_ = selectedQueueFamilies.dedicatedTransfer;

        VkPhysicalDeviceProperties selectedProperties{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &selectedProperties);
        deviceRuntimeStrategy_ = BuildDeviceRuntimeStrategy(
            selectedProperties,
            selectedDeviceLocalHeapBytes_);
        std::cout
            << "Selected Vulkan GPU: name=\"" << selectedProperties.deviceName
            << "\", type=" << DeviceTypeName(selectedProperties.deviceType)
            << ", score=" << selectedDeviceScore_
            << ", local_heap_mb="
            << static_cast<double>(selectedDeviceLocalHeapBytes_) / (1024.0 * 1024.0)
            << ", strategy=" << deviceRuntimeStrategy_.name
            << '\n';
    }

    if (physicalDevice_ == VK_NULL_HANDLE)
    {
        if (gpuPreference_ != GpuPreference::Auto || !gpuNameFilter_.empty())
        {
            throw std::runtime_error(
                std::string("Failed to find a Vulkan GPU matching preference=") +
                GpuPreferenceName(gpuPreference_) +
                ", name_filter=\"" +
                gpuNameFilter_ +
                "\".");
        }

        throw std::runtime_error("Failed to find a GPU with swapchain support.");
    }
}

void VulkanContext::CreateLogicalDevice()
{
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<std::uint32_t> uniqueQueueFamilies =
    {
        graphicsQueueFamilyIndex_,
        presentQueueFamilyIndex_,
        transferQueueFamilyIndex_,
    };

    float queuePriority = 1.0f;
    for (std::uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceVulkan12Features supportedVulkan12Features{};
    supportedVulkan12Features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures.pNext = &supportedVulkan12Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &supportedFeatures);

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.drawIndirectFirstInstance = VK_TRUE;
    deviceFeatures.pipelineStatisticsQuery =
        supportedFeatures.features.pipelineStatisticsQuery;
    pipelineStatisticsAvailable_ =
        supportedFeatures.features.pipelineStatisticsQuery == VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.timelineSemaphore = VK_TRUE;
    vulkan12Features.drawIndirectCount = VK_TRUE;
    vulkan12Features.hostQueryReset = supportedVulkan12Features.hostQueryReset;
    hostQueryResetAvailable_ = supportedVulkan12Features.hostQueryReset == VK_TRUE;

    std::vector<const char*> enabledExtensions = kDeviceExtensions;
    memoryBudgetExtensionEnabled_ =
        HasDeviceExtension(physicalDevice_, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (memoryBudgetExtensionEnabled_)
    {
        enabledExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &vulkan12Features;
    createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    if (kEnableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<std::uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan logical device.");
    }

    vkGetDeviceQueue(device_, graphicsQueueFamilyIndex_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentQueueFamilyIndex_, 0, &presentQueue_);
    vkGetDeviceQueue(device_, transferQueueFamilyIndex_, 0, &transferQueue_);

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    timestampPeriodNanoseconds_ = properties.limits.timestampPeriod;
    graphicsTimestampValidBits_ = QueueTimestampValidBits(
        physicalDevice_,
        graphicsQueueFamilyIndex_);
    transferTimestampValidBits_ = QueueTimestampValidBits(
        physicalDevice_,
        transferQueueFamilyIndex_);
    graphicsTimestampsAvailable_ =
        graphicsTimestampValidBits_ > 0 && timestampPeriodNanoseconds_ > 0.0f;
    transferTimestampsAvailable_ =
        transferTimestampValidBits_ > 0 &&
        timestampPeriodNanoseconds_ > 0.0f &&
        hostQueryResetAvailable_;

    std::cout
        << "Vulkan 1.2 features enabled: timeline_semaphore=yes"
        << ", draw_indirect_count=yes"
        << ", draw_indirect_first_instance=yes"
        << ", gpu_timestamp=" << (graphicsTimestampsAvailable_ ? "yes" : "no")
        << ", pipeline_statistics=" << (pipelineStatisticsAvailable_ ? "yes" : "no")
        << ", host_query_reset=" << (hostQueryResetAvailable_ ? "yes" : "no")
        << ", memory_budget=" << (memoryBudgetExtensionEnabled_ ? "yes" : "no")
        << '\n';
}

void VulkanContext::CreateSwapChain()
{
    SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(physicalDevice_, surface_);

    VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities, window_);

    const std::uint32_t preferredImageCount =
        presentMode == VK_PRESENT_MODE_MAILBOX_KHR ||
        presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR
        ? 3u
        : swapChainSupport.capabilities.minImageCount + 1;
    std::uint32_t imageCount = std::max(
        swapChainSupport.capabilities.minImageCount,
        preferredImageCount);
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    std::uint32_t queueFamilyIndices[] =
    {
        graphicsQueueFamilyIndex_,
        presentQueueFamilyIndex_,
    };

    if (graphicsQueueFamilyIndex_ != presentQueueFamilyIndex_)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan swapchain.");
    }

    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
    swapChainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, swapChainImages_.data());

    swapChainImageFormat_ = surfaceFormat.format;
    swapChainExtent_ = extent;

    std::cout
        << "Swapchain created: present_mode=" << PresentModeName(presentMode)
        << ", frame_limit="
        << (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "off" : "display_limited")
        << ", low_latency="
        << (presentMode != VK_PRESENT_MODE_FIFO_KHR ? "yes" : "fallback")
        << ", requested_images=" << createInfo.minImageCount
        << ", actual_images=" << swapChainImages_.size()
        << '\n';
}

void VulkanContext::CreateImageViews()
{
    swapChainImageViews_.resize(swapChainImages_.size());

    for (std::size_t i = 0; i < swapChainImages_.size(); ++i)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages_[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat_;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device_, &createInfo, nullptr, &swapChainImageViews_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swapchain image view.");
        }
    }
}

void VulkanContext::CreateRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan render pass.");
    }
}

void VulkanContext::CreateGpuProxyDescriptorSetLayout()
{
    if (gpuProxyDescriptorSetLayout_ != VK_NULL_HANDLE)
    {
        return;
    }

    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
        device_,
        &layoutInfo,
        nullptr,
        &gpuProxyDescriptorSetLayout_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create GPU proxy descriptor set layout.");
    }
}

void VulkanContext::CreateGpuPageDescriptorSetLayout()
{
    if (gpuPageDescriptorSetLayout_ != VK_NULL_HANDLE)
    {
        return;
    }

    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(
            device_,
            &layoutInfo,
            nullptr,
            &gpuPageDescriptorSetLayout_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create GPU Page descriptor set layout.");
    }
}

void VulkanContext::CreateGraphicsPipeline(
    bool createProxy,
    bool createGpuPage,
    bool createOverlay)
{
    const auto vertexShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/pointcloud.vert.spv");
    const auto fragmentShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/pointcloud.frag.spv");

    VkShaderModule vertexShaderModule = VulkanShader::CreateShaderModule(device_, vertexShaderCode);
    VkShaderModule fragmentShaderModule = VulkanShader::CreateShaderModule(device_, fragmentShaderCode);

    VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
    vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexShaderStageInfo.module = vertexShaderModule;
    vertexShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
    fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentShaderStageInfo.module = fragmentShaderModule;
    fragmentShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] =
    {
        vertexShaderStageInfo,
        fragmentShaderStageInfo,
    };

    const auto bindingDescription = GetPointBindingDescription();
    const auto attributeDescriptions = GetPointAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkDynamicState dynamicStates[] =
    {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPushConstantRange cameraPushConstantRange{};
    cameraPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    cameraPushConstantRange.offset = 0;
    cameraPushConstantRange.size = sizeof(CameraPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &cameraPushConstantRange;

    if (pipelineLayout_ == VK_NULL_HANDLE &&
        vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, vertexShaderModule, nullptr);
        throw std::runtime_error("Failed to create Vulkan pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;

    if (graphicsPipeline_ == VK_NULL_HANDLE &&
        vkCreateGraphicsPipelines(
            device_,
            pipelineCache_,
            1,
            &pipelineInfo,
            nullptr,
            &graphicsPipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, vertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create Vulkan graphics pipeline.");
    }

    vkDestroyShaderModule(device_, fragmentShaderModule, nullptr);
    vkDestroyShaderModule(device_, vertexShaderModule, nullptr);

    if (createProxy && gpuProxyGraphicsPipeline_ == VK_NULL_HANDLE)
    {
    CreateGpuProxyDescriptorSetLayout();
    const auto proxyVertexShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/tile_proxy.vert.spv");
    const auto proxyFragmentShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/pointcloud.frag.spv");

    VkShaderModule proxyVertexShaderModule =
        VulkanShader::CreateShaderModule(device_, proxyVertexShaderCode);
    VkShaderModule proxyFragmentShaderModule =
        VulkanShader::CreateShaderModule(device_, proxyFragmentShaderCode);

    VkPipelineShaderStageCreateInfo proxyVertexShaderStageInfo{};
    proxyVertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    proxyVertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    proxyVertexShaderStageInfo.module = proxyVertexShaderModule;
    proxyVertexShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo proxyFragmentShaderStageInfo{};
    proxyFragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    proxyFragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    proxyFragmentShaderStageInfo.module = proxyFragmentShaderModule;
    proxyFragmentShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo proxyShaderStages[] =
    {
        proxyVertexShaderStageInfo,
        proxyFragmentShaderStageInfo,
    };

    VkPipelineVertexInputStateCreateInfo proxyVertexInputInfo{};
    proxyVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineLayoutCreateInfo proxyPipelineLayoutInfo{};
    proxyPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    proxyPipelineLayoutInfo.setLayoutCount = 1;
    proxyPipelineLayoutInfo.pSetLayouts = &gpuProxyDescriptorSetLayout_;
    proxyPipelineLayoutInfo.pushConstantRangeCount = 1;
    proxyPipelineLayoutInfo.pPushConstantRanges = &cameraPushConstantRange;

    if (vkCreatePipelineLayout(
        device_,
        &proxyPipelineLayoutInfo,
        nullptr,
        &gpuProxyGraphicsPipelineLayout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, proxyFragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, proxyVertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU proxy graphics pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo proxyPipelineInfo = pipelineInfo;
    proxyPipelineInfo.pStages = proxyShaderStages;
    proxyPipelineInfo.pVertexInputState = &proxyVertexInputInfo;
    proxyPipelineInfo.layout = gpuProxyGraphicsPipelineLayout_;

    if (vkCreateGraphicsPipelines(
        device_,
        pipelineCache_,
        1,
        &proxyPipelineInfo,
        nullptr,
        &gpuProxyGraphicsPipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, proxyFragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, proxyVertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU proxy graphics pipeline.");
    }

    vkDestroyShaderModule(device_, proxyFragmentShaderModule, nullptr);
    vkDestroyShaderModule(device_, proxyVertexShaderModule, nullptr);

    const auto computeShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/tile_cull.comp.spv");
    VkShaderModule computeShaderModule =
        VulkanShader::CreateShaderModule(device_, computeShaderCode);

    VkPushConstantRange tileCullPushConstantRange{};
    tileCullPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tileCullPushConstantRange.offset = 0;
    tileCullPushConstantRange.size = sizeof(TileCullPushConstants);

    VkPipelineLayoutCreateInfo computePipelineLayoutInfo{};
    computePipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computePipelineLayoutInfo.setLayoutCount = 1;
    computePipelineLayoutInfo.pSetLayouts = &gpuProxyDescriptorSetLayout_;
    computePipelineLayoutInfo.pushConstantRangeCount = 1;
    computePipelineLayoutInfo.pPushConstantRanges = &tileCullPushConstantRange;

    if (vkCreatePipelineLayout(
        device_,
        &computePipelineLayoutInfo,
        nullptr,
        &gpuProxyComputePipelineLayout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, computeShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU proxy compute pipeline layout.");
    }

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineInfo{};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.stage = computeShaderStageInfo;
    computePipelineInfo.layout = gpuProxyComputePipelineLayout_;

    if (vkCreateComputePipelines(
        device_,
        pipelineCache_,
        1,
        &computePipelineInfo,
        nullptr,
        &gpuProxyComputePipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, computeShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU proxy compute pipeline.");
    }

    vkDestroyShaderModule(device_, computeShaderModule, nullptr);
    }

    if (createGpuPage && gpuPageGraphicsPipeline_ == VK_NULL_HANDLE)
    {
    CreateGpuPageDescriptorSetLayout();
    const auto pageVertexShaderCode =
        VulkanShader::ReadBinaryFile("shaders/spv/page_pointcloud.vert.spv");
    VkShaderModule pageVertexShaderModule =
        VulkanShader::CreateShaderModule(device_, pageVertexShaderCode);
    VkShaderModule pageFragmentShaderModule =
        VulkanShader::CreateShaderModule(device_, fragmentShaderCode);

    VkPipelineShaderStageCreateInfo pageVertexShaderStageInfo{};
    pageVertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pageVertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    pageVertexShaderStageInfo.module = pageVertexShaderModule;
    pageVertexShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo pageFragmentShaderStageInfo{};
    pageFragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pageFragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    pageFragmentShaderStageInfo.module = pageFragmentShaderModule;
    pageFragmentShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo pageShaderStages[] =
    {
        pageVertexShaderStageInfo,
        pageFragmentShaderStageInfo,
    };

    VkPipelineVertexInputStateCreateInfo pageVertexInputInfo{};
    pageVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineColorBlendAttachmentState pageColorBlendAttachment = colorBlendAttachment;
    pageColorBlendAttachment.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo pageColorBlending = colorBlending;
    pageColorBlending.pAttachments = &pageColorBlendAttachment;

    VkPipelineLayoutCreateInfo pageGraphicsLayoutInfo{};
    pageGraphicsLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pageGraphicsLayoutInfo.setLayoutCount = 1;
    pageGraphicsLayoutInfo.pSetLayouts = &gpuPageDescriptorSetLayout_;
    pageGraphicsLayoutInfo.pushConstantRangeCount = 1;
    pageGraphicsLayoutInfo.pPushConstantRanges = &cameraPushConstantRange;
    if (vkCreatePipelineLayout(
            device_,
            &pageGraphicsLayoutInfo,
            nullptr,
            &gpuPageGraphicsPipelineLayout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, pageFragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, pageVertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU Page graphics pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo pageGraphicsPipelineInfo = pipelineInfo;
    pageGraphicsPipelineInfo.pStages = pageShaderStages;
    pageGraphicsPipelineInfo.pVertexInputState = &pageVertexInputInfo;
    pageGraphicsPipelineInfo.pColorBlendState = &pageColorBlending;
    pageGraphicsPipelineInfo.layout = gpuPageGraphicsPipelineLayout_;
    if (vkCreateGraphicsPipelines(
            device_,
            pipelineCache_,
            1,
            &pageGraphicsPipelineInfo,
            nullptr,
            &gpuPageGraphicsPipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, pageFragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, pageVertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU Page graphics pipeline.");
    }

    vkDestroyShaderModule(device_, pageFragmentShaderModule, nullptr);
    vkDestroyShaderModule(device_, pageVertexShaderModule, nullptr);

    const auto pageComputeShaderCode =
        VulkanShader::ReadBinaryFile("shaders/spv/page_cull.comp.spv");
    VkShaderModule pageComputeShaderModule =
        VulkanShader::CreateShaderModule(device_, pageComputeShaderCode);

    VkPushConstantRange pageCullPushConstantRange{};
    pageCullPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pageCullPushConstantRange.offset = 0;
    pageCullPushConstantRange.size = sizeof(PageCullPushConstants);

    VkPipelineLayoutCreateInfo pageComputeLayoutInfo{};
    pageComputeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pageComputeLayoutInfo.setLayoutCount = 1;
    pageComputeLayoutInfo.pSetLayouts = &gpuPageDescriptorSetLayout_;
    pageComputeLayoutInfo.pushConstantRangeCount = 1;
    pageComputeLayoutInfo.pPushConstantRanges = &pageCullPushConstantRange;
    if (vkCreatePipelineLayout(
            device_,
            &pageComputeLayoutInfo,
            nullptr,
            &gpuPageComputePipelineLayout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, pageComputeShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU Page compute pipeline layout.");
    }

    VkPipelineShaderStageCreateInfo pageComputeStageInfo{};
    pageComputeStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pageComputeStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pageComputeStageInfo.module = pageComputeShaderModule;
    pageComputeStageInfo.pName = "main";

    VkComputePipelineCreateInfo pageComputePipelineInfo{};
    pageComputePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pageComputePipelineInfo.stage = pageComputeStageInfo;
    pageComputePipelineInfo.layout = gpuPageComputePipelineLayout_;
    if (vkCreateComputePipelines(
            device_,
            pipelineCache_,
            1,
            &pageComputePipelineInfo,
            nullptr,
            &gpuPageComputePipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, pageComputeShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create GPU Page compute pipeline.");
    }
    vkDestroyShaderModule(device_, pageComputeShaderModule, nullptr);
    }

    if (createOverlay && selectionOverlayPipeline_ == VK_NULL_HANDLE)
    {
    CreateSelectionOverlayBuffer();
    const auto overlayVertexShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/selection_overlay.vert.spv");
    const auto overlayFragmentShaderCode = VulkanShader::ReadBinaryFile("shaders/spv/selection_overlay.frag.spv");

    VkShaderModule overlayVertexShaderModule =
        VulkanShader::CreateShaderModule(device_, overlayVertexShaderCode);
    VkShaderModule overlayFragmentShaderModule =
        VulkanShader::CreateShaderModule(device_, overlayFragmentShaderCode);

    VkPipelineShaderStageCreateInfo overlayVertexShaderStageInfo{};
    overlayVertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    overlayVertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    overlayVertexShaderStageInfo.module = overlayVertexShaderModule;
    overlayVertexShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo overlayFragmentShaderStageInfo{};
    overlayFragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    overlayFragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    overlayFragmentShaderStageInfo.module = overlayFragmentShaderModule;
    overlayFragmentShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo overlayShaderStages[] =
    {
        overlayVertexShaderStageInfo,
        overlayFragmentShaderStageInfo,
    };

    const auto overlayBindingDescription = GetOverlayBindingDescription();
    const auto overlayAttributeDescriptions = GetOverlayAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo overlayVertexInputInfo{};
    overlayVertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    overlayVertexInputInfo.vertexBindingDescriptionCount = 1;
    overlayVertexInputInfo.pVertexBindingDescriptions = &overlayBindingDescription;
    overlayVertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(overlayAttributeDescriptions.size());
    overlayVertexInputInfo.pVertexAttributeDescriptions = overlayAttributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo overlayInputAssembly = inputAssembly;
    overlayInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineLayoutCreateInfo overlayPipelineLayoutInfo{};
    overlayPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    if (vkCreatePipelineLayout(
        device_,
        &overlayPipelineLayoutInfo,
        nullptr,
        &selectionOverlayPipelineLayout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, overlayFragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, overlayVertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create selection overlay pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo overlayPipelineInfo = pipelineInfo;
    overlayPipelineInfo.pStages = overlayShaderStages;
    overlayPipelineInfo.pVertexInputState = &overlayVertexInputInfo;
    overlayPipelineInfo.pInputAssemblyState = &overlayInputAssembly;
    overlayPipelineInfo.layout = selectionOverlayPipelineLayout_;

    if (vkCreateGraphicsPipelines(
        device_,
        pipelineCache_,
        1,
        &overlayPipelineInfo,
        nullptr,
        &selectionOverlayPipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, overlayFragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, overlayVertexShaderModule, nullptr);
        DestroyGraphicsPipeline();
        throw std::runtime_error("Failed to create selection overlay pipeline.");
    }

    vkDestroyShaderModule(device_, overlayFragmentShaderModule, nullptr);
    vkDestroyShaderModule(device_, overlayVertexShaderModule, nullptr);
    }
}

void VulkanContext::CreateFramebuffers()
{
    swapChainFramebuffers_.resize(swapChainImageViews_.size());

    for (std::size_t i = 0; i < swapChainImageViews_.size(); ++i)
    {
        VkImageView attachments[] = { swapChainImageViews_[i] };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass_;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent_.width;
        framebufferInfo.height = swapChainExtent_.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &swapChainFramebuffers_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swapchain framebuffer.");
        }
    }
}

void VulkanContext::CreateCommandPool()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex_;

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan command pool.");
    }
}

void VulkanContext::CreatePerformanceQueryResources()
{
    if (performanceQueryResourcesCreated_)
    {
        return;
    }
    performanceQueryResourcesCreated_ = true;

    if (graphicsTimestampsAvailable_)
    {
        VkQueryPoolCreateInfo timestampInfo{};
        timestampInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        timestampInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        timestampInfo.queryCount = kTimestampQueriesPerFrame * MaxFramesInFlight;
        if (vkCreateQueryPool(
                device_,
                &timestampInfo,
                nullptr,
                &graphicsTimestampQueryPool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create the graphics timestamp query pool.");
        }
    }

    if (pipelineStatisticsAvailable_)
    {
        VkQueryPoolCreateInfo statisticsInfo{};
        statisticsInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        statisticsInfo.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        statisticsInfo.queryCount = MaxFramesInFlight;
        statisticsInfo.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
        if (vkCreateQueryPool(
                device_,
                &statisticsInfo,
                nullptr,
                &pipelineStatisticsQueryPool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create the pipeline statistics query pool.");
        }
    }

    if (transferTimestampsAvailable_)
    {
        VkQueryPoolCreateInfo transferInfo{};
        transferInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        transferInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        transferInfo.queryCount =
            kTransferTimestampQueriesPerSlot * TilePageStagingRingSize;
        if (vkCreateQueryPool(
                device_,
                &transferInfo,
                nullptr,
                &transferTimestampQueryPool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create the transfer timestamp query pool.");
        }
    }
}

void VulkanContext::DestroyPerformanceQueryResources()
{
    if (graphicsTimestampQueryPool_ != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device_, graphicsTimestampQueryPool_, nullptr);
        graphicsTimestampQueryPool_ = VK_NULL_HANDLE;
    }
    if (pipelineStatisticsQueryPool_ != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device_, pipelineStatisticsQueryPool_, nullptr);
        pipelineStatisticsQueryPool_ = VK_NULL_HANDLE;
    }
    if (transferTimestampQueryPool_ != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device_, transferTimestampQueryPool_, nullptr);
        transferTimestampQueryPool_ = VK_NULL_HANDLE;
    }
    frameGpuQueriesPending_.fill(false);
    frameVisibleCountPending_.fill(false);
    frameHadActivePages_.fill(false);
    performanceQueryResourcesCreated_ = false;
}

void VulkanContext::CreateTransferTimelineSemaphore()
{
    if (tilePageTransferTimelineSemaphore_ != VK_NULL_HANDLE)
    {
        return;
    }

    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &timelineInfo;
    if (vkCreateSemaphore(
            device_,
            &semaphoreInfo,
            nullptr,
            &tilePageTransferTimelineSemaphore_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the Vulkan transfer timeline semaphore.");
    }
}

void VulkanContext::CreateTransferUploadResources()
{
    if (transferCommandPool_ != VK_NULL_HANDLE)
    {
        return;
    }
    CreateTransferTimelineSemaphore();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = transferQueueFamilyIndex_;
    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &transferCommandPool_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create the Vulkan transfer command pool.");
    }

    std::array<VkCommandBuffer, TilePageStagingRingSize> transferCommandBuffers{};
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = transferCommandPool_;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = TilePageStagingRingSize;
    if (vkAllocateCommandBuffers(
            device_,
            &allocateInfo,
            transferCommandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate Vulkan transfer command buffers.");
    }

    for (std::size_t slotIndex = 0; slotIndex < tilePageStagingRing_.size(); ++slotIndex)
    {
        TilePageStagingSlot& slot = tilePageStagingRing_[slotIndex];
        slot.commandBuffer = transferCommandBuffers[slotIndex];
        slot.buffer.Create(
            physicalDevice_,
            device_,
            kTilePageStagingBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    std::cout
        << "GPU Page transfer pipeline created: staging_ring="
        << TilePageStagingRingSize
        << ", staging_mb_each="
        << static_cast<double>(kTilePageStagingBytes) / (1024.0 * 1024.0)
        << ", queue=" << (dedicatedTransferQueue_ ? "dedicated" : "shared")
        << ", timeline=yes"
        << '\n';
}

void VulkanContext::DestroyTransferUploadResources()
{
    for (TilePageStagingSlot& slot : tilePageStagingRing_)
    {
        slot.buffer.Destroy();
        slot.commandBuffer = VK_NULL_HANDLE;
        slot.completionValue = 0;
        slot.timingBytes = 0;
        slot.timingPending = false;
    }

    if (tilePageTransferTimelineSemaphore_ != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(device_, tilePageTransferTimelineSemaphore_, nullptr);
        tilePageTransferTimelineSemaphore_ = VK_NULL_HANDLE;
    }
    if (transferCommandPool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device_, transferCommandPool_, nullptr);
        transferCommandPool_ = VK_NULL_HANDLE;
    }

    tilePageTransferNextValue_ = 0;
    gpuPageMetadataCompletedTransferValue_ = 0;
    tilePageTransferSubmittedBytes_ = 0;
    tilePageTransferSubmitCount_ = 0;
    tilePageTransferDeferredCount_ = 0;
    tilePageTransferGpuMeasuredBytes_ = 0;
    tilePageTransferCpuWaitMilliseconds_ = 0.0;
    tilePageTransferGpuMilliseconds_ = 0.0;
    tilePageTransferLastBatchGpuMilliseconds_ = 0.0;
    nextTilePageStagingSlot_ = 0;
}

void VulkanContext::CreateSelectionOverlayBuffer()
{
    if (selectionOverlayVertexBuffers_[0].Handle() != VK_NULL_HANDLE)
    {
        return;
    }

    constexpr VkDeviceSize bufferSize =
        sizeof(OverlayVertex) * kMaximumOverlayVertices;

    for (VulkanBuffer& selectionOverlayVertexBuffer : selectionOverlayVertexBuffers_)
    {
        selectionOverlayVertexBuffer.Create(
            physicalDevice_,
            device_,
            bufferSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    selectionOverlayVertexCount_ =
        static_cast<std::uint32_t>(kMaximumOverlayVertices);
}

void VulkanContext::CreateOrUpdateGpuProxyDescriptors()
{
    if (gpuProxyDescriptorPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gpuProxyDescriptorPool_, nullptr);
        gpuProxyDescriptorPool_ = VK_NULL_HANDLE;
        gpuProxyDescriptorSets_.fill(VK_NULL_HANDLE);
    }

    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 4 * MaxFramesInFlight;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = MaxFramesInFlight;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &gpuProxyDescriptorPool_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create GPU proxy descriptor pool.");
    }

    std::array<VkDescriptorSetLayout, MaxFramesInFlight> layouts{};
    layouts.fill(gpuProxyDescriptorSetLayout_);

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = gpuProxyDescriptorPool_;
    allocateInfo.descriptorSetCount = MaxFramesInFlight;
    allocateInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(device_, &allocateInfo, gpuProxyDescriptorSets_.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate GPU proxy descriptor set.");
    }

    VkDescriptorBufferInfo tileMetadataInfo{};
    tileMetadataInfo.buffer = gpuTileMetadataBuffer_.Handle();
    tileMetadataInfo.offset = 0;
    tileMetadataInfo.range = gpuTileMetadataBuffer_.Size();

    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        VkDescriptorBufferInfo visibleTileInfo{};
        visibleTileInfo.buffer = gpuVisibleTileIdsBuffers_[frameIndex].Handle();
        visibleTileInfo.offset = 0;
        visibleTileInfo.range = gpuVisibleTileIdsBuffers_[frameIndex].Size();

        VkDescriptorBufferInfo indirectInfo{};
        indirectInfo.buffer = gpuProxyIndirectBuffers_[frameIndex].Handle();
        indirectInfo.offset = 0;
        indirectInfo.range = gpuProxyIndirectBuffers_[frameIndex].Size();

        VkDescriptorBufferInfo proxyMaskInfo{};
        proxyMaskInfo.buffer = gpuProxyMaskBuffers_[frameIndex].Handle();
        proxyMaskInfo.offset = 0;
        proxyMaskInfo.range = gpuProxyMaskBuffers_[frameIndex].Size();

        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = gpuProxyDescriptorSets_[frameIndex];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].pBufferInfo = &tileMetadataInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = gpuProxyDescriptorSets_[frameIndex];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].pBufferInfo = &visibleTileInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = gpuProxyDescriptorSets_[frameIndex];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].pBufferInfo = &indirectInfo;

        descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[3].dstSet = gpuProxyDescriptorSets_[frameIndex];
        descriptorWrites[3].dstBinding = 3;
        descriptorWrites[3].descriptorCount = 1;
        descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[3].pBufferInfo = &proxyMaskInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(),
            0,
            nullptr);
    }
}

void VulkanContext::CreateOrUpdateGpuPageDescriptors()
{
    if (gpuPageDescriptorPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gpuPageDescriptorPool_, nullptr);
        gpuPageDescriptorPool_ = VK_NULL_HANDLE;
        gpuPageDescriptorSets_.fill(VK_NULL_HANDLE);
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 5 * MaxFramesInFlight;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = MaxFramesInFlight;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &gpuPageDescriptorPool_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create GPU Page descriptor pool.");
    }

    std::array<VkDescriptorSetLayout, MaxFramesInFlight> layouts{};
    layouts.fill(gpuPageDescriptorSetLayout_);

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = gpuPageDescriptorPool_;
    allocateInfo.descriptorSetCount = MaxFramesInFlight;
    allocateInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &allocateInfo, gpuPageDescriptorSets_.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate GPU Page descriptor sets.");
    }

    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        VkDescriptorBufferInfo metadataInfo{};
        metadataInfo.buffer = gpuPageMetadataBuffers_[frameIndex].Handle();
        metadataInfo.offset = 0;
        metadataInfo.range = gpuPageMetadataBuffers_[frameIndex].Size();

        VkDescriptorBufferInfo indirectInfo{};
        indirectInfo.buffer = gpuPageIndirectBuffers_[frameIndex].Handle();
        indirectInfo.offset = 0;
        indirectInfo.range = gpuPageIndirectBuffers_[frameIndex].Size();

        VkDescriptorBufferInfo countInfo{};
        countInfo.buffer = gpuPageIndirectCountBuffers_[frameIndex].Handle();
        countInfo.offset = 0;
        countInfo.range = sizeof(std::uint32_t);

        VkDescriptorBufferInfo pointPoolInfo{};
        pointPoolInfo.buffer = tilePagePoolBuffer_.Handle();
        pointPoolInfo.offset = 0;
        pointPoolInfo.range = tilePagePoolBuffer_.Size();

        VkDescriptorBufferInfo colormapInfo{};
        colormapInfo.buffer = gpuPageColormapBuffer_.Handle();
        colormapInfo.offset = 0;
        colormapInfo.range = gpuPageColormapBuffer_.Size();

        std::array<VkWriteDescriptorSet, 5> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = gpuPageDescriptorSets_[frameIndex];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &metadataInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = gpuPageDescriptorSets_[frameIndex];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &indirectInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = gpuPageDescriptorSets_[frameIndex];
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &countInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = gpuPageDescriptorSets_[frameIndex];
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &pointPoolInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = gpuPageDescriptorSets_[frameIndex];
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &colormapInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(writes.size()),
            writes.data(),
            0,
            nullptr);
    }
}

void VulkanContext::CreateCommandBuffers()
{
    commandBuffers_.resize(MaxFramesInFlight);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<std::uint32_t>(commandBuffers_.size());

    if (vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate Vulkan command buffers.");
    }
}

void VulkanContext::CreateSyncObjects()
{
    imageAvailableSemaphores_.resize(MaxFramesInFlight);
    inFlightFences_.resize(MaxFramesInFlight);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MaxFramesInFlight; ++i)
    {
        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan synchronization objects.");
        }
    }

    CreateRenderFinishedSemaphores();
}

void VulkanContext::CreateRenderFinishedSemaphores()
{
    renderFinishedSemaphores_.resize(swapChainImages_.size());

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (std::size_t i = 0; i < renderFinishedSemaphores_.size(); ++i)
    {
        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create per-image render-finished semaphore.");
        }
    }
}

void VulkanContext::DestroyGraphicsPipeline()
{
    if (gpuPageComputePipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gpuPageComputePipeline_, nullptr);
        gpuPageComputePipeline_ = VK_NULL_HANDLE;
    }

    if (gpuPageComputePipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, gpuPageComputePipelineLayout_, nullptr);
        gpuPageComputePipelineLayout_ = VK_NULL_HANDLE;
    }

    if (gpuPageGraphicsPipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gpuPageGraphicsPipeline_, nullptr);
        gpuPageGraphicsPipeline_ = VK_NULL_HANDLE;
    }

    if (gpuPageGraphicsPipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, gpuPageGraphicsPipelineLayout_, nullptr);
        gpuPageGraphicsPipelineLayout_ = VK_NULL_HANDLE;
    }

    if (gpuProxyComputePipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gpuProxyComputePipeline_, nullptr);
        gpuProxyComputePipeline_ = VK_NULL_HANDLE;
    }

    if (gpuProxyComputePipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, gpuProxyComputePipelineLayout_, nullptr);
        gpuProxyComputePipelineLayout_ = VK_NULL_HANDLE;
    }

    if (gpuProxyGraphicsPipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gpuProxyGraphicsPipeline_, nullptr);
        gpuProxyGraphicsPipeline_ = VK_NULL_HANDLE;
    }

    if (gpuProxyGraphicsPipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, gpuProxyGraphicsPipelineLayout_, nullptr);
        gpuProxyGraphicsPipelineLayout_ = VK_NULL_HANDLE;
    }

    if (selectionOverlayPipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, selectionOverlayPipeline_, nullptr);
        selectionOverlayPipeline_ = VK_NULL_HANDLE;
    }

    if (selectionOverlayPipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, selectionOverlayPipelineLayout_, nullptr);
        selectionOverlayPipelineLayout_ = VK_NULL_HANDLE;
    }

    if (graphicsPipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        graphicsPipeline_ = VK_NULL_HANDLE;
    }

    if (pipelineLayout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::DestroyGpuProxyResources()
{
    gpuProxyDrawEnabled_ = false;
    gpuTileCount_ = 0;
    gpuTileMetadataBytes_ = 0;

    gpuTileMetadataBuffer_.Destroy();
    for (int frameIndex = 0; frameIndex < MaxFramesInFlight; ++frameIndex)
    {
        gpuVisibleTileIdsBuffers_[frameIndex].Destroy();
        gpuProxyIndirectBuffers_[frameIndex].Destroy();
        gpuProxyMaskBuffers_[frameIndex].Destroy();
    }
    pendingGpuProxyMask_.clear();
    gpuProxyMaskDirty_.fill(false);

    if (gpuProxyDescriptorPool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gpuProxyDescriptorPool_, nullptr);
        gpuProxyDescriptorPool_ = VK_NULL_HANDLE;
        gpuProxyDescriptorSets_.fill(VK_NULL_HANDLE);
    }
}

void VulkanContext::DestroyRenderFinishedSemaphores()
{
    for (VkSemaphore semaphore : renderFinishedSemaphores_)
    {
        vkDestroySemaphore(device_, semaphore, nullptr);
    }

    renderFinishedSemaphores_.clear();
}

void VulkanContext::CleanupSwapChain()
{
    for (VkFramebuffer framebuffer : swapChainFramebuffers_)
    {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    swapChainFramebuffers_.clear();

    for (VkImageView imageView : swapChainImageViews_)
    {
        vkDestroyImageView(device_, imageView, nullptr);
    }
    swapChainImageViews_.clear();
    swapChainImages_.clear();

    if (swapChain_ != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device_, swapChain_, nullptr);
        swapChain_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::RecreateSwapChain()
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device_);

    DestroyRenderFinishedSemaphores();
    CleanupSwapChain();
    CreateSwapChain();
    CreateImageViews();
    CreateFramebuffers();
    CreateRenderFinishedSemaphores();
}

void VulkanContext::RecordGpuProxyCull(
    VkCommandBuffer commandBuffer,
    const Bounds2D& viewportBounds,
    std::uint32_t frameIndex)
{
    if (!gpuProxyDrawEnabled_ || frameIndex >= gpuProxyDescriptorSets_.size())
    {
        return;
    }

    VulkanBuffer& visibleTileIdsBuffer = gpuVisibleTileIdsBuffers_[frameIndex];
    VulkanBuffer& indirectBuffer = gpuProxyIndirectBuffers_[frameIndex];
    VulkanBuffer& proxyMaskBuffer = gpuProxyMaskBuffers_[frameIndex];
    const VkDescriptorSet descriptorSet = gpuProxyDescriptorSets_[frameIndex];

    if (descriptorSet == VK_NULL_HANDLE ||
        visibleTileIdsBuffer.Handle() == VK_NULL_HANDLE ||
        indirectBuffer.Handle() == VK_NULL_HANDLE ||
        proxyMaskBuffer.Handle() == VK_NULL_HANDLE)
    {
        return;
    }

    vkCmdFillBuffer(
        commandBuffer,
        indirectBuffer.Handle(),
        0,
        sizeof(VkDrawIndirectCommand),
        0);

    const std::uint32_t instanceCount = 1;
    vkCmdUpdateBuffer(
        commandBuffer,
        indirectBuffer.Handle(),
        sizeof(std::uint32_t),
        sizeof(instanceCount),
        &instanceCount);

    std::array<VkBufferMemoryBarrier, 2> beforeCompute{};

    beforeCompute[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    beforeCompute[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    beforeCompute[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    beforeCompute[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[0].buffer = indirectBuffer.Handle();
    beforeCompute[0].offset = 0;
    beforeCompute[0].size = sizeof(VkDrawIndirectCommand);

    beforeCompute[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    beforeCompute[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    beforeCompute[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    beforeCompute[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[1].buffer = proxyMaskBuffer.Handle();
    beforeCompute[1].offset = 0;
    beforeCompute[1].size = proxyMaskBuffer.Size();

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<std::uint32_t>(beforeCompute.size()),
        beforeCompute.data(),
        0,
        nullptr);

    TileCullPushConstants pushConstants{};
    pushConstants.viewportBounds = glm::vec4(
        static_cast<float>(viewportBounds.minX),
        static_cast<float>(viewportBounds.minY),
        static_cast<float>(viewportBounds.maxX),
        static_cast<float>(viewportBounds.maxY));
    pushConstants.tileCount = gpuTileCount_;
    pushConstants.maxVisibleTiles = gpuTileCount_;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpuProxyComputePipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gpuProxyComputePipelineLayout_,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
    vkCmdPushConstants(
        commandBuffer,
        gpuProxyComputePipelineLayout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(TileCullPushConstants),
        &pushConstants);

    const std::uint32_t groupCount =
        (gpuTileCount_ + kGpuProxyCullGroupSize - 1u) / kGpuProxyCullGroupSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);

    std::array<VkBufferMemoryBarrier, 2> computeToDraw{};

    computeToDraw[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    computeToDraw[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToDraw[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    computeToDraw[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[0].buffer = visibleTileIdsBuffer.Handle();
    computeToDraw[0].offset = 0;
    computeToDraw[0].size = visibleTileIdsBuffer.Size();

    computeToDraw[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    computeToDraw[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToDraw[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    computeToDraw[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[1].buffer = indirectBuffer.Handle();
    computeToDraw[1].offset = 0;
    computeToDraw[1].size = sizeof(VkDrawIndirectCommand);

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0,
        0,
        nullptr,
        static_cast<std::uint32_t>(computeToDraw.size()),
        computeToDraw.data(),
        0,
        nullptr);
}

void VulkanContext::RecordGpuPageCull(
    VkCommandBuffer commandBuffer,
    const Bounds2D& viewportBounds,
    double renderOriginX,
    double renderOriginY,
    std::uint32_t frameIndex)
{
    if (activeGpuPageDrawCount_ == 0 || activeTilePageLodLevel_ < 0 ||
        frameIndex >= gpuPageDescriptorSets_.size())
    {
        return;
    }

    VulkanBuffer& metadataBuffer = gpuPageMetadataBuffers_[frameIndex];
    VulkanBuffer& indirectBuffer = gpuPageIndirectBuffers_[frameIndex];
    VulkanBuffer& countBuffer = gpuPageIndirectCountBuffers_[frameIndex];
    const VkDescriptorSet descriptorSet = gpuPageDescriptorSets_[frameIndex];
    if (descriptorSet == VK_NULL_HANDLE ||
        metadataBuffer.Handle() == VK_NULL_HANDLE ||
        indirectBuffer.Handle() == VK_NULL_HANDLE ||
        countBuffer.Handle() == VK_NULL_HANDLE ||
        gpuPageComputePipeline_ == VK_NULL_HANDLE)
    {
        throw std::runtime_error("GPU Page culling resources are incomplete.");
    }

    vkCmdFillBuffer(
        commandBuffer,
        countBuffer.Handle(),
        0,
        sizeof(std::uint32_t),
        0);

    std::array<VkBufferMemoryBarrier, 2> beforeCompute{};
    beforeCompute[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    beforeCompute[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    beforeCompute[0].dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    beforeCompute[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[0].buffer = countBuffer.Handle();
    beforeCompute[0].offset = 0;
    beforeCompute[0].size = sizeof(std::uint32_t);

    beforeCompute[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    beforeCompute[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    beforeCompute[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    beforeCompute[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeCompute[1].buffer = metadataBuffer.Handle();
    beforeCompute[1].offset = 0;
    beforeCompute[1].size =
        sizeof(GpuPageDrawMetadata) * static_cast<VkDeviceSize>(activeGpuPageDrawCount_);

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<std::uint32_t>(beforeCompute.size()),
        beforeCompute.data(),
        0,
        nullptr);

    PageCullPushConstants pushConstants{};
    pushConstants.viewportBounds = glm::vec4(
        ConservativeMinimum(viewportBounds.minX - renderOriginX),
        ConservativeMinimum(viewportBounds.minY - renderOriginY),
        ConservativeMaximum(viewportBounds.maxX - renderOriginX),
        ConservativeMaximum(viewportBounds.maxY - renderOriginY));
    pushConstants.renderOriginSplit = SplitWorldOrigin(renderOriginX, renderOriginY);
    pushConstants.pageCount = activeGpuPageDrawCount_;
    pushConstants.maxDrawCount = gpuPageDrawCapacity_;
    pushConstants.activeLodLevel = activeTilePageLodLevel_;

    vkCmdBindPipeline(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gpuPageComputePipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        gpuPageComputePipelineLayout_,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
    vkCmdPushConstants(
        commandBuffer,
        gpuPageComputePipelineLayout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(PageCullPushConstants),
        &pushConstants);

    const std::uint32_t groupCount =
        (activeGpuPageDrawCount_ + kGpuPageCullGroupSize - 1u) /
        kGpuPageCullGroupSize;
    vkCmdDispatch(commandBuffer, groupCount, 1, 1);

    std::array<VkBufferMemoryBarrier, 2> computeToDraw{};
    computeToDraw[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    computeToDraw[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToDraw[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    computeToDraw[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[0].buffer = indirectBuffer.Handle();
    computeToDraw[0].offset = 0;
    computeToDraw[0].size =
        sizeof(VkDrawIndirectCommand) * static_cast<VkDeviceSize>(activeGpuPageDrawCount_);

    computeToDraw[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    computeToDraw[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeToDraw[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    computeToDraw[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    computeToDraw[1].buffer = countBuffer.Handle();
    computeToDraw[1].offset = 0;
    computeToDraw[1].size = sizeof(std::uint32_t);

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0,
        0,
        nullptr,
        static_cast<std::uint32_t>(computeToDraw.size()),
        computeToDraw.data(),
        0,
        nullptr);
}

void VulkanContext::RecordGpuPageVisibleCountReadback(
    VkCommandBuffer commandBuffer,
    std::uint32_t frameIndex)
{
    if (activeGpuPageDrawCount_ == 0 || activeTilePageLodLevel_ < 0 ||
        frameIndex >= gpuPageIndirectCountBuffers_.size())
    {
        return;
    }

    VulkanBuffer& countBuffer = gpuPageIndirectCountBuffers_[frameIndex];
    VulkanBuffer& readbackBuffer = gpuPageVisibleCountReadbackBuffers_[frameIndex];

    VkBufferMemoryBarrier countToTransfer{};
    countToTransfer.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    countToTransfer.srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    countToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    countToTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    countToTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    countToTransfer.buffer = countBuffer.Handle();
    countToTransfer.offset = 0;
    countToTransfer.size = sizeof(std::uint32_t);
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &countToTransfer,
        0,
        nullptr);

    VkBufferCopy copyRegion{};
    copyRegion.size = sizeof(std::uint32_t);
    vkCmdCopyBuffer(
        commandBuffer,
        countBuffer.Handle(),
        readbackBuffer.Handle(),
        1,
        &copyRegion);

    VkBufferMemoryBarrier transferToHost{};
    transferToHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    transferToHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    transferToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    transferToHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transferToHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    transferToHost.buffer = readbackBuffer.Handle();
    transferToHost.offset = 0;
    transferToHost.size = sizeof(std::uint32_t);
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0,
        nullptr,
        1,
        &transferToHost,
        0,
        nullptr);
}

void VulkanContext::CollectFrameGpuPerformance(
    std::uint32_t frameIndex,
    FrameTimingStats& timing)
{
    if (frameIndex >= MaxFramesInFlight)
    {
        return;
    }

    if (frameGpuQueriesPending_[frameIndex])
    {
        if (graphicsTimestampQueryPool_ != VK_NULL_HANDLE)
        {
            std::array<std::uint64_t, kTimestampQueriesPerFrame> timestamps{};
            const std::uint32_t firstQuery = frameIndex * kTimestampQueriesPerFrame;
            const VkResult result = vkGetQueryPoolResults(
                device_,
                graphicsTimestampQueryPool_,
                firstQuery,
                kTimestampQueriesPerFrame,
                sizeof(timestamps),
                timestamps.data(),
                sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (result == VK_SUCCESS)
            {
                timing.gpuFrameMilliseconds = TimestampMilliseconds(
                    timestamps[FrameBeginTimestamp],
                    timestamps[FrameEndTimestamp],
                    graphicsTimestampValidBits_,
                    timestampPeriodNanoseconds_);
                timing.gpuCullMilliseconds = TimestampMilliseconds(
                    timestamps[CullBeginTimestamp],
                    timestamps[CullEndTimestamp],
                    graphicsTimestampValidBits_,
                    timestampPeriodNanoseconds_);
                timing.gpuDrawMilliseconds = TimestampMilliseconds(
                    timestamps[DrawBeginTimestamp],
                    timestamps[DrawEndTimestamp],
                    graphicsTimestampValidBits_,
                    timestampPeriodNanoseconds_);
                timing.gpuTimestampsValid = true;
            }
        }

        if (pipelineStatisticsQueryPool_ != VK_NULL_HANDLE)
        {
            std::array<std::uint64_t, 3> statistics{};
            const VkResult result = vkGetQueryPoolResults(
                device_,
                pipelineStatisticsQueryPool_,
                frameIndex,
                1,
                sizeof(statistics),
                statistics.data(),
                sizeof(statistics),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (result == VK_SUCCESS)
            {
                timing.inputAssemblyVertices = statistics[0];
                timing.vertexShaderInvocations = statistics[1];
                timing.computeShaderInvocations = statistics[2];
                timing.pipelineStatisticsValid = true;
            }
        }
        frameGpuQueriesPending_[frameIndex] = false;
    }

    if (frameVisibleCountPending_[frameIndex])
    {
        if (frameHadActivePages_[frameIndex])
        {
            gpuPageVisibleCountReadbackBuffers_[frameIndex].Download(
                &timing.visiblePageCount,
                sizeof(timing.visiblePageCount));
        }
        timing.visiblePageCountValid = true;
        frameVisibleCountPending_[frameIndex] = false;
        frameHadActivePages_[frameIndex] = false;
    }
}

void VulkanContext::RecordCommandBuffer(
    VkCommandBuffer commandBuffer,
    std::uint32_t imageIndex,
    const glm::mat4& viewProjection,
    const Bounds2D& viewportBounds,
    double renderOriginX,
    double renderOriginY,
    const BoxSelectionState& boxSelection,
    const std::vector<glm::vec2>& profileOverlayNdc)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin recording command buffer.");
    }

    const std::uint32_t timestampBase = currentFrame_ * kTimestampQueriesPerFrame;
    if (graphicsTimestampQueryPool_ != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(
            commandBuffer,
            graphicsTimestampQueryPool_,
            timestampBase,
            kTimestampQueriesPerFrame);
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            graphicsTimestampQueryPool_,
            timestampBase + FrameBeginTimestamp);
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            graphicsTimestampQueryPool_,
            timestampBase + CullBeginTimestamp);
    }
    if (pipelineStatisticsQueryPool_ != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(
            commandBuffer,
            pipelineStatisticsQueryPool_,
            currentFrame_,
            1);
        vkCmdBeginQuery(
            commandBuffer,
            pipelineStatisticsQueryPool_,
            currentFrame_,
            0);
    }

    RecordGpuProxyCull(commandBuffer, viewportBounds, currentFrame_);
    RecordGpuPageCull(
        commandBuffer,
        viewportBounds,
        renderOriginX,
        renderOriginY,
        currentFrame_);

    if (graphicsTimestampQueryPool_ != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            graphicsTimestampQueryPool_,
            timestampBase + CullEndTimestamp);
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            graphicsTimestampQueryPool_,
            timestampBase + DrawBeginTimestamp);
    }

    VkClearValue clearColor = { {{0.02f, 0.03f, 0.06f, 1.0f}} };

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass_;
    renderPassInfo.framebuffer = swapChainFramebuffers_[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent_;
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent_.width);
    viewport.height = static_cast<float>(swapChainExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = swapChainExtent_;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    CameraPushConstants cameraPushConstants{};
    cameraPushConstants.viewProjection = viewProjection;
    cameraPushConstants.selectionRect = glm::vec4(
        boxSelection.minNdcX,
        boxSelection.minNdcY,
        boxSelection.maxNdcX,
        boxSelection.maxNdcY);
    cameraPushConstants.selectionParams = glm::vec4(
        boxSelection.active ? 1.0f : 0.0f,
        boxSelection.flashIntensity,
        static_cast<float>(activePointAttributeIndex_),
        0.0f);
    cameraPushConstants.renderOrigin = SplitWorldOrigin(renderOriginX, renderOriginY);

    VkDeviceSize offsets[] = { 0 };
    if (gpuProxyDrawEnabled_)
    {
        const VkDescriptorSet descriptorSet = gpuProxyDescriptorSets_[currentFrame_];
        VulkanBuffer& indirectBuffer = gpuProxyIndirectBuffers_[currentFrame_];

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gpuProxyGraphicsPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            gpuProxyGraphicsPipelineLayout_,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);
        vkCmdPushConstants(
            commandBuffer,
            gpuProxyGraphicsPipelineLayout_,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(CameraPushConstants),
            &cameraPushConstants);
        vkCmdDrawIndirect(
            commandBuffer,
            indirectBuffer.Handle(),
            0,
            1,
            sizeof(VkDrawIndirectCommand));
    }
    if (pointVertexCount_ > 0)
    {
        CameraPushConstants pointPushConstants = cameraPushConstants;
        pointPushConstants.renderOrigin = glm::vec4(0.0f);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout_,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(CameraPushConstants),
            &pointPushConstants);

        VkBuffer vertexBuffers[] = { pointVertexBuffers_[currentFrame_].Handle() };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdDraw(commandBuffer, pointVertexCount_, 1, 0, 0);
    }

    if (activeGpuPageDrawCount_ > 0 && activeTilePageLodLevel_ >= 0)
    {
        const VkDescriptorSet descriptorSet = gpuPageDescriptorSets_[currentFrame_];
        VulkanBuffer& indirectBuffer = gpuPageIndirectBuffers_[currentFrame_];
        VulkanBuffer& countBuffer = gpuPageIndirectCountBuffers_[currentFrame_];
        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            gpuPageGraphicsPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            gpuPageGraphicsPipelineLayout_,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);
        vkCmdPushConstants(
            commandBuffer,
            gpuPageGraphicsPipelineLayout_,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(CameraPushConstants),
            &cameraPushConstants);
        vkCmdDrawIndirectCount(
            commandBuffer,
            indirectBuffer.Handle(),
            0,
            countBuffer.Handle(),
            0,
            gpuPageDrawCapacity_,
            sizeof(VkDrawIndirectCommand));
    }

    if (boxSelection.active || !profileOverlayNdc.empty())
    {
        const std::vector<OverlayVertex> overlayVertices =
            CreateOverlayVertices(boxSelection, profileOverlayNdc);
        VulkanBuffer& selectionOverlayVertexBuffer = selectionOverlayVertexBuffers_[currentFrame_];
        if (!overlayVertices.empty())
        {
            selectionOverlayVertexBuffer.Upload(
                overlayVertices.data(),
                sizeof(overlayVertices[0]) * overlayVertices.size());

            vkCmdBindPipeline(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                selectionOverlayPipeline_);

            VkBuffer overlayVertexBuffers[] = { selectionOverlayVertexBuffer.Handle() };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, overlayVertexBuffers, offsets);
            vkCmdDraw(
                commandBuffer,
                static_cast<std::uint32_t>(overlayVertices.size()),
                1,
                0,
                0);
        }
    }

    vkCmdEndRenderPass(commandBuffer);

    if (graphicsTimestampQueryPool_ != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            graphicsTimestampQueryPool_,
            timestampBase + DrawEndTimestamp);
    }
    if (pipelineStatisticsQueryPool_ != VK_NULL_HANDLE)
    {
        vkCmdEndQuery(
            commandBuffer,
            pipelineStatisticsQueryPool_,
            currentFrame_);
    }

    RecordGpuPageVisibleCountReadback(commandBuffer, currentFrame_);

    if (graphicsTimestampQueryPool_ != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            graphicsTimestampQueryPool_,
            timestampBase + FrameEndTimestamp);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer.");
    }
}

} // namespace gpv
