// Vulkan 缓冲区声明：封装 VkBuffer、显存分配和简单 CPU 写入上传。
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>

namespace gpv
{

class VulkanBuffer
{
public:
    VulkanBuffer() = default;
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    void Create(
        VkPhysicalDevice physicalDevice,
        VkDevice device,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        const std::uint32_t* queueFamilyIndices = nullptr,
        std::uint32_t queueFamilyIndexCount = 0);
    void Upload(const void* data, VkDeviceSize size);
    void UploadAt(const void* data, VkDeviceSize size, VkDeviceSize offset);
    void Download(void* data, VkDeviceSize size, VkDeviceSize offset = 0) const;
    void Destroy();

    VkBuffer Handle() const;
    VkDeviceSize Size() const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mappedMemory_ = nullptr;
};

} // namespace gpv
