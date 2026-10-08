// Vulkan 缓冲区实现：封装缓冲创建、显存分配、共享模式和 CPU 写入。
#include "VulkanBuffer.h"

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <stdexcept>

namespace gpv
{
namespace
{

std::uint32_t FindMemoryType(
    VkPhysicalDevice physicalDevice,
    std::uint32_t typeFilter,
    VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        const bool typeMatches = (typeFilter & (1u << i)) != 0;
        const bool propertiesMatch =
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;

        if (typeMatches && propertiesMatch)
        {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable Vulkan memory type.");
}

} // namespace

VulkanBuffer::~VulkanBuffer()
{
    Destroy();
}

void VulkanBuffer::Create(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    const std::uint32_t* queueFamilyIndices,
    std::uint32_t queueFamilyIndexCount)
{
    Destroy();

    device_ = device;
    size_ = size;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_;
    bufferInfo.usage = usage;
    if (queueFamilyIndices != nullptr && queueFamilyIndexCount > 1)
    {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = queueFamilyIndexCount;
        bufferInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan buffer.");
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device_, buffer_, &memoryRequirements);

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = FindMemoryType(
        physicalDevice,
        memoryRequirements.memoryTypeBits,
        properties);

    if (vkAllocateMemory(device_, &allocateInfo, nullptr, &memory_) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate Vulkan buffer memory.");
    }

    vkBindBufferMemory(device_, buffer_, memory_, 0);

    if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        if (vkMapMemory(device_, memory_, 0, size_, 0, &mappedMemory_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to map Vulkan buffer memory.");
        }
    }
}

void VulkanBuffer::Upload(const void* data, VkDeviceSize size)
{
    UploadAt(data, size, 0);
}

void VulkanBuffer::UploadAt(const void* data, VkDeviceSize size, VkDeviceSize offset)
{
    if (offset > size_ || size > size_ - offset)
    {
        throw std::runtime_error("Upload size is larger than Vulkan buffer size.");
    }
    if (size == 0)
    {
        return;
    }
    if (mappedMemory_ == nullptr)
    {
        throw std::runtime_error("Vulkan buffer memory is not host visible.");
    }

    auto* destination = static_cast<std::byte*>(mappedMemory_) + offset;
    std::memcpy(destination, data, static_cast<std::size_t>(size));
}

void VulkanBuffer::Download(void* data, VkDeviceSize size, VkDeviceSize offset) const
{
    if (offset > size_ || size > size_ - offset)
    {
        throw std::runtime_error("Download size is larger than Vulkan buffer size.");
    }
    if (size == 0)
    {
        return;
    }
    if (mappedMemory_ == nullptr)
    {
        throw std::runtime_error("Vulkan buffer memory is not host visible.");
    }

    const auto* source = static_cast<const std::byte*>(mappedMemory_) + offset;
    std::memcpy(data, source, static_cast<std::size_t>(size));
}

void VulkanBuffer::Destroy()
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }

    if (buffer_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }

    if (memory_ != VK_NULL_HANDLE)
    {
        if (mappedMemory_ != nullptr)
        {
            vkUnmapMemory(device_, memory_);
            mappedMemory_ = nullptr;
        }
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }

    size_ = 0;
    mappedMemory_ = nullptr;
    device_ = VK_NULL_HANDLE;
}

VkBuffer VulkanBuffer::Handle() const
{
    return buffer_;
}

VkDeviceSize VulkanBuffer::Size() const
{
    return size_;
}

} // namespace gpv
