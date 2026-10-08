// Vulkan 着色器工具声明：负责读取 SPIR-V 文件并创建 VkShaderModule。
#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <filesystem>
#include <vector>

namespace gpv
{

class VulkanShader
{
public:
    static std::vector<char> ReadBinaryFile(const std::filesystem::path& filePath);
    static VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code);
};

} // namespace gpv
