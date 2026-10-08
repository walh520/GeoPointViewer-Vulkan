// Vulkan 着色器工具实现：为 HLSL 编译得到的 SPIR-V 提供加载入口。
#include "VulkanShader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace gpv
{

std::vector<char> VulkanShader::ReadBinaryFile(const std::filesystem::path& filePath)
{
    std::filesystem::path resolvedPath = filePath;

    if (!std::filesystem::exists(resolvedPath))
    {
        const std::filesystem::path fromBuildDirectory = std::filesystem::path("..") / ".." / filePath;
        if (std::filesystem::exists(fromBuildDirectory))
        {
            resolvedPath = fromBuildDirectory;
        }
    }

    std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open shader file: " + filePath.string());
    }

    const std::size_t fileSize = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    return buffer;
}

VkShaderModule VulkanShader::CreateShaderModule(VkDevice device, const std::vector<char>& code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan shader module.");
    }

    return shaderModule;
}

} // namespace gpv
