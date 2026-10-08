# GeoPointViewer-Vulkan

东方杯项目中的 C++ / Vulkan 点云可视化模块，在二维视口中展示地理点数据，支持属性着色、点选、框选和剖面分析叠加。

## 主要功能

- 基于屏幕空间的 LOD 选择与异步视口查询
- Device-local GPU Page Pool，配合 HLSL 计算着色器进行可见性裁剪
- 间接绘制与 Timeline Semaphore 同步
- CPU Tile / Point Buffer 渲染路径
- 交互选择、剖面分析与运行时性能记录

## 构建

目标平台为 Windows x64，使用 C++20。所需环境：

- Visual Studio 与 MSVC v145 工具集；也可在项目中重定向到已安装的兼容工具集
- Vulkan SDK，包含 DXC 着色器编译器
- GLFW 3.x 开发包
- GLM 头文件

配置环境变量：

- `VULKAN_SDK`：Vulkan SDK 安装目录
- `GLFW_ROOT`：包含 `include`、`lib-vc2022` 和 `bin` 的 GLFW 开发目录

项目使用 `$(VULKAN_SDK)\Include` 作为头文件搜索路径之一。请确保编译器能够找到 GLM；需要时在 Visual Studio 中补充 GLM 的包含目录。

在仓库根目录编译着色器：

```bat
call shaders\compile_hlsl.bat
```

脚本优先使用环境变量 `DXC` 指定的编译器，其次使用 Vulkan SDK 中的 DXC，最后尝试 PATH 中的 `dxc.exe`。生成的 SPIR-V 文件位于 `shaders/spv`。

打开 `GeoPointViewer.slnx` 或 `GeoPointViewer.vcxproj`，选择 `Release|x64` 构建。构建步骤会将外部 GLFW 开发包中的 `glfw3.dll` 复制到程序输出目录。

## 运行

以仓库根目录为工作目录运行生成的程序，确保能够读取 `shaders/spv`：

```bat
<程序输出目录>\GeoPointViewer.exe --cache <本地缓存目录>
```

常用选项：

- `--check-cache`：检查缓存契约
- `--profile-check`：执行剖面检查
- `--gpu auto|integrated|discrete`：选择 GPU 类型
- `--performance-csv <output.csv>`：导出运行时性能记录
- `--help`：查看完整命令行说明

## 数据接口

数据需由使用者自行准备，并确保拥有使用权限。本仓库不包含地震观测数据、派生缓存、真实样本、截图、录屏或运行日志。数据预处理和 Qt 上层工具独立维护。

缓存目录需要包含 `metadata.json` 及其描述的本地文件，LOD 契约版本须为 6 或更新版本。记录布局、Tile 索引和 GPU Page 格式可参考：

- `src/cache/SeismicCacheTypes.h`
- `src/cache/SeismicCacheReader.cpp`
- `src/cache/GpuReadyPageCache.cpp`

## 代码结构

- `src/cache`：缓存契约、数据读取和视口查询
- `src/core`：配置、输入、异步查询和帧调度
- `src/vulkan`：Vulkan 资源、管线、传输、同步和绘制
- `src/interaction`：交互选择
- `src/profile`：剖面分析
- `src/performance`：运行时性能记录
- `shaders/hlsl`：顶点、片元和计算着色器

## 第三方依赖

Vulkan SDK、GLFW、GLM 和 Visual Studio 工具链需单独安装。本仓库不捆绑其头文件或二进制文件；使用或分发相关依赖时，请遵守各自的许可证。
