@echo off
setlocal

if defined DXC set "DXC_EXE=%DXC%"
if not defined DXC_EXE if defined VULKAN_SDK set "DXC_EXE=%VULKAN_SDK%\Bin\dxc.exe"
if not defined DXC_EXE set "DXC_EXE=dxc.exe"
set OUT_DIR=%~dp0spv

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

"%DXC_EXE%" -spirv -T vs_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\pointcloud.vert.spv" "%~dp0hlsl\pointcloud.vert.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T ps_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\pointcloud.frag.spv" "%~dp0hlsl\pointcloud.frag.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T vs_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\tile_proxy.vert.spv" "%~dp0hlsl\tile_proxy.vert.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T vs_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\selection_overlay.vert.spv" "%~dp0hlsl\selection_overlay.vert.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T ps_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\selection_overlay.frag.spv" "%~dp0hlsl\selection_overlay.frag.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T cs_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\tile_cull.comp.spv" "%~dp0hlsl\tile_cull.comp.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T vs_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\page_pointcloud.vert.spv" "%~dp0hlsl\page_pointcloud.vert.hlsl"
if errorlevel 1 exit /b 1

"%DXC_EXE%" -spirv -T cs_6_0 -E main -fspv-target-env=vulkan1.0 -Fo "%OUT_DIR%\page_cull.comp.spv" "%~dp0hlsl\page_cull.comp.hlsl"
if errorlevel 1 exit /b 1

echo HLSL shaders compiled.
