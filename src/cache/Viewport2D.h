// 二维视口声明：维护世界坐标 bbox 并生成 Vulkan 正交投影矩阵。
#pragma once

#include "SeismicCacheTypes.h"

#include <glm/glm.hpp>

namespace gpv
{

class Viewport2D
{
public:
    void SetWorldBounds(const Bounds2D& bounds);
    void FitWorld();
    void SetBounds(const Bounds2D& bounds);
    void PanPixels(double deltaX, double deltaY, int width, int height);
    void ZoomAt(double worldX, double worldY, double scale);
    void ZoomToBox(const Bounds2D& bounds);

    Bounds2D ScreenToWorld(double cursorX, double cursorY, int width, int height) const;
    glm::mat4 BuildViewProjection(double renderOriginX, double renderOriginY) const;

    const Bounds2D& Bounds() const;

private:
    Bounds2D worldBounds_;
    Bounds2D bounds_;
};

} // namespace gpv
