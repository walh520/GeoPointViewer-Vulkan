// 二维视口实现：负责屏幕像素和世界坐标之间的换算。
#include "Viewport2D.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace gpv
{
namespace
{

constexpr double kMinViewportSpan = 0.000001;

double Width(const Bounds2D& bounds)
{
    return bounds.maxX - bounds.minX;
}

double Height(const Bounds2D& bounds)
{
    return bounds.maxY - bounds.minY;
}

Bounds2D NormalizeBounds(const Bounds2D& bounds)
{
    Bounds2D result = bounds;
    if (result.maxX < result.minX)
    {
        std::swap(result.minX, result.maxX);
    }
    if (result.maxY < result.minY)
    {
        std::swap(result.minY, result.maxY);
    }
    return result;
}

Bounds2D ClampToWorld(Bounds2D bounds, const Bounds2D& worldBounds)
{
    const double width = std::max(Width(bounds), kMinViewportSpan);
    const double height = std::max(Height(bounds), kMinViewportSpan);
    const double worldWidth = Width(worldBounds);
    const double worldHeight = Height(worldBounds);

    if (width >= worldWidth)
    {
        bounds.minX = worldBounds.minX;
        bounds.maxX = worldBounds.maxX;
    }
    else
    {
        if (bounds.minX < worldBounds.minX)
        {
            bounds.minX = worldBounds.minX;
            bounds.maxX = bounds.minX + width;
        }
        if (bounds.maxX > worldBounds.maxX)
        {
            bounds.maxX = worldBounds.maxX;
            bounds.minX = bounds.maxX - width;
        }
    }

    if (height >= worldHeight)
    {
        bounds.minY = worldBounds.minY;
        bounds.maxY = worldBounds.maxY;
    }
    else
    {
        if (bounds.minY < worldBounds.minY)
        {
            bounds.minY = worldBounds.minY;
            bounds.maxY = bounds.minY + height;
        }
        if (bounds.maxY > worldBounds.maxY)
        {
            bounds.maxY = worldBounds.maxY;
            bounds.minY = bounds.maxY - height;
        }
    }

    return bounds;
}

} // namespace

void Viewport2D::SetWorldBounds(const Bounds2D& bounds)
{
    worldBounds_ = NormalizeBounds(bounds);
    FitWorld();
}

void Viewport2D::FitWorld()
{
    bounds_ = worldBounds_;
}

void Viewport2D::SetBounds(const Bounds2D& bounds)
{
    bounds_ = ClampToWorld(NormalizeBounds(bounds), worldBounds_);
}

void Viewport2D::PanPixels(double deltaX, double deltaY, int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        return;
    }

    const double worldPerPixelX = Width(bounds_) / static_cast<double>(width);
    const double worldPerPixelY = Height(bounds_) / static_cast<double>(height);

    Bounds2D moved = bounds_;
    moved.minX -= deltaX * worldPerPixelX;
    moved.maxX -= deltaX * worldPerPixelX;
    moved.minY += deltaY * worldPerPixelY;
    moved.maxY += deltaY * worldPerPixelY;
    SetBounds(moved);
}

void Viewport2D::ZoomAt(double worldX, double worldY, double scale)
{
    scale = std::clamp(scale, 0.05, 20.0);

    Bounds2D zoomed;
    zoomed.minX = worldX - (worldX - bounds_.minX) * scale;
    zoomed.maxX = worldX + (bounds_.maxX - worldX) * scale;
    zoomed.minY = worldY - (worldY - bounds_.minY) * scale;
    zoomed.maxY = worldY + (bounds_.maxY - worldY) * scale;
    SetBounds(zoomed);
}

void Viewport2D::ZoomToBox(const Bounds2D& bounds)
{
    Bounds2D target = NormalizeBounds(bounds);
    if (Width(target) <= kMinViewportSpan || Height(target) <= kMinViewportSpan)
    {
        return;
    }

    SetBounds(target);
}

Bounds2D Viewport2D::ScreenToWorld(double cursorX, double cursorY, int width, int height) const
{
    Bounds2D point{};
    if (width <= 0 || height <= 0)
    {
        point.minX = bounds_.minX;
        point.maxX = bounds_.minX;
        point.minY = bounds_.minY;
        point.maxY = bounds_.minY;
        return point;
    }

    const double normalizedX = cursorX / static_cast<double>(width);
    const double normalizedY = cursorY / static_cast<double>(height);
    const double worldX = bounds_.minX + normalizedX * Width(bounds_);
    const double worldY = bounds_.maxY - normalizedY * Height(bounds_);

    point.minX = worldX;
    point.maxX = worldX;
    point.minY = worldY;
    point.maxY = worldY;
    return point;
}

glm::mat4 Viewport2D::BuildViewProjection(double renderOriginX, double renderOriginY) const
{
    return glm::ortho(
        static_cast<float>(bounds_.minX - renderOriginX),
        static_cast<float>(bounds_.maxX - renderOriginX),
        static_cast<float>(bounds_.maxY - renderOriginY),
        static_cast<float>(bounds_.minY - renderOriginY),
        -1.0f,
        1.0f);
}

const Bounds2D& Viewport2D::Bounds() const
{
    return bounds_;
}

} // namespace gpv
