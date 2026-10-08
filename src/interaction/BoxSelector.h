// 框选状态声明：记录屏幕空间框选区域并传给渲染端做高亮。
#pragma once

namespace gpv
{

struct BoxSelectionState
{
    bool active = false;
    float minNdcX = 0.0f;
    float minNdcY = 0.0f;
    float maxNdcX = 0.0f;
    float maxNdcY = 0.0f;
    float flashIntensity = 1.0f;
};

} // namespace gpv
