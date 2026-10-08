// GPU Page 剔除计算着色器：筛选可见 Page 并紧凑生成间接绘制命令。
struct PageDrawMetadata
{
    float4 localBounds;
    float4 originSplit;
    uint4 draw;
    uint4 format;
};

struct PageCullParams
{
    float4 viewportBounds;
    float4 renderOriginSplit;
    uint pageCount;
    uint maxDrawCount;
    int activeLodLevel;
    uint padding1;
};

[[vk::push_constant]]
ConstantBuffer<PageCullParams> cullParams;

[[vk::binding(0, 0)]]
StructuredBuffer<PageDrawMetadata> pages : register(t0);

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint4> drawCommands : register(u0);

[[vk::binding(2, 0)]]
RWByteAddressBuffer drawCount : register(u1);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint pageIndex = dispatchThreadId.x;
    if (pageIndex >= cullParams.pageCount)
    {
        return;
    }

    PageDrawMetadata page = pages[pageIndex];
    float2 relativeOrigin =
        (page.originSplit.xy - cullParams.renderOriginSplit.xy) +
        (page.originSplit.zw - cullParams.renderOriginSplit.zw);
    float4 relativeBounds = page.localBounds + relativeOrigin.xyxy;
    bool outside =
        relativeBounds.z < cullParams.viewportBounds.x ||
        relativeBounds.x > cullParams.viewportBounds.z ||
        relativeBounds.w < cullParams.viewportBounds.y ||
        relativeBounds.y > cullParams.viewportBounds.w ||
        page.draw.y == 0 ||
        page.format.w == 0 ||
        int(page.format.z) != cullParams.activeLodLevel;
    if (outside)
    {
        return;
    }

    uint drawIndex = 0;
    drawCount.InterlockedAdd(0, 1, drawIndex);
    if (drawIndex < cullParams.maxDrawCount)
    {
        drawCommands[drawIndex] = uint4(
            page.draw.y,
            1,
            0,
            pageIndex);
    }
}
