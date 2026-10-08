// GPU Page 点云顶点着色器：按间接命令索引 Page 元数据并保持浮动原点精度。
#pragma pack_matrix(row_major)

struct CameraPushConstants
{
    float4x4 viewProjection;
    float4 selectionRect;
    float4 selectionParams;
    float4 renderOrigin;
};

struct PageDrawMetadata
{
    float4 localBounds;
    float4 originSplit;
    uint4 draw;
    uint4 format;
};

[[vk::push_constant]]
ConstantBuffer<CameraPushConstants> cameraConstants;

[[vk::binding(0, 0)]]
StructuredBuffer<PageDrawMetadata> pages : register(t0);

[[vk::binding(3, 0)]]
ByteAddressBuffer pagePointData : register(t3);

[[vk::binding(4, 0)]]
StructuredBuffer<uint> colormap256 : register(t4);

struct VSInput
{
    uint vertexId : SV_VertexID;
    uint instanceId : SV_InstanceID;
};

struct VSOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
    [[vk::location(1)]] float flashIntensity : TEXCOORD1;
    [[vk::builtin("PointSize")]] float pointSize : PSIZE;
};

VSOutput main(VSInput input)
{
    PageDrawMetadata page = pages[input.instanceId];
    uint pointOffset = page.draw.x + input.vertexId * page.format.y;
    float2 pointPosition;
    uint color0;
    uint color1;
    if (page.format.x == 2)
    {
        uint quantizedXY = pagePointData.Load(pointOffset);
        uint packedIndices = pagePointData.Load(pointOffset + 4);
        float2 normalizedPosition = float2(
            quantizedXY & 0xFFFF,
            quantizedXY >> 16) / 65535.0f;
        pointPosition = lerp(page.localBounds.xy, page.localBounds.zw, normalizedPosition);
        color0 = colormap256[packedIndices & 0xFF];
        color1 = colormap256[(packedIndices >> 8) & 0xFF];
    }
    else
    {
        pointPosition = asfloat(pagePointData.Load2(pointOffset));
        color0 = pagePointData.Load(pointOffset + 8);
        color1 = pagePointData.Load(pointOffset + 12);
    }

    float2 relativeOrigin =
        (page.originSplit.xy - cameraConstants.renderOrigin.xy) +
        (page.originSplit.zw - cameraConstants.renderOrigin.zw);
    float2 localPosition = pointPosition + relativeOrigin;
    float4 clipPosition = mul(
        float4(localPosition, 0.0f, 1.0f),
        cameraConstants.viewProjection);
    float2 ndcPosition = clipPosition.xy / clipPosition.w;
    float4 selectionRect = cameraConstants.selectionRect;

    bool selectionActive = cameraConstants.selectionParams.x > 0.5f;
    bool insideSelection =
        ndcPosition.x >= selectionRect.x &&
        ndcPosition.x <= selectionRect.z &&
        ndcPosition.y >= selectionRect.y &&
        ndcPosition.y <= selectionRect.w;

    VSOutput output;
    output.position = clipPosition;
    uint packedColor = cameraConstants.selectionParams.z < 0.5f ? color0 : color1;
    output.color = float4(
        packedColor & 0xFF,
        (packedColor >> 8) & 0xFF,
        (packedColor >> 16) & 0xFF,
        (packedColor >> 24) & 0xFF) / 255.0f;
    output.flashIntensity = selectionActive && insideSelection
        ? cameraConstants.selectionParams.y
        : 1.0f;
    output.pointSize = 1.0f;
    return output;
}
