// GPU tile proxy 顶点着色器：根据 compute culling 输出的可见 tile id 绘制 proxy 点。
#pragma pack_matrix(row_major)

struct CameraPushConstants
{
    float4x4 viewProjection;
    float4 selectionRect;
    float4 selectionParams;
    float4 renderOrigin;
};

struct TileMetadata
{
    float4 bounds;
    float4 centerAttrCount;
    float4 color;
    uint4 idAndOffset;
    uint4 lengthRowColFlags;
};

[[vk::push_constant]]
ConstantBuffer<CameraPushConstants> cameraConstants;

[[vk::binding(0, 0)]]
StructuredBuffer<TileMetadata> tiles : register(t0);

[[vk::binding(1, 0)]]
StructuredBuffer<uint> visibleTileIds : register(t1);

struct VSOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
    [[vk::location(1)]] float flashIntensity : TEXCOORD1;
    [[vk::builtin("PointSize")]] float pointSize : PSIZE;
};

VSOutput main(uint vertexId : SV_VertexID)
{
    uint tileIndex = visibleTileIds[vertexId];
    TileMetadata tile = tiles[tileIndex];
    float3 worldPosition = float3(
        tile.centerAttrCount.xy - cameraConstants.renderOrigin.xy,
        0.0f);
    float4 clipPosition = mul(float4(worldPosition, 1.0f), cameraConstants.viewProjection);
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
    output.color = tile.color;
    output.flashIntensity = selectionActive && insideSelection
        ? cameraConstants.selectionParams.y
        : 1.0f;
    output.pointSize = 1.0f;
    return output;
}
