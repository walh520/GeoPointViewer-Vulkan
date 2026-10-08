struct TileMetadata
{
    float4 bounds;
    float4 centerAttrCount;
    float4 color;
    uint4 idAndOffset;
    uint4 lengthRowColFlags;
};

struct TileCullParams
{
    float4 viewportBounds;
    uint tileCount;
    uint maxVisibleTiles;
    uint padding0;
    uint padding1;
};

[[vk::push_constant]]
ConstantBuffer<TileCullParams> cullParams;

[[vk::binding(0, 0)]]
StructuredBuffer<TileMetadata> tiles : register(t0);

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> visibleTileIds : register(u0);

[[vk::binding(2, 0)]]
RWByteAddressBuffer indirectDraw : register(u1);

[[vk::binding(3, 0)]]
StructuredBuffer<uint> proxyMask : register(t2);

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint tileIndex = dispatchThreadId.x;
    if (tileIndex >= cullParams.tileCount)
    {
        return;
    }

    TileMetadata tile = tiles[tileIndex];
    if (proxyMask[tileIndex] == 0)
    {
        return;
    }

    bool outside =
        tile.bounds.z < cullParams.viewportBounds.x ||
        tile.bounds.x > cullParams.viewportBounds.z ||
        tile.bounds.w < cullParams.viewportBounds.y ||
        tile.bounds.y > cullParams.viewportBounds.w ||
        tile.lengthRowColFlags.w == 0;

    if (outside)
    {
        return;
    }

    uint visibleIndex = 0;
    indirectDraw.InterlockedAdd(0, 1, visibleIndex);
    if (visibleIndex < cullParams.maxVisibleTiles)
    {
        visibleTileIds[visibleIndex] = tileIndex;
    }
}
