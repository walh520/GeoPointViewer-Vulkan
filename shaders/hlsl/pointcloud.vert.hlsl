// 点云顶点着色器：读取正式 cache GPU 点位置和颜色并投影到屏幕。
#pragma pack_matrix(row_major)

struct CameraPushConstants
{
    float4x4 viewProjection;
    float4 selectionRect;
    float4 selectionParams;
    float4 renderOrigin;
};

[[vk::push_constant]]
ConstantBuffer<CameraPushConstants> cameraConstants;

struct VSInput
{
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float4 color0 : COLOR0;
    [[vk::location(2)]] float4 color1 : COLOR1;
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
    VSOutput output;
    float2 localPosition = input.position + cameraConstants.renderOrigin.xy;
    float4 clipPosition = mul(float4(localPosition, 0.0f, 1.0f), cameraConstants.viewProjection);
    float2 ndcPosition = clipPosition.xy / clipPosition.w;
    float4 selectionRect = cameraConstants.selectionRect;

    bool selectionActive = cameraConstants.selectionParams.x > 0.5f;
    bool insideSelection =
        ndcPosition.x >= selectionRect.x &&
        ndcPosition.x <= selectionRect.z &&
        ndcPosition.y >= selectionRect.y &&
        ndcPosition.y <= selectionRect.w;

    output.position = clipPosition;
    output.color = cameraConstants.selectionParams.z < 0.5f
        ? input.color0
        : input.color1;
    output.flashIntensity = selectionActive && insideSelection
        ? cameraConstants.selectionParams.y
        : 1.0f;
    output.pointSize = 1.0f;
    return output;
}
