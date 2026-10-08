// 框选覆盖层顶点着色器：直接使用 NDC 坐标绘制屏幕矩形。
struct VSInput
{
    [[vk::location(0)]] float2 position : POSITION;
};

float4 main(VSInput input) : SV_Position
{
    return float4(input.position, 0.0f, 1.0f);
}
