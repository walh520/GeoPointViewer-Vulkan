// 点云片元着色器：使用 CPU 映射后的 RGBA 颜色并叠加框选闪烁。
struct PSInput
{
    [[vk::location(0)]] float4 color : COLOR0;
    [[vk::location(1)]] float flashIntensity : TEXCOORD1;
};

float4 main(PSInput input) : SV_Target
{
    return float4(input.color.rgb * input.flashIntensity, input.color.a);
}
