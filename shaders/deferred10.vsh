#version 130

#define Density 0.5

out vec3 SunLightingColor;
out vec3 MoonLightingColor;
out vec3 LightingColor;

out vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/intersection.glsl"
#include "/libs/lighting/lighting_color.glsl"

void main() {
    //
    vec3 samplePosition = vec3(0.0, 1.0 + planet_radius, 0.0);
    SunLightingColor = CalculateSunLighting(samplePosition, worldSunVector, Density) * Sun_Light_Luminance;
    MoonLightingColor = CalculateSunLighting(samplePosition, worldMoonVector, Density) * Moon_Light_Luminance;
    LightingColor = SunLightingColor + MoonLightingColor;

    //
    texcoord = gl_MultiTexCoord0.st;

    gl_Position = ftransform();
}