#version 130

uniform sampler2D colortex4;

out vec2 texcoord;

out vec3 SunLightingColor;
out vec3 MoonLightingColor;
out vec3 LightingColor;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/intersection.glsl"
#include "/libs/lighting/lighting_color.glsl"

void main() {
    vec3 samplePosition = vec3(0.0, 1.0 + planet_radius, 0.0);
    SunLightingColor = CalculateSunLighting(samplePosition, worldSunVector, 0.7) * Sun_Light_Luminance;
    MoonLightingColor = CalculateSunLighting(samplePosition, worldMoonVector, 0.7) * Moon_Light_Luminance;
    LightingColor = SunLightingColor + MoonLightingColor;

    gl_Position = ftransform();

    gl_Position.xy = gl_Position.xy * 0.5 + 0.5;
    gl_Position.xy *= 0.25;
    gl_Position.xy = gl_Position.xy * 2.0 - 1.0;

    texcoord = gl_MultiTexCoord0.st;
}