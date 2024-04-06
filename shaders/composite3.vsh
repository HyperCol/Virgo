#version 130

#define Density 0.5

out vec3 SunLightingColor;
out vec3 MoonLightingColor;
out vec3 LightingColor;
out vec3 AmbientColor;

out vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/intersection.glsl"
#include "/libs/lighting/lighting_color.glsl"

struct AtmosphereDataStruct {
    float rD;
    vec3  rS;
    vec3  rE;

    float mD;
    vec3  mS;
    vec3  mE;

    vec3 E;
};

AtmosphereDataStruct GetAtmosphereData(in float height) {
    AtmosphereDataStruct a;

    a.rD = exp(-height / rayleigh_distribution);
    a.mD = exp(-height / mie_distribution);

    a.rS = rayleigh_scattering * a.rD;
    a.rE = a.rS + rayleigh_absorption * a.rD;

    a.mS = mie_scattering * a.mD;
    a.mE = a.mS + mie_absorption * a.mD;

    a.E = a.mE + a.rE;

    return a;
}

void main() {
    //vec3 s = (rayleigh_scattering + mie_scattering) / sum3(rayleigh_scattering + mie_scattering + rayleigh_absorption + mie_absorption);
    //     s = mix(vec3(1.0), s, vec3(0.4));

    AtmosphereDataStruct a = GetAtmosphereData(1000.0);

    vec3 L = a.rS + a.mS;

    vec3 s = exp(-20000.0 * a.E);
         s = (L - L * s) / a.E;

    float weatherLight = mix(HG(-0.9999, Fog_BackScattering_Phase) * (1.0 - Fog_Front_Scattering_Weight), 1.0, saturate(exp(-Fog_Light_Extinction_Distance * Rain_Fog_Density - Biome_Fog_Density * Fog_Light_Extinction_Distance)));
    //
    float luminance = 1.0 / HG(0.95, 0.76);

    vec3 samplePosition = vec3(0.0, 1.0 + planet_radius, 0.0);
    SunLightingColor = CalculateSunLighting(samplePosition, worldSunVector, Density) * (Sun_Light_Luminance * luminance * weatherLight);
    MoonLightingColor = CalculateSunLighting(samplePosition, worldMoonVector, Density) * (Moon_Light_Luminance * luminance * weatherLight);

    LightingColor = SunLightingColor + MoonLightingColor;

    //AmbientColor = (s * SunLightingColor) * (HG(worldSunVector.y, 0.76) * luminance) + (s * MoonLightingColor) * (HG(worldMoonVector.y, 0.76) * luminance);
    AmbientColor = s * SunLightingColor + s * MoonLightingColor;

    //SunLightingColor += SunLightingColor * s;
    //MoonLightingColor += MoonLightingColor * s;
    
    //
    texcoord = gl_MultiTexCoord0.st;

    gl_Position = ftransform();
}
//#include "/program/post.vsh"