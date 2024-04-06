#version 130

#define Dont_Randering_as_Land_Fog      //todo

in vec2 texcoord;

uniform sampler2D colortex3;
uniform sampler2D colortex4;

uniform int hideGUI;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/materialid.glsl"
#include "/libs/intersection.glsl"
#include "/libs/dither.glsl"
#include "/libs/lighting/lighting_common.glsl"
#include "/libs/lighting/lighting_color.glsl"
#include "/libs/lighting/shadowmap.glsl"

in vec3 SunLightingColor;
in vec3 MoonLightingColor;
in vec3 LightingColor;
in vec3 AmbientColor;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

struct GbuffersData {
    vec3 albedo;
    float alpha;

    vec2 lightmap;

    float roughness;
    float metalness;
    float metallic;
    vec3 F0;
};
/*
0 albedo
1 roughness metallic material tileID pomdata0 pomdata1
2 normal
3 diffuse ao
emissive lightmap0 lightmap1
4 skymap
5 specluar
6 specluar
7 taa exposure
*/
GbuffersData GetGbuffersData(in vec2 coord) {
    GbuffersData data;

    vec4 tex0 = texture(colortex0, coord);
    vec4 tex1 = texture(colortex1, coord);
    //vec4 tex2 = texture(colortex2, coord);

    data.albedo = LinearToGamma(tex0.rgb);
    data.alpha = tex0.a;
    
    data.lightmap = unpack2x8(tex1.y);

    vec2 unpack1x  = unpack2x8(tex1.x);
    data.roughness = pow2(1.0 - unpack1x.r);
    data.metalness = unpack1x.y;

    data.metallic = step(0.9, data.metalness);

    data.F0 = mix(vec3(max(0.02, data.metalness)), data.albedo, vec3(data.metallic));

    return data;
}

struct VectorStruct {
    float depth;
    float linearDepth;
    float viewDistance;

    vec3 viewPosition;
    vec3 worldPosition;
    vec3 viewDirection;
    vec3 eyeDirection;
    vec3 worldDirection;
    vec3 worldEyeDirection;

    vec3 texturedNormal;
    vec3 geometryNormal;
    vec3 visibleNormal;
    vec3 worldNormal;
    vec3 worldGeometryNormal;
    vec3 worldVisibleNormal;
};

VectorStruct CalculateVectorStruct(in vec2 coord, in float depth) {
    VectorStruct v;

    v.depth = depth;
    v.linearDepth = ExpToLinearDepth(depth);

    //v.viewPosition = GetViewPosition(coord, depth);
    v.viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(coord, depth) * 2.0 - 1.0));
    v.viewDistance = length(v.viewPosition);

    mat3 imv = mat3(gbufferModelViewInverse);

    v.worldPosition = imv * v.viewPosition;

    v.viewDirection     = v.viewPosition / v.viewDistance;
    v.eyeDirection      = -v.viewDirection;
    v.worldDirection    = v.worldPosition / v.viewDistance;
    v.worldEyeDirection = -v.worldDirection;

    v.worldPosition += gbufferModelViewInverse[3].xyz;

    v.texturedNormal    = DecodeSpheremap(texture(colortex2, coord).xy);
    v.geometryNormal    = DecodeSpheremap(texture(colortex2, coord).zw);
    v.worldNormal       = imv * v.texturedNormal;
    v.worldGeometryNormal = imv * v.geometryNormal;
    v.visibleNormal     = CalculateVisibleNormals(v.texturedNormal, v.eyeDirection);
    v.worldVisibleNormal = imv * v.visibleNormal;

    return v;
}

struct LandAtmosphericStruct {
    float r_density;
    vec3  r_scattering;
    vec3  r_extinction;

    float m_density;
    vec3  m_scattering;
    vec3  m_extinction;

    float f_density;
    vec3  f_scattering;

    vec3 extinction;

    vec3 r_lighting;
    vec3 m_lighting;
    vec3 f_lighting;
};

//temp
//clouds height=1000
//clouds thickness=500

LandAtmosphericStruct GetLandAtmosphericData(in float height) {
    LandAtmosphericStruct a;

    //tracing plane
    //float height = GetAltitudeClip(position, 1e-3);

    a.r_density    = exp(-height / rayleigh_distribution) * 1.0;
    a.r_scattering = rayleigh_scattering * a.r_density;
    a.r_extinction = a.r_scattering + rayleigh_absorption * a.r_density;

    a.m_density    = exp(-height / mie_distribution) * 1.0;
    a.m_scattering = mie_scattering * a.m_density;
    a.m_extinction = a.m_scattering + mie_absorption * a.m_density;

    //height *= 10.0;

    a.f_density    = 0.0;// + exp(-height / 12.0) * 10.0;
    a.f_scattering = vec3(0.001) * a.f_density;

    a.extinction = a.m_extinction + a.r_extinction + a.f_scattering;

    a.r_lighting = vec3(1.0);
    a.m_lighting = vec3(1.0);
    a.f_lighting = vec3(1.0);

    return a;
}

LandAtmosphericStruct GetLandAtmosphericData(in vec3 position) {
    float height = GetAltitudeClip(position, 1e-3);

    return GetLandAtmosphericData(height);
}

struct LandAtmosphericLight {
    float rayleigh;
    float mie;
    float fog;

    float rayleigh2;
    float mie2;
    float fog2;
};

LandAtmosphericLight GetAtmosphericLight(in float cosTheta, in float cosTheta2) {
    LandAtmosphericLight p;

    p.rayleigh  = RayleighPhase(cosTheta);
    p.rayleigh2 = RayleighPhase(cosTheta2);

    p.mie  = HG(cosTheta, 0.76);
    p.mie2 = HG(cosTheta2, 0.76);

    p.fog  = HG(cosTheta, 0.2);
    p.fog  = HG(cosTheta2, 0.2);

    return p;
}

//GetLandAtmosphericLight

vec3 LandASTransmittance(in vec3 rayOrigin, in vec3 L, in float density) {
    float tracingAtmosphere = -IntersectPlane(rayOrigin, L, vec3(0.0, atmosphere_radius, 0.0), vec3(0.0, 1.0, 0.0));
    if(tracingAtmosphere < 0.0) return vec3(1.0);

    float tracingPlanet = -IntersectPlane(rayOrigin, L, vec3(0.0, planet_radius, 0.0), vec3(0.0, 1.0, 0.0));

    float stepLength = tracingAtmosphere;

    float heightPoint = 0.02;
    float depthPoint = 0.5;

    float height = length(rayOrigin + L * stepLength * heightPoint) - planet_radius;
          height = 0.5;//max(1e-5, height);

    float density_rayleigh  = exp(-height / rayleigh_distribution) * density;
    float density_mie       = exp(-height / mie_distribution) * density;

    vec3 tau = (rayleigh_scattering + rayleigh_absorption) * density_rayleigh + (mie_scattering + mie_absorption) * density_mie;
    vec3 transmittance = exp(-tau * stepLength * depthPoint);

    return transmittance;
    
    /*
    vec2 tracingAtmosphere = RaySphereIntersection(rayOrigin, L, vec3(0.0), atmosphere_radius);
    if(tracingAtmosphere.y < 0.0) return vec3(1.0);

    vec2 tracingPlanet = RaySphereIntersection(rayOrigin, L, vec3(0.0), planet_radius);
    float planetShadow = tracingPlanet.x > 0.0 ? exp(-(tracingPlanet.y - tracingPlanet.x) * 0.00001) : 1.0;
    //if(tracingPlanet.x > 0.0) return vec3(0.0);

    float exposure = 1.0 / (exp2(Shadow_Light_Exposure) * exp2(Nature_Light_Exposure));

    float mu = 0.999;
    float phaser = (3.0 / 16.0 * invPi) * (1.0 + mu * mu);
    float phasem = min(1.0, HG(mu, 0.76));

    float depthPoint = 0.5;
    float heightPoint = 0.02;

    float stepLength = tracingAtmosphere.y;

    float height = length(rayOrigin + L * stepLength * heightPoint) - planet_radius;
          height = max(1e-5, height);

    float density_rayleigh  = exp(-height / rayleigh_distribution) * density;
    float density_mie       = exp(-height / mie_distribution) * density;

    vec3 tau = (rayleigh_scattering + rayleigh_absorption) * density_rayleigh + (mie_scattering + mie_absorption) * density_mie;
    vec3 transmittance = exp(-tau * stepLength * depthPoint);

    vec3 scattering = phaser * rayleigh_scattering * density_rayleigh / sum3(rayleigh_scattering) + phasem * mie_scattering * density_mie / sum3(mie_scattering);
         scattering *= transmittance;

    return (transmittance * phasem + scattering) * planetShadow * exposure;
    */
}
/*
void SimpleLandAtmosphericSattering(inout vec3 color, in vec3 rayDirection, in float rayLength) {
    //float rayLength = v.viewDistance;
    //if(v.depth >= 1.0) rayLength = tracingPlanet.x > 0.0 ? min(2000.0, tracingPlanet.x) : 2000.0;

    float rays = 0.0;
    LandAtmosphericStruct a = GetLandAtmosphericData(rayDirection * rayLength);
    vec3 transmittance = exp(-a.extinction * rayLength);

    vec3 r = rayleighPhase * rays + envRayleigh * LightingColor;
    vec3 m = ((miePhase * SunLightingColor + miePhase2 * MoonLightingColor) * rays + LightingColor * envPhase * envPhase);
    vec3 f = fogAmbientLight + (fogSunLight + fogMoonLight) * rays;

    vec3 scattering = (r * a.r_scattering + m * a.m_scattering + f * a.f_scattering) / a.extinction;

    color = mix(scattering, color, transmittance);
}
*/

const vec3 clouds_scattering = vec3(0.08);

float sea_octave(vec2 uv, float choppy) {
    uv += noise(uv);        
    vec2 wv = 1.0 - abs(sin(uv));
    vec2 swv = abs(cos(uv));    
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}

float mnoise(in vec2 coord) {
    return abs(noise(coord) - 0.5) * 2.0;
}

float mnoise(in vec3 coord) {
    return abs(noise(coord) - 0.5) * 2.0;
}


float GetCloudsType(in vec3 worldPosition) {
    float density = 0.0;

    vec2 coord = worldPosition.xz * 0.000125;

    density = (noise(coord) + noise(coord * 2.0) * 0.5) / 1.5;
    //density = (noise(coord + vec2(0.5, 0.0)) + noise(coord - vec2(0.5, 0.0)) + noise(coord + vec2(0.0, 0.5)) + noise(coord - vec2(0.0, 0.5))) * 0.25;
    density = saturate(rescale(density, 0.2, 0.4));

    return 0.0;//hideGUI == 0 ? density : 1.0 - density;
    //return hideGUI == 0 ? 1.0 : 0.0;
}

float GetCloudsShape(in vec3 worldPosition, in float linearHeight, in float t) {
    float clouds = 0.0;

    //vec3 globalWind = vec3(1.0, 0.0, 0.0);

    //worldPosition += globalWind * linearHeight * 0.0244 * 5.0 * 500.0;
    //worldPosition += globalWind * wind;

    //worldPosition *= 2.0;

    float type = 1.0;

    vec2 coord0 = (worldPosition.xz * 0.0007) * 1.0;
    //float shape = (noise(coord0 + vec2(1.0, 0.0)) * 0.5 + noise(coord0 - vec2(1.0, 1.0)) * 0.5 + noise(coord0 - vec2(-1.0, 1.0)) * 0.5 + noise(coord0)) / 2.5;
    //float shape = (noise(coord0 + vec2(1.0, 0.0)) * 0.5 + noise(coord0 + vec2(1.0, 1.0)) * 0.5 + noise(coord0 + vec2(0.0, 1.0)) * 0.5 + noise(coord0)) / 2.5;
    //float shape = (noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(0.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 3.0;
    //float shape = (noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 - vec2(1.0, 1.0)) + noise(coord0 + vec2(0.0, 1.0)) + noise(coord0)) / 4.0;
    //float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(-1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 4.0;
    //float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(-1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 4.0;
    //float shape = (noise(coord0) + noise(coord0 * 2.0) * 0.7 + noise(coord0 * 4.0) * 0.5) / 2.2;
    //float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(0.0, 1.0))+ noise(coord0 + vec2(1.0, 1.0))) / 4.0;
    //float shape = noise(coord0);

    //shape = (shape * 2.0 + noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(1.0, -1.0)) + noise(coord0 + vec2(-1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 6.0;
    float weight = 0.7;
    float size = 1.0;
    float shape = noise(coord0 + vec2(size, size)) + noise(coord0 + vec2(size, -size)) + noise(coord0 + vec2(-size, size)) + noise(coord0 + vec2(-size, -size));

          shape = (shape * weight + noise(coord0)) / (1.0 + 4.0 * weight);
    //shape = pow(shape, 0.8);
    //shape = (shape + noise(coord0 + vec2(1.0, .0))) / (2.0);
    //shape = rescale(shape, noise(coord0 + vec2(1.0, 1.0)) * 0.99, 1.0);
    //shape = (shape + noise(coord0 + vec2(1.0, .0)) + noise(coord0 + vec2(-1.0, .0))) / (3.0);
    //shape = rescale(shape, -(noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))), 1.0);
    //shape = rescale(shape, -noise(coord0 + vec2(1.0, 1.0)), 1.0);
    //shape = rescale(shape, -noise(coord0 - vec2(1.0, 1.0)), 1.0);

    //float shape = rescale(noise(coord0), -(noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(1.0, -1.0))), 1.0);

    //float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(-1.0, 1.0)) * 0.0 + noise(coord0 + vec2(-1.0, -1.0)) * 0.0) / (2.0);
    //float shape = (noise(coord0) + noise(coord0 + vec2(-1.0, 0.0)) + noise(coord0 + vec2(1.0, 0.0))) / (3.0);

    //float shape = noise(coord0);
    //      shape = max(0.0, shape - noise(coord0 + vec2(1.0, 0.0))) * max(0.0, shape - noise(coord0 + vec2(0.0, 1.0)));
    //      shape = pow(shape, 0.2);

    //vec2 coord1 = (worldPosition.xz * 0.0015);
    //float shape2 = (noise(coord1) + noise(coord1 * 4.0) * 0.5) / (1.5);
    vec2 coord1 = worldPosition.xz * 0.0007 * 2.0;
    //float shape2 = (noise(coord1) + noise(coord1 * 2.0) * 0.7 + noise(coord1 * 4.0) * 0.5) / 2.2;
    //shape2 = pow(abs(shape2 * 2.0 - 1.0), 1.0);
#if 1
    //clouds = (shape + shape2 * 0.5) / 1.5;//saturate(shape - 1.0 + min(1.0, shape2 * 2.0));//saturate(rescale(shape, (1.0 - shape2) * 0.5, 1.0));
    //clouds = mix(shape, shape2, 0.33 * (hideGUI == 0 ? 1.0 : 0.0));
    //clouds = (shape + shape2 * 0.25) / 1.25;
    //clouds = saturate(rescale(clouds, shape2 * 0.3, 1.0));

    clouds = shape;

    #if 1
    float shape1 = (noise(coord1) + noise(coord1 * 2.0) * 0.5) / (1.5);
    clouds = rescale(clouds, rescale(shape1, 0.5, 1.0) * 0.5, 1.0);
    clouds = rescale(clouds, type * 0.7 * 0.7, 1.0);

    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.6 * type, 1.0, 1.0, 0.0 + saturate(clouds) * 0.25));
    #else
    float shape1 = (noise(coord1) + noise(coord1 * 2.0) * 0.5) / (1.5);

    clouds = (clouds + shape1 * 0.33) / (1.33);
    clouds = rescale(clouds, type * 0.5 * 0.95, 1.0);

    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.6 * type, 1.0, 1.0, 0.0 + saturate(clouds) * 0.75));
    #endif
    //clouds = rescale(clouds, rescale(shape2, 0.7, 1.0) * 0.25, 1.0);
    //clouds = rescale(clouds, type * 0.6, 1.0);

    //clouds = rescale(clouds, rescale(shape2 * 2.0 - 1.0, 0.5, 1.0) * 0.25, 1.0);
    //clouds = rescale(clouds, type * 0.6, 1.0);

    //float time = mod(frameTimeCounter * 0.5, 1.0);

    //clouds = rescale(clouds, -shape2 * 0.5, 1.0);
    //clouds = rescale(clouds, type * 0.65, 1.0);


    //clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.6 * type, 1.0, 1.0, 0.0 + saturate(clouds) * 0.75));
    //clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.6 * type, 1.0, 1.0, 0.5));
    //clouds = saturate(rescale(clouds, 0.1, 1.0));
    //clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.55 * type, 1.0, 1.0, 0.0 + clouds * 0.5));

    //clouds = (clouds + shape2 * 0.25) / 1.25;
    //clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.45, 0.0));
    //clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.55 * type, 1.0, 1.0, 0.0 + shape * 0.5));

    //clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.47, 0.0));
    //clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.55 * type, 1.0, 1.0, 0.3));
#else
    clouds = shape;
    clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.45, 0.0));
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.55 * type, 1.0, 1.0, 0.0 + 1.0 * 0.5));
#endif
    
#if 1

    vec3 direction3 = vec3(-0.5, -0.5, 0.0) * t * 0.0007 * 6.0;
    vec3 coord3 = worldPosition * 0.0007 * 6.0 + direction3;

    float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3) * 0.5 + noise(coord3 * 4.0 + direction3 * 3.0) * 0.25) / (1.75);
    clouds = saturate(rescale(clouds, noise0 * 0.2, 1.0));
#else
    vec3 direction3 = vec3(-0.5, -0.5, 0.0) * t * 0.0007 * 4.0;
    vec3 coord3 = worldPosition * 0.0007 * 4.0 + direction3;

    float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3 * 0.5) * 0.5) / (1.5);
    clouds = saturate(rescale(clouds, noise0 * 0.2, 1.0));
#endif
    //clouds *= step(linearHeight, 1.0 + 0.05);

/*
    //clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.55 * type, 1.0, 1.0, 0.0));
    vec2 coord0 = (worldPosition.xz * 0.0005);
    float shape = (noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(0.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 3.0;

    vec2 coord1 = (worldPosition.xz * 0.001 * 1.0);
    float shape2 = (noise(coord1) + noise(coord1 * 4.0) * 0.5) / (1.5);

    clouds = (shape + shape2 * 0.5) / 1.5;
    clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.45, 0.0));
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.55 * type, 1.0, 1.0, 0.0));
*/

    return clouds;
}

float GetCloudsSimple(in vec3 worldPosition, in float height, in float t) {
    //worldPosition *= 2.0;
    float linearHeight = rescale(height, 1000.0, 1500.0);

    vec3 globalWind = vec3(1.0, 0.0, 0.0);

    worldPosition += globalWind * linearHeight * 0.0244 * 5.0 * 500.0;
    worldPosition += globalWind * t;

    float clouds = GetCloudsShape(worldPosition, linearHeight, t);
          clouds = saturate(rescale(clouds, (0.2 * 0.0 + 0.15) * 0.4, 1.0));

    clouds = saturate(rescale(clouds, 0.0, 0.8));

    return clouds;
}

float GetClouds(in vec3 worldPosition, in float height, in float t) {
#if 0
    return GetCloudsSimple(worldPosition, height, t);
#else
    //worldPosition *= 2.0;
    float linearHeight = rescale(height, 1000.0, 1500.0);

    vec3 globalWind = vec3(1.0, 0.0, 0.0);

    worldPosition += globalWind * linearHeight * 0.0244 * 5.0 * 500.0;
    worldPosition += globalWind * t;

    float clouds = GetCloudsShape(worldPosition, linearHeight, t);

#if 0
    vec3 direction3 = vec3(-0.5, -0.5, 0.0) * t * 0.0007 * 6.0;
    vec3 coord3 = worldPosition * 0.0007 * 6.0 + direction3;

    float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3) * 0.5 + noise(coord3 * 4.0 + direction3 * 3.0) * 0.25) / (1.75);
    clouds = saturate(rescale(clouds, noise0 * 0.2, 1.0));
#endif

    vec3 coord4 = worldPosition * 0.0007 * 8.0;
    vec3 direction4 = vec3(-0.5, -0.5, 0.0) * t * 0.0;
    float noise1 = (noise(coord4) + noise(coord4 * 2.0 + direction4) * 0.7 + noise(coord4 * 4.0 + direction4 * 3.0) * 0.5) / 2.2;
    clouds = saturate(rescale(clouds, (noise1) * 0.15, 1.0));

    clouds = saturate(rescale(clouds, 0.0, 0.8));

    return clouds;
#endif
}

#if 0
float GetClouds(in vec3 worldPosition, in float height) {
    float wind = 0.0;//frameTimeCounter * 24.4 * 1.0;

    float linearHeight = rescale(height, 1000.0, 1500.0);

    float clouds = 0.0;

    vec3 globalWind = vec3(1.0, 0.0, 0.0);

    worldPosition += globalWind * linearHeight * 0.0244 * 5.0 * 500.0;
    worldPosition += globalWind * wind;

    float type = 1.0;//sin(frameTimeCounter) * 0.5 + 0.5;

    vec2 coord0 = (worldPosition.xz * 0.0005);
    float shape = (noise(coord0 + vec2(1.0, 0.0)) * 0.5 + noise(coord0 - vec2(1.0, 1.0)) * 0.5 + noise(coord0 - vec2(-1.0, 1.0)) * 0.5 + noise(coord0)) / 2.5;

    vec2 coord1 = coord0 * 4.0;
    float shape2 = (noise(coord1) + noise(coord1 * 2.0) * 0.5 + noise(coord1 * 4.0) * 0.25) / (1.75);

    clouds = (shape + shape2 * 0.3) / 1.3;
    clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.5, 0.0));
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.3 * type, 1.0, 1.0, 0.0));

    vec3 direction3 = vec3(-0.5, -0.5, 0.0) * wind * 0.001;
    vec3 coord3 = worldPosition * 0.004 + direction3;
    //float noise0 = (noise(coord3) + noise(coord3 * 3.0 + direction3 * 2.0) * 0.5 + noise(coord3 * 6.0 + direction3 * 5.0) * 0.25) / (1.75);
    //clouds += (noise0 - 1.0) * 0.3;
    float noise0 = (noise(coord3) + noise(coord3 * 3.0 + direction3 * 2.0) * 0.5) / 1.5;
    //clouds += (noise0 - 1.0) * 0.3;
    clouds = saturate(rescale(clouds, -(noise0 - 1.0) * 0.3, 1.0));

    float noise1 = noise(coord3 * 4.0 + direction3 * 3.0);
    clouds = saturate(rescale(clouds, noise1 * 0.06, 1.0));

    //clouds = saturate(rescale(clouds, (0.3 + 0.06) * 0.5, 1.0));

    clouds = saturate(rescale(clouds, 0.0, 0.7));

    return clouds;
}
#endif

float GetCloudsLightDepth(in vec3 rayOrigin, in vec3 rayDirection, in float dither, in vec3 camera, in float t) {
    int steps = Lower_Clouds_Light_Quality;
    float invsteps = 1.0 / float(steps);

    vec3 origin = vec3(0.0, planet_radius + rayOrigin.y, 0.0);

    vec2 top = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius + Lower_Clouds_Top);

    if(top.y < 0.0 || top.x > 0.0) return 0.0;
    float tracingTop = top.y;

    //float rayStep = clamp(tracingTop * topFront, 0.0, 1000.0) * invsteps;
    float rayStep = clamp(tracingTop, 0.0, Lower_Clouds_Light_Tracing_Max_Distance) * invsteps;
    //float rayStep = tracingTop * invsteps;
    float rayLength = 0.0;

    vec3 rayStart = rayOrigin + rayStep * dither;

    vec3 transmittance = vec3(1.0);
    float opticalDepth = 0.0;

    for(int i = 0; i < steps; i++) {
        vec3 rayPosition = rayStart + rayLength * rayDirection;

    #if Lower_Clouds_Distance != OFF
        if(length(rayPosition.xz - rayOrigin.xz) > float(Lower_Clouds_Distance)) break;
    #endif

        float density = GetCloudsSimple(rayPosition, length(rayPosition + vec3(0.0, planet_radius, 0.0) - vec3(camera.x, 0.0, camera.z)) - planet_radius, t);
        //float density = GetClouds(rayPosition, length(rayPosition + vec3(0.0, planet_radius, 0.0) - vec3(camera.x, 0.0, camera.z)) - planet_radius, t);
        opticalDepth += density;

        rayLength += rayStep;
    }

    return opticalDepth * rayStep * Lower_Clouds_Scattering;
}
/*
void GetSimpleLandAtmosphericScattering(inout vec3 color, in float rayLength) {
    float rays = 0.0;
    LandAtmosphericStruct a = GetLandAtmosphericData(rayDirection * rayLength);
    vec3 transmittance = exp(-a.extinction * rayLength);

    vec3 r = rayleighPhase * rays + envRayleigh * LightingColor;
    vec3 m = ((miePhase * SunLightingColor + miePhase2 * MoonLightingColor) * rays + LightingColor * envPhase * envPhase);
    vec3 f = fogAmbientLight + (fogSunLight + fogMoonLight) * rays;

    vec3 scattering = (r * a.r_scattering + m * a.m_scattering + f * a.f_scattering) / a.extinction;

    color = mix(scattering, color, transmittance);
}
*/

struct CloudsStruct{
    vec2 bottom;
    vec2 top;

    float start;
    float end;

    float stepSize;

    vec3 origin;
    vec3 direction;
    vec3 planet;
    float t;

    vec3 scattering;
    vec3 transmittance;
};


void CalculateCloudMedium(inout vec3 color, in vec3 rayOrigin, in vec3 rayStart, in vec3 rayDirection) {
    vec2 bottom = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), planet_radius + 4000.0);
    vec2 top = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), planet_radius + 4500.0);
    /*
    float tracingBottom = bottom.x > 0.0 ? bottom.x : bottom.y;
    float tracingTop = top.x > 0.0 ? top.x : top.y;

    float start = tracingBottom;
    float end = tracingTop;

    color
    */
}

float GetClouds2Shape(in vec3 worldPosition, in float height, in float t) {
    float type = 1.0;

    float clouds = 0.0;

    vec2 coord0 = (worldPosition.xz * 0.0007) * 1.0;
    float shape = noise(coord0);
    shape = (shape * 2.0 + noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(1.0, -1.0)) + noise(coord0 + vec2(-1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 6.0;

    vec2 coord1 = worldPosition.xz * 0.0007 * 2.0;
    float shape1 = (noise(coord1) + noise(coord1 * 3.0) * 0.7 + noise(coord1 * 6.0) * 0.5) / 2.2;

    vec3 coord2 = worldPosition * 0.0007 * 4.0;
    float shape2 = (noise(coord2.xz) + noise(coord2.xz * 2.0) * 0.) / 1.;

    //clouds = shape;
    //clouds = rescale(clouds, -shape1 * 0.99, 1.0);

    clouds = shape;
    clouds = rescale(clouds, -shape1 * 0.99, 1.0);
    //clouds = (clouds + shape2 * 0.15) / (1.0 + 0.15);
    clouds = rescale(clouds, -shape2 * 0.5, 1.0);

    //clouds = rescale(clouds, -shape2 * 0.25, 1.0);
    //clouds = (clouds + shape1 * 0.25 + shape2 * 0.) / (1.25 + 0.);

    clouds = rescale(clouds, type * 0.75, 1.0);

    float linearHeight = rescale(height, 4000.0, 4300.0);
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.7 * type, 1.0, 1.0, 0.0));
    //clouds = (rescale(clouds, shape2 * 0.1, 1.0));
    //clouds = (rescale(clouds, 0.05, 1.0));
    
    //vec3 coord4 = worldPosition * 0.0007 * 12.0;
    //vec3 direction4 = vec3(-0.5, -0.5, 0.0) * t * 0.007 * 12.0;
    //float noise1 = (noise(coord4) + noise(coord4 * 2.0 + direction4) * 0.7 + noise(coord4 * 4.0 + direction4 * 0.25) * 0.5) / 2.2;
    //clouds = saturate(rescale(clouds, (noise1) * 0.15, 1.0));

    return (clouds);
}

float GetClouds2Simple(in vec3 worldPosition, in float height, in float t) {
    float clouds = GetClouds2Shape(worldPosition, height, t);
    clouds = rescale(clouds, 0.15 * 0.5, 1.0);

    return saturate(clouds);
}

float GetClouds2(in vec3 worldPosition, in float height, in float t) {
    float type = 1.0;
#if 0
    float type = 1.0;

    float clouds = 0.0;

    vec2 coord0 = (worldPosition.xz * 0.0007) * 1.0;
    float shape = noise(coord0);

    shape = (shape * 1.0 + noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(1.0, -1.0)) + noise(coord0 + vec2(-1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 5.0;
    //shape = (shape * 2.0 + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(-1.0, 0.0)) + noise(coord0 + vec2(0.0, 1.0)) + noise(coord0 + vec2(0.0, -1.0))) / 6.0;
    //shape = pow(shape, 0.8);

    vec2 coord1 = worldPosition.xz * 0.0014;
    float shape2 = (noise(coord1) + noise(coord1 * 2.0) * 0.5 + noise(coord1 * 4.0) * 0.25) / (1.75);
          shape2 = abs(shape2 - 0.5) * 2.0;
          //shape2 = pow(shape2, 0.9);

    clouds = shape;

    clouds = (clouds + shape2 * 0.25) / 1.25;
    clouds = rescale(clouds, type * 0.45, 1.0);

    float linearHeight = rescale(height, 4000.0, 4300.0);
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.6 * type, 1.0, 1.0, 0.0 + max(0.0, clouds) * 0.99));
    clouds = saturate(rescale(clouds, 0.05, 1.0));

    /*
    vec3 direction3 = vec3(-0.5, -0.5, 0.0) * t * 0.002;
    vec3 coord3 = worldPosition * 0.002 + direction3;
    float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3 * 1.0) * 0.5) / 1.5;
    //clouds = saturate(rescale(clouds, noise0 * 0.1, 1.0));
    */
#endif

#if 0
    float clouds = GetClouds2Shape(worldPosition, height, t);

    vec3 coord4 = worldPosition * 0.0007 * 12.0;
    vec3 direction4 = vec3(-0.5, -0.5, 0.0) * t * 0.007 * 12.0;
    float noise1 = (noise(coord4) + noise(coord4 * 2.0 + direction4) * 0.7 + noise(coord4 * 4.0 + direction4 * 0.25) * 0.5) / 2.2;
    clouds = rescale(clouds, (noise1) * 0.15, 1.0);
#else
    float linearHeight = rescale(height, Medium_Clouds_Bottom, Medium_Clouds_Top);
    //float clouds = GetCloudsShape(worldPosition, linearHeight, t);

    float clouds = 1.0;

    float weight = 0.7;
    float size = 1.0;
    vec2 coord0 = (worldPosition.xz * 0.0007) * 1.0;
    float shape = noise(coord0 + vec2(size, size)) + noise(coord0 + vec2(size, -size)) + noise(coord0 + vec2(-size, size)) + noise(coord0 + vec2(-size, -size));
          shape = (shape * weight + noise(coord0)) / (1.0 + 4.0 * weight);
          //shape = saturate(rescale(pow(shape, 0.25), 0.5, 1.0));

    vec2 coord1 = coord0 * 1.5;
    float shape1 = (noise(coord1) + noise(coord1 * 2.0) * 0.7 + noise(coord1 * 4.0) * 0.5) / 2.2;

    vec2 coord2 = coord0 * 4.0;
    float shape2 = (noise(coord2) + noise(coord2 * 2.0) * 0.5) / 1.5;
    //shape2 = abs(shape2 - 0.5) * 2.0;
    //shape1 = (shape1 + shape2 * 0.5) / 1.5;

    clouds = shape;

    //clouds = rescale(clouds, -shape1 * 0.99, 1.0);
    clouds = rescale(clouds, -shape1 * 0.99, 1.0);
    clouds = rescale(clouds, -rescale(shape2, 0.3, 1.0) * 0.12, 1.0);
    //clouds = rescale(clouds, -(1.0 - abs(shape2 * 2.0 - 1.0)) * 0.99, 1.0);
    //clouds = rescale(clouds, (abs(shape2 - 0.5) * 2.0) * 0.25, 1.0);
    //clouds = (clouds + shape2 * 0.25) / 1.25;
    //clouds = rescale(clouds, -abs(shape2 - 0.5) * 2.0 * 0.25, 1.0);
    clouds = rescale(clouds, type * 0.7 * 0.99, 1.0);

    //clouds = (clouds + shape1 * 0.5) / 1.5;
    //clouds = rescale(clouds, type * 0.5, 1.0);

    /*
    #if 0
    clouds = rescale(clouds, -shape1 * 0.99, 1.0);
    clouds = rescale(clouds, type * 0.7 * 0.95, 1.0);
    #else
    clouds = (clouds + shape1 * 0.33) / (1.33);
    clouds = rescale(clouds, type * 0.5, 1.0);
    #endif
    */
/*
    vec2 direction3 = vec2(-0.5, 0.0) * t * 0.0007 * 3.0;
    vec2 coord3 = worldPosition.xz * 0.0007 * 3.0 + direction3;
    float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3) * 0.5) / (1.5);
          noise0 = abs(noise0 - 0.5) * 2.0;
    clouds = (clouds + noise0 * 0.99) / 1.99;
*/
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8 - 0.6 * type, 1.0, 1.0, 0.1));

    #if 0
    vec3 direction3 = vec3(-0.5, -0.5, 0.0) * t * 0.0007 * 5.0;
    vec3 coord3 = worldPosition * 0.0007 * 5.0 + direction3;
    float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3) * 0.5) / (1.5);
    clouds = saturate(rescale(clouds, noise0 * 0.1, 1.0));
    #endif

    vec3 coord4 = worldPosition * 0.0007 * 6.0;
    vec3 direction4 = vec3(-0.5, -0.5, 0.0) * t * 0.0;
    float noise1 = (noise(coord4) + noise(coord4 * 2.0 + direction4) * 0.7 + noise(coord4 * 4.0 + direction4 * 3.0) * 0.5) / 2.2;
    clouds = saturate(rescale(clouds, (noise1) * 0.15, 1.0));

    clouds = saturate(rescale(clouds, 0.0, 0.8));
#endif

#if 0
    vec2 coord0 = (worldPosition.xz * 0.0005);
    float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 - vec2(1.0, 1.0)) + noise(coord0 + vec2(0.0, 1.0))) / 4.0;
    //float shape = (noise(coord0) * 0.0 + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(-1.0, -1.0)) + noise(coord0 + vec2(-1.0, 1.0))) / 3.0;


    //clouds = shape;
    //clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.4, 0.0));

    vec2 coord2 = coord0 * 3.0;
    //float noise1 = (sea_octave(coord2, 4.0) + sea_octave(coord2 * 2.0, 4.0)) / 2.0;
    //float noise1 = (sea_octave(coord2 + vec2(1.0), 1.0) + sea_octave(coord2 - vec2(1.0), 1.0) * 0.5) / 2.0;
    //float noise1 = (noise(coord2) + noise(coord2 * 2.0) * 0.5 + noise(coord2 * 4.0) * 0.25) / (1.75);
    //float noise1 = 1.0 - mnoise(coord2);
    float noise1 = (mnoise(coord2) + mnoise(coord2 * 2.0) * 0.5) / 1.5;
          //noise1 = (abs(noise1 - 0.5) * 2.0);

    clouds = shape;
    clouds = (clouds + pow(noise1, 0.3) * 0.15) / 1.15;
    //clouds *= saturate(rescale(pow(noise1, 0.05), 0.3, 1.0));
    clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.5, 0.0));
    //noise1 = saturate(remap(1.0 - noise1, 0.5, 1.0, 0.0, 0.3));
    //clouds = saturate(rescale(clouds, noise1, 1.0));

    //clouds *= saturate(rescale(noise1, 0.05, 0.5));//pow(noise1, 0.7);
    //clouds = saturate(rescale(clouds, pow((1.0 - noise1), 4.0) * 0.3, 1.0));
    //clouds = 0.3;
    //noise1 = saturate(rescale(1.0 - noise1, 0.8, 1.0));
    //clouds = saturate(rescale(clouds, noise1 * 0.5, 1.0));

    float linearHeight = rescale(height, 4000.0, 4300.0);
    clouds *= min(1.0, rescale(linearHeight, 0.0, 0.01 + 0.05)) * saturate(remap(linearHeight, 0.8, 1.0, 1.0, 0.0 + shape * 0.));


    vec3 direction3 = vec3(-1.0, -1.0, 0.0) * t * 0.001;
    vec3 coord3 = worldPosition * 0.0005 * 4.0 + direction3;
    float noise0 = (noise(coord3) + noise(coord3 * 3.0 + direction3 * 2.0) * 0.6 + noise(coord3 * 6.0 + direction3 * 5.0) * 0.36) / (1.96);
    //clouds = saturate(rescale(clouds, (1.0 - noise0) * 0.5, 1.0));

#endif
    //clouds *= step(4000.0 - 0.5, height) * step(height, 4300.0 + 0.5);

    #if 0
    vec2 coord2 = (worldPosition.xz * 0.0004);
    float shape = (noise(coord2 + vec2(1.0, 0.0)) + noise(coord2 + vec2(0.0, 1.0)) + noise(coord2)) / 3.0;
    shape = pow(abs(shape - 0.5) * 2.0, 0.4);

    vec3 direction1 = vec3(0.0);
    vec3 coord1 = worldPosition * 0.002 + direction1;
    float shape2 = (noise(coord1.xz) + noise(coord1.xz * 2.0 + direction1.xz * 1.0) * 0.5 + noise(coord1.xz * 4.0 + direction1.xz * 3.0) * 0.25) / (1.75);
    //shape2 = pow(abs(shape2 - 0.5) * 2.0, 0.3);

    //clouds = saturate(rescale(shape2 - 0.5, -shape * 2.0, 0.5) * 0.8);
    clouds = saturate(rescale(shape, -shape2 * 4.0, 1.0) * 0.7);
    //clouds = (shape * 1.0 + shape2 * 1.0) * 0.5;
    //clouds = pow(clouds, 0.7);
    clouds = saturate(rescale(clouds, 0.5, 0.95));

    vec3 direction3 = vec3(-1.0, -1.0, 0.0) * t * 0.001;
    //vec3 coord3 = worldPosition * 0.008 + direction3;
    //float noise0 = (noise(coord3) + noise(coord3 * 2.0 + direction3 * 1.0) * 0.6 + noise(coord3 * 4.0 + direction3 * 3.0) * 0.36) / (1.96);
    vec3 coord3 = worldPosition * 0.004 + direction3;
    float noise0 = (noise(coord3) + noise(coord3 * 3.0 + direction3 * 2.0) * 0.7 + noise(coord3 * 6.0 + direction3 * 5.0) * 0.49) / (2.19);
    clouds = saturate(rescale(clouds, noise0 * 0.4, 1.0));

    clouds = saturate(rescale(clouds, 0.01, 0.6));
    #endif
    #if 0
    vec2 coord0 = (worldPosition.xz * 0.0005);
    float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 - vec2(1.0, 1.0)) + noise(coord0 + vec2(0.0, 1.0))) / 4.0;

    vec2 coord1 = coord0 * 4.0;
    float shape2 = (noise(coord1) + noise(coord1 * 2.0) * 0.5) / (1.5);
    //shape2 = 1.0 - abs(shape2 - 0.5) * 2.0;
    //clouds = saturate(rescale(shape, max(0.0, shape2 * 0.5), 1.0));
    clouds = (shape + shape2 * 0.) / 1.;
    //clouds = shape;
    clouds = saturate(rescale(clouds - 1.0, -1.0 + type * 0.35, 0.0));
    
    vec2 coord2 = coord0 * 8.0;
    //float shape3 = (sea_octave(coord2, 1.0) + sea_octave(coord2 * 1.5, 1.0) * 0.5) / 1.5;
          //shape3 += (sea_octave((coord2 + d) * 2.0, 4.0) + sea_octave((coord2 - d) * 2.0, 4.0)) * 0.5;
    //clouds = saturate(rescale(clouds, shape3 * 0.4, 1.0));

    //float noise1 = (noise(coord2) + noise(coord2 * 2.0) * 0.5) / 1.5;
    float noise1 = sea_octave(coord2 - vec2(1.0), 4.0);
          noise1 = (noise1 + noise(coord2 * 2.0) * 0.5) / 1.5;
    //clouds = (clouds + noise1 * 0.25) / 1.25;
    clouds = saturate(rescale(clouds, noise1 * 0.4, 1.0));
    //clouds = 1.0 - abs(noise1 - 0.5) * 2.0;

    vec3 direction3 = vec3(-1.0, -1.0, 0.0) * t * 0.001;
    vec3 coord3 = worldPosition * 0.01 * vec3(1.0, 1.0, 1.0) + direction3;
    float noise0 = (noise(coord3) + noise(coord3 * 3.0 + direction3 * 2.0) * 0.5) / (1.5);
    clouds = saturate(rescale(clouds, (1.0 - noise0) * 0.2, 1.0));
    
    clouds = saturate(rescale(clouds, 0.0, 0.9));
#endif

    return saturate(clouds);
}

float GetCloudsLightDepth2(in vec3 rayOrigin, in vec3 rayDirection, in float dither, in vec3 camera, in float t) {
    int steps = 6;
    float invsteps = 1.0 / float(steps);

    vec3 origin = vec3(0.0, planet_radius + rayOrigin.y, 0.0);

    vec2 top = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius + Medium_Clouds_Top);

    if(top.y < 0.0 || top.x > 0.0) return 1000.0;
    float tracingTop = top.y;

    //float rayStep = clamp(tracingTop * topFront, 0.0, 1000.0) * invsteps;
    float rayStep = clamp(tracingTop, 0.0, 1000.0) * invsteps;
    //float rayStep = tracingTop * invsteps;
    float rayLength = 0.0;

    vec3 rayStart = rayOrigin + rayStep * dither;

    vec3 transmittance = vec3(1.0);
    float opticalDepth = 0.0;

    for(int i = 0; i < steps; i++) {
        vec3 rayPosition = rayStart + rayLength * rayDirection;

        float density = GetClouds2(rayPosition, length(rayPosition + vec3(0.0, planet_radius, 0.0) - vec3(camera.x, 0.0, camera.z)) - planet_radius, t);
        opticalDepth += density;

        rayLength += rayStep;
    }

    return opticalDepth * rayStep;
}

void main() {
    float fframeCounter = float(frameCounter);
    const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    //vec2 stageJitter = offset[frameCounter % 4] + offset[(frameCounter / 4) % 4] * 2.0;
    vec2 stageJitter = (float2R2(float(frameCounter) + 1011.5)) * (1.0 / 0.25 - 1.0);

    vec2 fragCoord = texcoord * resolution;

    vec2 coord = floor(fragCoord * 0.25) * 4.0;
         coord = round(fragCoord + stageJitter) * texelSize;

    VectorStruct v = CalculateVectorStruct(coord, 1.0);

    //if(v.worldPosition.y + cameraPosition.y - 63.0 < 32.0) color += vec3(1.0, 0.0, 0.0) * 0.001;

    vec2 frameCountOffset = float2R2(fframeCounter + 0.5) * 2.0 - 1.0;

    ivec2 iseed = ivec2(texcoord * resolution + frameCountOffset * 2.0);

    vec3 background = color;

    float dither = GetBlueNoise(iseed);
    float dither2 = GetBlueNoise1(iseed);

    float cosTheta = dot(sunVector, v.viewDirection);
    float cosTheta2 = dot(moonVector, v.viewDirection);
    float cosTheta3 = dot(worldLightVector, worldUpVector);

    float mieLightS = HG(0.999, 0.2);
    float miePhase = HG(cosTheta, 0.76);
    float miePhase2 = HG(cosTheta2, 0.76);
    float rayleighPhase = RayleighPhase(cosTheta);
    float envPhase  = HG(dot(worldLightVector, worldUpVector), 0.76);
    float envRayleigh = RayleighPhase(cosTheta3) * envPhase;

    //float phaseLunarCorona = HG(cosTheta2)

    float fogBackScattering = HG(0.999, 0.1) * invPi;
    float fogBackScatteringWeight = 0.9;

    float fogPhase = mix(HG(cosTheta, 0.9), fogBackScattering, fogBackScatteringWeight);
    //float fogPhase = max(HG(cosTheta, 0.9), fogBackScattering);
    vec3  fogSunLight = fogPhase * SunLightingColor;

    float fogPhase2 = mix(HG(cosTheta2, 0.9), fogBackScattering, fogBackScatteringWeight);
    vec3  fogMoonLight = fogPhase * MoonLightingColor;

    //vec3 fogAmbientLight = max(HG(cosTheta3, 0.9), fogBackScattering) * LightingColor;
    vec3 fogAmbientLight = mix(HG(cosTheta3, 0.9), fogBackScattering, fogBackScatteringWeight) * LightingColor;

    vec3 rayDirection = v.worldDirection;
    //vec2 tracingPlanet = RaySphereIntersection(vec3(0.0, (cameraPosition.y - 63.0) * 1.0 + planet_radius, 0.0), rayDirection, vec3(0.0), planet_radius);

    //float lightingPhase0 = (HG(cosTheta, 0.8) * 0.5 + HG(cosTheta, -0.1) / HG(0.9, 0.8));
    //float lightingPhase0 = mix(HG(cosTheta, -0.2) / HG(0.9999, -0.2), HG(cosTheta, 0.8), 0.9);
    //float lightingPhase0 = HG(cosTheta, -0.1) / HG(0.9999, -0.1) * 0.5 * invPi;
    float lightingPhase0 = mix(HG(cosTheta, 0.1) / HG(0.9999, 0.1) * invPi, HG(cosTheta, 0.8), 0.1);
    float lightingPhase1 = mix(HG(cosTheta2, 0.1) / HG(0.9999, 0.1) * invPi, HG(cosTheta2, 0.8), 0.1);
    float lightingPhaseA = invPi;// * invPi;

    vec4 cloudsScattering = vec4(vec3(0.0), 1.0);
    float cloudsDepth = 0.0;
    float cloudsViewDistance = 0.0;

/*
t1 
-frontFace:start
-backFace:end
*/

/*
raid fram 240
meadow:147
*/
    //if(v.depth == 1.0) color = vec3(0.0);

    float worldScale = World_Scale;

    vec3 wolrdScale3 = vec3(World_Scale, Altitude_Scale, World_Scale);
         wolrdScale3.y *= max(1.0, float(heldBlockLightValue));

    float t = 0.0;//frameTimeCounter * 24.4 * 1.0;

    float rescaleDistance = length(v.worldPosition * wolrdScale3);

    vec3 rayOrigin = vec3(cameraPosition.x, cameraPosition.y - 63.0, cameraPosition.z) + gbufferModelViewInverse[3].xyz;
         rayOrigin *= wolrdScale3;

    //rayOrigin.y *= max(1.0, float(heldBlockLightValue));

    vec3 origin = vec3(0.0, rayOrigin.y + planet_radius, 0.0);

    vec2 tracingPlanet = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius);
    vec2 tracingAtmosphere = RaySphereIntersection(origin, rayDirection, vec3(0.0), atmosphere_radius);
    #ifdef Enabled_Medium_Clouds
    CloudsStruct m;
    
    m.origin = rayOrigin;
    m.planet = vec3(0.0, planet_radius, 0.0);
    m.direction = rayDirection;

    m.bottom = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius + Medium_Clouds_Bottom);
    m.top = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius + Medium_Clouds_Top);

    m.start = m.bottom.x > 0.0 ? m.bottom.x : m.bottom.y;
    m.end = m.top.x > 0.0 ? m.top.x : m.top.y;

    vec3 p = rayOrigin + m.start * rayDirection + (m.end - m.start) * rayDirection * dither * 0.2;
    float h = length(p + vec3(0.0, planet_radius, 0.0) - vec3(rayOrigin.x, 0.0, rayOrigin.z)) - planet_radius;

    if(v.depth < 1.0) {
        m.start = m.start > rescaleDistance ? 0.0 : m.start;
    }

    if(tracingPlanet.y > 0.0 && m.start > tracingPlanet.x) m.start = 0.0;

    if(m.start > 0.0) {
        int steps = 6;
        float invsteps = 1.0 / float(steps);

        m.stepSize = (m.end - m.start) * invsteps;
        m.t = m.stepSize * dither + m.start;

        vec3 cloudsSunLighting = CalculateSunLighting(vec3(0.0, planet_radius + rayOrigin.y, 0.0) + m.end * rayDirection, worldSunVector, 0.8 / 1.5) * Sun_Light_Luminance;

        LandAtmosphericStruct a = GetLandAtmosphericData(length(rayDirection * m.start + vec3(0.0, planet_radius + rayOrigin.y, 0.0)) - planet_radius);
        vec3 fogTransmittance = exp(-a.extinction * m.start);
        vec3 fogScattering = (1.0 - fogTransmittance) / (a.extinction);

        vec3 cloudsAmbient0 = cloudsSunLighting * (a.r_scattering * rayleighPhase + a.m_scattering * miePhase * 0.25) * mix(1.0 / a.extinction, vec3(1.0 / sum3(a.extinction)), fogTransmittance);
        vec3 cloudsLight0 = cloudsSunLighting * min(1.0, lightingPhase0);

        vec3 scattering = vec3(0.0);
        vec3 transmittance = vec3(1.0);

        for(int i = 0; i < steps; i++) {
        vec3 p = m.t * m.direction + m.origin;

        float h = length(p + m.planet - vec3(m.origin.x, 0.0, m.origin.z)) - planet_radius;
        float density = GetClouds2(p, h, t);
        
        if(density > 0.0) {
            float s = Medium_Clouds_Scattering * density;
            float opticalDepth = s * m.stepSize;
            float stepT = exp(-opticalDepth);

            float lightExt = GetCloudsLightDepth2(p, worldSunVector, 0.2, rayOrigin, t) * Medium_Clouds_Scattering;
            //vec3 ambient = (a.r_scattering * rayleighPhase + a.m_scattering * miePhase * 0.25) * (fogScattering + fogTransmittance / sum3(a.extinction));
            float powder = 1.0 - exp(-opticalDepth * 2.0 - lightExt * 2.0);

            vec3 stepS = cloudsLight0 * (exp(-lightExt) + exp(-lightExt * 0.25) * 0.7) / 1.7 * powder;
            stepS += cloudsAmbient0 * exp(-lightExt * 0.02);

            //color = color * transmittance + scattering * s / s * powder;
            scattering += stepS * (1.0 - stepT) * transmittance * s / s;
            transmittance *= stepT;
        }

        m.t += m.stepSize;
        }

        //scattering = mix(color, scattering, fogTransmittance);
        color = color * transmittance + scattering;
    }
    #endif

#ifdef Enabled_Lower_Clouds
    vec2 bottom = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius + Lower_Clouds_Bottom);
    vec2 top = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius + Lower_Clouds_Top);

    float bottomFront = 0.0;
    //float tracingBottom = IntersectPlane(vec3(0.0, rayOrigin.y, 0.0), rayDirection, vec3(0.0, 1000.0, 0.0), vec3(0.0, -1.0, 0.0), bottomFront);
    float tracingBottom = bottom.x > 0.0 ? bottom.x : bottom.y;
    float tracingTop = top.x > 0.0 ? top.x : top.y;

    float topFront = 0.0;
    //float tracingTop = IntersectPlane(vec3(0.0, rayOrigin.y, 0.0), rayDirection, vec3(0.0, 1500.0, 0.0), vec3(0.0, -1.0, 0.0), topFront);

//if(v.depth == 1.0) {
    int steps = Lower_Clouds_Quality;
    float invsteps = 1.0 / float(steps);

    float start = tracingBottom;
    float end = tracingTop;

    bool plane = false;

    //float lightingPhase0 = mix(HG(cosTheta, 0.8), HG(cosTheta, -0.1), 0.8);
    //float lightingPhase0 = HG(cosTheta, 0.8);
    //float lightingPhase0 = (HG(cosTheta, 0.8) + HG(cosTheta, -0.1));

    vec3 heightFactor = vec3(1.0);

    if(rayOrigin.y > 1000.0 && rayOrigin.y < 1500.0) {
        start = 0.05;
        end = min(tracingTop > 0.0 ? tracingTop : 5000.0, tracingBottom > 0.0 ? tracingBottom : 5000.0);
        //end = min(tracingTop, tracingBottom);
    }

    if(start > end) {
        float temp = start;
        start = end;
        end = temp;
    }

    #ifndef Dont_Randering_as_Land_Fog
    if(v.depth < 1.0) {
        end = min(end, rescaleDistance);

        start = start > rescaleDistance ? 0.0 : start;
        //plane = true;
        //heightFactor = vec3(0.0, 1.0, 0.0);
    }
    #endif

    if(tracingPlanet.y > 0.0 && start > tracingPlanet.x) start = 0.0;

    cloudsViewDistance = max(tracingAtmosphere.y > 0.0 ? tracingAtmosphere.y : 50000.0, tracingPlanet.x > 0.0 ? tracingPlanet.x : 50000.0);

    if(start > 0.0) {
    float totalLength = end - start;
//
    float scatteringTracing = max(start - (min(tracingAtmosphere.x, tracingAtmosphere.y) > 0.0 ? tracingAtmosphere.x : 0.0), 0.0);
/*
    float cosTheta = dot(worldSunVector, rayDirection);
    float cosTheta2 = dot(worldMoonVector, rayDirection);
    float cosTheta3 = dot(worldLightVector, worldUpVector);

    float mieLightS = HG(0.999, 0.2);
    float miePhase = HG(cosTheta, 0.76);
    float miePhase2 = HG(cosTheta2, 0.76);
    float rayleighPhase = RayleighPhase(cosTheta);
    float envPhase  = HG(dot(worldLightVector, worldUpVector), 0.76);
    float envRayleigh = RayleighPhase(cosTheta3) * envPhase;
*/
    float rays = 0.0;
    //length(rayPosition + vec3(0.0, planet_radius, 0.0) - vec3(rayOrigin.x, 0.0, rayOrigin.z)) - planet_radius
    //rayOrigin + rayDirection * scatteringTracing
    LandAtmosphericStruct a = GetLandAtmosphericData(length(rayDirection * start + vec3(0.0, planet_radius + rayOrigin.y, 0.0)) - planet_radius);
    vec3 fogTransmittance = exp(-a.extinction * scatteringTracing);
//

    vec3 lightSamplePosition = vec3(0.0, planet_radius + rayOrigin.y, 0.0) + (start + totalLength * 0.5) * rayDirection;

    vec3 cloudsSunLighting = CalculateSunLighting(lightSamplePosition, worldSunVector, 0.5) * Sun_Light_Luminance;// / HG(0.999, 0.6); //0.8
    vec3 fogScattering0 = (a.r_scattering * rayleighPhase + a.m_scattering * miePhase * 0.25) * (1.0 - fogTransmittance) / a.extinction * cloudsSunLighting;
    vec3 lightingColor0 = mix(background, cloudsSunLighting * min(1.0, lightingPhase0), exp(-a.extinction * scatteringTracing * 0.5));
    vec3 cloudsAmbient0 = cloudsSunLighting * fogTransmittance * (a.r_scattering * rayleighPhase + a.m_scattering * miePhase * 0.25) / sum3(a.extinction);
         cloudsAmbient0 += fogScattering0;

    vec3 cloudsLighting0 = cloudsSunLighting + AmbientColor * lightingPhaseA;
    vec3 cloudsDir0 = cloudsLighting0 * lightingPhase0;

    vec3 cloudsLighting1 = CalculateSunLighting(lightSamplePosition, worldMoonVector, 0.5) * Moon_Light_Luminance + AmbientColor * lightingPhaseA;
    vec3 cloudsDir1 = cloudsLighting1 * lightingPhase1;

    //vec2 envmapCoord = EncodeOctahedralmap(vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
         //envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;//


    //float rpa = RayleighPhase(0.999);
    //float mpa = HG(0.999, 0.76);
    //vec3 skyLighting0 = ;

    float stepLength = totalLength * invsteps;

    vec3 rayStep = rayDirection * stepLength;
    vec3 rayPosition = rayOrigin + rayDirection * start + rayStep * dither;

    float transmittance  = 1.0;
    vec3 scattering     = vec3(0.0);

    //float clouds_type = GetCloudsType(rayOrigin + rayDirection * end);

    float d = 0.0;

    for(int i = 0; i < steps; i++) {
        #if Lower_Clouds_Distance != OFF
        if(length(rayPosition.xz - rayOrigin.xz) > float(Lower_Clouds_Distance)) break;
        #endif

        if(transmittance < 1e-5) break;

        float h = length(rayPosition + vec3(0.0, planet_radius, 0.0) - vec3(rayOrigin.x, 0.0, rayOrigin.z)) - planet_radius;
        float density = GetClouds(rayPosition, h, t);

        //vec3 camera = rayOrigin;
        //float density = GetCloudsSimple(rayPosition, length(rayPosition + vec3(0.0, planet_radius, 0.0) - vec3(camera.x, 0.0, camera.z)) - planet_radius, t);

        if(density > 0.0) {
            float sigma_s = density * Lower_Clouds_Scattering;
            float opticalDepth = sigma_s * stepLength;

            float stepTransmittance = exp(-opticalDepth);

            float lightExt0 = GetCloudsLightDepth(rayPosition, worldSunVector, 0.5, rayOrigin, t);
            vec3 lighting0 = ((exp(-lightExt0) + exp(-lightExt0 * 0.25) * 0.7) / 1.7) * cloudsDir0;
            //vec3 lighting0 = ((exp(-lightExt0) + exp(-lightExt0 * 0.25) * 0.7) / 1.7) * lightingColor0;

            float lightExt1 = GetCloudsLightDepth(rayPosition, worldMoonVector, 0.5, rayOrigin, t);
            vec3 lighting1 = ((exp(-lightExt1) + exp(-lightExt1 * 0.25) * 0.7) / 1.7) * cloudsDir1;

            float powderEffect0 = 1.0 - exp(-lightExt0 * 2.0 - opticalDepth * 2.0);
            float powderEffect1 = 1.0 - exp(-lightExt1 * 2.0 - opticalDepth * 2.0);

            float s = (1.0 - stepTransmittance) * transmittance * sigma_s / sigma_s;
            //if((h - 1000.0) / 500.0 > 0.9) s *= vec3(1.0, 0.0, 0.0);

            //scattering += s * lighting0 * powderEffect;
            //scattering += s * cloudsAmbient0 * (exp(-lightExt0 * 0.02) + exp(-lightExt0 * 0.01) * 0.7) / 1.7;

            scattering += (s * powderEffect0) * lighting0;
            scattering += (s * powderEffect1) * lighting1;

            #if Enabled_Lunar_Corona
            vec3 sigma_s2 = vec3(0.05, 0.1, 0.2) * Lower_Clouds_Scattering * 10.0;
            vec3 lightExt1 = (lightExt0 / Lower_Clouds_Scattering) * sigma_s2;
            //scattering += s * cloudsSunLighting * HG(cosTheta, 0.9) * 0.5 * exp(-lightExt1) * powderEffect / sigma_s * sigma_s2 * density;
            #endif
            // + vec3(0.05, 0.1, 0.3) * Lower_Clouds_Scattering * 0.1
            //scattering += s * powderEffect * 0.01;

            //scattering += (lighting0 - lighting0 * stepTransmittance) * transmittance * powderEffect / clamped_s * sigma_s;// / max(vec3(1e-7), sigma_s) * clouds * clouds_scattering * transmittance;

            transmittance *= stepTransmittance;

            //d += opticalDepth;
            cloudsViewDistance = min(cloudsViewDistance, length(rayPosition - rayOrigin));
        }

        rayPosition += rayStep;
    }

        //scattering = scattering * (fogTransmittance * 0.5 + 0.5) + fogScattering0 * 0.5;

        //scattering = mix(color, scattering, fogTransmittance);

        //color = vec3(0.0);
        color = color * transmittance + scattering;
        cloudsScattering = vec4(scattering, transmittance);
        //if(transmittance.r < 1.0) color += fogScattering;
        //color = fogScattering;
        //color = cloudsSunLighting * (1.0 - fogTransmittance) * 0.05;
        //if(scattering.r >= 1.0) color = vec3(1.0, 0.0, 0.0);
        //color = skyLighting0;
        //color = fogScattering0;

        //color = lightingPhase0 * vec3(1.0 / 10.0);
    }
//}
#endif

#if 0
    float rayLength = v.viewDistance;
    if(v.depth >= 1.0) rayLength = tracingPlanet.x > 0.05 ? tracingPlanet.x : 2000.0;

    float rays = 0.0;
    LandAtmosphericStruct a = GetLandAtmosphericData(rayDirection * rayLength);
    vec3 transmittance = exp(-a.extinction * rayLength);

    vec3 r = rayleighPhase * rays + envRayleigh * LightingColor;
    vec3 m = ((miePhase * SunLightingColor + miePhase2 * MoonLightingColor) * rays + LightingColor * envPhase * envPhase);
    vec3 f = fogAmbientLight + (fogSunLight + fogMoonLight) * rays;

    vec3 scattering = (r * a.r_scattering + m * a.m_scattering + f * a.f_scattering) / a.extinction;

    color = mix(scattering, color, transmittance);
#endif

#if 0
if(true) {
    int steps = 12;
    float invsteps = 1.0 / float(steps);

    float tracingDistance = v.viewDistance;
    if(v.depth >= 1.0) tracingDistance = tracingPlanet.x > 0.05 && tracingPlanet.y > 0.0 ? tracingPlanet.x : 100.0;

    float stepLength = tracingDistance * invsteps;

    vec3 rayStep = rayDirection * stepLength;
    vec3 rayStart = rayStep * dither;

    int count = 0;

    vec3 scattering = vec3(0.0);
    vec3 transmittance = vec3(1.0);

    float windSpeed = 3.0;
    vec3 windDirection = vec3(1.0, 0.3, 0.5) * frameTimeCounter * windSpeed;

    for(int i = 0; i < steps; i++) {
        vec3 rayPosition = rayStart + rayStep * float(i);
        //if(length(rayPosition) > v.viewDistance) break;

        vec3 shadowCoord = CalculateShadowCoord(rayPosition);
             shadowCoord = RemapShadowCoord(shadowCoord);
        bool doShadowmap = abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < Shadow_Depth_Mul;

        shadowCoord = shadowCoord * 0.5 + 0.5;

        float visibility = doShadowmap ? step(shadowCoord.z, texture(shadowtex1, shadowCoord.xy).x) : 1.0;

        LandAtmosphericStruct a = GetLandAtmosphericData(rayPosition);

        vec3 rayleighLight = (visibility * rayleighPhase + envRayleigh) * LightingColor;

        vec3 mieLight = visibility * (miePhase * SunLightingColor + miePhase2 * MoonLightingColor);
             mieLight += envPhase * envPhase * LightingColor;

        vec3 fogLighting = visibility * (fogSunLight + fogMoonLight) + fogAmbientLight;

        float height = GetAltitudeClip(rayPosition, 1e-3);

        vec3 worldPosition = rayPosition + gbufferModelViewInverse[3].xyz + cameraPosition;

        vec3 coord = (worldPosition + windDirection) * 0.5;
        vec2 coord2 = (worldPosition.xz + windDirection.xz) * 0.01;

        float clouds = 1.0;
        
        clouds = (noise(coord2) + noise(coord2 * 2.0) * 0.5 + noise(coord2 * 4.0) * 0.25) / 1.75;
        clouds = saturate(rescale(clouds, min(0.999, height / 1500.0), 1.0));

        if(length(rayPosition) < 128.0) { 
            //clouds += 1.0 - (noise(coord) + noise(coord * 2.0) * 0.5 + noise(coord * 4.0) * 0.25) / 1.75;
        }

        clouds = max(0.0, rescale(clouds, 0.5, 1.0));
        //clouds *= exp(-height / 12.0);

        float density = 1.0 + clouds * 10.0;// * exp(-height / 12.0) * 10.0;

        scattering += transmittance * (a.r_scattering * rayleighLight + a.m_scattering * mieLight + a.f_scattering * density * fogLighting) * stepLength;
        transmittance *= exp(-stepLength * a.extinction);
    }

    color *= transmittance;
    color = scattering;
}
#endif

#if 0
    //vec3 rayDirection = v.worldDirection;

    //vec2 tracingPlanet = RaySphereIntersection(vec3(0.0, max(0.05, (cameraPosition.y - 63.0) * 1.0) + planet_radius, 0.0), rayDirection, vec3(0.0), planet_radius);
    float tracingDistance = v.viewDistance;
    if(v.depth >= 1.0) tracingDistance = tracingPlanet.x > 0.05 && tracingPlanet.y > 0.0 ? tracingPlanet.x : 500.0;

    vec3 windDirection = vec3(frameTimeCounter * 10.0, 0.0, 0.0);

    vec3 rayPosition = rayDirection * tracingDistance + cameraPosition + gbufferModelViewInverse[3].xyz;
    vec2 coord2 = (rayPosition.xz + windDirection.xz) * 0.01;
    float clouds = (noise(coord2) + noise(coord2 * 2.0) * 0.5 + noise(coord2 * 4.0) * 0.25);
    clouds = mix(0.05, 1.95, pow(clouds, 10.0));//max(0.0, rescale(clouds, 0.5, 1.0));
    color = mix(color, vec3(1.0), vec3(clouds));
#endif

    //if(tracingPlanet.x > 0.0) color += vec3(0.01);

    //color = LandASTransmittance(v.worldPosition + vec3(0.0, ((cameraPosition.y - 63.0) * 1.0) + planet_radius, 0.0), worldSunVector, 1.0) * 0.05;
#if 0
    float t = (IntersectPlane(v.worldPosition * 1.0 + vec3(0.0, cameraPosition.y - 0.0, 0.0), v.worldDirection, vec3(0.0, 80.0, 0.0), vec3(0.0, 1.0, 0.0)));
    if(t > 0.0 && t < v.viewDistance) {
        color += vec3(0.1);
    }
#endif

    color = GammaToLinear(color * MappingToSDR);

    cloudsDepth = nvec3(gbufferProjection * nvec4(cloudsViewDistance * v.viewPosition)).z * 0.5 + 0.5;

    cloudsScattering.rgb = GammaToLinear(cloudsScattering.rgb * MappingToSDR);

    gl_FragData[0] = cloudsScattering;
    gl_FragData[1] = vec4(cloudsViewDistance, vec3(1.0));
}
/* DRAWBUFFERS:56 */