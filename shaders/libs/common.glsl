const float Pi = 3.14159265;
const float invPi = 1.0 / Pi;

const float MappingToHDR = 3000.0;
const float MappingToSDR = 1.0 / MappingToHDR;

const float VoxelHDR = 16.0;
const float VoxelSDR = 1.0 / VoxelHDR;

uniform float near;
uniform float far;

uniform float aspectRatio;

uniform float viewWidth;
uniform float viewHeight;

uniform vec2 resolution;
uniform vec2 texelSize;
//vec2 resolution = vec2(viewWidth, viewHeight);
//vec2 texelSize = 1.0 / vec2(viewWidth, viewHeight);

//custom uniform
uniform vec3 lightVector;
uniform vec3 worldLightVector;
uniform vec3 sunVector;
uniform vec3 worldSunVector;
uniform vec3 upVector;
uniform vec3 worldUpVector;
uniform vec3 moonVector;
uniform vec3 worldMoonVector;

uniform vec2 jitter;
uniform vec2 jitter1;
//end custom uniform

vec4 nvec4(in vec3 x) {
    return vec4(x, 1.0);
}

vec3 nvec3(in vec4 x) {
    return x.xyz / x.w;
}

vec3 LinearToGamma(in vec3 color) {
    return pow(color, vec3(2.2));
}

vec3 GammaToLinear(in vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

float saturate(in float v) {
    return clamp(v, 0.0, 1.0);
}

vec2 saturate(in vec2 v) {
    return clamp(v, vec2(0.0), vec2(1.0));
}

vec3 saturate(in vec3 v) {
    return clamp(v, vec3(0.0), vec3(1.0));
}

void ApplyTAAJitter(inout vec4 coord) {
    #ifdef Enabled_Temporal_AA
    coord.xy += jitter * coord.w;
    #endif
}

vec2 ApplyTAAJitter(in vec2 coord) {
#ifdef Enabled_Temporal_AA
    //return ((coord * 2.0 - 1.0) - jitter) * 0.5 + 0.5;
    return coord - jitter * 0.5;
#else
    return coord;
#endif
}

vec2 RemoveTAAJitter(in vec2 coord) {
#ifdef Enabled_Temporal_AA
    return coord + jitter * 0.5;
#else
    return coord;
#endif
}

vec2 PreviousToCurrentJitter(in vec2 coord) {
    #ifdef Enabled_Temporal_AA
    return coord + jitter1 * 0.5 - jitter * 0.5;
    #else
    return coord;
    #endif
}

const vec3  rayleigh_scattering         = vec3(5.8e-6, 1.35e-5, 3.31e-5);
const vec3  rayleigh_absorption         = vec3(0.0);//vec3(3.426, 8.298, 0.356) * 0.06 * 1e-5;
const float rayleigh_distribution       = 8000.0;

const vec3  mie_scattering              = vec3(3.6e-6, 3.6e-6, 3.6e-6);
const vec3  mie_absorption              = mie_scattering * 0.11;
const float mie_distribution            = 1200.0;

const vec3  ozone_absorption            = vec3(0.65e-6, 1.881e-6, 0.085e-6);

const float planet_radius               = 6360e3;
const float atmosphere_radius           = 6420e3;

#ifdef FSH
    #define io in
#else
    #define io out
#endif