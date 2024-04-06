const int   shadowMapResolution     = 2048;

//const float shadowDistance          = 160.0;
//const float shadowDistanceRenderMul = -1.0;

const int   voxelMapWidth           = 2048;
const int   shadowMapWidth          = shadowMapResolution - voxelMapWidth;

const float shadowTexelSize = 1.0 / float(shadowMapResolution);

//const vec2 shadowMapOffset = vec2(1.0 / float(shadowMapResolution / shadowMapWidth), 0.0);
//const float shadowMapSize = float(shadowMapWidth) * shadowTexelSize;
const vec2 shadowMapOffset = vec2(0.0, 0.0);
const float shadowMapSize = 1.0;

#define Shadow_Depth_Mul 0.2

uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;

vec3 CalculateShadowCoord(in vec3 worldPosition) {
    vec4 shadowCoord = shadowProjection * shadowModelView * vec4(worldPosition, 1.0);

    return shadowCoord.xyz / shadowCoord.w;
}

float ShadowMapDistortion(in vec2 coord) {
    //vec2 q = abs(coord) - vec2(0.0) + 0.25;
    //float l = length(q) + min(max(q.x, q.y),0.0) - 0.25;

    return 1.0 / mix(1.0, length(coord), 0.9);
}

//float sdRoundBox( vec2 p, vec2 b, float r ) {
//    vec2 q = abs(p) - b + r;
//    return length(q) + min(max(q.x, q.y),0.0) - r;
//}

float GetWarpX(in vec2 uv, in vec3 pos, in float depth) {
    float k = uv.x;
    float n = 1.0;

    float t1 = k / n;

    float t2 = k / n;
    
    float warp = t1 - t2;
    
    return k + warp;
}

vec3 RemapShadowCoord(in vec3 shadowCoord) {
    vec3 coord = shadowCoord;
    
    coord.xy *= ShadowMapDistortion(coord.xy);

    coord.xy = coord.xy * 0.5 + 0.5;
    coord.xy *= shadowMapSize;
    //coord.xy += shadowMapOffset;
    coord.xy = coord.xy * 2.0 - 1.0;

    coord.z *= Shadow_Depth_Mul;
    
    return coord;
}

vec3 RemapShadowCoord(in vec3 shadowCoord, in vec3 playerViewPosition) {
    vec3 coord = shadowCoord;

    coord.xy *= ShadowMapDistortion(coord.xy);

    coord.xy = coord.xy * 0.5 + 0.5;
    coord.xy *= shadowMapSize;
    //coord.xy += shadowMapOffset;
    coord.xy = coord.xy * 2.0 - 1.0;

    coord.z *= Shadow_Depth_Mul;
    
    return coord;
}

void RemapShadowCoord(inout vec3 shadowCoord, inout float distortion) {
    distortion = ShadowMapDistortion(shadowCoord.xy);
    shadowCoord.xy *= distortion;

    shadowCoord.xy = shadowCoord.xy * 0.5 + 0.5;
    shadowCoord.xy *= shadowMapSize;
    //shadowCoord.xy += shadowMapOffset;
    shadowCoord.xy = shadowCoord.xy * 2.0 - 1.0;

    shadowCoord.z *= Shadow_Depth_Mul;
}

float GetShadowDepth(in float depth) {
    return ((depth * 2.0 - 1.0) / Shadow_Depth_Mul) * 0.5 + 0.5;//(depth * 2.0 - 1.0) / Shadow_Depth_Mul * 0.5 + 0.5;
}

float GetShadowLinearDepth(float depth) {
    vec2 viewDepth = mat2(shadowProjectionInverse[2].zw, shadowProjectionInverse[3].zw) * vec2(depth * 2.0 - 1.0, 1.0);
    return -viewDepth.x / viewDepth.y;
}

float ShadowLinearToExpDepth(float linerDepth) {
    vec2 expDepth = mat2(shadowProjection[2].zw, shadowProjection[3].zw) * vec2(-linerDepth, 1.0);

    return (expDepth.x / expDepth.y) * 0.5 + 0.5;
}