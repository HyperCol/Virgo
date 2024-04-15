#version 130

#define Reflection_Temporal_Upsample

#ifdef Reflection_Temporal_Upsample
#define Reflection_Render_Scale 0.5
#else
#define Reflection_Render_Scale 1.0
#endif

uniform sampler2D colortex3;
uniform sampler2D colortex4;
//uniform sampler2D colortex5;
uniform sampler2D colortex6;

uniform int hideGUI;
uniform int heldItemId;
uniform int heldItemId2;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/dither.glsl"
#include "/libs/materialid.glsl"
#include "/libs/tonemapping.glsl"
#include "/libs/lighting/brdf.glsl"

#if 1
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

struct GbuffersData {
    vec2 lightmap;
    float tileID;

    vec3 albedo;
    float alpha;

    vec3 geometryNormal;
    vec3 texturedNormal;

    float roughness;
    float metalness;
    float metallic;
    float emissive;
    float material;
    vec3 F0;
};

GbuffersData GetGbuffersData(in vec2 coord) {
    GbuffersData data;

    vec4 tex0 = texture(colortex0, coord);
    vec4 tex1 = texture(colortex1, coord);
    vec4 tex2 = texture(colortex2, coord);

    vec2 unpack1r  = unpack2x8(tex1.r);
    vec2 unpack1b  = unpack2x8(tex1.b);

    data.tileID = round(unpack1b.y * 255.0);

    data.lightmap = unpack2x8(tex1.g);

    data.albedo = LinearToGamma(tex0.rgb);
    data.alpha = tex0.a;

    data.texturedNormal = DecodeSpheremap(tex2.xy);
    data.geometryNormal = DecodeSpheremap(tex2.zw);

    data.roughness = pow2(1.0 - unpack1r.r);
    data.metalness = unpack1r.y;
    data.metallic = step(0.9, data.metalness);
    data.emissive = unpack1b.x;
    data.material = tex2.b;

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
    /*
    vec3 texturedNormal;
    vec3 geometryNormal;
    vec3 visibleNormal;
    vec3 worldNormal;
    vec3 worldGeometryNormal;
    vec3 worldVisibleNormal;
    */
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
    /*
    v.texturedNormal    = DecodeSpheremap(texture(colortex2, coord).xy);
    v.geometryNormal    = DecodeSpheremap(texture(colortex2, coord).zw);
    v.worldNormal       = imv * v.texturedNormal;
    v.worldGeometryNormal = imv * v.geometryNormal;
    v.visibleNormal     = CalculateVisibleNormals(v.texturedNormal, v.eyeDirection);
    v.worldVisibleNormal = imv * v.visibleNormal;
    */

    return v;
}
#endif
/*
float GetPixelPDF(in vec3 e, in vec3 r, in vec3 n, in float roughness) {
    vec3 h = normalize(r + e);

    float ndoth = max(0.0, dot(n, h));
    float d = DistributionTerm(ndoth, roughness) * ndoth;

    return max(1e-6, d);//max(d / (4.0 * abs(dot(e, h)) + 1e-6), 1e-6);
}
*/

vec3 GetClosest(in sampler2D tex, in vec2 coord, in float depth0) {
    vec3 closest = vec3(0.0, 0.0, depth0);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            vec2 sampleCoord = coord + vec2(i, j) * texelSize;
            float sampleDepth = texture(tex, sampleCoord).x;
            //      sampleDepth = sampleDepth >= 1.0 ? texture(dhDepthTex0, sampleCoord).x : sampleDepth;

            if(sampleDepth < closest.z) {
                closest = vec3(i, j, sampleDepth);
            }
        }
    }

    closest.xy = closest.xy * texelSize + coord;

    //return vec3(coord, texture(depthtex0, coord).x);

    return closest;
}

void main() {
    //float depth = texture(depthtex0, texcoord).x;

    //if(depth == 1.0) {
    //    coord = GetClosest(depthtex0, texcoord, 1.0).xy;
    //}

    GbuffersData data = GetGbuffersData(texcoord);
    VectorStruct v = CalculateVectorStruct(texcoord, texture(depthtex0, texcoord).x);

    vec3 reflectDirection = normalize(reflect(v.viewDirection, data.texturedNormal));
    vec3 fr = SpecularLighting(reflectDirection, v.eyeDirection, data.texturedNormal, data.F0, data.roughness, 1.0);

    //vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;
    //vec2 stageJitter = float2R2(float(frameCounter));

    vec2 coord = texcoord;//min(coord * Reflection_Render_Scale, vec2(Reflection_Render_Scale) - texelSize);

    const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

#ifdef Reflection_Temporal_Upsample
    coord -= offset[frameCounter % 4] * texelSize;
#endif

    vec2 halfCoord = min(coord * Reflection_Render_Scale, vec2(Reflection_Render_Scale - texelSize));

    ivec2 texelPosition = ivec2(halfCoord * resolution);

    ivec2 screenClip = ivec2(resolution * Reflection_Render_Scale);

    float a = data.roughness;

    vec3 specular = vec3(0.0);//LinearToGamma(texelFetch(colortex4, texelPosition, 0).rgb) * MappingToHDR;
    
    float totalWeight = 0.0;
    vec3 averageColor = vec3(0.0);

    float importantSample = 0.0;
    float rayDepth = 1.0;

    int radius = 1;

#ifndef Reflection_Temporal_Upsample
    //specular = LinearToGamma(texture(colortex4, halfCoord).rgb) * MappingToHDR;
    //rayDepth = texture(colortex6, halfCoord).x;
    specular = LinearToGamma(texelFetch(colortex4, texelPosition, 0).rgb) * MappingToHDR;
    rayDepth = texelFetch(colortex6, texelPosition, 0).x;
#else
    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j);

            float sampleDepth = texelFetch(colortex5, sampleTexel, 0).z;
            if(sampleDepth >= 1.0) continue;

            vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex5, sampleTexel, 0).xy);
            vec3 sampleViewPosition = GetViewPosition(texcoord + vec2(i, j) * texelSize, sampleDepth);

            float sampleRoughness = texelFetch(colortex4, sampleTexel, 0).a;

            vec3 rayDirection = normalize(reflect(normalize(sampleViewPosition), sampleNormal));

            float weight = GetPixelPDF(v.eyeDirection, rayDirection, data.texturedNormal, data.roughness);
                  //weight = 1.0;
                  //weight *= i == 0 && j == 0 ? 1.0 : 0.0;

            vec3 sampleColor = LinearToGamma(texelFetch(colortex4, sampleTexel, 0).rgb) * MappingToHDR;

            averageColor += sampleColor * weight;
            totalWeight += weight;

            weight *= step(abs(sampleRoughness - data.roughness), 0.02);

            if(importantSample < weight) {
                specular = sampleColor;
                //totalWeight = 1.0;

                float sampleRayDepth = texelFetch(colortex6, sampleTexel, 0).x;

                rayDepth = sampleRayDepth;
                importantSample = weight;
            }

        }
    }

    averageColor /= totalWeight > 0.0 ? totalWeight : 1.0;

    specular = importantSample < 2e-6 ? averageColor : specular;
#endif

#ifdef Reflection_Temporal_Accumulation
    specular = GammaToLinear(specular);
    specular = KarisToneMapping(specular);
#else
    specular = GammaToLinear(specular * MappingToSDR);
#endif

    float pdf0 = GetPixelPDF(1.0, data.roughness) - 1.0;// - GetPixelPDF(1.0, pow2(1.0 - 0.95));

    gl_FragData[0] = vec4(specular, data.roughness);
    //gl_FragData[1] = vec4(pdf0 >= 0.0 ? rayDepth : v.depth, vec3(1.0));
    gl_FragData[1] = vec4(rayDepth, vec3(1.0));
    //gl_FragData[1] = vec4(v.depth, vec3(1.0));
}
/* DRAWBUFFERS:46 */