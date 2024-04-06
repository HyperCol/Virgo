#version 130

#define Reflection_Render_Scale 0.5

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
    vec2 coord = texcoord;
    float depth = texture(depthtex0, coord).x;

    //if(depth == 1.0) {
    //    coord = GetClosest(depthtex0, texcoord, 1.0).xy;
    //}

    GbuffersData data = GetGbuffersData(coord);

    VectorStruct v = CalculateVectorStruct(coord, texture(depthtex0, coord).x);

    vec3 reflectDirection = normalize(reflect(v.viewDirection, data.texturedNormal));
    vec3 fr = SpecularLighting(reflectDirection, v.eyeDirection, data.texturedNormal, data.F0, data.roughness, 1.0);

    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    vec2 halfCoord = min(coord * Reflection_Render_Scale, vec2(Reflection_Render_Scale) - texelSize);
    ivec2 texelPosition = ivec2(halfCoord * resolution);

    ivec2 screenClip = ivec2(resolution * Reflection_Render_Scale);

    vec3 specular = vec3(0.0);//LinearToGamma(texelFetch(colortex4, texelPosition, 0).rgb) * MappingToHDR;
    float totalWeight = 0.0;

    float importantSample = 0.0;
    float rayDepth = 1.0;

    int radius = 1;

    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j);
            //if(sampleTexel.x > )

            float sampleDepth = texelFetch(colortex5, sampleTexel, 0).z;
            if(sampleDepth >= 1.0) continue;

            vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex5, sampleTexel, 0).xy);
            vec3 sampleViewPosition = GetViewPosition(texcoord + vec2(i, j) * texelSize, sampleDepth);
            vec3 rayDirection = normalize(reflect(v.viewDirection, sampleNormal));

            //float weight = GetPixelPDF(saturate(dot(sampleNormal, data.texturedNormal)), data.roughness) / max(1e-7, dot(v.eyeDirection, sampleNormal) * 4.0);
            float weight = GetPixelPDF(v.eyeDirection, normalize(reflect(normalize(sampleViewPosition), sampleNormal)), data.texturedNormal, data.roughness);
            vec3 sampleColor = LinearToGamma(texelFetch(colortex4, sampleTexel, 0).rgb) * MappingToHDR;

            specular += sampleColor * weight;
            totalWeight += weight;

            float sampleRayDepth = texelFetch(colortex6, sampleTexel, 0).x;

            if(importantSample < weight) {
                rayDepth = sampleRayDepth;
                importantSample = weight;
            }

        }
    }

    specular /= totalWeight == 0.0 ? 1.0 : totalWeight;

    //if(hideGUI == 1) {
    //    specular = LinearToGamma(texelFetch(colortex4, texelPosition, 0).rgb) * MappingToHDR;
    //}

    color += specular * fr;
    if(hideGUI == 1) color = specular;
    //color = fr;

    color = GammaToLinear(color * MappingToSDR);

    specular = GammaToLinear(specular * MappingToSDR);


    vec3 h = normalize(reflectDirection + v.eyeDirection);
    float ndoth = saturate(dot(h, data.texturedNormal));
    float d = DistributionTerm(ndoth, data.roughness);

    gl_FragData[0] = vec4(specular, 1.0);
    gl_FragData[1] = vec4(d >= 0.0 ? rayDepth : v.depth, vec3(1.0));
}
/* DRAWBUFFERS:46 */