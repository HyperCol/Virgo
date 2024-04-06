#version 130

uniform sampler2D colortex3;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/lighting/brdf.glsl"

//
uniform sampler2D depthtex0;
uniform sampler2D colortex2;

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

    v.viewPosition = GetViewPosition(coord, depth);
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

    return v;
}
//

uniform int hideGUI;

void main() {
    VectorStruct v = CalculateVectorStruct(texcoord, texture(depthtex0, texcoord).x);

    ivec2 texelPosition = ivec2(texcoord * resolution);

    float roughness = 0.999;

    int radius = 1;

    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    vec3 color = texelFetch(colortex3, texelPosition, 0).rgb;
    float totalWeight = 1.0;

    for(int i = -radius; i <= radius; i += 1) {
        for(int j = -radius; j <= radius; j += 1) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j);
            if(i == 0 && j == 0) continue;

            vec3 sampleColor = texelFetch(colortex3, sampleTexel, 0).rgb;

            m1 += sampleColor;
            m2 += sampleColor * sampleColor;

            float sampleDepth = texelFetch(depthtex0, sampleTexel, 0).x;
            vec3 sampleViewPosition = GetViewPosition(vec2(sampleTexel) * texelSize, sampleDepth);
            vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, sampleTexel, 0).xy);

            vec3 rayDirection = normalize(sampleViewPosition);

            float cosTheta = max(0.0, dot(sampleNormal, v.texturedNormal));

            float weight = GetPixelPDF(cosTheta, roughness) * GetPixelPDF(max(0.0, dot(rayDirection, v.viewDirection)), 0.999);// * max(0.0, -dot(rayDirection, sampleNormal)) * max(0.0, dot(rayDirection, v.texturedNormal));
                  weight /= max(1.0, pow2(length(sampleViewPosition - v.viewPosition)));
                  //weight = 1.0;
                  weight = hideGUI == 1 ? 0.0 : weight;

            color += sampleColor * weight;
            totalWeight += weight;
        }
    }

    color /= totalWeight;

    float variance = sum3(sqrt(max(vec3(0.0), m1 * m1 - m2)));

    //if(hideGUI == 1) {
        //color = variance < 0.5 ? color : vec3(1.0, 0.0, 0.0);
    //}

    gl_FragData[0] = vec4(color, variance);
}

/* DRAWBUFFERS:3 */