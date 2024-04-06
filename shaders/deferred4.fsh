#version 130

#define GI_Render_Scale 0.25

uniform sampler2D colortex3;
//uniform sampler2D colortex5;
uniform sampler2D colortex10;
uniform sampler2D colortex11;

uniform sampler2D depthtex0;

in vec2 texcoord;

const bool colortex10Clear = false;
const bool colortex11Clear = false;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/lighting/brdf.glsl"

//
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

vec3 GetClosest(in vec2 coord) {
    vec3 closest = vec3(0.0, 0.0, 1.0);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            float depth = texture(depthtex0, coord + vec2(i, j) * texelSize).x;

            if(depth < closest.z) {
                closest = vec3(i, j, depth);
            }
        }
    }

    closest.xy = closest.xy * texelSize + coord;

    return closest;
}

in mat4 previousModelViewInverse;
in mat4 previousProjectionInverse;

float AccumulationGeometryTerm(in vec2 previousCoord, in VectorStruct v, in float sigma) {
    vec2 previousCoordJitter = PreviousToCurrentJitter(previousCoord);

    float previousDepth = texture(colortex11, previousCoordJitter).x;
    float previousLinear = ExpToLinearDepth(previousDepth);

    vec3 worldPosition = v.worldPosition + cameraPosition;
    vec3 prevWorldPosition = worldPosition - previousCameraPosition;

    vec3 prevViewPos = nvec3(previousProjectionInverse * nvec4(vec3(ApplyTAAJitter(previousCoord), previousDepth) * 2.0 - 1.0));
    vec3 prevSampleWorldPosition = mat3(previousModelViewInverse) * prevViewPos + previousModelViewInverse[3].xyz;

    vec3 halfVector = prevSampleWorldPosition.xyz - prevWorldPosition;

    float planeDistance = abs(dot(v.worldNormal, halfVector));
    float penumbra = exp(-100.0 * planeDistance / v.linearDepth);

#if 0
    float pdf  = abs(dot(n, v.viewDirection)) + 1e-6;
    float pdf2 = abs(dot(n, normalize(prevViewPos))) + 1e-6;
#else
    float pdfa = 0.999;

    float pdf  = GetPixelPDF(1.0, pdfa);

    float visiblityTS = dot(normalize(prevViewPos), v.viewDirection) / max(1.0, pow2(length(halfVector)));
    float pdf2 = max(1e-6, GetPixelPDF(visiblityTS, pdfa));
#endif

    float normalWeight = (pdf2 / pdf) / 0.9999;
          normalWeight = min(1.0, pow(normalWeight, 1.0));
          //normalWeight *= -dot(normalize(prevViewPos), normal.geometry) > 0.0 ? 1.0 : 0.0;
          //normalWeight = 1.0;

    return saturate(penumbra) * normalWeight;
}

void main() {
    VectorStruct v = CalculateVectorStruct(texcoord, texture(depthtex0, texcoord).x);

    ivec2 halfTexelPosition = ivec2(texcoord * resolution * GI_Render_Scale);
    vec3 currentColor = vec3(0.0);

#if 1
    float totalWeight = 0.0;

    for(int i = -2; i <= 2; i++) {
        for(int j = -2; j <= 2; j++) {
            ivec2 sampleTexel = ivec2(i, j) + halfTexelPosition;

            vec3 normal = DecodeSpheremap(texelFetch(colortex5, sampleTexel, 0).yz);
            float sampleDepth = texelFetch(colortex5, sampleTexel, 0).x;
            vec3 samplePosition = GetViewPosition(vec2(sampleTexel) * texelSize / GI_Render_Scale, sampleDepth);

            float weight = max(1e-5, GetPixelPDF(max(0.0, dot(normal, v.texturedNormal)), 0.3)) * GetPixelPDF(max(0.0, dot(normalize(samplePosition), v.viewDirection)), 0.99);

            vec3 halfVector = samplePosition - v.viewPosition;
            weight *= min(1.0, exp(-100.0 * abs(dot(halfVector, v.texturedNormal)) / length(samplePosition)));

            currentColor += texelFetch(colortex3, sampleTexel, 0).rgb * weight;
            totalWeight += weight;
        }
    }

    if(totalWeight > 0.0) {
        currentColor /= totalWeight;
    }
#endif

    vec3 closest = GetClosest(texcoord);
    vec2 velocity = GetVelocity(closest);
    vec2 previousCoord = texcoord - velocity;
    float inscreen = step(abs(previousCoord.x - 0.5), 0.5) * step(abs(previousCoord.y - 0.5), 0.5);

    vec2 previousCoordJitter = PreviousToCurrentJitter(previousCoord);

    float previousDepth = texture(colortex11, previousCoordJitter).x;
    float previousLinear = ExpToLinearDepth(previousDepth);

    vec3 worldPosition = v.worldPosition + cameraPosition;

    vec3 prevViewPos = nvec3(previousProjectionInverse * nvec4(vec3(ApplyTAAJitter(previousCoord), previousDepth) * 2.0 - 1.0));
    vec3 prevSampleWorldPosition = mat3(previousModelViewInverse) * prevViewPos + previousModelViewInverse[3].xyz + previousCameraPosition;

    vec3 halfVector = prevSampleWorldPosition.xyz - worldPosition;

    vec3 previousNormal = mat3(previousModelViewInverse) * DecodeSpheremap(texture(colortex11, previousCoord).yz);

    float planeDistance = abs(dot(v.worldGeometryNormal, halfVector));
    float penumbra = exp(-100.0 * planeDistance / v.linearDepth);

    vec3 historyColor = texture(colortex10, previousCoord).rgb;

    float historyLength = texture(colortex10, previousCoord).a;
          historyLength = historyLength * 255.0 + 1.0;
          historyLength = min(historyLength, 255.0);

    historyLength *= min(1.0, penumbra);

    float c1 = max(0.0, -dot(v.texturedNormal, normalize(prevViewPos)));
    float c2 = max(1e-5, -dot(v.texturedNormal, v.viewDirection));
    //historyLength *= min(1.0, c1 / c2 * 1.01);

    float ntermAlpha = 0.3;
    historyLength *= min(1.0, GetPixelPDF(max(0.0, dot(v.worldNormal, previousNormal)), ntermAlpha) / GetPixelPDF(0.9, ntermAlpha));

    historyLength *= inscreen;
    historyLength = max(1.0, historyLength);

#if Diffuse_Accumulation_Frame > 0
    float alpha = mix(1.0, max(1.0 / float(Diffuse_Accumulation_Frame), 1.0 / historyLength), inscreen);
#else
    float alpha = 1.0;
#endif

    vec3 accumulation = mix(historyColor, currentColor, alpha);

    gl_FragData[0] = vec4(accumulation, 1.0);
    gl_FragData[1] = vec4(accumulation, historyLength / 255.0);
    gl_FragData[2] = vec4(v.depth, texture(colortex2, texcoord).xy, 1.0);
}

/* RENDERTARGETS: 3,10,11 */