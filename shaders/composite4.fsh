#version 130

const float maxDistance = 2048.0;

uniform sampler2D colortex3;
uniform sampler2D colortex4;

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

//https://www.shadertoy.com/view/4XfSDB
vec3 frustumClip(vec3 from, vec3 to, in float near, in float far, in vec2 s, in float inf) {
    vec3 dir = to - from;
    vec3 signDir = sign(dir);

    float nfSlab = signDir.z * (far - near) * 0.5 + (far + near) * 0.5;
    float lenZ = (nfSlab - from.z) / dir.z;
    if (dir.z == 0.0) lenZ = inf;

    vec2 ss = sign(dir.xy - s * dir.z) * s;
    vec2 denom = ss * dir.z - dir.xy;
    vec2 lenXY = (from.xy - ss * from.z) / denom;
    if (lenXY.x < 0.0 || denom.x == 0.0) lenXY.x = inf;
    if (lenXY.y < 0.0 || denom.y == 0.0) lenXY.y = inf;

    float len = min(min(1.0f, lenZ), min(lenXY.x, lenXY.y));
    vec3 clippedVS = from + dir * len;
    
    return clippedVS;
}

vec4 ScreenSpaceReflection(in vec3 rayStart, in vec3 rayDirection, in vec3 normal, in float dither, inout float rayLinearDepth) {
    int steps = 20;
    float invsteps = 1.0 / float(steps);

    float thickness = 0.5;
    float planeCheck = 2e-3;

    bool ClipToNear = rayStart.z + rayDirection.z * maxDistance > -near;
    float rayLength = ClipToNear ? (-near - rayStart.z) / rayDirection.z : maxDistance;

    float flipZ = -1.0;

    vec3 rayEnd = rayStart + rayDirection * maxDistance;
         rayEnd = flipZ * frustumClip(rayStart * flipZ, rayEnd * flipZ, near, maxDistance, vec2(1.0) / vec2(gbufferProjection[0][0], gbufferProjection[1][1]), maxDistance);

    vec4    H0 = gbufferProjection * nvec4(rayStart);
    vec4    H1 = gbufferProjection * nvec4(rayEnd);

    float   k0 = 1.0 / H0.w;
    float   Q0 = rayStart.z * k0;
    vec2    P0 = H0.xy * k0;

    float   k1 = 1.0 / H1.w;
    float   Q1 = rayEnd.z * k1;
    vec2    P1 = H1.xy * k1;

    vec2 delta = (P1 - P0);

    float stepSize = invsteps;
          //stepSize = ClipToNear ? 0.05 : 1.0;

	vec4 pqk    = vec4(P0, Q0, k0);
	vec4 dPQK   = vec4(delta, Q1 - Q0, k1 - k0) * stepSize;

    //pqk += dPQK * mix(dither, 1.0, 0.9) - dPQK;

    float rayZmin = -Q0 / k0;

    bool intersect = false;
    ivec2 hitTexel = ivec2(0.0);
    float ltracing = 0.0;

    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        float planeDistance = abs(dot(normal, samplePosition - rayStart)) / sampleLinear;
        if(planeDistance < planeCheck) continue;

        float rayZmax = sqrt(coord.z * coord.z);

        if(rayZmax > sampleLinear && (rayZmin - sampleLinear) / sampleLinear < thickness) {
            intersect = true;
        } else {
            rayZmin = rayZmax;
        }
    }

    vec4 color = vec4(0.0);

    if(intersect) {
        vec2 coord = pqk.xy;

    #if 1
        int steps2 = 4;
        float invsteps2 = 1.0 / float(steps2);

        vec4 p0 = pqk - dPQK;
        vec4 p1 = pqk;

        #if 1
        pqk -= dPQK;
        dPQK *= invsteps2;

        for(int i = 0; i < steps2; i++) {
            pqk += dPQK;

            vec3 coord = pqk.xyz;
                 coord.xy = coord.xy * 0.5 + 0.5;
                 coord.z = coord.z / pqk.w;

            //if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

            ivec2 texelPosition = ivec2(coord.xy * resolution);

			float rayZmax = sqrt(coord.z * coord.z);

            float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
            float sampleLinear = ExpToLinearDepth(sampleDepth);

            if(rayZmax > sampleLinear) {
                pqk -= dPQK;
                dPQK *= 0.5;
                p1 = pqk;
            }
        }

        #else
        for(int i = 0; i < steps2; i++) {
            vec4 p = p0 * 0.5 + p1 * 0.5;

            vec3 coord = p.xyz;
                 coord.xy = coord.xy * 0.5 + 0.5;
                 coord.z = coord.z / p.w;

            //if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

            ivec2 texelPosition = ivec2(coord.xy * resolution);

			float rayZmax = sqrt(coord.z * coord.z);
            float sampleLinear = ExpToLinearDepth(texelFetch(depthtex0, texelPosition, 0).x);

            if(rayZmax > sampleLinear) {
                p1 = p;
            } else {
                p0 = p;
            }
        }
        #endif
    #endif

        coord = p1.xy * 0.5 + 0.5;
        ivec2 hitTexel = ivec2(coord * resolution);

        float sampleDepth = texelFetch(depthtex0, hitTexel, 0).x;

        if(sampleDepth < 1.0) {
            color = vec4(LinearToGamma(texelFetch(colortex3, hitTexel, 0).rgb) * MappingToHDR, 1.0);
            //rayLinearDepth = ExpToLinearDepth(sampleDepth);
            //rayLinearDepth = ltracing;
            rayLinearDepth = -p1.z / p1.w * abs(rayDirection.z) - rayStart.z;
            //rayLinearDepth = length(nvec3(gbufferProjectionInverse * nvec4(vec3(pqk.xy / pqk.w, LinearToExpDepth(pqk.z / pqk.w) * 2.0 - 1.0))));
        }
    }
    

    return color;
}

uniform float screenBrightness;

void main() {
    float fframeCounter = float(frameCounter);

    const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    vec2 stageJitter = float2R2(fframeCounter) * 2.0 - 1.0;

    vec2 coord = texcoord;

#if 1
    #if 1
    coord += offset[frameCounter % 4] * texelSize;
    #else
    coord += (float2R2(fframeCounter) * 2.0 - 1.0) * texelSize;
    #endif
#endif

    GbuffersData data = GetGbuffersData(coord);
    VectorStruct v = CalculateVectorStruct(coord, texture(depthtex0, coord).x);

    //perfect surface:smoothness=0.9999

    if(data.tileID == F_Water) {
        //not perfect surface
        float smoothness = 0.997;
        data.roughness = pow2(1.0 - smoothness);
        data.metalness = 0.02;
    }

    vec2 frameCountOffset = float2R2(fframeCounter + 0.5) * 2.0 - 1.0;

#ifdef Enabled_Temporal_AA

#else
    //vec2 frameCountOffset = vec2(0.0);
#endif

    //frameCountOffset = vec2(0.0);

    ivec2 iseed = ivec2(texcoord * resolution + frameCountOffset * 64.0);
    float dither = GetBlueNoise(iseed);
    float dither1 = GetBlueNoise1(iseed);

    //vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    vec3 rayDirection = normalize(reflect(v.viewDirection, data.texturedNormal));
    //vec2 envCoord = EncodeOctahedralmap(rayDirection) * 0.5 + 0.5;

    //store visible normal before reflection
    //vec3 worldNormal = mat3(gbufferModelViewInverse) * data.texturedNormal;
    //vec3 geometryNormal = mat3(gbufferModelViewInverse) * data.geometryNormal;
    vec3 visibleNormal = mat3(gbufferModelViewInverse) * data.texturedNormal;//CalculateVisibleNormals(worldNormal, v.worldEyeDirection);
    mat3 tbn = tbnNormal(visibleNormal);

    float skyVisibility = saturate(rescale(data.lightmap.y, 0.5 / 15.0, 14.0 / 15.0));
          skyVisibility = pow(skyVisibility, 5.0);

    vec3 specular = vec3(0.0);
    float weight = 0.0;

    float rayMinLinearDepth = maxDistance;

    float resultDepth = 1.0;
    float importantWeight = 0.0;
    
if(v.depth < 1.0) {
    int steps = 1;
    float invsteps = 1.0 / float(steps);

    bool prefect = false;

    for(int i = 0; i < steps; i++) {
        vec4 rayPDF = ImportanceSampleGGX(vec2((float(i) + dither) * invsteps, dither1), data.roughness);

        //if(rayPDF.z > 0.9999) {
        //    prefect = true;
        //    rayPDF = vec4(0.0, 0.0, 1.0, 1.0);
        //}

        vec3 H = normalize(tbn * rayPDF.xyz);

        vec3 rayDirection = normalize(reflect(v.worldDirection, H));
             rayDirection = CalculateVisibleNormals(rayDirection, visibleNormal);
             rayDirection = mat3(gbufferModelView) * rayDirection;

        float vdoth = abs(dot(rayDirection, H));
        float ndoth = saturate(dot(visibleNormal, H));

        float pdf = DistributionTerm(ndoth, data.roughness) * ndoth / (4.0 * vdoth + 1e-5);

        float rayLength = maxDistance;
        vec4 ssr = ScreenSpaceReflection(v.viewPosition, rayDirection, data.texturedNormal, 0.5, rayLength);

        vec2 envmapCoord = EncodeOctahedralmap(mat3(gbufferModelViewInverse) * rayDirection) * 0.5 + 0.5;
             envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);
        vec3 skyEnvmap = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR * skyVisibility;

        //if(dot(ssr.rgb, vec3(1.0 / 3.0)) > 1.0) ssr.a = 0.0;

        specular += mix(skyEnvmap, ssr.rgb, ssr.a) * pdf;
        weight += pdf;

        if(pdf >= importantWeight) {
            rayMinLinearDepth = min(rayMinLinearDepth, rayLength);
            //resultDepth = nvec3(gbufferProjection * nvec4(v.viewPosition + v.viewDirection * rayLength)).z * 0.5 + 0.5;
            importantWeight = pdf;
        }

        //if(rayPDF.z > 0.9999 && pdf > 1000.0) {
            //specular = vec3(1.0, 0.0, 0.0);
            //weight = 1.0;
        //    break;
        //}
    }

    if(weight > 0.0) {
        specular /= weight;
    }
}

    //specular = vec3(resultDepth / 100.0);

    specular = GammaToLinear(specular * MappingToSDR);
    specular = saturate(specular);

    //resultDepth = -v.viewPosition.z + abs(rayMinLinearDepth) * -v.viewDirection.z;
    //resultDepth = v.linearDepth + sqrt(rayMinLinearDepth * rayMinLinearDepth) * -rayDirection.z;
    resultDepth = LinearToExpDepth(rayMinLinearDepth);
    //resultDepth = nvec3(gbufferProjection * nvec4(v.viewPosition + rayDirection * resultDepth)).z * 0.5 + 0.5;

    gl_FragData[0] = vec4(specular, data.roughness);
    gl_FragData[1] = vec4(texture(colortex2, texcoord).rg, v.depth, 1.0);
    gl_FragData[2] = vec4(resultDepth, vec3(1.0));
}
/* DRAWBUFFERS:456 */