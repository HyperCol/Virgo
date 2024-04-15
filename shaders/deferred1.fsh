#version 130

#define GI_Render_Scale 0.25

in vec2 texcoord;

in vec3 SunLightingColor;
in vec3 MoonLightingColor;
in vec3 LightingColor;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/lighting/brdf.glsl"
#include "/libs/lighting/lighting_common.glsl"
#include "/libs/dither.glsl"

uniform sampler2D colortex4;

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;
uniform sampler2D shadowtex0;

//

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

float CalculateSkyLightmap(in float l) {
    float lightmap = saturate(rescale(l, 0.5 / 15.0, 14.0 / 15.0));
    return lightmap * lightmap;
}

//uniform mat4 shadowModelViewInverse;

uniform int hideGUI;

vec3 RSMSunIndirectLighting(in vec3 worldPosition, in vec3 worldNormal, in float dither) {
    mat3 shadowView = mat3(shadowProjectionInverse);
    mat3 worldToShadowView = mat3(shadowModelView);

    vec3 shadowViewNormal = worldToShadowView * worldNormal;
    vec3 shadowViewLight = worldToShadowView * worldLightVector;

    vec3 shadowCoord = CalculateShadowCoord(worldPosition + worldNormal * 0.001);

    float distortion = 1.0;
    //RemapShadowCoord(shadowCoord, distortion);
    //shadowCoord = shadowCoord * 0.5 + 0.5;
    //shadowCoord.z -= 0.000001;

    //vec3 diffuse = saturate(texture(shadowcolor1, shadowCoord.xy).xyz * 2.0 - 1.0);

    float roughness = 0.7;
    float totalWeight = 0.0;

    //vec3 shading = LinearToGamma(texture(shadowcolor0, shadowCoord.xy).rgb) * totalWeight;

    vec3 diffuse = vec3(0.0);

    float radius = 16.0 / 1000.0 * 2.0;

    int steps = 8;
    float invsteps = 1.0 / float(steps);


        //float r = pow(float(i + 1) * invsteps, 0.7);
        //float a = (float(i) + 0.5) * (sqrt(5.0) - 1.0) * Pi;
        //vec2 offset = vec2(cos(a), sin(a)) * r * radius;
    vec2 offsetSeed = (float2R2(dither * 1000.0 + 0.5) * 2.0 - 1.0);

    vec3 shadowViewPosition = shadowView * shadowCoord;

    for(int i = 0; i < steps; i++) {
    #if 0
        //vec2 offset = offsetSeed * radius * float(i + 1);
        vec2 offset = float2R2(float(i) + dither) * 2.0 - 1.0;
             offset *= radius;
    #else
        float r = pow(float(i + 1) * invsteps, 0.9);
        float a = (float(i) + dither) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r * radius;
    #endif
        //float distortion = 1.0;
        vec3 coord = RemapShadowCoord(shadowCoord + vec3(offset, 0.0));
             coord.z /= Shadow_Depth_Mul;
             coord = coord * 0.5 + 0.5;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) continue;

        float depth = GetShadowDepth(texture(shadowtex0, coord.xy).x);
        if(depth + 0.005 < coord.z) continue;

        vec3 shadowNormal = mat3(shadowModelView) * (texture(shadowcolor1, coord.xy).xyz * 2.0 - 1.0);
        vec3 albedo = LinearToGamma(texture(shadowcolor0, coord.xy).rgb);

        vec3 sampleShadowView = shadowView * (vec3((shadowCoord.xy * 0.5 + 0.5) + offset, depth) * 2.0 - 1.0);
        vec3 halfVector = sampleShadowView - shadowViewPosition;
        float vectorLength = length(halfVector);
        vec3 rayDirection = halfVector / vectorLength;

        float attenuation = 1.0 / max(1.0, halfVector.z * halfVector.z);
 
        float cosTheta = max(0.0, dot(shadowNormal, -rayDirection));
        float visibility = max(0.0, dot(shadowViewNormal, rayDirection));

        float weight = cosTheta / max(1.0, pow2(length(halfVector.xy)));

        float d = max(0.0, dot(shadowViewLight, shadowNormal)) * invPi;

        diffuse += albedo * (d * attenuation * weight * visibility);
        totalWeight += weight;
    }

    if(totalWeight > 0.0) {
        diffuse /= totalWeight;
    }

    return diffuse;
}

//
bool ScreenSpaceReflection(inout ivec2 hitTexel, in vec3 origin, in vec3 direction, in vec3 normal, in float dither) {
    int steps = 20;
    float invsteps = 1.0 / float(steps);

    vec3 rayDirection = direction;
    vec3 rayStart = origin;
    vec3 viewDirection = normalize(rayStart);
    float viewLinear = sqrt(rayStart.z * rayStart.z);

    float planeCheck = viewLinear * 0.01;

    float thickness = 0.5;

    bool intersect = false;

#if 0
    float rayLength = 0.2;

    vec3 rayPosition = rayStart;
    vec3 rayStep = direction * rayLength;

    float rayZmin = viewLinear;

    for(int i = 0; i < steps && intersect == false; i++) {
        vec3 coord = nvec3(gbufferProjection * nvec4(rayPosition)).xyz;
             coord.xy -= jitter * 0.5;
             coord = coord * 0.5 + 0.5;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        float rayLinear = sqrt(rayPosition.z * rayPosition.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        //vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        float planeDistance = abs(dot(normal, samplePosition - rayStart));

        if(planeDistance > planeCheck) {
            float rayZmax = rayLinear;

            intersect = rayZmax > sampleLinear && rayZmin < sampleLinear + thickness;

            hitTexel = texelPosition;

            rayZmin = rayLinear;
        }

        rayPosition += rayStep;
    }
#else
    float maxDistance = 0.1;
    float rayLength = rayStart.z + rayDirection.z * maxDistance > -near ? (-near - rayStart.z) / rayDirection.z : maxDistance;

    rayStart += normal * viewLinear * 0.004;

    vec3 rayEnd = rayStart + rayDirection * rayLength;

    vec4    H0 = gbufferProjection * nvec4(rayStart); 
    float   k0 = 1.0 / H0.w;
    float   Q0 = rayStart.z * k0;
    vec2    P0 = H0.xy * k0;
    float   L0 = length(rayStart) * k0;

    vec4    H1 = gbufferProjection * nvec4(rayEnd);
    float   k1 = 1.0 / H1.w;
    float   Q1 = rayEnd.z * k1;
    vec2    P1 = H1.xy * k1;
    float   L1 = length(rayEnd) * k1;

    vec2 delta = (P1 - P0) * resolution;
    //if(length(delta * resolution) < 1.0) delta = sign(delta) * step(vec2(0.5), abs(delta * resolution)) * texelSize;

    float stepSize = 1.0;
          //stepSize *= min(1.0 / abs(delta.x), 1.0 / abs(delta.y)) * 20.0;
          //stepSize *= 10.0;
          //stepSize *= 2.0 - min(1.0, -rayStart.z * 0.01);
          //stepSize *= 10.0 / max(1.0, viewLinear);
          //stepSize = min(texelSize.x, texelSize.y) * 30.0;
          //stepSize = min(texelSize.x, texelSize.y) * 200.0;
          //stepSize *= 1.0 + (1.0 - min(1.0, viewLinear / 10.0)) * 30.0;

	vec4 pqk    = vec4(P0, Q0, k0);
	vec4 dPQK   = vec4(P1 - P0, Q1 - Q0, k1 - k0) * stepSize;

    float currentLength = L0 * 0.0;
    float stepLength = (L1 - L0) * stepSize;

    pqk += dPQK * dither - dPQK;
    //ivec2 hitTexel = ivec2(0.0);

    float rayZmin = -Q0 / k0;

    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;
        //currentLength += stepLength;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        float rayLinear = sqrt(coord.z * coord.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        //vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        float planeDistance = abs(dot(normal, samplePosition - rayStart));
        if(planeDistance < planeCheck) continue;

        float rayZmax = rayLinear;

        intersect = rayZmax > sampleLinear && rayZmin < sampleLinear + thickness;

        hitTexel = texelPosition;
        rayZmin = rayLinear;
    }

    //if(intersect) {
    //    ssrRayLength = currentLength / pqk.w - L0 / k0;
    //    //ssrRayLength = length(GetViewPosition(vec2(hitTexel) * texelSize, texelFetch(depthtex0, hitTexel, 0).x));
    //}

#endif

    return intersect;
}

vec3 CalculateSkyLighting(in vec3 n) {
    vec3 color = vec3(0.0);
    float totalWeight = 0.0;

    float roughness = 0.999;

    int steps = 12;
    float invsteps = 1.0 / float(steps);

    vec2 envmapCoord = EncodeOctahedralmap(n) * 0.5 + 0.5;
         envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

    //float weight0 = 1.0;//GetPixelPDF(1.0, roughness);//DistributionTerm(roughness, 1.0) * 1.0;
    //color += LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR * weight0;
    //totalWeight += weight0;

    mat3 tbn = tbnNormal(n);

    float weatherAlpha = saturate(exp(-Rain_Fog_Density * Fog_Light_Extinction_Distance - Biome_Fog_Density * Fog_Light_Extinction_Distance));
    float weatherLight = HG(0.9999, Fog_BackScattering_Phase) * Fog_Front_Scattering_Weight;

    for(int i = 0; i < steps; i++) {
        //chashi
        float r = pow(float(i + 1) * invsteps, 0.7);
        float a = (float(i) + 0.5) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r;

        float cosTheta = sqrt(1.0 - dot(offset, offset));
        vec3 direction = normalize(tbn * vec3(offset.x, offset.y, 1.0));

        float weight = 1.0;//GetPixelPDF(max(0.0, dot(n, direction)), roughness);

        vec2 envmapCoord = EncodeOctahedralmap(direction) * 0.5 + 0.5;
             envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

        float occlusion = saturate(rescale(abs(dot(direction, worldUpVector)), -3.0, 1.0));

        vec3 sampleColor = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;
             //sampleColor = occlusion * sampleColor + (1.0 - occlusion) * topSample;

        //weather
        vec3 weatherColor = (sampleColor + LightingColor) * weatherLight;
        sampleColor = weatherAlpha * sampleColor + weatherColor - weatherColor * weatherAlpha;

        color += sampleColor * weight * occlusion;
        totalWeight += weight;
    }

    color /= totalWeight;

    //vec3 weatherColor = color + LightingColor;
    //color = weatherAlpha * color + (weatherColor - weatherColor * weatherAlpha) * (HG(-0.9999, Fog_BackScattering_Phase) * 0.7 * invPi);

    return color;
}

float ComputeAO(vec3 P, vec3 N, vec3 S) {
    vec3 V = S - P;
    float VdotV = dot(V, V);
    float NdotV = dot(N, V) / sqrt(VdotV);

    // Use saturate(x) instead of max(x,0.f) because that is faster
    return saturate(NdotV - 0.001) / max(1.0, VdotV);
    //return saturate(NdotV * GetPixelPDF(NdotV, 0.999) * 3.0 - 0.001) / max(1.0, VdotV);
}

float CalculateAO(in vec3 viewPosition, in vec3 normal) {
    float ao = 1.0;

    int steps = 24;

    ao = 0.0;
    float totalWeight = 0.0;

    for(int i = 0; i < steps; i++) {
        vec2 sampleCoord = texcoord + (float2R2(float(i) + 0.5) * 2.0 - 1.0) * max(texelSize.x, 0.2 / sqrt(viewPosition.z * viewPosition.z));
        float sampleDepth = texture(depthtex0, sampleCoord).x;
        //if(abs(sampleCoord))
        vec3 samplePosition = nvec3(gbufferProjectionInverse * vec4(sampleCoord * 2.0 - 1.0, sampleDepth * 2.0 - 1.0, 1.0));

        vec3 halfVector = samplePosition - viewPosition;
        float vecLength = length(halfVector);
        vec3 rayDirection = halfVector / vecLength;

        float weight = 1.0;//GetPixelPDF(max(0.0, dot(normalize(samplePosition), normalize(viewPosition))), 0.999);

        ao += ComputeAO(viewPosition, normal, samplePosition) * weight;
        totalWeight += weight;
    }

    ao /= totalWeight;
    //ao = min(1.0, ao / GetPixelPDF(1.0, 0.1));

    return 1.0 - ao;
}

float ScreenSpaceTracing(inout vec3 diffuse, in vec3 rayStart, in vec3 rayDirection, in vec3 normal, in vec3 ambientColor) {
    float maxDistance = 1.5;
    float thickness = 0.5;

    int steps = 8;
    float invsteps = 1.0 / float(steps);

    float rayLength = rayStart.z + rayDirection.z * maxDistance > -near ? (-near - rayStart.z) / rayDirection.z : maxDistance;
    vec3 rayEnd = rayStart + rayDirection * invsteps * maxDistance;

    vec4    H0 = gbufferProjection * nvec4(rayStart); 
    float   k0 = 1.0 / H0.w;
    float   Q0 = rayStart.z * k0;
    vec2    P0 = H0.xy * k0;

    vec4    H1 = gbufferProjection * nvec4(rayEnd);
    float   k1 = 1.0 / H1.w;
    float   Q1 = rayEnd.z * k1;
    vec2    P1 = H1.xy * k1;

    vec2 delta = abs(P1 - P0) * resolution;
    float stepSize = 1.0;//20.0 / min(delta.x, delta.y);

	vec4 pqk    = vec4(P0, Q0, k0);
	vec4 dPQK   = vec4(P1 - P0, Q1 - Q0, k1 - k0) * stepSize;
    float rayZmin = -Q0 / k0;

    bool intersect = false;
    ivec2 hitTexel = ivec2(0);

    float closest = 1000.0;
    vec3 fakeLightPosition = vec3(0.0);

    float occlusion = 1.0;

    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        float rayLinear = sqrt(coord.z * coord.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        //vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        //float planeDistance = abs(dot(normal, samplePosition - rayStart));
        //if(planeDistance < planeCheck) continue;

        float rayZmax = rayLinear;

        float d = rayZmax - sampleLinear;

        if(d > 0.0 && d < closest) {
            //fakeLightPosition = vec3(samplePosition.xy, coord.z);
            fakeLightPosition = samplePosition;
            closest = d;
        }

        intersect = rayZmax > sampleLinear && rayZmin < sampleLinear + thickness;

        hitTexel = texelPosition;
        rayZmin = rayLinear;
    }

    if(intersect) {
        occlusion = 0.0;

        vec3 albedo = LinearToGamma(texelFetch(colortex0, hitTexel, 0).rgb);
        vec2 lightmap = unpack2x8(texelFetch(colortex1, hitTexel, 0).y);
        float skyLightmap = CalculateSkyLightmap(lightmap.y);
        diffuse = ambientColor * albedo * skyLightmap;

        vec3 viewPosition = GetViewPosition(vec2(hitTexel) * texelSize, texelFetch(depthtex0, hitTexel, 0).x);
        vec3 worldPosition = mat3(gbufferModelViewInverse) * viewPosition + gbufferModelViewInverse[3].xyz;

        vec3 normal = DecodeSpheremap(texelFetch(colortex2, hitTexel, 0).xy);
        vec3 geometryNormal = DecodeSpheremap(texelFetch(colortex2, hitTexel, 0).zw);

        vec3 shadowCoord = CalculateShadowCoord(worldPosition + mat3(gbufferModelViewInverse) * normal * 0.001);
        vec3 coord = RemapShadowCoord(shadowCoord);
             //coord.z /= Shadow_Depth_Mul;
             coord = coord * 0.5 + 0.5;
             coord.z -= 0.0005 * Shadow_Depth_Mul;;

        float shading = step(coord.z, texture(shadowtex0, coord.xy).x) * invPi * max(0.0, dot(normal, lightVector)) * SimpleGeometryTerm(dot(geometryNormal, lightVector));
        diffuse += LightingColor * albedo * shading * step(1e-5, dot(normal, -rayDirection));
    } else {
        if(closest < 1000.0) {
            occlusion = max(0.0, dot(rayDirection, normal)) / max(1.0, pow2(length(fakeLightPosition - rayStart)));
            occlusion = 1.0 - min(occlusion, 1.0);
        }

        //vec2 envmapCoord = EncodeOctahedralmap(mat3(gbufferModelViewInverse) * rayDirection) * 0.5 + 0.5;
        //     envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);
        //diffuse = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR * occlusion;
    }

    return occlusion;
}
#if 0
bool CalculateGI(inout vec3 diffuse, in vec3 viewPosition, in vec3 normal, in vec3 rayDirection, in float dither) {
    int steps = 8;
    float invsteps = 1.0 / float(steps);

    vec3 rayStart = viewPosition;
    vec3 viewDirection = normalize(rayStart);
    float thickness = 0.3;

    vec3 rayStep = rayDirection * invsteps * 3.0;
    vec3 rayPosition = rayStart;

    ivec2 hitTexel = ivec2(0);

    bool intersect = false;
    bool doao = false;

    vec3 fakeLightPosition = rayStart + rayDirection * 2000.0;
    float closest = 1000.0;

    vec3 skyLight = CalculateSkyLighting(vec3(0.0, 1.0, 0.0));
    float occlusion = 1.0;//(diffuse, rayStart, rayDirection, normal, skyLight);

#if 0
#if 1
    float viewLinear = sqrt(rayStart.z * rayStart.z);
    float planeCheck = viewLinear * 0.01;

    float maxDistance = 2.0;
    float rayLength = rayStart.z + rayDirection.z * maxDistance > -near ? (-near - rayStart.z) / rayDirection.z : maxDistance;

    vec3 rayEnd = rayStart + rayDirection * invsteps * maxDistance;

    vec4    H0 = gbufferProjection * nvec4(rayStart); 
    float   k0 = 1.0 / H0.w;
    float   Q0 = rayStart.z * k0;
    vec2    P0 = H0.xy * k0;

    vec4    H1 = gbufferProjection * nvec4(rayEnd);
    float   k1 = 1.0 / H1.w;
    float   Q1 = rayEnd.z * k1;
    vec2    P1 = H1.xy * k1;

    vec2 delta = abs(P1 - P0) * resolution;
    float stepSize = 1.0;//20.0 / min(delta.x, delta.y);

	vec4 pqk    = vec4(P0, Q0, k0);
	vec4 dPQK   = vec4(P1 - P0, Q1 - Q0, k1 - k0) * stepSize;
    float rayZmin = -Q0 / k0;

    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        float rayLinear = sqrt(coord.z * coord.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        //vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        float planeDistance = abs(dot(normal, samplePosition - rayStart));
        //if(planeDistance < planeCheck) continue;

        float rayZmax = rayLinear;

        float d = rayZmax - sampleLinear;

        if(d > 0.0 && doao == false) {
            doao = true;
            //fakeLightPosition = vec3(samplePosition.xy, coord.z);
            fakeLightPosition = samplePosition;
            closest = d;
        }

        intersect = rayZmax > sampleLinear && rayZmin < sampleLinear + thickness;

        hitTexel = texelPosition;
        rayZmin = rayLinear;
    }

#else
    for(int i = 0; i < steps; i++) {
        rayPosition += rayStep;

        vec3 coord = nvec3(gbufferProjection * nvec4(rayPosition)) * 0.5 + 0.5;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) {
            rayPosition += rayDirection * 1000.0;
            break;
        }

        float rayLength = length(rayPosition);

        ivec2 sampleTexel = ivec2(coord.xy * resolution);

        float sampleDepth = texelFetch(depthtex0, sampleTexel, 0).x;
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLength = length(samplePosition);

        float delta = rayLength - sampleLength;

        if(delta > 0.0 && delta < closest) {
            fakeLightPosition = rayPosition;
            closest = delta;
        }

        if(delta > 0.0 && delta < thickness) {
            hitTexel = sampleTexel;
            intersect = true;
            break;
        }
    }
#endif
    vec3 skyLight = CalculateSkyLighting(vec3(0.0, 1.0, 0.0));

    vec2 envmapCoord = EncodeOctahedralmap(mat3(gbufferModelViewInverse) * rayDirection) * 0.5 + 0.5;
         envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

    //occlusion = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;
    //occlusion *= CalculateSkyLighting(mat3(gbufferModelViewInverse) * normal);
    //occlusion = vec3(1.0);

    float sampleDepth = texelFetch(depthtex0, hitTexel, 0).x;
    vec3 samplePosition = GetViewPosition(vec2(hitTexel) * texelSize, sampleDepth);

    float ao = max(0.0, dot(rayDirection, normal)) / max(1.0, pow2(length(fakeLightPosition - rayStart)));
          ao = 1.0 - min(ao, 1.0);

    //diffuse *= (hideGUI == 0 ? ao : 1.0);

    if(intersect) {
        vec3 albedo = LinearToGamma(texelFetch(colortex0, hitTexel, 0).rgb);
        vec2 lightmap = unpack2x8(texelFetch(colortex1, hitTexel, 0).y);
        float skyLightmap = CalculateSkyLightmap(lightmap.y);
        diffuse = skyLight * albedo * skyLightmap;

        vec3 viewPosition = GetViewPosition(vec2(hitTexel) * texelSize, texelFetch(depthtex0, hitTexel, 0).x);
        vec3 worldPosition = mat3(gbufferModelViewInverse) * viewPosition + gbufferModelViewInverse[3].xyz;

        vec3 normal = DecodeSpheremap(texelFetch(colortex2, hitTexel, 0).xy);
        vec3 geometryNormal = DecodeSpheremap(texelFetch(colortex2, hitTexel, 0).zw);

        vec3 shadowCoord = CalculateShadowCoord(worldPosition + mat3(gbufferModelViewInverse) * normal * 0.001);
        vec3 coord = RemapShadowCoord(shadowCoord);
             //coord.z /= Shadow_Depth_Mul;
             coord = coord * 0.5 + 0.5;
             coord.z -= 0.0005 * Shadow_Depth_Mul;

        float shading = step(coord.z, texture(shadowtex0, coord.xy).x) * invPi * max(0.0, dot(normal, lightVector)) * SimpleGeometryTerm(dot(geometryNormal, lightVector));
        diffuse += LightingColor * albedo * shading;
    }
#endif

    //if(hideGUI == 1) occlusion = vec3(1.0);

    /*
    float viewLinear = sqrt(rayStart.z * rayStart.z);

    float planeCheck = viewLinear * 0.01;


    bool intersect = false;
    float maxDistance = 0.1;
    float rayLength = rayStart.z + rayDirection.z * maxDistance > -near ? (-near - rayStart.z) / rayDirection.z : maxDistance;

    rayStart += normal * viewLinear * 0.004;

    vec3 rayEnd = rayStart + rayDirection * rayLength;

    vec4    H0 = gbufferProjection * nvec4(rayStart); 
    float   k0 = 1.0 / H0.w;
    float   Q0 = rayStart.z * k0;
    vec2    P0 = H0.xy * k0;
    float   L0 = length(rayStart) * k0;

    vec4    H1 = gbufferProjection * nvec4(rayEnd);
    float   k1 = 1.0 / H1.w;
    float   Q1 = rayEnd.z * k1;
    vec2    P1 = H1.xy * k1;
    float   L1 = length(rayEnd) * k1;

    vec2 delta = (P1 - P0) * resolution;
    //if(length(delta * resolution) < 1.0) delta = sign(delta) * step(vec2(0.5), abs(delta * resolution)) * texelSize;

    float stepSize = 1.0;
          //stepSize *= min(1.0 / abs(delta.x), 1.0 / abs(delta.y)) * 20.0;
          //stepSize *= 10.0;
          //stepSize *= 2.0 - min(1.0, -rayStart.z * 0.01);
          //stepSize *= 10.0 / max(1.0, viewLinear);
          //stepSize = min(texelSize.x, texelSize.y) * 30.0;
          //stepSize = min(texelSize.x, texelSize.y) * 200.0;
          //stepSize *= 1.0 + (1.0 - min(1.0, viewLinear / 10.0)) * 30.0;

	vec4 pqk    = vec4(P0, Q0, k0);
	vec4 dPQK   = vec4(P1 - P0, Q1 - Q0, k1 - k0) * stepSize;

    float currentLength = L0 * 0.0;
    float stepLength = (L1 - L0) * stepSize;

    pqk += dPQK * dither - dPQK;
    //ivec2 hitTexel = ivec2(0.0);

    float rayZmin = -Q0 / k0;

    ivec2 hitTexel = ivec2(0);

    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;
        //currentLength += stepLength;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        float rayLinear = sqrt(coord.z * coord.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        //vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        float planeDistance = abs(dot(normal, samplePosition - rayStart));
        if(planeDistance < planeCheck) continue;

        float rayZmax = rayLinear;

        intersect = rayZmax > sampleLinear && rayZmin < sampleLinear + thickness;

        hitTexel = texelPosition;
        rayZmin = rayLinear;
    }
    */
/*
    if(intersect) { 
        occlusion = LinearToGamma(texelFetch(colortex0, hitTexel, 0).rgb);
    } else {
        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        float sampleDepth = texelFetch(depthtex0, hitTexel, 0).x;
        vec3 samplePosition = GetViewPosition(vec2(hitTexel) * texelSize, sampleDepth);

        vec3 rayPosition = nvec3(gbufferProjectionInverse * nvec4(coord * 2.0 - 1.0));

        vec3 H = samplePosition - viewPosition;

        occlusion = max(0.0, 1.0 - max(0.0, dot(rayDirection, normal)) / max(1.0, dot(H, H))) * vec3(1.0);
    }
*/
    return intersect;    
}
#endif
void main() {
    vec2 fragCoord = texcoord * resolution;
    ivec2 texelPosition = ivec2(fragCoord);
    GbuffersData data = GetGbuffersData(texcoord);
    VectorStruct v = CalculateVectorStruct(texcoord, texelFetch(depthtex0, texelPosition, 0).x);

    float fframeCounter = float(frameCounter);

//#if Diffuse_Accumulation_Frame > 0
    vec2 timeOffset = (float2R2(fframeCounter + 0.5) * 2.0 - 1.0);
//#else
    //vec2 timeOffset = vec2(0.0);
//#endif

    vec2 seed = fragCoord * GI_Render_Scale;
    ivec2 iseed = ivec2(seed + timeOffset * 64.0);

    float dither = GetBlueNoise(iseed);
    float dither1 = GetBlueNoise1(iseed);
    float dither2 = GetBlueNoise2(iseed);

    //dither = R2dither(fframeCounter, dither * 255.0);
    //dither1 = R2dither(fframeCounter, dither1 * 255.0);
    //dither2 = R2dither(fframeCounter, dither2 * 255.0);

    //dither1 = R2dither(seed, fframeCounter);
    //dither2 = R2dither(seed, fframeCounter);

    mat3 tbn = tbnNormal(v.worldNormal);

    vec4 rayPDF = ImportanceSampleGGX(vec2(dither1, dither2), 0.9999);
    vec3 rayDirection = normalize(tbn * rayPDF.xyz);
         rayDirection = CalculateVisibleNormals(rayDirection, v.worldEyeDirection);

    vec2 envmapCoord = EncodeOctahedralmap(rayDirection) * 0.5 + 0.5;
         envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

    ivec2 hitTexel = ivec2(0);
    bool ssdoHit = false;//ScreenSpaceReflection(hitTexel, v.viewPosition, mat3(gbufferModelView) * rayDirection, v.geometryNormal, 1.0);//

    //vec3 ambientLighting = 

    //if(ssdoHit) {
    //    ambientLighting *= LinearToGamma(texelFetch(colortex0, hitTexel, 0).rgb);
    //}

    //vec3 ao = ssdoHit ? (texelFetch(colortex0, hitTexel, 0).rgb) : vec3(1.0);
    //vec3 diffuse = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;
    //vec3 diffuse = vec3(1.0);//CalculateSkyLighting(v.worldGeometryNormal);

    //diffuse *= skyLightmap;

    vec3 skyLight = CalculateSkyLighting(vec3(0.0, -1.0, 0.0));
         //skyLight = vec3(1.0);

    //diffuse = vec3(0.0);

    vec3 rayStart = v.worldPosition;
    vec3 tracedNormal = v.worldNormal;
#if 0
    if(ssdoHit) {
        //diffuse = vec3(0.0);
        
        vec3 albedo = LinearToGamma(texelFetch(colortex0, hitTexel, 0).rgb);

        float depth = texelFetch(depthtex0, hitTexel, 0).x;

        vec3 normal = DecodeSpheremap(texelFetch(colortex2, hitTexel, 0).xy);
        vec3 geometryNormal = DecodeSpheremap(texelFetch(colortex2, hitTexel, 0).zw);
        vec3 viewPosition = GetViewPosition(vec2(hitTexel) * texelSize, depth);

        vec2 lightmap = unpack2x8(texelFetch(colortex1, hitTexel, 0).y);
        float skyLightmap = CalculateSkyLightmap(lightmap.y);

/*
        //vec3 shadowCoord = RemapShadowCoord(CalculateShadowCoord(mat3(gbufferModelViewInverse) * viewPosition + gbufferModelViewInverse[3].xyz));
        //     shadowCoord = shadowCoord * 0.5 + 0.5;

        diffuse = skyLight * albedo * skyLightmap;
        //diffuse += step(shadowCoord.z, texture(shadowtex0, shadowCoord.xy).x) * albedo * invPi;

        vec3 worldPosition = mat3(gbufferModelViewInverse) * viewPosition + gbufferModelViewInverse[3].xyz;
        vec3 worldNormal = mat3(gbufferModelViewInverse) * normal;

        //diffuse = vec3(0.0);

        vec3 shadowCoord = CalculateShadowCoord(worldPosition + worldNormal * 0.001);
        vec3 coord = RemapShadowCoord(shadowCoord);
             //coord.z /= Shadow_Depth_Mul;
             coord = coord * 0.5 + 0.5;
             coord.z -= 0.0005 * Shadow_Depth_Mul;;

        float shading = step(coord.z, texture(shadowtex0, coord.xy).x) * invPi * max(0.0, dot(normal, lightVector)) * SimpleGeometryTerm(dot(geometryNormal, lightVector));

        //diffuse += albedo * LightingColor * shading;

        rayStart = worldPosition;
        tracedNormal = worldNormal;
        */
    }
    #endif

    vec3 diffuse = vec3(1.0);
    //diffuse *= LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;
    //diffuse *= CalculateSkyLighting(v.worldGeometryNormal);

    float skyLightmap = CalculateSkyLightmap(data.lightmap.y);
    //diffuse *= skyLightmap;

    //float ao = CalculateAO(v.viewPosition, v.texturedNormal);
    //diffuse *= ao;

    vec3 direction = mat3(gbufferModelView) * rayDirection;
         //direction = normalize(reflect(v.viewDirection, v.texturedNormal));

    //bool intersect = CalculateGI(diffuse, v.viewPosition, v.texturedNormal, direction, 1.0);
    //diffuse += (rayStart, tracedNormal, dither) * LightingColor * (hideGUI == 1 ? 0.0 : 1.0);

    float weatherLight = mix(HG(-0.9999, Fog_BackScattering_Phase) * (1.0 - Fog_Front_Scattering_Weight), 1.0, saturate(exp(-Fog_Light_Extinction_Distance * Rain_Fog_Density - Biome_Fog_Density * Fog_Light_Extinction_Distance)));

    mat3 mvm = mat3(gbufferModelView);

    diffuse = vec3(0.0);
    float occlusion = 0.0;

    int steps = 4;
    float invsteps = 1.0 / float(steps);

    for(int i = 0; i < steps; i++) {
        vec3 rayStart = v.viewPosition;

        vec4 rayPDF = ImportanceSampleGGX(vec2((float(i) + dither1) * 0.25, dither2), 0.9999);
        vec3 rayDirection = normalize(tbn * rayPDF.xyz);
            rayDirection = CalculateVisibleNormals(rayDirection, v.worldEyeDirection);
            rayDirection = mvm * rayDirection;

        vec3 color = vec3(0.0);
        occlusion += ScreenSpaceTracing(color, rayStart, rayDirection, v.texturedNormal, skyLight);

        diffuse += color * weatherLight;
    }

    diffuse *= invsteps;
    occlusion *= invsteps;

    diffuse += CalculateSkyLighting(normalize(worldLightVector * 0. + v.worldGeometryNormal)) * occlusion;
    diffuse += (RSMSunIndirectLighting(rayStart, tracedNormal, dither) * LightingColor) * (occlusion * weatherLight);// * (hideGUI == 1 ? 0.0 : 1.0);

    diffuse = GammaToLinear(diffuse);

    gl_FragData[0] = vec4(diffuse, 1.0);
    gl_FragData[1] = vec4(v.depth, texture(colortex2, texcoord).xy, 1.0);
}
/* DRAWBUFFERS:35 */