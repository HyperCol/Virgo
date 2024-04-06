#version 130

in vec2 texcoord;

uniform sampler2D colortex3;
uniform sampler2D colortex4;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/materialid.glsl"
#include "/libs/dither.glsl"
#include "/libs/intersection.glsl"
#include "/libs/lighting/lighting_common.glsl"
#include "/libs/lighting/lighting_color.glsl"
#include "/libs/lighting/shadowmap.glsl"
#include "/libs/lighting/brdf.glsl"

in vec3 SunLightingColor;
in vec3 MoonLightingColor;
in vec3 LightingColor;

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

struct GbuffersData {
    vec3 albedo;
    float alpha;

    vec2 lightmap;

    vec3 texturedNormal;
    vec3 geometryNormal;

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
5 specular
6 specular
7 taa exposure
*/
GbuffersData GetGbuffersData(in vec2 coord) {
    GbuffersData data;

    vec4 tex0 = texture(colortex0, coord);
    vec4 tex1 = texture(colortex1, coord);
    vec4 tex2 = texture(colortex2, coord);

    data.albedo = LinearToGamma(tex0.rgb);
    data.alpha = tex0.a;
    
    data.texturedNormal = DecodeSpheremap(tex2.rg);
    data.geometryNormal = DecodeSpheremap(tex2.ba);

    data.lightmap = unpack2x8(tex1.y);

    vec2 unpack1x  = unpack2x8(tex1.x);
    data.roughness = pow2(1.0 - unpack1x.r);
    data.metalness = unpack1x.y;

    data.metallic = step(0.9, data.metalness);

    data.F0 = mix(vec3(max(0.02, data.metalness)), data.albedo, vec3(data.metallic));

    return data;
}
/*
ViewVector GetViewVector(in vec2 texcoord, in float depth) {
    ViewVector v;

    v.depth = depth;
    v.position = GetViewPosition(texcoord, depth);
    v.l = length(v.position);
    v.linearDepth = sqrt(v.position.z * v.position.z);

    return v;
}

struct VectorStruct2 {
    vec3 worldPosition;
    vec3 gameWorldPosition;
};
*/
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

    v.viewPosition = GetViewPosition(coord, depth);
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

#ifdef DISTANT_HORIZONS 
//DH support
uniform sampler2D dhDepthTex0;
//uniform sampler2D dhDepthTex1;

//uniform float dhNearPlane;
//uniform float dhFarPlane;

//uniform int dhRenderDistance;

//uniform mat4 dhProjection;
uniform mat4 dhProjectionInverse;
//uniform mat4 dhPreviousProjection;

void GetDHVector(inout VectorStruct v, in vec2 coord) {
    if(v.depth >= 1.0) {
        v.depth = texture(dhDepthTex0, coord).x;
        v.viewPosition = nvec3(dhProjectionInverse * vec4(ApplyTAAJitter(coord) * 2.0 - 1.0, v.depth * 2.0 - 1.0, 1.0));
        v.worldPosition = mat3(gbufferModelViewInverse) * v.viewPosition;
        v.viewDistance = length(v.viewPosition);
        v.linearDepth = sqrt(v.viewPosition.z * v.viewPosition.z);
    }
}
#else
void GetDHVector(inout VectorStruct v, in vec2 coord) {
}
#endif

vec3 CalculateSkyLighting(in vec3 n) {
    vec3 color = vec3(0.0);
    float totalWeight = 0.0;

    float roughness = 0.999;

    int steps = 12;
    float invsteps = 1.0 / float(steps);

    vec2 envmapCoord = EncodeOctahedralmap(n) * 0.5 + 0.5;
         envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

    float weight0 = 1.0;//GetPixelPDF(1.0, roughness);//DistributionTerm(roughness, 1.0) * 1.0;
    color += LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR * weight0;
    totalWeight += weight0;

    mat3 tbn = tbnNormal(n);

    for(int i = 0; i < steps; i++) {
        //chashi
        float r = pow(float(i + 1) * invsteps, 0.7);
        float a = (float(i) + 0.5) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r;

        float cosTheta = sqrt(1.0 - dot(offset, offset));
        vec3 direction = normalize(tbn * vec3(offset.x, offset.y, 1.0));

        float weight = max(0.0, dot(direction, n));//GetPixelPDF(max(0.0, dot(n, direction)), roughness);

        vec2 envmapCoord = EncodeOctahedralmap(direction) * 0.5 + 0.5;
             envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

        color += LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR * weight;
        totalWeight += weight;
    }

    color /= totalWeight;

    return color;
}

float ScreenSpaceContactShadow(in vec3 rayStart, in vec3 rayDirection, in vec3 normal, in float dither) {
    float shading = 1.0;

    float ndotl = dot(normal, rayDirection);

    //float dither = GetBlueNoise(texcoord);
    //      dither = R2dither(dither * 128.0, float(frameCounter));
    //float dither = 0.5;

    int steps = 12;
    float invsteps = 1.0 / float(steps);

    //float viewLength = length(rayStart);
    float viewLinear = sqrt(rayStart.z * rayStart.z);

    //const float maxDistance = 0.022097;
    const float maxDistance = 0.25;

    //float result = 1.0;
    //if(rayStart.z + rayDirection.z * 128.0 < -near) result = 0.0;
    //return result;

    #if 1
if(ndotl > 0.05) {
    float rayLength = maxDistance;

    //rayStart += rayDirection * -rayStart.z / ndotl * 0.01;

#ifdef SS_Contact_Shadow_Tracing_Clip
    //clamp raylength to near clip plane
    rayLength = rayStart.z + rayDirection.z * maxDistance > -near ? (-near - rayStart.z) / rayDirection.z : rayLength;
#endif
    
    vec3 rayEnd = rayStart + rayDirection * rayLength;
         //rayEnd = rayEnd + normal / ndotl / 100.0 * rayLength;

    vec4    H0 = gbufferProjection * nvec4(rayStart);
    vec4    H1 = gbufferProjection * nvec4(rayEnd);

    float   k0 = 1.0 / H0.w;
    float   Q0 = rayStart.z * k0;
    vec2    P0 = H0.xy * k0;

    float   k1 = 1.0 / H1.w;
    float   Q1 = rayEnd.z * k1;
    vec2    P1 = H1.xy * k1;

    vec2 delta = (P1 - P0);
    //if(length(delta * resolution) < 1.0) delta = sign(delta) * step(vec2(0.5), abs(delta * resolution)) * texelSize;


    //float stepSize = 4.0 / max(abs(delta.x * resolution.x), abs(delta.y * resolution.y));
    float stepSize = invsteps * 50.0 / max(abs(delta.x * resolution.x), abs(delta.y * resolution.y));

    float thickness = maxDistance * 4.2 * stepSize;
    float sampleBias = stepSize * 0.001 / ndotl;
    
    float planeCheck = 2e-3;//0.001 / ndotl * stepSize;

    //float planeCheck = linear / 500.0 * (1.0 - ndotl);

	vec4 pqk    = vec4(P0, Q0, k0);
	vec4 dPQK   = vec4(delta, Q1 - Q0, k1 - k0) * stepSize;

    pqk += dPQK * mix(dither, 1.0, 0.0) - dPQK;

    bool intersect = false;
    ivec2 hitTexel = ivec2(0.0);

    float rayZmin = -Q0 / k0;
    
    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        //float rayLinear = sqrt(coord.z * coord.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);


        #ifdef DISTANT_HORIZONS 
        if(sampleDepth >= 1.0) {
            //sampleDepth = texelFetch(dhDepthTex0, texelPosition, 0).x;
            samplePosition = nvec3(dhProjectionInverse * nvec4(vec3(coord.xy, texelFetch(dhDepthTex0, texelPosition, 0).x) * 2.0 - 1.0));
        }
        #endif

        //samplePosition -= sampleVPBias;
    
        //vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);//length(samplePosition);

        float planeDistance = abs(dot(normal, samplePosition - rayStart)) / sampleLinear;
        if(planeDistance < planeCheck) continue;
        
    #if 0
        float rayZmax = sqrt(coord.z * coord.z);

        float test = rayZmax - sampleLinear;
        intersect = test > 0.0;

        if(intersect/* && test < thickness*/) {
            shading = planeDistance;//pow(sum3(texelFetch(colortex0, texelPosition, 0).rgb), 2.2);
        }
    #else
        float rayZmax = sqrt(coord.z * coord.z);

        intersect = rayZmax > sampleLinear && rayZmin - sampleLinear < thickness;

        rayZmin = rayZmax;
    #endif
    }

    shading = intersect ? 0.0 : shading;
}
    //return intersect ? 0.0 : 1.0;
    return shading;
#else
    float stepSize = 0.5;
    float thickness = invsteps * stepSize * 2.0;

    vec3 rayStep = rayDirection * stepSize * invsteps;
    vec3 rayPosition = rayStart + rayStep * dither + normal * 0.01;

    float planeCheck = sqrt(rayStart.z * rayStart.z) * 0.002;

    for(int i = 0; i < steps; i++) {
        vec3 coord = GetFragCoord(rayPosition);
        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5 || coord.z >= 1.0) break;

        float raylength = length(rayPosition);
        //if(raylength > viewLength) break;

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLength = length(samplePosition);

        float test = (raylength - sampleLength);
        float test2 = abs(dot(normal, samplePosition - rayStart));

        if(test2 >= planeCheck && test > 0.0 && test < thickness) {
            shading = 0.0;
            break;
        }

        rayPosition += rayStep;
    }

    return shading;
    #endif
}

vec3 CalculateShadow(in vec3 worldPosition, in vec3 worldNormal) {
    float fframeCounter = float(frameCounter);

    ivec2 iseed = ivec2(texcoord * resolution);

    float dither = GetBlueNoise(iseed);
    float dither2 = GetBlueNoise1(iseed);

    dither = R2dither(dither * 255.0, fframeCounter);
    dither2 = R2dither(dither2 * 255.0, fframeCounter);

    vec3 shading = vec3(1.0);

    float distortion = 1.0;

    vec3 shadowCoord = CalculateShadowCoord(worldPosition + worldNormal * 0.001);

    RemapShadowCoord(shadowCoord, distortion);

if(abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < 1.0) {
    shadowCoord = shadowCoord * 0.5 + 0.5;
    shadowCoord.z -= 0.00025 * Shadow_Depth_Mul;

#if Soft_Shadow == OFF
    //float shading = GetShading(shadowtex1, shadowCoord);

    shading = CalculateTintShadow(shadowCoord);

#else
    shading = vec3(0.0);

    float receiver = GetShadowLinearDepth(GetShadowDepth(shadowCoord.z));

    float blocker = 0.0;
    int blockerCount = 0;

    float blocker1 = 0.0;
    int blocker1Counter = 0;

    float radius = shadowTexelSize * distortion * 0.0625;
    float blurRadius = 32.0;

    for(int i = 0; i < 8; i++) {
        float r = pow(float(i + 1) / 8.0, 0.75);
        float a = (float(i) + dither) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r * radius * blurRadius * 0.5;

        float depth0 = GetShadowLinearDepth(GetShadowDepth(texture(shadowtex1, shadowCoord.xy + offset).x));

        if(depth0 < receiver) {
            blocker += depth0;
            blockerCount++;
        }

        float depth1 = GetShadowLinearDepth(GetShadowDepth(texture(shadowtex0, shadowCoord.xy + offset).x));

        if(depth1 < receiver) {
            blocker1 += depth1;
            blocker1Counter++;
        }
    }   

    float penumbra = 0.0;

    if(blocker1Counter > 0) blocker1 /= float(blocker1Counter);

    if(blockerCount > 0) {
        blocker /= float(blockerCount);
        penumbra = (receiver - blocker) / max(1e-5, blocker);
    }

    //if(blocker < blocker1) {
        float penumbra1 = (receiver - blocker1) / max(1e-5, blocker1);
        penumbra = max(penumbra1, penumbra);
        //if(penumbra1 > penumbra) return vec3(1.0, 0.0, 0.0);
        //return vec3(1.0, 0.0, 0.0);
    //}

    //penumbra = clamp(penumbra * 4.0 * blurRadius, 1.0, 32.0) * radius;
    penumbra = clamp(penumbra * 4.0, 1.0 / blurRadius, 1.0) * blurRadius * radius;

    //chashi xvlie
    for(int i = 0; i < 12; i++) {
        float r = pow(float(i + 1) / 12.0, 0.75);
        float a = (float(i) + dither) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r * penumbra;

        shading += CalculateTintShadow(shadowCoord + vec3(offset, -length(offset) * 0.5));
        //shading += GetShading(shadowtex1, shadowCoord + vec3(offset, -length(offset) * 0.5));
    }

    shading /= 12.0;
#endif
}
    return shading;
}
#if 0
//bool ScreenSpaceReflection(inout ivec2 hitTexel, in vec3 origin, in vec3 direction, in vec3 normal, inout float ssrRayLength, in float dither) {
bool ScreenSpaceReflection(inout ivec2 hitTexel, in vec3 origin, in vec3 direction, in vec3 normal, in float dither) {
    int steps = 20;
    float invsteps = 1.0 / float(steps);

    //mat3 mv = mat3(gbufferModelView);
    //mat3 imv = mat3(gbufferModelViewInverse);
    //vec3 motion = gbufferModelView[3].xyz;

    vec3 rayDirection = direction;
    vec3 rayStart = origin;

    //vec3 rayDirection = mv * tracedRay.direction;
    //vec3 rayDirection = normalize(reflect(normalize(tracedRay.origin), normal));
    //vec3 rayStart = mv * (tracedRay.origin - cameraPosition) + motion;

    float viewLinear = sqrt(rayStart.z * rayStart.z);

    float maxDistance = 1.0 / 10.0;
    float rayLength = rayStart.z + rayDirection.z * maxDistance > -near ? (-near - rayStart.z) / rayDirection.z : maxDistance;

    rayStart += normal * viewLinear * 0.004;

    vec3 rayEnd = rayStart + rayDirection * rayLength;

    vec3 viewDirection = normalize(rayStart);

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

    float thickness = exp2(0.5);

    float planeCheck = viewLinear * 0.01;

    float stepSize = 1.0;
          //stepSize *= min(1.0 / abs(delta.x), 1.0 / abs(delta.y));
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

    bool intersect = false;
    //ivec2 hitTexel = ivec2(0.0);

    float rayZmin = -Q0 / k0;

    for(int i = 0; i < steps && intersect == false; i++) {
        pqk += dPQK;
        currentLength += stepLength;

        vec3 coord = pqk.xyz;
             coord.xy = coord.xy * 0.5 + 0.5;
             coord.z = coord.z / pqk.w;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) break;

        float rayLinear = sqrt(coord.z * coord.z);

        ivec2 texelPosition = ivec2(coord.xy * resolution);
        float sampleDepth = texelFetch(depthtex0, texelPosition, 0).x;
        vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
        vec3 samplePosition = GetViewPosition(coord.xy, sampleDepth);
        float sampleLinear = sqrt(samplePosition.z * samplePosition.z);

        float planeDistance = abs(dot(normal, samplePosition - rayStart));
        if(planeDistance < planeCheck) continue;

        float rayZmax = rayLinear;

        intersect = rayZmax > sampleLinear && rayZmin < sampleLinear + thickness;

        //float test = (rayLinear - sampleLinear);
        //intersect = test > 0.0 && test / sampleLinear < thickness;
        //intersect = test > 0.0 && test / rayLinear < thickness;
        //intersect = test > 0.0;// && test < thickness;

        hitTexel = texelPosition;
        rayZmin = rayLinear;
    }

    //if(intersect) {
    //    ssrRayLength = currentLength / pqk.w - L0 / k0;
    //    //ssrRayLength = length(GetViewPosition(vec2(hitTexel) * texelSize, texelFetch(depthtex0, hitTexel, 0).x));
    //}

    return intersect;
}
#endif

//uniform mat4 shadowModelViewInverse;
#if 0
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

    float radius = 4.0 / 1000.0;

    int steps = 8;
    float invsteps = 1.0 / float(steps);


        //float r = pow(float(i + 1) * invsteps, 0.7);
        //float a = (float(i) + 0.5) * (sqrt(5.0) - 1.0) * Pi;
        //vec2 offset = vec2(cos(a), sin(a)) * r * radius;
    vec2 offsetSeed = (float2R2(dither * 1000.0 + 0.5) * 2.0 - 1.0);

    for(int i = 0; i < steps; i++) {
    #if 1
        //vec2 offset = offsetSeed * radius * float(i + 1);
        vec2 offset = float2R2(float(i) + dither) * 2.0 - 1.0;
             offset *= radius * 3.0;
    #else
        float r = pow(float(i + 1) * invsteps, 0.9);
        float a = (float(i) + dither) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r * radius * 4.0;
    #endif

        vec3 coord = RemapShadowCoord(shadowCoord + vec3(offset, 0.0));
             coord.z /= Shadow_Depth_Mul;
             coord = coord * 0.5 + 0.5;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) continue;

        float depth = GetShadowDepth(texture(shadowtex0, coord.xy).x) - 0.01 * 0.0;
        if(depth < coord.z) continue;

        vec3 shadowNormal = mat3(shadowModelView) * (texture(shadowcolor1, coord.xy).xyz * 2.0 - 1.0);
        vec3 albedo = LinearToGamma(texture(shadowcolor0, coord.xy).rgb);

        vec3 rayPosition = shadowView * (vec3((shadowCoord.xy * 0.5 + 0.5) + offset, depth) * 2.0 - 1.0);
        vec3 halfVector = rayPosition - (shadowView * shadowCoord);
        float vectorLength = length(halfVector);
        vec3 rayDirection = halfVector / vectorLength;

        float attenuation = 1.0 / max(1.0, pow2(vectorLength));
 
        float cosTheta = max(0.0, dot(shadowNormal, -rayDirection));
        float visibility = max(0.0, dot(shadowViewNormal, rayDirection));

        float weight = visibility * cosTheta;

        float d = max(0.0, dot(shadowViewLight, shadowNormal)) * invPi;

        diffuse += albedo * (d * attenuation * weight);
        totalWeight += weight;
    }

    if(totalWeight > 0.0) {
        diffuse /= totalWeight;
    }

    return diffuse;
}
#else
vec3 RSMSunIndirectLighting(in vec3 worldPosition, in vec3 worldNormal, in float dither) {
    vec3 shadowViewNormal = mat3(shadowModelView) * worldNormal;

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

    float radius = 1.0 / 1000.0;

    int steps = 3;
    float invsteps = 1.0 / float(steps);

    mat3 shadowView = mat3(shadowProjectionInverse);

        //float r = pow(float(i + 1) * invsteps, 0.7);
        //float a = (float(i) + 0.5) * (sqrt(5.0) - 1.0) * Pi;
        //vec2 offset = vec2(cos(a), sin(a)) * r * radius;
    vec2 offsetSeed = (float2R2(dither * 1000.0 + 0.5) * 2.0 - 1.0);

    for(int i = 0; i < steps; i++) {
    #if 1
        vec2 offset = offsetSeed * radius * float(i + 1);
    #else
        float r = pow(float(i + 1) * invsteps, 0.7);
        float a = (float(i) + 0.5) * (sqrt(5.0) - 1.0) * Pi;
        vec2 offset = vec2(cos(a), sin(a)) * r * radius * 3.0;
    #endif

        vec3 coord = RemapShadowCoord(shadowCoord + vec3(offset, 0.0));
             coord.z /= Shadow_Depth_Mul;
             coord = coord * 0.5 + 0.5;

        if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) continue;

        float depth = GetShadowDepth(texture(shadowtex0, coord.xy).x) - 0.01 * 0.0;

        vec3 blockerNormal = mat3(shadowModelView) * (texture(shadowcolor1, coord.xy).xyz * 2.0 - 1.0);

        vec3 halfVector = (shadowView * (vec3((shadowCoord.xy * 0.5 + 0.5) + offset, depth) * 2.0 - 1.0)) - (shadowView * shadowCoord);
             //halfVector.z = -halfVector.z;
        vec3 rayDirection = normalize(halfVector);

        float attenuation = 1.0 / max(1.0, pow2(length(halfVector)));

        float visibility = max(0.0, dot(shadowViewNormal, rayDirection)) * max(0.0, dot(blockerNormal, -rayDirection));
        float weight = step(coord.z, depth);

        diffuse += LinearToGamma(texture(shadowcolor0, coord.xy).rgb) * (weight * attenuation * visibility);
        totalWeight += weight;
    }

    if(totalWeight > 0.0) {
        diffuse /= totalWeight;
    }

    return diffuse;
}
#endif
float CalculateSkyLightmap(in float l) {
    float lightmap = saturate(rescale(l, 0.5 / 15.0, 14.0 / 15.0));
    //return mix(pow5(lightmap), saturate(rescale(lightmap, 0.0, 1.0)), 0.5);
    return lightmap * lightmap;
}

uniform int hideGUI;

//

#if Torch_Light_Temperature == 1000
    #define Torch_Light_Color vec3(1.0, 0.0401, 0.0)
#elif Torch_Light_Temperature == 1500
    #define Torch_Light_Color vec3(1.0, 0.1515, 0.0)
#elif Torch_Light_Temperature == 2000
    #define Torch_Light_Color vec3(1.0, 0.2484, 0.0061)
#elif Torch_Light_Temperature == 2500
    #define Torch_Light_Color vec3(1.0, 0.3577, 0.064)
#elif Torch_Light_Temperature == 3000
    #define Torch_Light_Color vec3(1.0, 0.4589, 0.1483)
#elif Torch_Light_Temperature == 3500
    #define Torch_Light_Color vec3(1.0, 0.5515, 0.2520)
#elif Torch_Light_Temperature == 4000
    #define Torch_Light_Color vec3(1.0, 0.6354, 0.3684)
#elif Torch_Light_Temperature == 4500
    #define Torch_Light_Color vec3(1.0, 0.7111, 0.4919)
#elif Torch_Light_Temperature == 5000
    #define Torch_Light_Color vec3(1.0, 0.7792, 0.6180)
#elif Torch_Light_Temperature == 5500
    #define Torch_Light_Color vec3(1.0, 0.8403, 0.7437)
#elif Torch_Light_Temperature == 6000
    #define Torch_Light_Color vec3(1.0, 0.8952, 0.8666)
#elif Torch_Light_Temperature == 6500
    #define Torch_Light_Color vec3(1.0, 0.9445, 0.9853)
#else
    #define Torch_Light_Color vec3(1.0)
#endif

struct LandAtmosphericStruct {
    float rdensity;
    vec3  rscattering;
    vec3  rextinction;

    float mdensity;
    vec3  mscattering;
    vec3  mextinction;

    float fdensity;
    vec3  fscattering;
};

LandAtmosphericStruct GetLandAtmosphericData(in vec3 position) {
    LandAtmosphericStruct a;

    float height = GetAltitudeClip(position, 1e-3);

    a.rdensity    = exp(-height / rayleigh_distribution);
    a.rscattering = rayleigh_scattering * a.rdensity;
    a.rextinction = a.rscattering + rayleigh_absorption * a.rdensity;

    a.mdensity    = exp(-height / mie_distribution);
    a.mscattering = mie_scattering * a.mdensity;
    a.mextinction = a.mscattering + mie_absorption * a.mdensity;

    a.fdensity = 0.0;
    a.fscattering = vec3(0.0);

    return a;
}

vec3 SunLimbdarkening(in float centerToEdge) {
    vec3 u = vec3(1.0);
    vec3 a = vec3(0.397, 0.503, 0.652);

    float mu = sqrt(1.0 - pow2(1.0 - centerToEdge));

    vec3 factor = 1.0 - u * (1.0 - pow(vec3(mu), a));

    return factor;
}

void main() {
    GbuffersData data = GetGbuffersData(texcoord);
    VectorStruct v = CalculateVectorStruct(texcoord, texture(depthtex0, texcoord).x);

    vec3 visibleNormal = CalculateVisibleNormals(data.texturedNormal, v.eyeDirection);

    vec3 worldNormal = mat3(gbufferModelViewInverse) * data.texturedNormal;
    vec3 worldGeometryNormal = mat3(gbufferModelViewInverse) * data.geometryNormal;

    GetDHVector(v, texcoord);
    
    /*
    if(v.depth >= 1.0) {
        v.depth = dh0.depth;
        v.viewPosition = dh0.viewPosition;
        v.viewDistance = dh0.viewDistance;
        v.worldPosition = mat3(gbufferModelViewInverse) * v.viewPosition + gbufferModelViewInverse[3].xyz;
    }*/

    //data.albedo = vec3(1.0);

    vec3 color = vec3(0.0);
    //color = data.albedo;

    float fframeCounter = float(frameCounter);

    vec2 frameCountOffset = float2R2(fframeCounter + 0.5) * 2.0 - 1.0;

    ivec2 iseed = ivec2(texcoord * resolution + frameCountOffset * 64.0);

    float dither = GetBlueNoise(iseed);
    float dither1 = GetBlueNoise1(iseed);
    float dither2 = GetBlueNoise2(iseed);

    //dither = R2dither(dither * 1000.0, fframeCounter);
    //dither1 = R2dither(dither1 * 1000.0, fframeCounter);
    //dither2 = R2dither(dither2 * 1000.0, fframeCounter);

    vec3 specular = vec3(0.0);

    vec2 envmapCoord = EncodeOctahedralmap(v.worldDirection) * 0.5 + 0.5;
         envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

    vec3 skyEnvmap = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;

    float theta = dot(v.viewDirection, sunVector);

    if(v.depth >= 1.0) {
        vec3 planetOrigin = vec3(0.0, planet_radius, 0.0);
        vec3 rayOrigin = vec3(0.0, Altitude_Start, 0.0);

        vec3 rayDirection = v.worldDirection;

        vec2 tracingAtmosphere = RaySphereIntersection(planetOrigin + rayOrigin, rayDirection, vec3(0.0), atmosphere_radius);
        vec2 tracingPlanet = RaySphereIntersection(planetOrigin + rayOrigin, rayDirection, vec3(0.0), planet_radius);

        float height = length(tracingAtmosphere.y * rayDirection * 0.25 + planetOrigin + rayOrigin) - planet_radius;

        vec3 asTransmittance = SimpleLightTransmittance(tracingAtmosphere.y, height, 0.5);

        float sunAngle = saturate(rescale(theta, 1.0 - 0.0005, 1.0));
        color = (SunLimbdarkening(sunAngle) * asTransmittance) * (step(1e-5, sunAngle) * step(tracingPlanet.x, 0.0) * step(tracingPlanet.y, 0.0) * 10.0);

        color += skyEnvmap;
    } else {
        vec3 shadowmap = CalculateShadow(v.worldPosition, worldGeometryNormal);

        //shadowmap = vec3(1.0);
        if(v.depth > 0.7) {
            //temp
            shadowmap *= ScreenSpaceContactShadow(v.viewPosition, lightVector, data.geometryNormal, dither);
        }

        float ndotl = max(0.0, dot(lightVector, data.geometryNormal));

        vec3 shading = BRDFLighting(v.eyeDirection, lightVector, data.albedo, data.texturedNormal, data.F0, data.roughness, data.metallic);

        //temp rain setting
        //float lightExt = 0.0005 * max(0.0, IntersectPlane(vec3(0.0, v.worldPosition.y + cameraPosition.y - 63.0, 0.0), worldSunVector, vec3(0.0, 4000.0, 0.0), vec3(0.0, -1.0, 0.0)));
        //shading *= saturate(HG(-0.9999, Fog_BackScattering_Phase) * 0.7 * Pi);
        shading *= mix(HG(-0.9999, Fog_BackScattering_Phase) * (1.0 - Fog_Front_Scattering_Weight), 1.0, saturate(exp(-Fog_Light_Extinction_Distance * Rain_Fog_Density - Biome_Fog_Density * Fog_Light_Extinction_Distance)));

        color = shading * LightingColor * shadowmap * SimpleGeometryTerm(ndotl);

/*
        mat3 tbn = tbnNormal(v.worldNormal);

        vec4 rayPDF = ImportanceSampleGGX(vec2(dither1, dither2), 0.9999);
        vec3 rayDirection = normalize(tbn * rayPDF.xyz);
             rayDirection = CalculateVisibleNormals(rayDirection, v.worldEyeDirection);

        vec2 envmapCoord = EncodeOctahedralmap(rayDirection) * 0.5 + 0.5;
             envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

        ivec2 texelPosition = ivec2(0);
        bool ssdoHit = ScreenSpaceReflection(texelPosition, v.viewPosition, mat3(gbufferModelView) * rayDirection, v.geometryNormal, 1.0);

        vec3 ambientLighting = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;

        if(ssdoHit) {
            ambientLighting *= LinearToGamma(texelFetch(colortex0, texelPosition, 0).rgb);
        }
*/
        //float skyLightmap = saturate(rescale(data.lightmap.y, 0.5 / 15.0, 1.0));
        //      skyLightmap = pow5(skyLightmap) * skyLightmap + skyLightmap * 0.3;

        vec3 SkyLightColor = CalculateSkyLighting(worldGeometryNormal);
        //     SkyLightColor = ambientLighting;

        ivec2 halfTexelPosition = ivec2(texcoord * resolution);
        vec3 ao = LinearToGamma(texelFetch(colortex3, halfTexelPosition, 0).rgb);
        SkyLightColor = ao;
        //color = vec3(0.0);

        vec3 rayDirection = normalize(reflect(v.viewDirection, visibleNormal));

        vec3 fr = SpecularLighting(rayDirection, v.eyeDirection, visibleNormal, data.F0, data.roughness, 1.0);
        vec3 dl = (1.0 - data.metallic) * (1.0 - fr);

        float skyLightmap = CalculateSkyLightmap(data.lightmap.y);

        //vec3 sunIndirect = vec3(0.0);//RSMSunIndirectLighting(v.worldPosition, v.worldNormal, dither) * LightingColor;
        color += data.albedo * SkyLightColor * dl * skyLightmap;

        //color = SkyLightColor * data.albedo;

        //temp rain setting
        //vec3 fogAmbient = LightingColor * data.albedo;
        //color += (fogAmbient - exp(-4000.0 * 0.0005) * fogAmbient) * (saturate(rescale(dot(upVector, v.texturedNormal), -1.15, 1.0)) * invPi * HG(-0.9999, -0.15) * 0.7) * skyLightmap;

        //color += LightingColor * (max(0.0, dot(sunVector, v.texturedNormal)) * invPi * HG(0.9999, 0.9) * 0.3 * exp(-lightExt));

        color += saturate(rescale(data.lightmap.x, 0.5 / 15.0, 2.0 / 15.0)) / max(1.0, pow2(15.0 - data.lightmap.x * 15.0)) * texture(colortex1, texcoord).a * Torch_Light_Color * data.albedo * invPi * 0.3 * dl;

        color += BRDFLighting(v.eyeDirection, v.eyeDirection, v.viewDistance * 2.0, data.albedo, data.texturedNormal, data.F0, data.roughness, data.metallic) * Torch_Light_Color * max(float(heldBlockLightValue), float(heldBlockLightValue2)) / 15.0;

        //color = data.albedo * LightingColor * invP;

        //color = data.albedo * texture(colortex1, texcoord).z * 0.05 * data.lightmap.x;

        //float fadeOut = 4.0;
        //float s = 1.0 - min(1.0, exp(-max(0.0, v.linearDepth - far + fadeOut) / fadeOut));

        //color = mix(color, skyEnvmap, vec3(s * (hideGUI == 0 ? 1.0 : 0.0)));
        //color = vec3(1.0 / 21.0);
        //vec3 
        //LandAtmosphericStruct a = GetLandAtmosphericData()

        //color = shadowmap / 10.0 * invPi * step(0.05, ndotl);

        vec2 envmapCoord = EncodeOctahedralmap(mat3(gbufferModelViewInverse) * rayDirection) * 0.5 + 0.5;
             envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

        vec3 skyEnvmap = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;

        specular = (skyEnvmap * fr) * saturate(rescale(data.lightmap.y, 0.5, 1.0));
#if 0
        float cameraHeight = max(1.0, v.worldPosition.y + cameraPosition.y - 63.0);//1.625;
        float rayleighDensity = exp(-cameraHeight / rayleigh_distribution);
        float mieDensity = exp(-cameraHeight / mie_distribution);

        float sunIndirect  = HG(dot(worldSunVector, worldUpVector), 0.76);
        float moonIndirect = HG(dot(worldMoonVector, worldUpVector), 0.76);
        float envIndirect  = HG(dot(worldLightVector, worldUpVector), 0.76);

        float cosTheta = dot(sunVector, v.viewDirection);
        float cosTheta2 = dot(moonVector, v.viewDirection);

        float backScattering = 0.0;//HG(0.999, 0.1);
        float frontScattering = 1.0 - min(backScattering, 1.0);
        
        vec3 mieLight = (backScattering + HG(cosTheta, 0.76) * frontScattering) * SunLightingColor + (backScattering + HG(cosTheta2, 0.76) * frontScattering) * MoonLightingColor;
        vec3 rayleighLight = RayleighPhase(cosTheta) * LightingColor;

        //mieLight *= max(envIndirect, visibility);
        //mieLight *= HG(0.7, 0.76);
        //mieLight *= backScattering;
        mieLight *= envIndirect;
        rayleighLight *= envIndirect;

        vec3 mieScat = mie_scattering * mieDensity;
        vec3 rayleighScat = rayleigh_scattering * rayleighDensity;
        vec3 mExt = mieScat + mie_absorption * mieDensity;
        vec3 rExt = rayleighScat + rayleigh_absorption * rayleighDensity;

        vec3 scattering = mieScat / mExt * mieLight + rayleighScat / rExt * rayleighLight;

        vec3 fogExt = vec3(0.0);
        vec3 s = 1.0 - min(vec3(1.0), exp(-v.viewDistance * 20.0 * (mExt + rExt + fogExt)));

        color = mix(color, scattering * 20.0, (s));
#endif
#if 0
        if(true) {
        int steps = 24;
        float invsteps = 1.0 / float(steps);

        float stepLength = v.viewDistance * invsteps;

        vec3 rayDirection = v.worldDirection;
        vec3 rayStep = rayDirection * stepLength;
        vec3 rayStart = rayStep * dither;

        float cosTheta = dot(sunVector, v.viewDirection);
        float cosTheta2 = dot(moonVector, v.viewDirection);
        float cosTheta3 = dot(worldLightVector, worldUpVector);

        float mieLightS = HG(0.999, 0.2);
        float miePhase = HG(cosTheta, 0.76);
        float miePhase2 = HG(cosTheta2, 0.76);
        float rayleighPhase = RayleighPhase(cosTheta);
        float envPhase  = HG(dot(worldLightVector, worldUpVector), 0.76);
        float envRayleigh = RayleighPhase(cosTheta3) * envPhase;

        float rays = 0.0;

        vec3 scattering = vec3(0.0);
        vec3 transmittance = vec3(1.0);

        for(int i = 0; i < steps; i++) {
            vec3 rayPosition = rayStart + rayStep * float(i);

            vec3 shadowCoord = CalculateShadowCoord(rayPosition);
            vec3 coord = RemapShadowCoord(shadowCoord);
                //coord.z /= Shadow_Depth_Mul;
                coord = coord * 0.5 + 0.5;

            float visibility = step(coord.z, texture(shadowtex1, coord.xy).x);
            //rays += visibility;

            //float currentLength = length(rayPosition);

            float height = GetAltitudeClip(rayPosition, 1e-3);

            float   rayleighDensity = exp(-height / rayleigh_distribution);
            vec3    rayleighScat    = rayleigh_scattering * rayleighDensity;
            vec3    rayleighExt     = rayleighScat + rayleigh_absorption * rayleighDensity;

            float   mieDensity      = exp(-height / mie_distribution);
            vec3    mieScat         = mie_scattering * mieDensity;
            vec3    mieExt          = mieScat + mie_absorption * mieDensity;

            vec3 rayleighLight = (visibility * rayleighPhase + envRayleigh) * LightingColor;

            vec3 mieLight = visibility * (miePhase * SunLightingColor + miePhase2 * MoonLightingColor);
                 mieLight += envPhase * envPhase * LightingColor;

            scattering += transmittance * (rayleighScat * rayleighLight + mieScat * mieLight) * stepLength * 2000.0 * 1.0;
            transmittance *= exp(-stepLength * (mieExt + rayleighExt) * 2000.0);
        }

        color *= transmittance;
        color += scattering;

    }
    #endif
        //

        //color = scattering;
        //color = vec3(visibility);
        //color = vec3(rays * invsteps);

        //Fog_Quality
        //Low:simple scattering
        //-Volmetric_Light
        //-off
        //-low noise light
        //-medium-ultra 8 step 8

        //if(isEyeInWater == 3) {
            //color = vec3(0.0);
        //}

        //color = vec3(shadowmap) * invPi * 0.1 * (hideGUI == 0 ? data.albedo : vec3(1.0));

        //color = saturate((v.worldNormal));

        //color = ao;

        //color = vec3(CalculateSkyLightmap(data.lightmap.y));

        //color = data.albedo;

        //color += pow5(1.0 - data.lightmap.x) * saturate(rescale(data.lightmap.x, 0.5 / 15.0, 1.0)) * data.albedo * pow(vec3(1.0, 0.782, 0.344), vec3(4.0)) * 0.05;

        //color = height;

        //color = shading;

        //color = vec3(1.0) * max(0.0, texture(colortex1, texcoord).z * 2.0 - 1.0) * 0.01;
    }

    //color = saturate(v.worldNormal) / 21.0;
    //color = vec3(v.linearDepth / 1000.0);

    color = GammaToLinear(color * MappingToSDR);

    //float eta = 1.333 / 1.0;
    //float angle = dot(normalize(reflect(v.viewDirection, v.texturedNormal)), -upVector) * 0.5 + 0.5;//pow2(dot(normalize(reflect(v.viewDirection, v.texturedNormal)), -upVector));//

    gl_FragData[0] = vec4(color, 1.0);
    gl_FragData[1] = vec4(specular, 1.0);
}
/* DRAWBUFFERS:35 */