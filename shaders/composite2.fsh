#version 130

uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex9;

uniform int hideGUI;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/dither.glsl"
#include "/libs/materialid.glsl"
#include "/libs/intersection.glsl"
#include "/libs/lighting/lighting_common.glsl"
#include "/libs/lighting/shadowmap.glsl"
#include "/libs/lighting/brdf.glsl"

#if 1
in vec3 SunLightingColor;
in vec3 MoonLightingColor;
in vec3 LightingColor;
in vec3 AmbientColor;
#endif

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

#if 0
float FresnelReflectAmount (float n1, float n2, vec3 normal, vec3 incident)
{
    #if DO_FRESNEL
        // Schlick aproximation
        float r0 = (n1-n2) / (n1+n2);
        r0 *= r0;
        float cosX = -dot(normal, incident);
        if (n1 > n2)
        {
            float n = n1/n2;
            float sinT2 = n*n*(1.0-cosX*cosX);
            // Total internal reflection
            if (sinT2 > 1.0)
                return 1.0;
            cosX = sqrt(1.0-sinT2);
        }
        float x = 1.0-cosX;
        float ret = r0+(1.0-r0)*x*x*x*x*x;

        // adjust reflect multiplier for object reflectivity
        ret = (OBJECT_REFLECTIVITY + (1.0-OBJECT_REFLECTIVITY) * ret);
        return ret;
    #else
    	return OBJECT_REFLECTIVITY;
    #endif
}
#endif

struct AtmosphereDataStruct {
    float r_density;
    vec3  r_scattering;
    vec3  r_extinction;

    float m_density;
    vec3  m_scattering;
    vec3  m_extinction;

    float f_density;
    vec3  f_scattering;

    vec3 extinction;
};
AtmosphereDataStruct GetAtmosphereData(in float height, in float density) {
    AtmosphereDataStruct a;

    a.r_density = exp(-height / rayleigh_distribution) * density;
    a.m_density = exp(-height / mie_distribution) * density;

    a.r_scattering = rayleigh_scattering * a.r_density;
    a.r_extinction = a.r_scattering + rayleigh_absorption * a.r_density;

    a.m_scattering = mie_scattering * a.m_density;
    a.m_extinction = a.m_scattering + mie_absorption * a.m_density;

    a.f_density    = 1.0;// + exp(-height / 8.0) * 10.0;
    a.f_scattering = vec3(Rain_Fog_Density * a.f_density + Biome_Fog_Density * a.f_density);

    a.extinction = a.m_extinction + a.r_extinction + a.f_scattering;

    return a;
}

void LandAtmosphericScattering(inout vec3 color, in VectorStruct v, in VectorStruct secV, in float skyLightmap, in float blocksLM, in float dither, in float TIR) {
    int steps = 20;
    float invsteps = 1.0 / float(steps);

    const float tracingQualityDropoff = 1.0;

    vec3 scattering = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);
    vec3 f = vec3(0.0);

    const float g = 0.76;

    vec3 rayDirection = v.worldDirection;

    float cosTheta = dot(worldSunVector, rayDirection);
    float cosTheta2 = dot(worldMoonVector, rayDirection);
    float ambPhaseAngle = -0.9999;

    //ambient color merge in lightcolor
    vec3 lightcolor0 = SunLightingColor + AmbientColor;
    vec3 lightcolor1 = MoonLightingColor + AmbientColor;
    vec3 lightcolor2 = LightingColor + AmbientColor;

    vec3 mLightDir = lightcolor0 * HG(cosTheta, g) + MoonLightingColor * HG(cosTheta2, g);
    vec3 mLightAmb = lightcolor2 * HG(ambPhaseAngle, g);

    vec3 rLightDir = lightcolor2 * RayleighPhase(cosTheta);
    vec3 rLightAmb = lightcolor2 * RayleighPhase(ambPhaseAngle);

    float dir = Fog_Front_Scattering_Weight;
    float amb = 1.0 - Fog_Front_Scattering_Weight;

    vec3 fLightDir = lightcolor0 * (HG(cosTheta, Fog_FrontScattering_Phase) * dir) + MoonLightingColor * (HG(cosTheta2, Fog_FrontScattering_Phase * Fog_Moon_Light_Phase_Multiplier) * dir);
    vec3 fLightAmb = lightcolor0 * (HG(cosTheta, Fog_BackScattering_Phase) * amb) + MoonLightingColor * (HG(cosTheta2, Fog_BackScattering_Phase * Fog_Moon_Light_Phase_Multiplier) * amb);

    float defaultFrontPhase = HG(0.9999, Fog_FrontScattering_Phase) * dir;
    float defaultBackPhase = HG(0.9999, Fog_BackScattering_Phase) * amb;

    float depth1 = texture(depthtex1, texcoord).x;
    float isSky = step(1.0, depth1);

    float skyTracingLength = 2000.0;

    float end = v.viewDistance;
    float start = 0.0;

    float InWater = isEyeInWater == 1 ? 1.0 : 0.0;
    float notInAir = isEyeInWater > 0 ? 1.0 : 0.0;
    end += max(0.0, secV.viewDistance - v.viewDistance) * InWater;
    start += v.viewDistance * InWater;

    vec3 blocksTransmittance = vec3(1.0);
    float transBlocks = 0.0;

    vec3 origin = vec3(0.0, max(0.0, cameraPosition.y - 63.0) + planet_radius, 0.0);
    vec2 tracingPlanet = RaySphereIntersection(origin, rayDirection, vec3(0.0), planet_radius);

    //planet surface tracing distance, intersect plane buz simplified height calculation
    float t = IntersectPlane(vec3(0.0, cameraPosition.y, 0.0), secV.worldDirection, vec3(0.0, 63.0, 0.0), vec3(0.0, 1.0, 0.0));
    //skyTracingLength = t > 0.0 ? min(t, skyTracingLength) : skyTracingLength;
    skyTracingLength = skyTracingLength + (t - skyTracingLength) * step(0.0, t) * isSky;
    end = (end + (skyTracingLength - end) * isSky);

    float skyOcclusionLevel = 15.0 - 15.0 * skyLightmap;
    float eyeSkyLight = float(eyeBrightness.y) / 240.0;

    float eyeBlkLevel = float(heldBlockLightValue);
    float eyeBlkLight = eyeBlkLevel / 15.0;

    float blocksLight = saturate(rescale(blocksLM, 0.5 / 15.0, 1.0)) * (1.0 - isSky);

    //end = min(end, 160.0);

if(end - start > 0.0) {
    float stepLength = (end - start) * invsteps;
    vec3 rayStep =  rayDirection * stepLength;
    vec3 rayStart = rayStep * dither + rayDirection * start;

    for(int i = 0; i < steps; i++) {
        vec3 rayPosition = rayStart + rayStep * float(i);

        float rayDistance = length(rayPosition);
        //if(rayDistance * (1.0 - isSky) > end) break;

        vec3 shadowCoord = CalculateShadowCoord(rayPosition);
             shadowCoord = RemapShadowCoord(shadowCoord);
        bool doShadowmap = abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < Shadow_Depth_Mul;

        shadowCoord = shadowCoord * 0.5 + 0.5;
        float visibility = doShadowmap ? step(shadowCoord.z, texture(shadowtex1, shadowCoord.xy).x) : 1.0;

        float bgDistance = end - rayDistance;

        //LandAtmosphericStruct a = GetLandAtmosphericData(max(0.0, rayPosition.y + cameraPosition.y - 63.0), 1.0);
        AtmosphereDataStruct a = GetAtmosphereData(max(0.0, rayPosition.y + cameraPosition.y - 63.0), 10.0);

        vec3 alpha = exp(-stepLength * a.extinction);

        vec3 clampedE = a.extinction + step(a.extinction, vec3(0.0));

        vec3 l = vec3(1.0);
        vec3 s = transmittance * (l - l * alpha) / clampedE;

        s *= mix(vec3(1.0), blocksTransmittance, vec3(step(v.viewDistance, rayDistance - 1e-5)));

        vec3 blocksFogLight = vec3(0.0);
        //blocksFogLight += (vec3(1.0, 0.782, 0.344) * exp(-bgDistance * a.extinction)) * (defaultBackPhase / pow2(max(1.0, bgDistance)) * step(bgDistance + 0.05, blocksLight));// * step(bgDistance + 0.05, blocksLight * 15.0)
        //blocksFogLight += (vec3(1.0, 0.782, 0.344) * transmittance) * (eyeBlkLight * saturate(eyeBlkLevel - rayDistance) / pow2(max(1.0, rayDistance)) * defaultBackPhase * step(stepLength, rayDistance));

        r += s * a.r_scattering * (visibility * rLightDir + rLightAmb);
        m += s * a.m_scattering * (visibility * mLightDir + mLightAmb);
        //f += s * a.f_scattering * (visibility * fLightDir + fLightAmb + blocksFogLight);

        transmittance *= alpha;
    }

    //r *= SunLightingColor;// * RayleighPhase(cosTheta);
    //m *= SunLightingColor;// * HG(cosTheta, 0.76);

    //if(minComponent(transmittance) <= 0.0) transmittance = vec3(1.0, 0.0, 0.0);

    //f *= vec3(0.9, 1.0, 1.0);
    //f *= 189.0 / vec3(189.0, 183.0, 161.0);

    //if(hideGUI == 0)
    vec3 rawcolor = color;

    color = color * transmittance + (r + m + f);
    //color = mix(color, rawcolor, vec3(saturate(TIR)));
}
    //color += saturate(eyeBlkLevel - rayEnd) / pow2(max(1.0, rayEnd)) * 0.01;
}

uniform int heldItemId;
uniform int heldItemId2;

const vec3[9] waterColorTest = vec3[9](
vec3(63.0, 118.0, 228.0) / 255.0,
vec3(76.0, 101.0, 89.0) / 255.0,
vec3(69.0, 173.0, 242.0) / 255.0,
vec3(67.0, 213.0, 238.0) / 255.0,
vec3(61.0, 87.0, 214.0) / 255.0,
vec3(57.0, 56.0, 201.0) / 255.0,
vec3(14.0, 78.0, 207.0) / 255.0,
vec3(58.0, 122.0, 106.0) / 255.0,
vec3(93.0, 183.0, 239.0) / 255.0
);

#ifdef DISTANT_HORIZONS 
uniform sampler2D dhDepthTex0;
uniform sampler2D dhDepthTex1;

//uniform float dhNearPlane;
//uniform float dhFarPlane;

//uniform int dhRenderDistance;

//uniform mat4 dhProjection;
uniform mat4 dhProjectionInverse;
//uniform mat4 dhPreviousProjection;

void GetDHVector(inout VectorStruct v, in vec2 coord, in float dhDepth) {
    if(v.depth >= 1.0) {
        v.depth = dhDepth;
        v.viewPosition = nvec3(dhProjectionInverse * vec4(ApplyTAAJitter(coord) * 2.0 - 1.0, dhDepth * 2.0 - 1.0, 1.0));
        v.worldPosition = mat3(gbufferModelViewInverse) * v.viewPosition;
        v.viewDistance = length(v.viewPosition);
        v.linearDepth = sqrt(v.viewPosition.z * v.viewPosition.z);
    }
}
#endif

struct EyeInMediaData {
    vec3 p0;
    vec3 p1;

    vec3 directLight;
    vec3 ambientLight;

    float sigma_s;
    float sigma_a;
    vec3 sigma_e;

    vec3 absorption;
    vec3 albedo;
};

void WaterScattering(inout vec3 scattering, inout vec3 transmittance, in EyeInMediaData m, in float dither) {
    int steps = 12;
    float invsteps = 1.0 / float(steps);

if(m.sigma_s + m.sigma_a > 0.0) {
    vec3 rayStep = (m.p1 - m.p0);

    float stepLength = min(100.0, length(rayStep)) * invsteps;
    rayStep = normalize(rayStep) * stepLength;

    vec3 rayOrigin = rayStep * dither + m.p0;

    vec3 preCalculateAlpha = exp(-m.sigma_e * stepLength);
    vec3 clampedE = m.sigma_e + step(m.sigma_e, vec3(0.0));

    //scattering = vec3(0.0);
    //transmittance = vec3(1.0);

    for(int i = 0; i < steps; i++) {
        vec3 rayPosition = rayOrigin + rayStep * float(i);

        vec3 shadowCoord = CalculateShadowCoord(rayPosition);
             shadowCoord = RemapShadowCoord(shadowCoord);
        bool doShadowmap = abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < Shadow_Depth_Mul;

        shadowCoord = shadowCoord * 0.5 + 0.5;
        float visibility = doShadowmap ? step(shadowCoord.z - defaultShadowBias, texture(shadowtex1, shadowCoord.xy).x) : 1.0;

        vec3 luminance0 = vec3(0.0);

        float lightDepth = max(0.0, GetShadowLinearDepth(shadowCoord.z) - GetShadowLinearDepth(texture(shadowtex0, shadowCoord.xy).x));
        float powderEffect = saturate(1.0 - exp(-1.602 * lightDepth));

        //water
        luminance0 += m.sigma_s * (m.directLight * (visibility * powderEffect) + m.ambientLight);

        scattering += (luminance0 - luminance0 * preCalculateAlpha) * transmittance / clampedE;
        transmittance *= preCalculateAlpha;
    }
}
}

void main() {
    GbuffersData data = GetGbuffersData(texcoord);
    VectorStruct v = CalculateVectorStruct(texcoord, texture(depthtex0, texcoord).x);
    VectorStruct v1 = CalculateVectorStruct(texcoord, texture(depthtex1, texcoord).x);

#ifdef DISTANT_HORIZONS 
    GetDHVector(v, texcoord, texture(dhDepthTex0, texcoord).x);
    GetDHVector(v1, texcoord, texture(dhDepthTex1, texcoord).x);
#endif

    float fIsSky = step(1.0, v.depth);

    float fframeCounter = float(frameCounter);

#ifdef Enabled_Temporal_AA
    vec2 frameCountOffset = float2R2(fframeCounter + 0.5) * 2.0 - 1.0;
#else
    vec2 frameCountOffset = vec2(0.0);
#endif

    ivec2 iseed = ivec2(texcoord * resolution + frameCountOffset * 64.0);
    float dither = GetBlueNoise(iseed);

    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    vec3 worldNormal = mat3(gbufferModelViewInverse) * data.texturedNormal;

    bool isSky = v.depth > 0.0;
    bool isWater = BoolMask(F_Water, data.tileID);
    bool isIce = BoolMask(F_Ice, data.tileID);
    bool isStainedGlass = BoolMask(F_Stained_Glass, data.tileID);
    bool isStainedGlassPane = BoolMask(F_Stained_Glass_Pane, F_Stained_Glass_Pane_End, data.tileID);
    bool isTransBlocks = BoolMask(F_Translucent, F_Translucent_End, data.tileID) || isStainedGlassPane;
    bool isMagicThing = BoolMask(F_Nether_Portal_Start, F_Nether_Portal_End, data.tileID);
    
    //IOR=(F(0)-1+2*sqrt[F(0)])/(1-F(0)).
    float n1 = 1.0;
    float n2 = isTransBlocks ? 1.0 / ((2.0 / (sqrt(data.metalness) + 1.0)) - 1.0) : 1000.0;

    if(isWater) {
        n2 = 1.333;
    }

    if(isEyeInWater == 1) {
        n1 = 1.333;
        n2 = isWater ? 1.0 : n2;
    }

    float weatherLight = mix(HG(-0.9999, Fog_BackScattering_Phase) * (1.0 - Fog_Front_Scattering_Weight), 1.0, saturate(exp(-Fog_Light_Extinction_Distance * Rain_Fog_Density - Biome_Fog_Density * Fog_Light_Extinction_Distance)));

    float ndotlight = dot(data.texturedNormal, lightVector);
    float vdotlight = dot(v.viewDirection, lightVector);
    float vdotsun = dot(sunVector, v.viewDirection);
    float vdotmoon = dot(moonVector, v.viewDirection);
    float vdotu = dot(upVector, v.viewDirection);
    float udotlight = worldLightVector.y;

    float eyeSkylight = saturate(rescale(eyeBrightness.y / 240.0, 0.5 / 15.0, 1.0));

    vec3 mergeLight0 = SunLightingColor + AmbientColor;
    vec3 mergeLight1 = MoonLightingColor + AmbientColor;

    float dir = Fog_Front_Scattering_Weight;
    float amb = 1.0 - Fog_Front_Scattering_Weight;

    float asG = 0.76;

    vec3 mieDir = mergeLight0 * HG(vdotsun, asG) + mergeLight1 * HG(vdotmoon, asG);
    vec3 mieAmb = (LightingColor + AmbientColor) * HG(udotlight, asG);

    vec3 rayleighDir = mergeLight0 * RayleighPhase(vdotsun) + mergeLight1 * RayleighPhase(vdotmoon);
    vec3 rayleighAmb = (LightingColor + AmbientColor) * RayleighPhase(udotlight);

    vec3 fogDir = mergeLight0 * (HG(vdotsun, Fog_FrontScattering_Phase) * dir) + mergeLight1 * (HG(vdotmoon, Fog_FrontScattering_Phase * Fog_Moon_Light_Phase_Multiplier) * dir);
    vec3 fogAmb = mergeLight0 * (HG(0.999, Fog_BackScattering_Phase) * amb) + mergeLight1 * (HG(0.999, Fog_BackScattering_Phase * Fog_Moon_Light_Phase_Multiplier) * amb);

    //vec3 fLightDir = lightcolor0 * (HG(cosTheta, Fog_FrontScattering_Phase) * dir) + MoonLightingColor * (HG(cosTheta2, Fog_FrontScattering_Phase * Fog_Moon_Light_Phase_Multiplier) * dir);
    //vec3 fLightAmb = lightcolor0 * (HG(cosTheta, Fog_BackScattering_Phase) * amb) + MoonLightingColor * (HG(cosTheta2, Fog_BackScattering_Phase * Fog_Moon_Light_Phase_Multiplier) * amb);

    vec3 mediaAlbedo = LinearToGamma(waterColorTest[0]);
    //vec3 mediaDir = (HG(vdotsun, 0.66) * 0.1 + HG(vdotsun, -0.1) * 0.05) * (SunLightingColor * mediaAlbedo);
    vec3 mediaDir = ((HG(vdotsun, -0.1) * 0.5 + HG(vdotsun, 0.66)) / 1.5 * invPi) * (SunLightingColor * mediaAlbedo);
    vec3 mediaAmb = (HG(vdotu, 0.44) * invPi * eyeSkylight) * (AmbientColor * mediaAlbedo);

    vec3 albedo = LinearToGamma(waterColorTest[0]);
    vec3 absorption = mix(LinearToGamma(waterColorTest[0]), LinearToGamma(waterColorTest[1]), vec3(0.7));

    float s = true || isWater ? 0.04005 : 0.0;
    float a = 0.04005;
        
    vec3 e = s + a * -log(absorption);

    //float cosTheta = dot(v.viewDirection, data.texturedNormal);
    //float cosTheta = dot(normalize(reflect(v.viewDirection, data.texturedNormal)), -upVector);
    //float n = n1 / n2;

    //if(n*n*(1.0 - cosTheta * cosTheta) > 1.0) color = vec3(0.0);

    if(heldItemId2 == 1) color = vec3(0.0);
#if 0
    if(isWater && isEyeInWater == 0) {
        int steps = 12;
        float invsteps = 1.0 / float(steps);

        vec3 p0 = v.worldPosition;
        vec3 p1 = v1.worldPosition;

        float stepLength = length(p1 - p0) * invsteps;

        vec3 rayDirection = v1.worldDirection;
        vec3 rayStep = rayDirection * stepLength;
        vec3 rayOrigin = p0 + rayStep * dither;

        vec3 scattering = vec3(0.0);
        vec3 transmittance = vec3(1.0);

        vec3 preCalculateAlpha = exp(-e * stepLength);

        for(int i = 0; i < steps; i++) {
            vec3 rayPosition = rayOrigin + rayStep * float(i);

            vec3 shadowCoord = CalculateShadowCoord(rayPosition);
                 shadowCoord = RemapShadowCoord(shadowCoord);
            bool doShadowmap = abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < Shadow_Depth_Mul;

            shadowCoord = shadowCoord * 0.5 + 0.5;
            float visibility = doShadowmap ? step(shadowCoord.z - defaultShadowBias, texture(shadowtex1, shadowCoord.xy).x) : 1.0;

            vec3 luminance0 = vec3(0.0);
            vec3 clampedE = vec3(0.0);
            vec3 stepAlpha = vec3(1.0);

            float lightDepth = max(0.0, GetShadowLinearDepth(shadowCoord.z) - GetShadowLinearDepth(texture(shadowtex0, shadowCoord.xy).x));
            float powderEffect = saturate(1.0 - exp(-1.602 * lightDepth));

            //water
            clampedE = e;
            stepAlpha = preCalculateAlpha;
            luminance0 += s * (mediaDir * (visibility) + mediaAmb);
            clampedE += step(clampedE, vec3(0.0));

            scattering += (luminance0 - luminance0 * stepAlpha) * transmittance / clampedE;
            transmittance *= stepAlpha;
        }

        color = color * transmittance + scattering;
    }
#endif

#define Do_Fog_Twice

    EyeInMediaData media;

    media.p0 = isEyeInWater == 0 ? v.worldPosition : vec3(0.0);
    media.p1 = isEyeInWater == 0 ? v1.worldPosition : v.worldPosition;

    media.albedo = LinearToGamma(waterColorTest[0]);
    media.albedo = media.albedo / max(1e-5, maxComponent(media.albedo));
    media.absorption = mix(LinearToGamma(waterColorTest[0]), LinearToGamma(waterColorTest[1]), vec3(0.7));
    //media.absorption = media.absorption / maxComponent(media.absorption) * 0.95;
    media.directLight   = (mix(HG(vdotsun, -0.1), HG(vdotsun, 0.533), 0.3) * invPi * 0.7) * (SunLightingColor * media.albedo);
    media.directLight  += (HG(vdotsun, 0.8) * 0.02 * 0.3) * SunLightingColor;
    media.ambientLight  = (HG(vdotu, 0.44) * invPi * eyeSkylight) * (AmbientColor * media.albedo);
    media.sigma_s = isWater || isEyeInWater == 1 ? 0.2 : 0.0;
    media.sigma_a = isWater || isEyeInWater == 1 ? 0.1 : 0.0;
    media.sigma_e = media.sigma_s + media.sigma_a * -log(media.absorption);

#ifdef Do_Fog_Twice
//test0
//under water 84 83 82;126
//above water 85 84 84;121
if(isEyeInWater == 0){
    vec3 blocksTransmittance = vec3(1.0);
    vec3 blocksScattering = vec3(0.0);

    WaterScattering(blocksScattering, blocksTransmittance, media, dither);
    color = color * blocksTransmittance + blocksScattering;
}
//test1
//under water 89 87 86
//above water 95 94 94
#else
//test1
//under water 88 87 87
//above water 94 92 92

//test0
//under water 85 84 83;122-123
//above water 86 84 83;120-121
    vec3 blocksTransmittance = vec3(1.0);
    vec3 blocksScattering = vec3(0.0);

    WaterScattering(blocksScattering, blocksTransmittance, media, dither);

    if(isEyeInWater == 0) color = color * blocksTransmittance + blocksScattering;
#endif

    if(isWater || isTransBlocks || isMagicThing) {
        //temp. specular
        vec3 specular = texture(colortex5, texcoord).rgb;
        //color += specular;

        //translucent blocks sub-surface
        vec3 absorption = (data.albedo + 0.001) / (1.0 + 0.001);
             absorption = isTransBlocks || isStainedGlassPane ? data.albedo * 0.7 + 0.02 : absorption;//0.02

        float sigma_a = 0.0;
              sigma_a = BoolMask(F_Ice, data.tileID) ? 1.0 : sigma_a;
              sigma_a = isTransBlocks || isStainedGlassPane ? 1.4 : sigma_a;
              sigma_a = isMagicThing ? 0.2 : sigma_a;

        color *= exp(log(absorption) * sigma_a * data.alpha);

        //translucent blocks surface
        vec3 shadowCoord = CalculateShadowCoord(v.worldPosition + worldNormal * 0.001);
             shadowCoord = RemapShadowCoord(shadowCoord);
        bool doShadowmap = abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < Shadow_Depth_Mul;

        shadowCoord = shadowCoord * 0.5 + 0.5;
        float visibility = doShadowmap ? step(shadowCoord.z - defaultShadowBias, texture(shadowtex1, shadowCoord.xy).x) : 1.0;

        vec3 shadowmap = vec3(visibility);
        vec3 shading = SpecularLighting(worldLightVector, v.worldEyeDirection, worldNormal, data.F0, data.roughness, 50.0) * LightingColor;

        float sss = 0.9;    //solid <====> translucent , 0.0 to 1.0,aka data.material

        float sigma_s = 0.0;
              sigma_s = isIce ? 0.0 : sigma_s;
              sigma_s = isTransBlocks || isStainedGlassPane ? 0.33 : sigma_s;
        vec3 scattering = vec3(0.0);

        float solidShading = invPi * saturate(ndotlight);
        float translucentShading = (HG(vdotlight, 0.999) + HG(vdotlight, 0.3) * 0.05) / 1.05;

        scattering = (data.albedo * LightingColor * shadowmap) * mix(solidShading, translucentShading, sss);
        
        color = mix(color, scattering, vec3(saturate(sigma_s * data.alpha)));
        color += data.emissive * data.albedo / exp2(4.0 + Emissive_Light_Exposure);
        //color += LightingColor * (CalculateMask(F_Water, data.tileID) * HG(min(1.0, dot(lightVector, v.viewDirection) / (1.0 - 0.0005)), 0.999) / HG(0.9999, 0.999) * 0.1 * 0.5 * saturate(1.0 - dot(data.texturedNormal, lightVector)));

        vec3 reflectDirection = normalize(reflect(v.worldDirection, worldNormal));
        vec3 fr = SpecularLighting(reflectDirection, v.worldEyeDirection, worldNormal, data.F0, data.roughness, 1.0);
        color *= 1.0 - fr;

        color += shadowmap * shading * (isMagicThing ? 0.0 : 1.0);

        //temp. fake reflection
        vec2 envmapCoord = EncodeOctahedralmap(reflectDirection) * 0.5 + 0.5;
             envmapCoord = clamp(envmapCoord * 0.25, texelSize, vec2(0.25) - texelSize);

        vec3 skyEnvmap = LinearToGamma(texture(colortex4, envmapCoord).rgb) * MappingToHDR;
             //skyEnvmap += SunLightingColor * saturate(rescale(dot(worldSunVector, reflectDirection), 1.0 - 0.0005, 1.0)) * 100.0;

        //color += skyEnvmap * fr * (isEyeInWater == 0 ? 1.0 : 0.0);
    }

#if 1
        vec3 origin = vec3(0.0, max(0.0, cameraPosition.y - 63.0) + planet_radius + Altitude_Start, 0.0);
        vec2 tracingPlanet = RaySphereIntersection(origin, v1.worldDirection, vec3(0.0), planet_radius);

        int steps = 16;
        float invsteps = 1.0 / float(steps);

        vec3 p0 = vec3(0.0);
        vec3 p1 = v.worldPosition;

        p0 = isEyeInWater == 1 ? v.worldPosition : p0;
        p1 = isEyeInWater == 1 ? v1.worldPosition : p1;

        float totalLength = length(p1 - p0); 
        
        float t = IntersectPlane(vec3(0.0, cameraPosition.y, 0.0), v.worldDirection, vec3(0.0, 63.0, 0.0), vec3(0.0, 1.0, 0.0));
              t = tracingPlanet.x;
        //skyTracingLength = t > 0.0 ? min(t, skyTracingLength) : skyTracingLength;
        float skyTracingLength = 3000.0;
        //      skyTracingLength = skyTracingLength + (t - skyTracingLength) * step(0.0, t);
        //totalLength = (totalLength + max(0.0, skyTracingLength - totalLength) * fIsSky);
        //totalLength = isEyeInWater > 0 ? totalLength - max(0.0, totalLength - High_Density_Fog_Distance_Limit) : totalLength;

        if(v.depth >= 1.0) {
            //totalLength = skyTracingLength;
            totalLength = t > 0.0 ? t : skyTracingLength;//totalLength + max(t - totalLength, 0.0);
        }

        float stepLength = min(1000.0, totalLength) * invsteps;
        
        vec3 preCalculateAlpha = exp(-stepLength * e);

        vec3 rayStep = v1.worldDirection * (totalLength * invsteps);
        vec3 rayOrigin = p0 + rayStep * dither;

        vec3 transmittance = vec3(1.0);
        vec3 scattering = vec3(0.0);

        for(int i = 0; i < steps; i++) {
            vec3 rayPosition = rayOrigin + rayStep * float(i);
            float height = max(1.0, rayPosition.y + cameraPosition.y - 63.0);

            vec3 shadowCoord = CalculateShadowCoord(rayPosition);
                 shadowCoord = RemapShadowCoord(shadowCoord);
            bool doShadowmap = abs(shadowCoord.x) < 1.0 && abs(shadowCoord.y) < 1.0 && shadowCoord.z < Shadow_Depth_Mul;

            shadowCoord = shadowCoord * 0.5 + 0.5;
            float visibility = doShadowmap ? step(shadowCoord.z - defaultShadowBias, texture(shadowtex1, shadowCoord.xy).x) : 1.0;

            vec3 luminance0 = vec3(0.0);
            vec3 clampedE = vec3(0.0);
            vec3 stepAlpha = vec3(1.0);

            //if(isEyeInWater == 0) {
                float density = saturate((1.0 - height / 256.0) / 0.21875) * 2.47 + saturate(1.0 - height / 60.0) * 3.52;

                //#ifndef DISTANT_HORIZONS
                
                //#endif

                AtmosphereDataStruct a = GetAtmosphereData(height, density);

                clampedE = a.extinction;
                stepAlpha = exp(-a.extinction * stepLength);
                luminance0 += a.r_scattering * (rayleighDir * visibility + rayleighAmb);
                luminance0 += a.m_scattering * (mieDir * visibility + mieAmb);
                luminance0 += a.f_scattering * (fogDir * visibility + fogAmb);
                /*
            } else {
                float lightDepth = max(0.0, GetShadowLinearDepth(shadowCoord.z) - GetShadowLinearDepth(texture(shadowtex0, shadowCoord.xy).x));
                float powderEffect = saturate(1.0 - exp(-1.602 * lightDepth));

                clampedE = e;
                stepAlpha = preCalculateAlpha;
                luminance0 += s * (mediaDir * (visibility * powderEffect) + mediaAmb);
            }
            */

            clampedE += step(clampedE, vec3(0.0));

            scattering += (luminance0 - luminance0 * stepAlpha) * transmittance / clampedE;

                /*
            } else if(isEyeInWater == 1) {
            float lightDepth = max(0.0, GetShadowLinearDepth(shadowCoord.z) - GetShadowLinearDepth(texture(shadowtex0, shadowCoord.xy).x));
            vec3 powderEffect = saturate(1.0 - exp(-e * 2.0 * lightDepth * 5.0));
            luminance0 = light0 * (visibility * powderEffect * s) + ambient * s;
            } 
            scattering += (luminance0 - luminance0 * stepAlpha) * transmittance / e;
            */

            transmittance *= stepAlpha;
        }

        //if(hideGUI == 0) color = color * transmittance + scattering;
        color = color * transmittance + scattering;

        //color = t > 0.0 ? vec3(saturate(t / 100.0)) : vec3(1.0, 0.3, 0.0) / 100.0;
#endif

    //surface
    float eta = n1 / n2;
    float TIR = n2 < n1 ? 1.0 - eta * eta * (1.0 - pow2(dot(v.eyeDirection, data.texturedNormal))) : 1.0;
    //color *= saturate(TIR);
/*
#ifdef Do_Fog_Twice
if(isEyeInWater == 1){
    vec3 blocksTransmittance = vec3(1.0);
    vec3 blocksScattering = vec3(0.0);

    WaterScattering(blocksScattering, blocksTransmittance, media, dither);
    color = color * blocksTransmittance + blocksScattering;
}
#else
    if(isEyeInWater == 1) color = color * blocksTransmittance + blocksScattering;
#endif
*/
    //LandAtmosphericScattering(color, v, v1, 1.0, 0.0, dither, 1.0);

    //vec3 p = mat3(shadowProjection) * mat3(shadowModelView) * v.worldPosition;

    //color = vec3(saturate(length(p) / 100.0));

    color = GammaToLinear(color * MappingToSDR);

    gl_FragData[0] = vec4(color, 1.0);
}
/* DRAWBUFFERS:3 */