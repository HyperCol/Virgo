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

    vec3 blocksTransmittance;
    float blocksDistance;
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
        luminance0 += m.sigma_s * (m.directLight * (visibility) + m.ambientLight);
        luminance0 = m.blocksDistance - length(rayPosition) < 1e-5 ? luminance0 * m.blocksTransmittance : luminance0;

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

    if(v.depth >= 1.0) {
        //vec4 clouds = texture(colortex9, ApplyTAAJitter(texcoord));

        //color = color * clouds.a + clouds.rgb;
    }

    vec3 worldNormal = mat3(gbufferModelViewInverse) * data.texturedNormal;
    vec3 visibleNormal = CalculateVisibleNormals(worldNormal, v.worldEyeDirection);
    vec3 viewVisibleNormal = mat3(gbufferModelView) * visibleNormal;

    bool isWater = BoolMask(F_Water, data.tileID);
    bool isIce = BoolMask(F_Ice, data.tileID);
    bool isStainedGlass = BoolMask(F_Stained_Glass, data.tileID);
    bool isStainedGlassPane = BoolMask(F_Stained_Glass_Pane, F_Stained_Glass_Pane_End, data.tileID);
    bool isTransBlocks = BoolMask(F_Translucent, F_Translucent_End, data.tileID) || isStainedGlassPane;
    bool isMagicThing = BoolMask(F_Nether_Portal_Start, F_Nether_Portal_End, data.tileID);
    
    float ndotlight = dot(data.texturedNormal, lightVector);
    float vdotlight = dot(v.viewDirection, lightVector);
    float vdotsun = dot(sunVector, v.viewDirection);
    float vdotmoon = dot(moonVector, v.viewDirection);
    float vdotu = dot(upVector, v.viewDirection);
    float udotlight = worldLightVector.y;

    float eyeSkylight = saturate(rescale(eyeBrightness.y / 240.0, 0.5 / 15.0, 1.0));

    //if(heldItemId2 == 1) color = vec3(0.0);

    EyeInMediaData media;

    media.p0 = vec3(0.0);
    media.p1 = v.worldPosition + (v1.worldPosition - v.worldPosition) * (CalculateMask(F_Translucent, F_Translucent_End, data.tileID) + CalculateMask(F_Stained_Glass_Pane, F_Stained_Glass_Pane_End, data.tileID));

#if 0
    media.albedo = LinearToGamma(waterColorTest[0]);
    media.albedo = media.albedo / max(1e-5, maxComponent(media.albedo));
    media.absorption = mix(LinearToGamma(waterColorTest[0]), LinearToGamma(waterColorTest[1]), vec3(0.7));
    //media.absorption = media.absorption / maxComponent(media.absorption) * 0.5;
    media.directLight = (mix(HG(vdotsun, -0.1), HG(vdotsun, 0.66), 0.3) * invPi * 0.5) * (SunLightingColor * media.albedo);
    media.ambientLight = (HG(vdotu, 0.44) * invPi * eyeSkylight) * (AmbientColor * media.albedo);
    media.sigma_s = isWater || isEyeInWater == 1 ? 0.08 : 0.0;
    media.sigma_a = isWater || isEyeInWater == 1 ? 0.12 : 0.0;
    media.sigma_e = media.sigma_s + media.sigma_a * -log(media.absorption);
#else
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
#endif

    media.blocksTransmittance = vec3(1.0);
    media.blocksDistance = v.viewDistance;

    if(isTransBlocks || isMagicThing) {
        vec3 absorption = (data.albedo + 0.001) / (1.0 + 0.001);
             absorption = isStainedGlass || isStainedGlassPane ? absorption * 0.7 + 0.02 : absorption;

        float sigma_a = 1.4;
              sigma_a = BoolMask(F_Ice, data.tileID) ? 1.0 : sigma_a;
              sigma_a = isStainedGlassPane ? 1.4 : sigma_a;
              sigma_a = isMagicThing ? 0.2 : sigma_a;

        media.blocksTransmittance = exp(log(absorption) * sigma_a * data.alpha);
    }

if(isEyeInWater == 1){
    vec3 blocksTransmittance = vec3(1.0);
    vec3 blocksScattering = vec3(0.0);

    WaterScattering(blocksScattering, blocksTransmittance, media, dither);
    color = color * blocksTransmittance + blocksScattering;
}

    color = GammaToLinear(color * MappingToSDR);

    vec2 encodeNormal = EncodeSpheremap(viewVisibleNormal);

    gl_FragData[0] = vec4(encodeNormal, encodeNormal);
    gl_FragData[1] = vec4(color, 1.0);
}
/* DRAWBUFFERS:23 */