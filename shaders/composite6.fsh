#version 130

#define Reflection_Render_Scale 0.5

uniform sampler2D colortex3;
uniform sampler2D colortex4;
//uniform sampler2D colortex5;
uniform sampler2D colortex6;
uniform sampler2D colortex9;
uniform sampler2D colortex12;

uniform int hideGUI;
uniform int heldItemId;
uniform int heldItemId2;

in vec2 texcoord;

in mat4 previousModelViewInverse;
in mat4 previousProjectionInverse;

/*
const bool colortex9Clear = false;
const bool colortex12Clear = false;
*/
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

#define Clip_In_YCoCg

vec3 RGBToYCoCg(vec3 c) {
	// Y = R/4 + G/2 + B/4
	// Co = R/2 - B/2
	// Cg = -R/4 + G/2 - B/4
#ifdef Clip_In_YCoCg
    return vec3(c.x/4.0 + c.y/2.0 + c.z/4.0,
                c.x/2.0 - c.z/2.0,
                -c.x/4.0 + c.y/2.0 - c.z/4.0);
#else
    return c;
#endif
}

vec3 YCoCgToRGB(vec3 c) {
	// R = Y + Co - Cg
	// G = Y + Cg
	// B = Y - Co - Cg
#ifdef Clip_In_YCoCg
    return vec3(c.x + c.y - c.z,
                c.x + c.z,
	            c.x - c.y - c.z);
#else
    return c;
#endif
}

vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    vec3 p_clip = 0.5 * (maximum + minimum);
    vec3 e_clip = 0.5 * (maximum - minimum);

    vec3 v_clip = color - p_clip;
    vec3 v_unit = v_clip.xyz / e_clip;
    vec3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0) return p_clip + v_clip / ma_unit;
    
    return color;// point inside aabb
}

void ClipAABB(inout vec3 accumulation, in sampler2D tex, in vec2 coord, in float linearDepth, in float sigma) {
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    float totalWeight = 0.0;

    //vec2 alphaClip = vec2(1.0, -1.0);
    //vec3 minColor = vec3(1.0);
    //vec3 maxColor = vec3(0.0);

    ivec2 texelPosition = ivec2(coord * resolution);

    const float radius = 1.0;

    for(float i = -radius; i <= radius; i += 1.0) {
        for(float j = -radius; j <= radius; j += 1.0) {
            //vec4 sampleColor = texelFetch(tex, texelPosition + ivec2(i, j), 0);

            vec2 samplePosition = coord + vec2(i, j) * texelSize;
            vec3 sampleColor = texture(tex, samplePosition).rgb;
            float sampleLinear = (texture(colortex6, samplePosition).x);

            //vec3 color = LinearToGamma(sampleColor.rgb) * MappingToHDR;
            //sampleColor.rgb = LinearToGamma(sampleColor.rgb) * MappingToHDR;
            sampleColor.rgb = RGBToYCoCg(sampleColor.rgb);
            
            //if(all(equal(vec2(i, j), vec2(0.0))) || abs(sampleLinear - linearDepth) / sampleLinear < 20.0) {
                m1 += sampleColor;
                m2 += sampleColor * sampleColor;
                totalWeight += 1.0;
            //}
            /*
            minColor = min(minColor, color);
            maxColor = max(maxColor, color);

            alphaClip = vec2(min(alphaClip.x, sampleColor.a), max(alphaClip.y, sampleColor.a));
            */
        }
    }

    m1 /= totalWeight;
    m2 /= totalWeight;

    vec3 variance = sqrt(max(vec3(0.0), m2 - m1 * m1));

    //float sigma = 3.0;

    vec3 minColor = m1 - variance * sigma;
    vec3 maxColor = m1 + variance * sigma;
    
    accumulation.rgb = RGBToYCoCg(accumulation.rgb);
    accumulation.rgb = clipToAABB(accumulation.rgb, minColor.rgb, maxColor.rgb);
    accumulation.rgb = YCoCgToRGB(accumulation.rgb);
}

void main() {
    vec3 closest = GetClosest(colortex6, texcoord, 1.0);
    vec2 velocity = GetVelocity(closest);
    vec2 previousCoord = texcoord - velocity;

    vec3 currentColor = texture(colortex4, texcoord).rgb;

    vec3 previousColor = texture(colortex9, previousCoord).rgb;
    ClipAABB(previousColor, colortex4, texcoord, 1.0, 8.0);

    float historyLength = texture(colortex9, previousCoord).a * 255.0;
          historyLength = historyLength + 1.0;
          historyLength = min(255.0, historyLength);

    historyLength *= step(abs(previousCoord.x - 0.5), 0.5 - texelSize.x) * step(abs(previousCoord.y - 0.5), 0.5 - texelSize.y);

    float previousDepth = texture(colortex12, previousCoord).a;

    float depth = texture(depthtex0, texcoord).x;
    vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(vec3(ApplyTAAJitter(texcoord), depth) * 2.0 - 1.0));
    vec3 worldPosition = mat3(gbufferModelViewInverse) * viewPosition + cameraPosition + gbufferModelViewInverse[3].xyz;
    vec3 prevWorldPosition = worldPosition - previousCameraPosition;

    vec3 prevViewPos = nvec3(previousProjectionInverse * nvec4(vec3(ApplyTAAJitter(previousCoord), previousDepth) * 2.0 - 1.0));
    vec3 prevSampleWorldPosition = mat3(previousModelViewInverse) * prevViewPos + previousModelViewInverse[3].xyz;

    vec3 worldNormal = mat3(gbufferModelViewInverse) * DecodeSpheremap(texture(colortex2, texcoord).xy);

    float planeDistance = abs(dot(worldNormal, prevSampleWorldPosition - prevWorldPosition)) / length(prevViewPos);
    historyLength *= exp(-100.0 * planeDistance);

    historyLength = max(1.0, historyLength);

    float alpha = max(1.0 / (30.0 + 1.0), 1.0 / historyLength);
    vec3 accumulation = mix(previousColor, currentColor, vec3(alpha));

    GbuffersData data = GetGbuffersData(texcoord);
    VectorStruct v = CalculateVectorStruct(texcoord, texture(depthtex0, texcoord).x);

    vec3 reflectDirection = normalize(reflect(v.viewDirection, data.texturedNormal));
    vec3 fr = SpecularLighting(reflectDirection, v.eyeDirection, data.texturedNormal, data.F0, data.roughness, 1.0);

    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    color += LinearToGamma(accumulation) * MappingToHDR * fr;
    if(hideGUI == 1) color = LinearToGamma(accumulation) * MappingToHDR;

    color = GammaToLinear(color * MappingToSDR);

    gl_FragData[0] = vec4(color, 1.0);
    gl_FragData[1] = vec4(accumulation, historyLength / 255.0);
    gl_FragData[2] = vec4(vec3(1.0), depth);
}
/* RENDERTARGETS:3,9,12 */