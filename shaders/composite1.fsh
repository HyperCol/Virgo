#version 130

uniform sampler2D colortex3;

uniform sampler2D colortex6;
uniform sampler2D colortex15;

uniform int hideGUI;

in vec2 texcoord;

/*
const bool colortex15Clear = false;
*/

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/dither.glsl"

#if 1
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
/*
0 albedo
1 roughness metallic material tileID pomdata0 pomdata1
2 normal
3 diffuse ao
emissive lightmap0 lightmap1
4 skymap
5 specluar
6 specluar
7 taa exposure
*/
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
    v.visibleNormal     = CalculateVisibleNormals(v.texturedNormal, v.eyeDirection);
    v.worldVisibleNormal = imv * v.visibleNormal;

    return v;
}
#endif

vec4 SampleHistoryCatmullRom(in sampler2D tex, in vec2 uv, in vec2 size)
{
    // Source: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    // License: https://gist.github.com/TheRealMJP/bc503b0b87b643d3505d41eab8b332ae

    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv / size;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - 1.0;
    vec2 texPos3 = texPos1 + 2.0;
    vec2 texPos12 = texPos1 + offset12;

    texPos0 *= size;
    texPos3 *= size;
    texPos12 *= size;

    vec4 result = vec4(0.0);

    result += texture(tex, vec2(texPos0.x, texPos0.y)) * w0.x * w0.y;
    result += texture(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture(tex, vec2(texPos3.x, texPos0.y)) * w3.x * w0.y;

    result += texture(tex, vec2(texPos0.x, texPos12.y)) * w0.x * w12.y;
    result += texture(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture(tex, vec2(texPos3.x, texPos12.y)) * w3.x * w12.y;

    result += texture(tex, vec2(texPos0.x, texPos3.y)) * w0.x * w3.y;
    result += texture(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture(tex, vec2(texPos3.x, texPos3.y)) * w3.x * w3.y;

    return max(result, vec4(0.0));
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

void CloudsClipAABB(inout vec4 accumulation, in sampler2D tex, in vec2 coord, in float linearDepth, in float sigma) {
    vec4 m1 = vec4(0.0);
    vec4 m2 = vec4(0.0);
    float totalWeight = 0.0;

    //vec2 alphaClip = vec2(1.0, -1.0);
    //vec3 minColor = vec3(1.0);
    //vec3 maxColor = vec3(0.0);

    ivec2 texelPosition = ivec2(coord * resolution);

    const float radius = 1.0;

    for(float i = -radius; i <= radius; i += 1.0) {
        for(float j = -radius; j <= radius; j += 1.0) {
            //vec4 sampleColor = texelFetch(tex, texelPosition + ivec2(i, j), 0);

            vec2 samplePosition = coord + vec2(i, j) * texelSize * 3.0;
            vec4 sampleColor = texture(tex, samplePosition);
            float sampleLinear = (texture(colortex6, samplePosition).x);

            //vec3 color = LinearToGamma(sampleColor.rgb) * MappingToHDR;
            sampleColor.rgb = LinearToGamma(sampleColor.rgb) * MappingToHDR;
            sampleColor.rgb = RGBToYCoCg(sampleColor.rgb);
            
            if(all(equal(vec2(i, j), vec2(0.0))) || abs(sampleLinear - linearDepth) / sampleLinear < 20.0) {
                m1 += sampleColor;
                m2 += sampleColor * sampleColor;
                totalWeight += 1.0;
            }
            /*
            minColor = min(minColor, color);
            maxColor = max(maxColor, color);

            alphaClip = vec2(min(alphaClip.x, sampleColor.a), max(alphaClip.y, sampleColor.a));
            */
        }
    }

    m1 /= totalWeight;
    m2 /= totalWeight;

    vec4 variance = sqrt(max(vec4(0.0), m2 - m1 * m1));

    //float sigma = 3.0;

    vec4 minColor = m1 - variance * sigma;
    vec4 maxColor = m1 + variance * sigma;
    
    accumulation.a = clipToAABB(accumulation.aaa, minColor.aaa, maxColor.aaa).x;

    accumulation.rgb = RGBToYCoCg(accumulation.rgb);
    accumulation.rgb = clipToAABB(accumulation.rgb, minColor.rgb, maxColor.rgb);
    accumulation.rgb = YCoCgToRGB(accumulation.rgb);

    //if(totalWeight == 9.0) accumulation = vec4(1.0, 0.0, 0.0, 0.0);

#if 0
    vec3 variance = sqrt(max(vec3(0.0), m2 - m1 * m1));

    const float sigma = 2.0;

#if 0
    vec3 minColor = m1 - variance * sigma;
    vec3 maxColor = m1 + variance * sigma;
    accumulation.rgb = clipToAABB(accumulation.rgb, minColor, maxColor);
#else
    accumulation.rgb = clamp(accumulation.rgb, minColor, maxColor);
#endif

    float minAlpha = alphaClip.x;
    float maxAlpha = alphaClip.y;
    accumulation.a = clamp(accumulation.a, minAlpha, maxAlpha);
#endif
}

/*
tem 170 169 168
raw 146
*/

vec2 GetVelocityStage(in vec3 coord) {
    //vec4 velocityMap = texelFetch(colortex6, ivec2(coord.xy * resolution), 0);

    //const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    //coord.xy = ApplyTAAJitter(coord.xy);

    //coord.xy = coord.xy * 2.0 - 1.0;
    //coord.xy += float2R2(float(frameCounter) + 1011.5) * (1.0 / 0.25 - 1.0) * texelSize * 100.0;
    //coord.xy = coord.xy * 0.5 + 0.5;

    vec4 p = gbufferProjectionInverse * vec4(coord * 2.0 - 1.0, 1.0);
         p = gbufferModelViewInverse * vec4(p.xyz / p.w, 1.0);
         p.xyz += cameraPosition - previousCameraPosition;
         p = gbufferPreviousModelView * p;
         p = gbufferPreviousProjection * p;
         //p.xy += float2R2(float(frameCounter) - 1.0 + 1011.5) * 300.0 * texelSize;

    //vec2 c = p.xy / p.w * 0.5 + 0.5;
    
    vec2 velocity = coord.xy - (p.xy / p.w * 0.5 + 0.5);
    //if(velocityMap.a > 0.5) velocity = velocityMap.xy;

    return velocity;
}

vec3 CloestCloudsCoord(in vec2 coord, in vec3 direction) {
    vec3 closest = vec3(0.0, 0.0, texture(colortex6, coord).x);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            vec2 offset = vec2(i, j) * texelSize;
            float depth = texture(colortex6, coord + offset).x;

            if(depth < closest.z) {
                closest = vec3(offset, depth);
            }
        }
    }

    closest.z = nvec3(gbufferProjection * nvec4(direction * closest.z)).z * 0.5 + 0.5;

    return closest;
}

void main() {
    vec2 coord = RemoveTAAJitter(texcoord);
    VectorStruct v = CalculateVectorStruct(coord, texture(depthtex0, coord).x);

    float fframeCounter = float(frameCounter);

    const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    //vec2 stageJitter = offset[frameCounter % 4] + offset[(frameCounter / 4) % 4] * 2.0;
    vec2 stageJitter = (float2R2(float(frameCounter) + 1011.5)) * (1.0 / 0.25 - 1.0);

    vec2 fragCoord = texcoord * resolution - stageJitter;
    vec2 halfCoord = min(fragCoord * texelSize * 0.25, vec2(0.25 - texelSize));

    //vec2 cbCoord = floor(texcoord * resolution);
    //     cbCoord = round(cbCoord + stageJitter) + 4.0;
    //float cbHit = step(mod(cbCoord.x, 4.0), 1e-5) * step(mod(cbCoord.y, 4.0), 1e-5);

    float cloudsViewLength = texture(colortex6, halfCoord).x;
    float cloudsDepth = nvec3(gbufferProjection * nvec4(v.viewDirection * cloudsViewLength)).z * 0.5 + 0.5;

    vec3 closest = CloestCloudsCoord(halfCoord, v.viewDirection);

    vec2 velocity = GetVelocityStage(vec3(fragCoord * texelSize, cloudsDepth));
    //vec2 velocity = GetVelocity2(closest + vec3(texcoord, 0.0));
    vec2 previousCoord = texcoord - velocity;

    vec3 color = texture(colortex3, texcoord).rgb;
    color = LinearToGamma(color) * MappingToHDR;

    vec4 cloudsData = texture(colortex5, halfCoord);
         cloudsData.rgb = LinearToGamma(cloudsData.rgb) * MappingToHDR;

    vec4 accumulation = SampleHistoryCatmullRom(colortex15, previousCoord, texelSize);
    CloudsClipAABB(accumulation, colortex5, halfCoord, cloudsViewLength, 4.0);

    float blend = 0.9;
          blend *= step(abs(previousCoord.x - 0.5), 0.5) * step(abs(previousCoord.y - 0.5), 0.5) * step(v.depth, 1.0);

    accumulation = mix(cloudsData, accumulation, vec4(blend));

    if(v.depth >= 1.0) {
        color = accumulation.rgb + color * accumulation.a;
    }
    
    //color = vec3(ExpToLinearDepth(cloudsDepth) / 50000.0);

    //color = vec3(saturate(length(velocity * resolution) * 10.0));

    color = GammaToLinear(color * MappingToSDR);

    //gl_FragData[0] = vec4(color, 1.0);
    gl_FragData[0] = accumulation;
}
/* RENDERTARGETS:15 */