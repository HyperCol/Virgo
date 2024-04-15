#version 130

#define Reflection_Temporal_Upsample

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
#include "/libs/tonemapping.glsl"
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

const vec2 varOffset[13] = vec2[13](
vec2(0.0, 0.0), 
vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(-1.0, 0.0), vec2(0.0, -1.0),
vec2(1.0, 1.0), vec2(-1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, -1.0),
vec2(2.0, 0.0), vec2(0.0, 2.0), vec2(-2.0, 0.0), vec2(0.0, -2.0)
);

vec3 GetClosest(in sampler2D tex, in vec2 coord, in float depth0) {
    vec3 closest = vec3(0.0, 0.0, depth0);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            vec2 offset = vec2(i, j) * texelSize;
            vec2 sampleCoord = coord + offset;
            float sampleDepth = texture(tex, sampleCoord).x;
            //      sampleDepth = sampleDepth >= 1.0 ? texture(dhDepthTex0, sampleCoord).x : sampleDepth;

            if(sampleDepth < closest.z) {
                closest = vec3(offset, sampleDepth);
            }
        }
    }

    closest.xy = closest.xy + coord;

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

/*
void ClipAABB(inout vec3 accumulation, in sampler2D tex, in vec2 coord, in float sigma) {
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    float totalWeight = 0.0;

    //vec2 alphaClip = vec2(1.0, -1.0);
    //vec3 minColor = vec3(1.0);
    //vec3 maxColor = vec3(0.0);

    ivec2 texelPosition = ivec2(coord * resolution);

    const float radius = 2.0;

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
*/

void StageClipAABB(inout vec3 accumulation, in sampler2D tex, in vec3 e, in vec3 i, in vec3 n, in float pdf0, in float clipMin) {
    ivec2 texelPosition = ivec2(gl_FragCoord.xy);

    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    float totalWeight = 0.0;

#if 1
    for(int i = 0; i < 13; i++) {
        ivec2 sampleTexel = texelPosition + ivec2(varOffset[i]);

        vec3 sampleColor = texelFetch(tex, sampleTexel, 0).rgb;
             sampleColor = RGBToYCoCg(sampleColor);

        m1 += sampleColor;
        m2 += sampleColor * sampleColor;
        totalWeight += 1.0;        
    }
#else
    const float pdfThreshold = 0.001;
    const int radius = 2;

    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j);

            vec3 sampleColor = texelFetch(tex, sampleTexel, 0).rgb;
                 sampleColor = RGBToYCoCg(sampleColor);

            float sampleDepth = texelFetch(depthtex0, sampleTexel, 0).x;

            float weight = 1000.0;

            if(max(abs(float(i)), abs(float(j))) > 1.0) {
            vec3 sampleViewVector = normalize(GetViewPosition(vec2(sampleTexel) * texelSize, sampleDepth));
            vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, sampleTexel, 0).xy);
            weight = GetPixelPDF(e, normalize(reflect(sampleViewVector, sampleNormal)), n, texelFetch(tex, sampleTexel, 0).a);
            weight = weight / (weight + pdf0) * 4.0;
            }

            if(sampleDepth < 1.0 && weight >= pdfThreshold) {
                m1 += sampleColor;
                m2 += sampleColor * sampleColor;
                totalWeight += 1.0;
            }
        }
    }

    totalWeight += step(totalWeight, 0.0);
#endif

    m1 /= totalWeight;
    m2 /= totalWeight;

    vec3 variance = sqrt(max(vec3(0.0), m2 - m1 * m1));

#ifdef Reflection_Temporal_Upsample
    float sigma = clamp(GetPixelPDF(1.0, pow2(1.0 - 0.95)) / pdf0, clipMin, 10.0);
#else
    float sigma = clipMin;
#endif

    vec3 minColor = m1 - variance * sigma;
    vec3 maxColor = m1 + variance * sigma;

    accumulation.rgb = RGBToYCoCg(accumulation.rgb);
    accumulation.rgb = clipToAABB(accumulation.rgb, minColor.rgb, maxColor.rgb);
    accumulation.rgb = YCoCgToRGB(accumulation.rgb);
}

vec3 SampleHistoryCatmullRom(in sampler2D tex, in vec2 uv, in vec2 size) {
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

    return max(result.rgb, vec3(0.0));
}

vec2 CalculateDualMotionVector(in vec2 previousCoord, in float depth, in vec2 coord) {
    vec3 p = nvec3(previousProjectionInverse * nvec4(vec3(previousCoord, depth) * 2.0 - 1.0));
    //vec3 p = nvec3(previousProjectionInverse * nvec4(vec3(previousCoord, texture(tex, PreviousToCurrentJitter(previousCoord)).b) * 2.0 - 1.0));
         p = mat3(previousModelViewInverse) * p + previousModelViewInverse[3].xyz;
         p = p + previousCameraPosition - cameraPosition;
         p = mat3(gbufferModelView) * p + gbufferModelView[3].xyz;
         p = nvec3(gbufferProjection * nvec4(p)) * 0.5 + 0.5;

    vec2 velocity = coord.xy - p.xy;
         velocity *= step(vec2(1e-3), abs(velocity));

    return velocity;
}

void main() {
    vec2 coord = (texcoord);

    float fframeCounter = float(frameCounter);

    const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
#if 1
    vec2 stageJitter = offset[frameCounter % 4];
#else
    vec3 stageJitter = vec2(0.0);
#endif

    GbuffersData data = GetGbuffersData(coord);
    VectorStruct v = CalculateVectorStruct(coord, texture(depthtex0, coord).x);

    data.roughness = texture(colortex4, texcoord).a;

    vec3 rayDirection = normalize(reflect(v.viewDirection, data.texturedNormal));
    float pdf0 = GetPixelPDF(1.0, data.roughness);
    float pdfThreshold = pdf0 - 0.0;//GetPixelPDF(1.0, pow2(1.0 - 0.7));

    //view depth
    //vec3 closest = GetClosest(depthtex0, coord, 1.0);
    //vec3 closest = vec3(coord, texture(depthtex0, coord).x);

    //ray depth
    vec3 closest = GetClosest(colortex6, coord, 1.0);
    //vec3 closest = vec3(coord, texture(colortex6, coord).x);

    vec2 velocity = GetVelocity(closest);
         velocity *= step(vec2(1e-5), abs(velocity));
    //vec2 velocity = GetVelocity(pdfThreshold > 0.0 ? closestRay : closest);
    //vec2 velocity = GetVelocity(vec3(texcoord, texture(colortex6, texcoord).x));
         //velocity = vec2(0.0);

    vec2 previousCoord2 = coord - GetVelocity(vec3(coord, texture(depthtex0, coord).x));
    
    vec2 velocity2 = CalculateDualMotionVector(previousCoord2, texture(colortex12, PreviousToCurrentJitter(previousCoord2)).a, texcoord);
         velocity2 *= step(vec2(1e-3), abs(velocity2));
    //vec2 velocity2 = CalculateDualMotionVector(previousCoord, texture(colortex12, PreviousToCurrentJitter(previousCoord)).b, texcoord);
    //previousCoord += velocity2;
    velocity -= velocity2;

    float velocityLength = length(velocity * resolution);
    vec2 previousCoord = coord - velocity;

    vec3 currentColor = texture(colortex4, coord).rgb;

    float historyLength = texture(colortex9, previousCoord).a * 255.0;
          historyLength = historyLength + 1.0;
          historyLength = min(255.0, historyLength);

    historyLength *= step(abs(previousCoord.x - 0.5), 0.5 - texelSize.x) * step(abs(previousCoord.y - 0.5), 0.5 - texelSize.y);

    float depth = v.depth;
    float previousDepth = texture(colortex12, PreviousToCurrentJitter(previousCoord)).a;// - jitter * 0.5 + jitter1 * 0.5

    vec3 viewPosition = v.viewPosition;
    vec3 worldPosition = v.worldPosition + cameraPosition;
    vec3 prevWorldPosition = worldPosition - previousCameraPosition;

    vec3 prevViewPos = nvec3(previousProjectionInverse * nvec4(vec3(previousCoord, previousDepth) * 2.0 - 1.0));
    vec3 prevSampleWorldPosition = mat3(previousModelViewInverse) * prevViewPos + previousModelViewInverse[3].xyz;
    float previousViewLength = length(prevViewPos);

    vec3 worldNormal = mat3(gbufferModelViewInverse) * data.texturedNormal;

    vec3 previousWorldDir = (prevSampleWorldPosition - previousModelViewInverse[3].xyz) / previousViewLength;
    vec3 previousWorldNormal = DecodeOctahedralmap(texture(colortex12, previousCoord).xy);
    vec3 previousRayDir = normalize(reflect(previousWorldDir, previousWorldNormal));

    vec3 previousColor = texture(colortex9, previousCoord).rgb;
    //vec3 previousColor = SampleHistoryCatmullRom(colortex9, previousCoord, texelSize);

    float blocker = min(previousViewLength, v.viewDistance);
    
#if 1
    float planeDistance = abs(dot(worldNormal, prevSampleWorldPosition - prevWorldPosition)) / blocker;
    historyLength *= exp(-12.0 * planeDistance);

    float rayDepth = texture(colortex6, texcoord).x;

    ////removed
    //float linearRayDepth = ExpToLinearDepth(rayDepth);
    //float previousLinearRayDepth = ExpToLinearDepth(texture(colortex12, PreviousToCurrentJitter(previousCoord)).b);
    //historyLength *= exp(-max(0.0, abs(previousLinearRayDepth - linearRayDepth) - 1000.0) * 12.0 * step(0.05, length(velocity * resolution)));
    
    float previousRoughness = texture(colortex12, previousCoord).b;
    //historyLength = abs(data.roughness - previousRoughness) > 0.5 ? 1.0 : historyLength;
    float pdf1 = GetPixelPDF(1.0, previousRoughness);
    float roughnessWeight = abs(pdf1 - pdf0) / min(pdf0, pdf1) * 0.01;//(1.0 - abs(pdf1 - pdf0) / min(pdf0, pdf1)) * 10.0;
    historyLength = roughnessWeight > 1.0 ? 1.0 : historyLength;

    //float NoV = abs(dot(worldNormal, v.worldEyeDirection));
    //float NoVprev = abs(dot(previousWorldNormal, v.worldEyeDirection));
    ////float NoVprev = GetPixelPDF(v.worldEyeDirection, previousWorldNormal, worldNormal, 0.9999);
    ////float NoV = GetPixelPDF(v.worldEyeDirection, worldNormal, worldNormal, 0.9999);
    //float sizeQuality = (NoVprev + 1e-3) / (NoV + 1e-3); // this order because we need to fix stretching only, shrinking is OK

    //sizeQuality *= sizeQuality;
    //sizeQuality *= sizeQuality;
    //float footprintQuality = 1.0 * mix(0.1, 1.0, saturate(sizeQuality));
    //historyLength *= saturate(sizeQuality / 0.9999);

    //const vec2 clipA = vec2(0.01, 0.9);
    float aAdj = saturate(rescale(data.roughness, -0.04, 1.0)) - 1e-7;//max(0.04, data.roughness);
    float aP = GetPixelPDF(v.worldEyeDirection, previousRayDir, worldNormal, aAdj);
    float aC = GetPixelPDF(v.worldEyeDirection, mat3(gbufferModelViewInverse) * rayDirection, worldNormal, aAdj);
    float directionWeight = aP / (aP + aC) * 2.01;
    historyLength *= mix(saturate(directionWeight), 1.0, step(velocityLength, 0.05) * 0.9);

    float pdf = GetPixelPDF(v.eyeDirection, rayDirection, data.texturedNormal, data.roughness);
    float clipMin = 2.0;
    
    //vec2 cbCoord = floor(gl_FragCoord.xy) - 1.0;
    //float checkerBoard = mod(floor(cbCoord.x - stageJitter.x), 2.0) * mod(floor(cbCoord.y - stageJitter.y), 2.0);
    ////clipMin = mix(clipMin, 1.0, checkerBoard);
    ////historyLength *= mix(1.0, 0.5, checkerBoard);

    StageClipAABB(previousColor, colortex4, v.eyeDirection, rayDirection, data.texturedNormal, pdf, clipMin);
#endif

    historyLength = max(1.0, historyLength);

    float alpha = max(1.0 / Reflection_Accumulation_Frame, 1.0 / historyLength);
    vec3 accumulation = mix(previousColor, currentColor, vec3(alpha));
         //accumulation = currentColor;

    vec3 fr = SpecularLighting(rayDirection, v.eyeDirection, data.texturedNormal, data.F0, data.roughness, 1.0);
    vec3 specular = accumulation; //specular = currentColor;
         specular = LinearToGamma(InverseKarisToneMapping(specular));

    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    color += specular * fr;
    //if(hideGUI == 1) color = specular;
    if(hideGUI == 0) color = specular;

    //color = checkerBoard * vec3(0.1);

    //color = saturate(pdf0 / pdf1 / 0.9999) * pdf0 / threshold >= 1.0 ? vec3(1.0, 0.0, 0.0) : vec3(1.0);

    //color = vec3(saturate(rayDirection.z));

    //float w = rpdf0 / rpdf1 * pdf0;
    //color = vec3(saturate(w));
    //if(w >= 40.0) color = vec3(1.0, 0.0, 0.0);

    //color = vec3(saturate(planeDistance));
    //color = vec3(abs(ExpToLinearDepth(depth) - ExpToLinearDepth(previousDepth)));
    //color = saturate(length(prevSampleWorldPosition - prevWorldPosition)) * vec3(1.0);
    //color = vec3(step(1e-5, length(velocity * resolution)));
    //color = saturate(length(velocity * resolution) / 10.0) * vec3(1.0);

    //color = vec3(saturate(rpdf0 / rpdf1 / 0.9999 * 0.001));
    //if(heldItemId == 1)
    //color = pdfThreshold > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(1.0);
    //color = saturate(pdf0 / pdf1 <= 1.0 ? vec3(1.0 - pdf0 / pdf1) : vec3(1.0, 0.0, 0.0));

    //float pdf2 = GetPixelPDF(1.0, data.roughness);// / max(1e-6, dot(v.viewDirection, data.texturedNormal));
    //float pdf3 = GetPixelPDF(saturate(dot(previousViewNormal, data.texturedNormal)), previousRoughness);// / max(1e-6, dot(normalize(prevViewPos), previousViewNormal));

    //vec3 h = normalize(previousRayDir + v.eyeDirection);
    //float pdf2 = GetPixelPDF(1.0, data.roughness) / max(1e-6, dot(rayDirection, data.texturedNormal) * 4.0);
    //float pdf3 = GetPixelPDF(saturate(dot(previousViewNormal, data.texturedNormal)), previousRoughness) / max(1e-6, dot(previousRayDir, previousViewNormal) * 4.0);
    //float pdf2 = GetPixelPDF(v.eyeDirection, rayDirection, data.texturedNormal, data.roughness);
    //float pdf3 = GetPixelPDF(v.eyeDirection, previousRayDir, data.texturedNormal, previousRoughness);
    //color *= 0.1;
    //float w0 = (abs(dot(v.eyeDirection, data.texturedNormal)) + 1e-3) / (1e-3 + abs(dot(v.eyeDirection, previousViewNormal)));
    //float w0 = (GetPixelPDF(saturate(dot(data.texturedNormal, previousViewNormal)), previousRoughness) / max(1e-6, dot(v.eyeDirection, previousViewNormal) * 4.0)) / (DistributionTerm(1.0, data.roughness) / max(1e-6, dot(v.eyeDirection, data.texturedNormal)));
    //color = w0 / 0.9999 >= 1.0 ? vec3(1.0, 0.0, 0.0) : vec3(w0);
    //color = vec3(saturate(rpdfWeight));

    //color = sizeQuality < 1.0 ? vec3(sizeQuality) : vec3(1.0, 0.0, 0.0);
    //color = nweight < 1.0 ? vec3(nweight) : vec3(1.0, 0.0, 0.0);

    //float w1 = 1.0 + 1e-6 - abs(pdfP - pdfC) / pdfP;
    //w1 = texcoord.y > 0.5 ? 1.0 + 1e-6 - abs(data.roughness - previousRoughness) / previousRoughness * 1.4 : w1;
    //float w1 = 1.0 + 1e-6 - abs(data.roughness - previousRoughness) / previousRoughness;
    //float w1 = 1.0 + 1e-6 - abs(aP - aC) / min(aC, aP);
    //float w1 = 1.0 + 1e-6 - abs(data.roughness - previousRoughness) / min(previousRoughness, data.roughness);
    //float w1 = aP / aC / 0.9999;// + step(abs(v.linearDepth - ExpToLinearDepth(previousDepth)), 1e-3);// / clamp(length(prevSampleWorldPosition - prevWorldPosition), 1e-3, 1.0);
    //float w1 = sizeQuality / 0.9999;
    //float w1 = aP / (aP + aC) * 4.0 / min(1.0, 0.1 + length(velocity2 * resolution - velocity * resolution));
    //float w1 = (1.0 - abs(pdf1 - pdf0) / pdf1) * 2.0;
    //color = w1 >= 1.0 ? vec3(1.0, 0.0, 0.0) : vec3(saturate(w1));
    //color = abs(data.roughness - previousRoughness) > 0.015 ? vec3(0.0) : vec3(1.0);
    
    //specular = roughnessWeight < 1.0 ? vec3(1.0, 0.0, 0.0) : vec3(saturate(roughnessWeight));
    //specular = directionWeight >= 1.0 ? vec3(1.0, 0.0, 0.0) : vec3(saturate(directionWeight));
    //specular = vec3(step(0.05, velocityLength));

    color = GammaToLinear(color * MappingToSDR);

    specular = saturate(GammaToLinear(specular * MappingToSDR));

    gl_FragData[0] = vec4(specular, data.roughness);
    gl_FragData[1] = vec4(accumulation, historyLength / 255.0);
    //gl_FragData[2] = vec4(EncodeOctahedralmap(worldNormal), rayDepth, depth);
    gl_FragData[2] = vec4(EncodeOctahedralmap(worldNormal), data.roughness, depth);
}
/* RENDERTARGETS:4,9,12 */