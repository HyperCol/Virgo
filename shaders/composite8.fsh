#version 130

uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex7;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform int hideGUI;

in vec2 texcoord;

/*
const bool colortex2MipmapEnabled = true;

const bool colortex7Clear = false;
*/

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/tonemapping.glsl"
#include "/libs/materialid.glsl"

//DH Support
uniform sampler2D dhDepthTex0;

uniform mat4 dhProjection;
uniform mat4 dhProjectionInverse;
uniform mat4 dhPreviousProjection;

float GetDepth0(in vec2 coord) {
    float depth = texture(depthtex0, coord).x;
    return depth >= 1.0 ? texture(dhDepthTex0, coord).x : depth;
}

float GetDepth0(in ivec2 icoord) {
    float depth = texelFetch(depthtex0, icoord, 0).x;
    return depth >= 1.0 ? texelFetch(dhDepthTex0, icoord, 0).x : depth;
}

vec3 GetDHViewPosition(in vec3 coord) {
    return nvec3(dhProjectionInverse * vec4(ApplyTAAJitter(coord.xy) * 2.0 - 1.0, coord.z * 2.0 - 1.0, 1.0));
}

vec3 RGBToYCoCg(vec3 c) {
	// Y = R/4 + G/2 + B/4
	// Co = R/2 - B/2
	// Cg = -R/4 + G/2 - B/4

    return vec3(c.x/4.0 + c.y/2.0 + c.z/4.0,
                c.x/2.0 - c.z/2.0,
                -c.x/4.0 + c.y/2.0 - c.z/4.0);
}

vec3 YCoCgToRGB(vec3 c) {
	// R = Y + Co - Cg
	// G = Y + Cg
	// B = Y - Co - Cg

    return vec3(c.x + c.y - c.z,
                c.x + c.z,
	            c.x - c.y - c.z);
}

vec3 ReprojectSampler(in sampler2D tex, in vec2 pixelPos){
    vec4 color = vec4(0.0);

    vec2 position = resolution * pixelPos;
    vec2 centerPosition = floor(position - 0.5) + 0.5;

    vec2 f = position - centerPosition;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    float c = float(TAA_Sampling_Sharpeness) / 101.0;
    vec2 w0 =         -c  *  f3 + 2.0 * c          *  f2 - c  *  f;
    vec2 w1 =  (2.0 - c)  *  f3 - (3.0 - c)        *  f2            + 1.0;
    vec2 w2 = -(2.0 - c)  *  f3 + (3.0 - 2.0 * c)  *  f2 + c  *  f;
    vec2 w3 =          c  *  f3 - c                *  f2;
    vec2 w12 = w1 + w2;

    vec2 tc12 = texelSize * (centerPosition + w2 / w12);
    vec2 tc0 = texelSize * (centerPosition - 1.0);
    vec2 tc3 = texelSize * (centerPosition + 2.0);

    color = vec4((texture(tex, vec2(tc12.x, tc0.y)).rgb), 1.0) * (w12.x * w0.y) +
            vec4((texture(tex, vec2(tc0.x, tc12.y)).rgb), 1.0) * (w0.x * w12.y) +
            vec4((texture(tex, vec2(tc12.x, tc12.y)).rgb), 1.0) * (w12.x * w12.y) +
            vec4((texture(tex, vec2(tc3.x, tc12.y)).rgb), 1.0) * (w3.x * w12.y) +
            vec4((texture(tex, vec2(tc12.x, tc3.y)).rgb), 1.0) * (w12.x * w3.y);

    return saturate(color.rgb / color.a);
}

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

vec3 clipToAABB(vec3 color, vec3 minimum, vec3 maximum) {
    #ifndef TAA_No_Clip
    vec3 p_clip = 0.5 * (maximum + minimum);
    vec3 e_clip = 0.5 * (maximum - minimum);

    vec3 v_clip = color - p_clip;
    vec3 v_unit = v_clip.xyz / e_clip;
    vec3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0) return p_clip + v_clip / ma_unit;
    #endif
    
    return color;// point inside aabb
}

vec3 GetVariance(in vec2 coord, out vec3 minColor, out vec3 maxColor, in float sigma) {
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    ivec2 texelPosition = ivec2(coord * resolution);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            //vec3 sampleColor = RGBToYCoCg(texture(colortex3, coord + vec2(i, j) * texelSize).rgb);
            vec3 sampleColor = RGBToYCoCg(texelFetch(colortex3, texelPosition + ivec2(i, j), 0).rgb);

            m1 += sampleColor;
            m2 += sampleColor * sampleColor;
        }
    }

    m1 /= 9.0;
    m2 /= 9.0;

    vec3 variance = sqrt(max(vec3(0.0), m2 - m1 * m1));

    //const float sigma = 2.0;

    minColor = m1 - variance * sigma;
    maxColor = m1 + variance * sigma;

    return variance;
}

float random01(in float n) {
    float g = 1.6180339887498948482;
    return fract(n / g + 0.5);
}

vec2 float2R2(in float n) {
	float g = 1.32471795724474602596;
	vec2  a = 1.0 / vec2(g, g * g);

	return fract(0.5 + n * a);
}

float ColorEdgeDetection(in sampler2D tex, in vec2 coord, in float threshold) {
    ivec2 texelPosition = ivec2(coord * resolution);

    float delta = 0.0;

    vec3 c0 = texelFetch(tex, texelPosition, 0).rgb;
    //vec3 c0 = texture(tex, coord).rgb;

    //float threshold = 0.03;

    delta = maxComponent(abs(c0 - texelFetch(tex, texelPosition + ivec2(1, 0), 0).rgb));
    delta = max(delta, maxComponent(abs(c0 - texelFetch(tex, texelPosition - ivec2(1, 0), 0).rgb)));
    delta = max(delta, maxComponent(abs(c0 - texelFetch(tex, texelPosition + ivec2(0, 1), 0).rgb)));
    delta = max(delta, maxComponent(abs(c0 - texelFetch(tex, texelPosition - ivec2(0, 1), 0).rgb)));

    //delta = maxComponent(abs(c0 - texture(tex, coord + vec2(texelSize.x, 0.0)).rgb));
    //delta = max(delta, maxComponent(abs(c0 - texture(tex, coord - vec2(texelSize.x, 0.0)).rgb)));
    //delta = max(delta, maxComponent(abs(c0 - texture(tex, coord + vec2(0.0, texelSize.y)).rgb)));
    //delta = max(delta, maxComponent(abs(c0 - texture(tex, coord - vec2(0.0, texelSize.y)).rgb)));

    //delta = step(threshold, delta);
    delta = step(threshold, delta) * delta;

    return delta;
}

vec3 FXAA(in sampler2D tex, in vec2 coord) {
    ivec2 texelPosition = ivec2(coord * resolution);


    //vec3 c0 = texelFetch(tex, texelPosition, 0).rgb;
    vec3 c0 = texture(tex, coord).rgb;

    float threshold = 0.05;
    /*
    vec4 delta = vec4(0.0);

    vec3 c1 = texelFetch(tex, texelPosition + ivec2(1, 0), 0).rgb;
    delta += vec4(c1, 1.0) * step(threshold, maxComponent(abs(c0 - c1)));

    vec3 c2 = texelFetch(tex, texelPosition - ivec2(1, 0), 0).rgb;
    delta += vec4(c2, 1.0) * step(threshold, maxComponent(abs(c0 - c2)));

    vec3 c3 = texelFetch(tex, texelPosition + ivec2(0, 1), 0).rgb;
    delta += vec4(c3, 1.0) * step(threshold, maxComponent(abs(c0 - c3)));

    vec3 c4 = texelFetch(tex, texelPosition - ivec2(0, 1), 0).rgb;
    delta += vec4(c4, 1.0) * step(threshold, maxComponent(abs(c0 - c4)));

    vec3 color = (delta.rgb + c0) / (delta.a + 1.0);
    */
    
    float delta = 0.0;
    /*
    vec3 c1 = texelFetch(tex, texelPosition + ivec2(1, 0), 0).rgb;
    delta = maxComponent(abs(c0 - c1));

    vec3 c2 = texelFetch(tex, texelPosition - ivec2(1, 0), 0).rgb;
    delta = max(delta, maxComponent(abs(c0 - c2)));

    vec3 c3 = texelFetch(tex, texelPosition + ivec2(0, 1), 0).rgb;
    delta = max(delta, maxComponent(abs(c0 - c3)));

    vec3 c4 = texelFetch(tex, texelPosition - ivec2(0, 1), 0).rgb;
    delta = max(delta, maxComponent(abs(c0 - c4)));

    delta = step(threshold, delta);

    vec3 color = mix(c0, (c0 + c1 + c2 + c3 + c4) / 5.0, vec3(delta));
    */

    vec3 c1 = texture(tex, coord + vec2(texelSize.x, 0.0)).rgb;
    delta = maxComponent(abs(c0 - c1));

    vec3 c2 = texture(tex, coord - vec2(texelSize.x, 0.0)).rgb;
    delta = max(delta, maxComponent(abs(c0 - c2)));

    vec3 c3 = texture(tex, coord + vec2(0.0, texelSize.y)).rgb;
    delta = max(delta, maxComponent(abs(c0 - c3)));

    vec3 c4 = texture(tex, coord - vec2(0.0, texelSize.y)).rgb;
    delta = max(delta, maxComponent(abs(c0 - c4)));

    delta = step(threshold, delta);

    vec3 color = (c0 + c1 + c2 + c3 + c4) / 5.0;//mix(c0, (c0 + c1 + c2 + c3 + c4) / 5.0, vec3(delta));

    return color;
}

uniform float centerDepthSmooth;

float CalculateCoC(in float expFocus, in float expDepth) {
    float P = ExpToLinearDepth(expFocus);
    float Z = ExpToLinearDepth(expDepth);

    //float A = 2.8 * 20.0;
    //float F = 0.004 * 20.0;

    float A = Camera_Aperture;
    float F = Camera_Focal_Length;

    float d = abs(P - F);

	float maxBgdCoC = (A * F) / max(1e-5, abs(d));
	float CoC = (1.0 - P / Z) * maxBgdCoC;

    return CoC;
}

vec3 GetClosest(in sampler2D tex, in vec2 coord, in float depth0) {
    vec3 closest = vec3(0.0, 0.0, depth0);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            vec2 sampleCoord = coord + vec2(i, j) * texelSize;
            float sampleDepth = texture(tex, sampleCoord).x;
            //      sampleDepth = sampleDepth >= 1.0 ? texture(dhDepthTex0, sampleCoord).x : sampleDepth;

            if(sampleDepth < closest.z && sampleDepth > 0.7) {
                closest = vec3(i, j, sampleDepth);
            }
        }
    }

    closest.xy = closest.xy * texelSize + coord;

    //return vec3(coord, texture(depthtex0, coord).x);

    return closest;
}

vec2 GetVelocityStage(in vec2 coord) {
    //vec4 velocityMap = texelFetch(colortex5, ivec2(coord.xy * resolution), 0);

    //if(hideGUI == 0) coord.xy = ApplyTAAJitter(coord.xy);

    float depth = texture(depthtex0, coord).x;

    //vec3 closest = GetClosest(depthtex1, coord, depth >= 1.0 ? depth : 1.0);
    vec3 closest = GetClosest(depthtex0, coord, 1.0);
    vec3 viewPosition = nvec3(gbufferProjectionInverse * nvec4(closest * 2.0 - 1.0));

    //mat4 previousProjectionMat = gbufferPreviousProjection;

#ifdef DISTANT_HORIZONS 
    if(closest.z >= 1.0) {
        closest = GetClosest(dhDepthTex1, coord, 1.0);
        viewPosition = nvec3(dhProjectionInverse * nvec4(closest * 2.0 - 1.0));
        //previousProjectionMat = dhPreviousProjection;
    }
#endif

    //if(closest.)
    //vec3 closestDH = GetClosest(dhDepthTex0, coord);

    //vec4 p = gbufferProjectionInverse * vec4(coord * 2.0 - 1.0, 1.0);
    //     p = gbufferModelViewInverse * vec4(p.xyz / p.w, 1.0);
    vec4 p = gbufferModelViewInverse * nvec4(viewPosition);
         p.xyz += cameraPosition - previousCameraPosition;
         p = gbufferPreviousModelView * p;
         p = gbufferPreviousProjection * p;
         p /= p.w;

    //vec2 c = p.xy / p.w * 0.5 + 0.5;
    //p.xy -= jitter1;

    vec2 velocity = closest.xy - (p.xy * 0.5 + 0.5);
    //vec2 velocity = coord.xy - p.xy;
    //if(velocityMap.a > 0.5) velocity = velocityMap.xy;

    return velocity;
}

vec3 SampleClipToAABB(in vec3 color, in vec2 coord, in float sigma) {
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);

    ivec2 texelPosition = ivec2(coord * resolution);

    for(float i = -1.0; i <= 1.0; i += 1.0) {
        for(float j = -1.0; j <= 1.0; j += 1.0) {
            //vec3 sampleColor = RGBToYCoCg(texture(colortex3, coord + vec2(i, j) * texelSize).rgb);
            vec3 sampleColor = RGBToYCoCg(texelFetch(colortex3, texelPosition + ivec2(i, j), 0).rgb);

            m1 += sampleColor;
            m2 += sampleColor * sampleColor;
        }
    }

    m1 /= 9.0;
    m2 /= 9.0;

    vec3 variance = sqrt(max(vec3(0.0), m2 - m1 * m1));

    vec3 maxColor = m1 + variance * sigma;
    vec3 minColor = m1 - variance * sigma;

    return clipToAABB(color, minColor, maxColor);
}

void main() {
    vec3 currentColor = vec3(0.0);

    float stageID = round(unpack2x8Y(texture(colortex1, texcoord).b) * 255.0);
    float entity = CalculateMask(Stage_Entity, stageID);
    float hand = CalculateMask(Stage_Hand, stageID);

#ifdef Enabled_Temporal_AA
    vec2 coord = RemoveTAAJitter(texcoord);

    currentColor = RGBToYCoCg(texture(colortex3, coord).rgb);

    //float colorEdge = ColorEdgeDetection(colortex3, texcoord, 0.03);

    //vec3 closest = GetClosest(depthtex0, coord);
    //vec3 closestVP = nvec3(gbufferProjectionInverse * nvec4(closest * 2.0 - 1.0));

    //vec3 closestDH = GetClosest(dhDepthTex0, coord);

    vec2 velocity = GetVelocityStage(coord);

    velocity *= step(vec2(1e-5), abs(velocity));
    
    //vec4 velocityMap = texelFetch(colortex5, ivec2(texcoord * resolution), 0);
    //if(velocityMap.a > 0.5) velocity = velocityMap.xy;//velocity + (velocityMap.xy - velocity) / 30.0;

    vec2 velocityLimit = texelSize * 10.0;
    //if(hand == 1.0) velocity = clamp(velocity, -velocityLimit, velocityLimit);

    float velocityLength = length(velocity * resolution);

    vec2 previousCoord = texcoord - velocity;

    float InScreen = step(max(abs(previousCoord.x - 0.5), abs(previousCoord.y - 0.5)), 0.5);

    vec3 previousColor = RGBToYCoCg(SampleHistoryCatmullRom(colortex7, previousCoord, texelSize).rgb);
    //vec3 previousColor = RGBToYCoCg(texture(colortex7, previousCoord).rgb);

    float sigma = 2.0;
    //      sigma = mix(sigma, 1.0, entity);

    #ifndef TAA_Non_Clip
        previousColor = SampleClipToAABB(previousColor, coord, sigma);
    #endif

    float blend = 0.95 * InScreen;//(1.0 - 1.0 / 30.0) * InScreen;

    float velocityWeight = clamp(velocityLength * 32.0 - 0.05, 0.0, 0.08);
    //float velocityWeight = min(0.5, distance(cameraPosition, previousCameraPosition) / 4.316);

    float blocker = ExpToLinearDepth(texture(depthtex0, previousCoord).x);
    float depthWeight = min(0.2, abs(ExpToLinearDepth(texture(depthtex0, texcoord).x) - blocker) / blocker * 16.0);

    float entityWeight = 0.15 * entity;

    //blend -= max(velocityWeight, entityWeight);

    #ifndef TAA_Non_Clip
        //blend -= min(0.15, (1.0 - colorEdge) * step(0.05, velocityLength));
        blend -= clamp(velocityLength, 0.0, 0.1);
    #endif

    vec3 accumulation = mix(currentColor, previousColor, vec3(blend));

    accumulation = max(vec3(0.0), YCoCgToRGB(accumulation));
#else
    vec3 accumulation = texture(colortex3, texcoord).rgb;
#endif

    //result
    vec3 color = accumulation;
         color = LinearToGamma(InverseKarisToneMapping(color));

    //vec2 fragCoord = (texcoord - 0.5 - vec2(0.1)) * resolution;
    //if(max(abs(fragCoord.x), abs(fragCoord.y)) < 1.0) color = vec3(0.99, 0.0, 0.0) * float(heldBlockLightValue) / 15.0;

    //color = saturate(1.0 - colorEdge * 100.0) * vec3(1.0);
    //if(texcoord.y < 0.5) color = vec3(step(0.05, velocityLength));

    //color = vec3((colorEdge));

    //color = vec3(saturate(length(velocity * resolution * 0.01)));
    //color = vec3(step(1e-5, length(velocity * resolution)));

    color = GammaToLinear(color * MappingToSDR);

    float luminanceBlend = mod(frameTimeCounter, 1.0) - mod(frameTimeCounter * 0.5, 1.0);
          luminanceBlend /= 2.0;
          luminanceBlend = clamp(luminanceBlend, 1.0 / 300.0, 0.5);

    float luminance = textureLod(colortex2, vec2(0.5), log2(viewHeight)).a;
    float luminanceAccumulation = texture(colortex7, vec2(0.5)).a;
    
    luminance *= pow(MappingToHDR, 1.0 / 2.2);
    luminance = (log2(luminance * 100.0 / 12.5));
    luminance = clamp(luminance, -5.0, 5.0);
    luminance = rescale(luminance, -5.0, 5.0);
    //luminanceAccumulation = 

    luminanceAccumulation = mix(luminanceAccumulation, luminance, luminanceBlend);

    //float CoC = CalculateCoC(centerDepthSmooth, closest.z);
    //float CoC = CalculateCoC(centerDepthSmooth, texture(depthtex0, texcoord).x);
    
    float farest = 0.0;

    for(float i = 0.0; i <= 1.0; i += 1.0) {
        for(float j = 0.0; j <= 1.0; j += 1.0) {
            farest = max(farest, texture(depthtex0, texcoord + vec2(i, j) * texelSize).x);
        }
    }

    float CoC = CalculateCoC(centerDepthSmooth, farest);

    gl_FragData[0] = vec4(color, CoC);
    gl_FragData[1] = vec4(accumulation, luminanceAccumulation);
}
/* DRAWBUFFERS:37 */