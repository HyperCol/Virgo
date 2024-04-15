#version 130

uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex12;

uniform sampler2D depthtex0;

uniform float centerDepthSmooth;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/lighting/brdf.glsl"
#include "/libs/tonemapping.glsl"
#include "/libs/materialid.glsl"

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

void main() {
    float depth = texture(depthtex0, texcoord).x;

    float stageID = round(unpack2x8Y(texture(colortex1, texcoord).b) * 255.0);
    float entity = CalculateMask(Stage_Entity, stageID);
    float hand = CalculateMask(Stage_Hand, stageID);

    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

    //hand depth fix
    depth = depth * 2.0 - 1.0;
    #ifndef MC_HAND_DEPTH
    depth /= mix(1.0, 0.7, hand);
    #else
    depth /= mix(1.0, MC_HAND_DEPTH, hand);
    #endif
    depth = depth * 0.5 + 0.5;

    float CoC = CalculateCoC(centerDepthSmooth, depth);

    vec3 rawcolor = GammaToLinear(color * MappingToSDR);
    float luminance = luminance3(rawcolor);

    //color = GammaToLinear(color * MappingToSDR);
    color = KarisToneMapping(GammaToLinear(color));

    luminance *= mix(1.0, 0.05, hand);

    //gl_FragData[0] = vec4(rawcolor, luminance);
    //gl_FragData[1] = vec4(color, CoC);
    
    gl_FragData[0] = vec4(rawcolor, luminance);
    gl_FragData[1] = vec4(color, CoC);
}
/* DRAWBUFFERS:23 */