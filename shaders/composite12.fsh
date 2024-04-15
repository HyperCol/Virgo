#version 130

uniform sampler2D colortex3;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"

void main() {
    vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;

#ifdef Enabled_Temporal_AA
	ivec2 texelPosition = ivec2(clamp(texcoord * resolution, vec2(1.5), resolution - 1.5));

	vec4 neighborColor = vec4(0.0);
	neighborColor.rgb += LinearToGamma(texelFetch(colortex3, texelPosition + ivec2(1, 0), 0).rgb) * MappingToHDR;
	neighborColor.rgb += LinearToGamma(texelFetch(colortex3, texelPosition + ivec2(0, 1), 0).rgb) * MappingToHDR;
	neighborColor.rgb += LinearToGamma(texelFetch(colortex3, texelPosition + ivec2(-1, 0), 0).rgb) * MappingToHDR;
	neighborColor.rgb += LinearToGamma(texelFetch(colortex3, texelPosition + ivec2(0, -1), 0).rgb) * MappingToHDR;
	neighborColor.rgb *= 0.25;

	vec3 sharpen = color - neighborColor.rgb;
		 sharpen *= float(TAA_Post_Processing_Sharpeness) * 0.01 * 3.0;

	color = saturate(color + clamp(sharpen, vec3(-TAA_Post_Sharpen_Limit), vec3(TAA_Post_Sharpen_Limit)));	
#endif
	//color = pow(max(0.0, texcoord.y - 0.125), 2.2) * vec3(1.0 * 0.125);
	//if(texcoord.y < 0.125) color = vec3(1.0, 0.0, 0.0);

    color = GammaToLinear(color * MappingToSDR);

    gl_FragData[0] = vec4(color, 1.0);
}

/* DRAWBUFFERS:3 */