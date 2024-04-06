#version 130

uniform sampler2D colortex3;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/lighting/filter.glsl"

void main() {
#ifdef GI_A_Trous_3
    #if defined(GI_A_Trous_1) && defined(GI_A_Trous_2)
    gl_FragData[0] = SpatialFliter(colortex3, texcoord, 9, 1.0);
    #endif
#else
    gl_FragData[0] = texture(colortex3, texcoord);
#endif
}
/* DRAWBUFFERS:3 */