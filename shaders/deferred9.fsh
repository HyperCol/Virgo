#version 130

uniform sampler2D colortex3;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/lighting/filter.glsl"

void main() {
#ifdef GI_A_Trous_1
    gl_FragData[0] = SpatialFliter(colortex3, texcoord, 1, 1.0);
#else
    gl_FragData[0] = texture(colortex3, texcoord);
#endif
}
/* DRAWBUFFERS:3 */