uniform sampler2D colortex2;

uniform sampler2D depthtex0;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"

#ifdef Relfection_Atrous_Filter_3
    #if defined Relfection_Atrous_Filter_2 && defined Relfection_Atrous_Filter_1
        #define Iteration_Step_Size 9
    #endif
#endif

#include "/libs/lighting/filter_specular.glsl"

void main() {
    vec3 eyeDirection = -normalize(GetViewPosition(texcoord, texture(depthtex0, texcoord).x));

    vec3 texturedNormal = DecodeSpheremap(texture(colortex2, texcoord).xy);

    vec4 specularData = SpatialFilter(eyeDirection, texturedNormal);
    vec3 specular = GammaToLinear(specularData.rgb * MappingToSDR);

    gl_FragData[0] = vec4(specular, specularData.a);
}
/* DRAWBUFFERS:4 */