#version 130

//uniform sampler2D tex;

//in vec2 texcoord;

in vec3 normal;

in vec4 color;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"

void main() {
    vec2 encodeNormal = EncodeSpheremap(normal);

    gl_FragData[0] = color;// * texture(tex, texcoord);
    gl_FragData[1] = vec4(0.0, 0.0, 0.0, 1.0);
    gl_FragData[2] = vec4(encodeNormal, encodeNormal);
}
/* DRAWBUFFERS:012 */