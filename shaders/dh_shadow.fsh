#version 130

//uniform sampler2D tex;

//NOPE

//in vec2 texcoord;
//in vec2 lmcoord;

in vec3 normal;

in vec4 color;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"

void main() {
    gl_FragData[0] = color;
    gl_FragData[1] = vec4(normal * 0.5 + 0.5, 1.0);

    //gl_FragDepth = gl_FragCoord.z + 
}
/* DRAWBUFFERS:01 */