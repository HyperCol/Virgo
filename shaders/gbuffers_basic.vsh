#version 130

#include "/libs/setting.glsl"
#include "/libs/common.glsl"

//out vec2 texcoord;

out vec3 normal;

out vec4 color;

void main() {
    gl_Position = ftransform();
    ApplyTAAJitter(gl_Position);

    //texcoord = gl_MultiTexCoord0.xy;

    //normal = normalize(gl_NormalMatrix * gl_Normal);

    color = gl_Color;
}