#version 130

uniform sampler2D tex;

in vec2 texcoord;

in vec4 color;

void main() {
    gl_FragData[0] = vec4(0.0, 0.0, 0.0, 1.0);
}
/* DRAWBUFFERS:0 */