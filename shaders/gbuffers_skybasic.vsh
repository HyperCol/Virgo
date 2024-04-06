#version 130

out vec2 texcoord;

out vec4 color;

void main() {
    gl_Position = ftransform();

    color = gl_Color;

    texcoord = gl_MultiTexCoord0.xy;
}