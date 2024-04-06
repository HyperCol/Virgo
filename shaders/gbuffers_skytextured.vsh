#version 130

out vec2 texcoord;

out vec3 normal;

out vec4 color;

void main() {
    gl_Position = ftransform();

    color = gl_Color;

    normal = normalize(gl_NormalMatrix * gl_Normal);

    texcoord = gl_MultiTexCoord0.xy;
}