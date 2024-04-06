#version 330 compatibility

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec2[] vtexcoord;
in vec4[] fragCoord;

out vec2 texcoord;

void main() {
    for(int i = 0; i < 3; i++) {
        gl_Position = fragCoord[i];

        gl_Position.xy = gl_Position.xy * 0.5 + 0.5;
        gl_Position.xy *= 0.25;
        gl_Position.xy = gl_Position.xy * 2.0 - 1.0;

        texcoord = vtexcoord[i];

        EmitVertex();
    }   EndPrimitive();
}