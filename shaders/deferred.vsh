#version 130

//#define Stage_Scale 0.25
//#define Vertex_Lightcolor
//#include "/program/deferred_vertex.glsl"

#define texcoord vtexcoord

out vec2 texcoord;

out vec4 fragCoord;

void main() {
    texcoord = gl_MultiTexCoord0.st;

    gl_Position = ftransform();

    fragCoord = gl_Position;
}