#version 330 compatibility

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;

uniform vec3 cameraPosition;

uniform int blockEntityId;

uniform ivec2 atlasSize;

in int[3] vertexID;
in float[] isBlockEnentites;

in float[3] vtileID;

in vec2[3] vtexcoord;
in vec2[3] vmidcoord;
in vec2[3] vlmcoord;
in vec2[3] texelcoord;

in vec4[3] vcolor;

in vec4[3] shadowCoord;
in vec3[3] worldPosition;
in vec3[3] centerPosition;
in vec3[3] vworldNormal;

out float tileResolution;

out float tileID;

out float textureTile;

out vec2 tilecoord;
out vec2 texcoord;
out vec2 midcoord;

out vec3 worldNormal;

out vec4 color;

//#define Shadow_Map

float GetFragMinDistance(float p0, float p1, float p2) {
    return min(abs(p0), min(abs(p1), abs(p2)));
}

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/lighting/lighting_common.glsl"

void main() {
    const vec2[4] offset = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(1, 1),vec2(-1, 1));

    float tileID_0 = vtileID[0];

    textureTile = 0.0;

    for(int i = 0; i < 3; i++) {
        gl_Position = shadowCoord[i];
        gl_Position.xyz = RemapShadowCoord(gl_Position.xyz, worldPosition[i]);

        textureTile = 0.0;
        tileID      = tileID_0;
        tileResolution = 0.0;

        tilecoord   = gl_Position.xy * 0.5 + 0.5;
        texcoord    = vtexcoord[i];
        midcoord    = vec2(-1.0);

        color       = vcolor[i];

        worldNormal = vworldNormal[i];

        EmitVertex();
    } EndPrimitive();
}
