#version 330 compatibility

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in float[3] vTileMask;
in float[3] vhandness;

in vec2[3] vtexcoord;
in vec2[3] vmidcoord;
in vec2[3] vlmcoord;

in vec3[3] vnormal;
in vec3[3] vtangent;
in vec3[3] vbinormal;

in vec4[3] vcolor;

in vec3[3] worldPosition;
in vec4[3] vertexPosition;

out float tileMask;
out float FullSolidBlock;
out float TileResolution;

out float handness;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 tangent;
out vec3 binormal;

out vec3 viewDirection;
out vec3 lightDirection;

out vec4 color;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform vec3 cameraPosition;

uniform ivec2 atlasSize;

#if defined(MC_VERSION)
uniform vec3 worldLightVector;
#else
vec3 worldLightVector = normalize(mat3(gbufferModelViewInverse) * shadowLightPosition);
#endif

#define Enabled_Door_Parallax_Fix

#include "/libs/common.glsl"
#include "/libs/mask_check.glsl"

float sdBox2( vec3 p, vec3 b ) {
    vec3 q = abs(p) - b;
    return max(max(q.x,max(q.y,q.z)),0.0);
}

void main() {
    vec3 worldNormal = mat3(gbufferModelViewInverse) * vnormal[0];
    vec3 trainglePosition = (worldPosition[0] + worldPosition[1] + worldPosition[2]) / 3.0 + cameraPosition;
    vec3 blockCenter = floor(trainglePosition - worldNormal * 0.1) + 0.5;

    float vertexMinDistance = minComponent(vec3(length((worldPosition[0] - worldPosition[1])), length((worldPosition[0] - worldPosition[2])), length((worldPosition[1] - worldPosition[2]))));
    float vertexCenterDistance = sdBox2(trainglePosition - blockCenter, vec3(0.0));

    FullSolidBlock = 1.0;
    if(vertexMinDistance < 1.0 - 1e-3 || vertexCenterDistance < 0.5 - 1e-3) FullSolidBlock = 0.0;
    if(vTileMask[0] == Vine || vTileMask[0] == Leaves) FullSolidBlock = 0.0;

    vec2 f_atlasSize = vec2(atlasSize);
    
    vec2 midcoord = vmidcoord[0] * f_atlasSize;
    vec2 coord0 = vtexcoord[0] * f_atlasSize;
    vec2 coord1 = vtexcoord[1] * f_atlasSize;
    vec2 coord2 = vtexcoord[2] * f_atlasSize;

    TileResolution = min(length(coord0 - coord1), min(length(coord0 - coord2), length(coord1 - coord2)));

    vec2 p = midcoord - (coord0 + coord1 + coord2) / 3.0;

    vec3 n = mat3(gbufferModelViewInverse) * ((vnormal[0] + vnormal[1] + vnormal[2]) / 3.0);
    float t = dot(vec3(0.0, 1.0, 0.0), worldNormal);

    vec3 r0 = vec3(1.0, 0.0, 0.0);
    vec3 r1 = vec3(0.0, 1.0, 0.0);
    vec3 r2 = vec3(0.0, 0.0, 1.0);

    #ifdef Enabled_Door_Parallax_Fix
    if(vTileMask[0] == 96.0) {
        if((p.x > 0.0 && p.y > 0.0) || (p.x < 0.0 && p.y < 0.0)) {
            r1 *= (1.0 - step(abs(t), 0.5)) * 2.0 - 1.0;
            r2 *= step(t, 0.5) * 2.0 - 1.0;
        } else {
            r0 *= -((1.0 - step(t, 0.5)) * 2.0 - 1.0);
        }
    }
    #endif

    vec3 rotate = vec3(r0.x, r1.y, r2.z);
    lightDirection = mat3(gbufferModelView) * (worldLightVector);
    
    for(int i = 0; i < 3; i++) {
        gl_Position = vertexPosition[i];

        tileMask    = vTileMask[i];
        texcoord    = vtexcoord[i];
        lmcoord     = vlmcoord[i];
        normal      = vnormal[i];
        tangent     = vtangent[i];
        binormal    = vbinormal[i];
        color       = vcolor[i];
        handness = vhandness[i];

        viewDirection = mat3(gbufferModelView) * (worldPosition[i]);

        EmitVertex();   
    }   EndPrimitive();
}