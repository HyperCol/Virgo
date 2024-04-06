#version 130

uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjectionInverse;

uniform mat4 gbufferModelViewInverse;

uniform vec3 cameraPosition;

uniform ivec2 atlasSize;

uniform int entityId;
uniform int blockEntityId;

in vec3 mc_Entity;
in vec2 mc_midTexCoord;
in vec3 at_midBlock;

out int vertexID;

out float isBlockEnentites;

out float vtileID;

out vec2 vtexcoord;
out vec2 vmidcoord;
out vec2 vlmcoord;

out vec4 shadowCoord;
out vec3 worldPosition;
out vec3 centerPosition;
out vec3 vworldNormal;

out vec4 vcolor;

//#include "/libs/common.glsl"
//#include "/libs/function.glsl"

void main() {
    isBlockEnentites = 0.0;

    vtileID = mc_Entity.x;
    
    gl_Position = ftransform();
    shadowCoord = gl_Position;

    gl_Position = shadowModelViewInverse * shadowProjectionInverse * gl_Position;
    worldPosition = gl_Position.xyz;
    centerPosition = at_midBlock * (1.0 - isBlockEnentites);

    //gl_Position = shadowProjection * shadowModelView * gl_Position;

    vertexID = gl_VertexID % 4;

    vworldNormal = gl_Normal;

    vcolor = gl_Color;

    vtexcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vlmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    vmidcoord = mc_midTexCoord;
}