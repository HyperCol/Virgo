#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/materialid.glsl"

uniform int blockEntityId;
uniform ivec2 atlasSize;

#if MC_VERSION <= 11202 || !defined(MC_VERSION)
//enabled geometry shader
/*
    #define tileID vtileID
    #define isBlockEntity visBlockEntity
    #define vertex_emission vvertex_emission
    #define forced_emission vforced_emission
    #define handness vhandness
    #define prepare_material vprepare_material

    #define texcoord vtexcoord
    #define lmcoord vlmcoord
    //#define lmcoord vmidcoord

    #define normal vnormal
    #define tangent vtangent
    #define binormal vbinormal
    #define worldPosition vworldPosition

    #define color vcolor
    */
#endif

//vertex
in vec2 mc_midTexCoord;
in vec3 mc_Entity;
in vec3 at_midBlock;
in vec4 at_tangent;

out float tileID;
out float isBlockEntity;
out float vertex_emission;
out float forced_emission;

out vec2 texcoord;
out vec2 lmcoord;
//out vec2 midcoord;

out float handness;
out vec3 normal;
out vec3 tangent;
out vec3 binormal;

out vec3 viewPosition;

out vec4 color;

void main() {
    gl_Position = ftransform();
    ApplyTAAJitter(gl_Position);

    tileID = max(float(blockEntityId), mc_Entity.x);

    isBlockEntity = step(64.5, maxComponent(abs(at_midBlock)));

    vertex_emission = 0.0;
    forced_emission = 0.0;

    color = gl_Color;

    handness = at_tangent.w;
    normal = normalize(gl_NormalMatrix * gl_Normal);
    tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
    binormal = normalize(cross(tangent, normal) * handness);

    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;
    //midcoord = mc_midTexCoord;

    viewPosition = nvec3(gbufferProjectionInverse * nvec4(gl_Position.xyz / gl_Position.w));
}