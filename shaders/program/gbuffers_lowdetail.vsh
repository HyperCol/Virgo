out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 vP;

out vec4 color;

uniform mat4 gbufferProjectionInverse;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"

void main() {
    gl_Position = ftransform();
    ApplyTAAJitter(gl_Position);

    texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

    normal = normalize(gl_NormalMatrix * gl_Normal);

    vP = nvec3(gbufferProjectionInverse * nvec4(gl_Position.xyz / gl_Position.w));

    color = gl_Color;
}