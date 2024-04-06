#version 130

#include "/libs/setting.glsl"
#include "/libs/common.glsl"

uniform mat4 gbufferModelView;
uniform mat4 dhProjection;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;

out vec4 color;

void main() {
    gl_Position = ftransform();
    //gl_Position = dhProjection * gbufferModelView * gl_Vertex;
    ApplyTAAJitter(gl_Position);

    texcoord = gl_MultiTexCoord0.xy;
    lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

    normal = normalize(gl_NormalMatrix * gl_Normal);

    color = gl_Color;
}