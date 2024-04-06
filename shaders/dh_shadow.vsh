#version 130

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/lighting/lighting_common.glsl"

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform mat4 dhProjection;

//out vec2 texcoord;
//out vec2 lmcoord;

out vec3 normal;

out vec4 color;
//mat2(gbufferProjectionInverse[2].zw, gbufferProjectionInverse[3].zw) * vec2(depth * 2.0 - 1.0, 1.0);
void main() {
    //gl_Position = ftransform();
    //gl_Position = shadowProjection * shadowModelView * gl_Vertex;
    gl_Position = gl_ModelViewMatrix * gl_Vertex;
    gl_Position.z += -0.05;
    gl_Position = gl_ProjectionMatrix * gl_Position;
    gl_Position.xyz = RemapShadowCoord(gl_Position.xyz);
    //ApplyTAAJitter(gl_Position);

    //texcoord = gl_MultiTexCoord0.xy;
    //lmcoord = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

    normal = gl_Normal;

    color = gl_Color;
}