#version 130

#extension GL_ARB_gpu_shader5 : enable

uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

out vec2 texcoord;

out mat4 previousModelViewInverse;
out mat4 previousProjectionInverse;

void main() {
    previousModelViewInverse = inverse(gbufferPreviousModelView);
    previousProjectionInverse = inverse(gbufferPreviousProjection);

    texcoord = gl_MultiTexCoord0.xy;

    gl_Position = ftransform();
}