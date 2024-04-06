#version 130

#extension GL_ARB_gpu_shader5 : enable

out vec2 texcoord;

uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

out mat4 previousModelViewInverse;
out mat4 previousProjectionInverse;

void main() {
    gl_Position = ftransform();

    previousModelViewInverse = inverse(gbufferPreviousModelView);
    previousProjectionInverse = inverse(gbufferPreviousProjection);

    texcoord = gl_MultiTexCoord0.st;
}