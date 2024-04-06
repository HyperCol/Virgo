out vec2 texcoord;

void main() {
    gl_Position = ftransform();

    #if defined(Stage_Scale)
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * Stage_Scale * 2.0 - 1.0;
    #endif

    texcoord = gl_MultiTexCoord0.st;
}