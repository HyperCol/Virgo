uniform sampler2D tex;
uniform sampler2D normals;
uniform sampler2D specular;

uniform float alphaTestRef = 0.1;

in vec2 texcoord;
in vec2 lmcoord;

in vec3 normal;

in vec3 vP;

in vec4 color;

uniform mat4 gbufferProjection;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"

void main() {
    vec2 encodeNormal = EncodeSpheremap(normal);

    vec4 fragColor = color * texture(tex, texcoord);

    if(fragColor.a < alphaTestRef) discard;

    gl_FragData[0] = fragColor;
    gl_FragData[1] = vec4(0.0, pack2x8(lmcoord), 0.0, 1.0);
    gl_FragData[2] = vec4(encodeNormal, encodeNormal);

    //gl_FragDepth = 0.05;//nvec3(gbufferProjection * nvec4(vP)).z * 0.5 + 0.5;
}
/* DRAWBUFFERS:012 */