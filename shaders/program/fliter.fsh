uniform sampler2D colortex3;

in vec2 texcoord;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"

//#include "/libs/lighting/brdf.glsl"
#include "/libs/lighting/filter.glsl"

void main() {
    /*
    ivec2 texelPosition = ivec2(texcoord * resolution);

    float roughness = 0.999;

    float weight0 = GetPixelPDF(1.0, roughness);
    float totalWeight = weight0;

    vec3 color = texture(colortex3, texcoord).rgb * weight0;

    int radius = 1;

    for(int i = -radius; i <= radius; i += 1) {
        for(int j = -radius; j <= radius; j += 1) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j) * int_Step_Size;
            if()
        }
    }

    color /= totalWeight;
*/
    //gl_FragData[0] = SpatialFliter(colortex3, texcoord, int_Step_Size, 1.0);
    gl_FragData[0] = texture(colortex3, texcoord);
}

/* DRAWBUFFERS:3 */