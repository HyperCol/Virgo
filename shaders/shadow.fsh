#version 130

uniform sampler2D tex;
uniform sampler2D specular;

uniform mat4 shadowModelView;

uniform int blockEntityId;

uniform ivec2 atlasSize;

in float tileResolution;

in float tileID;
in float textureTile;

in vec2 tilecoord;
in vec2 texcoord;
in vec2 midcoord;

in vec3 worldNormal;

in vec4 color;

#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/materialid.glsl"

void main() {
    float v_tileID = round(tileID);

    float tile = round(textureTile);
    float rtileResolution = round(tileResolution);
    vec2 fatlasSize = vec2(atlasSize);

    vec4 textureColor = texture(tex, texcoord) * color;
#if 1
    //if(tilecoord.x < 0.5 || tilecoord.y > 0.5) discard;
    //if(!gl_FrontFacing) discard;

    float tintShadow = CalculateMask(15.0, v_tileID) + CalculateMask(145.0, 160.0, v_tileID);

    vec4 color = textureColor;
    if(color.a < 0.1) discard;

    //if(v_tileID == F_Water) color = vec4(color.rgb, 0.05);
    if(v_tileID == F_Water) color = vec4(vec3(0.05), 0.05);

    gl_FragData[0] = color;
    gl_FragData[1] = vec4((worldNormal) * 0.5 + 0.5, 0.1 + 0.9 * step(color.a, 0.999) * (1.0 - CalculateMask(F_Water, tileID)));

    gl_FragDepth = gl_FragCoord.z + (gl_FrontFacing ? 0.0 : 1e-5);
#else
    if(tile == 1.0) {
        if(tilecoord.x > 0.5 || v_tileID == 255.0) discard;

        vec2 packalpha = vec2(0.0, min(1.0, log2(tileResolution) / 15.0));

        vec2 texelIndex = floor(midcoord * fatlasSize - rtileResolution * 0.5);
             texelIndex /= tileResolution;
             texelIndex = floor(texelIndex + 0.5) / 255.0;

             //texelIndex = log2(texelIndex) / 255.0;

             //texelIndex = exp2(texelIndex * 255.0) * tileResolution;
             //texelIndex /= vec2(atlasSize);

        //vec2 texelIndex = floor(midcoord * vec2(atlasSize) / tileResolution);
        //     texelIndex = log2(texelIndex) / 255.0;
        //     texelIndex = exp2(texelIndex * 255.0) * tileResolution / vec2(atlasSize);

        /*
        vec2 texelIndex = floor(midcoord * vec2(atlasSize) / tileResolution) + 1.0;
             texelIndex = log2(texelIndex) / 255.0;

        vec2 coord = exp2(texelIndex * 255.0) / vec2(atlasSize) * tileResolution;
             coord = floor(coord * 255.0) / 255.0;
             */

        vec3 voxelColor = color.rgb;
        if(BoolMaskBetween(95.0, 99.0, tileID) || tileID == 1.0) {
            voxelColor = textureColor.rgb;
        }

        gl_FragData[0] = vec4(voxelColor, 1.0);
        gl_FragData[1] = vec4(texelIndex, v_tileID / 255.0, pack2x4(packalpha));
    } else if(tile == 2.0) {
        if(tilecoord.x < 0.5 || tilecoord.y > 0.5) discard;

        float tintShadow = CalculateMask(15.0, v_tileID) + CalculateMask(145.0, 160.0, v_tileID);

        vec4 color = textureColor;
        if(color.a < 0.1) discard;

        if(v_tileID == F_Water) color = vec4(color.rgb, 0.05);

        gl_FragData[0] = color;
        gl_FragData[1] = vec4(v_tileID / 255.0, 0.0, 0.0, 1.0);      
    } else if(tile == 3.0) {
        //if(v_tileID > 79.0 || (v_tileID == 18.0 || v_tileID == 14.0 || v_tileID == 15.0)) discard;
        if(v_tileID > 79.0 && v_tileID != 245.0) discard;

        gl_FragData[0] = vec4(vec3(0.0), 1.0);
        gl_FragData[1] = vec4(0.0, 0.0, v_tileID / 255.0, 1.0);
    }
#endif
}
/* DRAWBUFFERS:01 */