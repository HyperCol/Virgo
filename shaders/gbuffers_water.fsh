#version 130
#define Stage_ID F_Translucent

uniform sampler2D tex;
uniform sampler2D normals;
uniform sampler2D specular;

uniform ivec2 atlasSize;

in float tileID;

in float vertex_emission;
in float forced_emission;

in vec2 texcoord;
in vec2 lmcoord;
//in vec2 midcoord;

in float handness;
in vec3 normal;
in vec3 tangent;
in vec3 binormal;

in vec3 viewPosition;

in vec4 color;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/materialid.glsl"

#define PBR_Format LabPBR   //[LabPBR SEUS_Renewned]

#define LabPBR 0
#define SEUS_Renewned 1

struct GbuffersData {
    float tileID;

    vec2 lightmap;

    vec4 albedo;

    vec3 texturedNormal;
    vec3 geometryNormal;

    float smoothness;
    float metalness;
    float material;
    float emissive;

    float parallaxDepth;

    vec2 ddx;
    vec2 ddy;

    vec4 data0;
    vec4 data1;
    vec4 data2;
    vec4 data3;
};

#if !defined(Stage_ID)
    #define Stage_ID 0.0
#endif

void GbuffersDataPacking(inout GbuffersData data) {
    float adjTileID = data.tileID < 255.5 && data.tileID > 0.5 ? data.tileID / 255.0 : Stage_ID;
    //      adjTileID = data.tileID < 0.5 ? Stage_ID : adjTileID;

    data.data0 = data.albedo;
    data.data1 = vec4(pack2x8(data.smoothness, data.metalness), pack2x8(data.lightmap), pack2x8(data.emissive, adjTileID), 1.0);
    data.data2 = vec4(EncodeSpheremap(data.texturedNormal), EncodeSpheremap(data.geometryNormal));
}

GbuffersData DeafultGbuffersData(in vec2 coord, in vec2 lmcoord, in vec3 normal, in vec4 vertexColor, in vec2 ddx, in vec2 ddy) {
    GbuffersData data;

    data.tileID = round(tileID);

    data.albedo = textureGrad(tex, coord, ddx, ddy) * vertexColor;

    data.texturedNormal = normal;
    data.geometryNormal = normal;

    data.lightmap = lmcoord;

    vec4 raw = textureGrad(specular, coord, ddx, ddy);
    vec4 nomipmap = textureLod(specular, coord, 0);

    data.smoothness = raw.r;
    data.metalness  = raw.g;
    data.material   = nomipmap.b;

    float porosity = step(nomipmap.b * 255.0, 64.5) * nomipmap.b * (255.0 / 64.0);

    //data.smoothness = mix(data.smoothness, 0.999, 1.0 * (1.0 - porosity));

#if PBR_Format == LabPBR
    data.emissive   = max(vertex_emission * 254.0 / 255.0, nomipmap.a);
    data.emissive   = data.emissive > 0.999 ? 0.0 : data.emissive * 255.0 / 254.0;
#elif PBR_Format == SEUS_Renewned
    data.emissive   = vertex_emission > 0.0 ? min(vertex_emission, nomipmap.b) : nomipmap.b;
#else
    data.emissive = 0.0;
#endif

    //data.emissive = max(forced_emission, data.emissive);
    //data.emissive = 1.0;

    data.data0 = vec4(0.0);
    data.data1 = vec4(0.0);
    data.data2 = vec4(0.0);
    data.data3 = vec4(0.0);

    return data;
}

vec3 GetNormalFromHeightWSFaceted(vec3 vertexNormalWS, float heightInMeters)
{
    vec3 positionWS = viewPosition + heightInMeters * vertexNormalWS;
    vec3 bitangentWS = dFdx(positionWS);
    vec3 tangentWS = dFdy(positionWS);

    vec3 normalWS = normalize(cross(bitangentWS, tangentWS));

    // Result contains polygonal faceting
    return normalWS;
}

vec3 GetNormalFromHeightWS(vec3 vertexNormalWS, float heightInMeters)
{
    vec3 normalWS = GetNormalFromHeightWSFaceted(vertexNormalWS, heightInMeters);

    // Replace the faceted normal with the smooth vertex normal
    normalWS = normalWS - GetNormalFromHeightWSFaceted(vertexNormalWS, 0.0) + vertexNormalWS;
    return normalWS;
}

// Calculate and apply the normal in the blend region
vec3 ApplyDerivativeNormal(vec3 vertexNormalWS, float heightInMeters, float blendMask)
{
    vec3 heightNormalWS = GetNormalFromHeightWS(vertexNormalWS, heightInMeters);
    return normalize(normal + heightNormalWS * blendMask);
}

/* DRAWBUFFERS:012 */
#define Parallax_Mapping_Depth 0.25

//#define Parallax_Mapping_Style Voxel_Parallax_Mapping   //[Voxel_Parallax_Mapping Up_Scale_Alike]
//#define Voxel_Parallax_Mapping

//#define Voxel_Parallax_Mapping 1
//#define Up_Scale_Alike 2

float GradHeightmap(in vec2 coord, in vec2 ddx, in vec2 ddy) {
    float height = textureGrad(normals, coord, ddx, ddy).a - 1.0;

    return height * Parallax_Mapping_Depth;
}

float LowDetailDepthMap(in vec2 coord, in vec2 ddx, in vec2 ddy) {
    float depth = GradHeightmap(coord, ddx, ddy);

#if Parallax_Mapping_Depth_Low_Detail == OFF
    return depth;
#else
    #if Parallax_Mapping_Depth_Low_Detail == Default
        #if Parallax_Mapping_Quality < Medium
            float invsteps = 1.0 / 16.0;
        #elif Parallax_Mapping_Quality == Medium
            float invsteps = 1.0 / 32.0;
        #elif Parallax_Mapping_Quality > High
            float invsteps = 1.0 / 128.0;
        #else
            float invsteps = 1.0 / 64.0;
        #endif
    #else
        float invsteps = 4.0 / float(Parallax_Mapping_Depth_Low_Detail);
    #endif

    depth = depth / Parallax_Mapping_Depth + 1.0;
    depth = round(depth / invsteps) * invsteps;
    depth = (depth - 1.0) * Parallax_Mapping_Depth;

    return depth;
#endif
}

vec2 OffsetCoord(in vec2 coord, in vec2 offset, in vec2 size){
	vec2 offsetCoord = coord + mod(offset.xy, size);

	vec2 minCoord = vec2(coord.x - mod(coord.x, size.x), coord.y - mod(coord.y, size.y));
	vec2 maxCoord = minCoord + size;

    if(offsetCoord.x < minCoord.x){
        offsetCoord.x += size.x;
    }else if(maxCoord.x < offsetCoord.x){
        offsetCoord.x -= size.x;
    }

    if(offsetCoord.y < minCoord.y){
        offsetCoord.y += size.y;
    }else if(maxCoord.y < offsetCoord.y){
        offsetCoord.y -= size.y;
    }

	return offsetCoord;
}

vec2 normalFromHeight(in vec2 coord, float stepSize, in vec2 tileSize, in vec2 ddx, in vec2 ddy) {
    vec2 e = vec2(stepSize, 0.0);

    float px1 = LowDetailDepthMap(OffsetCoord(coord, -e.xy * tileSize, tileSize), ddx, ddy);
    float px2 = LowDetailDepthMap(OffsetCoord(coord,  e.xy * tileSize, tileSize), ddx, ddy);
    float py1 = LowDetailDepthMap(OffsetCoord(coord, -e.yx * tileSize, tileSize), ddx, ddy);
    float py2 = LowDetailDepthMap(OffsetCoord(coord,  e.yx * tileSize, tileSize), ddx, ddy);

    return vec2(px1 - px2, py1 - py2);
}

#ifdef Parallax_Mapping
vec2 CalculateParallaxMapping(vec2 coord, in vec2 ddx, in vec2 ddy, in float tileResolution, in vec3 viewPosition, inout float parallaxDepth, inout vec3 parallaxNormal) {
    float parallaxOffset = 0.0;
    vec3 parallaxTexNormal = vec3(0.0, 0.0, 1.0);

    mat3 tbn = mat3(tangent, binormal, normal);

    #if Parallax_Mapping_Quality < Medium
    int steps = 16;
    #elif Parallax_Mapping_Quality == Medium
    int steps = 32;
    #elif Parallax_Mapping_Quality > High
    int steps = 128;
    #else
    int steps = 64;
    #endif
    
    float invsteps = 1.0 / float(steps);

    vec2 parallaxCoord = coord;

    float texelHeight = LowDetailDepthMap(coord, ddx, ddy);
    //      texelHeight = floor((texelHeight / Parallax_Mapping_Depth + 1.0) / invsteps) * invsteps;
    //      texelHeight = (texelHeight - 1.0) * Parallax_Mapping_Depth;

    const float scaleLimit = 8.0;

    float level = log2(max(dot(ddx, ddx), dot(ddy, ddy))) * 0.5;

    float scale = tileResolution / (1.0 + max(0.0, level));
    float imagePixel = min(resolution.x, resolution.y) / sqrt(viewPosition.z * viewPosition.z);

    if (texelHeight < 0.0 && scale > scaleLimit) {
        vec2 atlasTexelSize = 1.0 / vec2(atlasSize);
        vec3 rayDirection = normalize(viewPosition * tbn);
    
#ifdef Voxel_Parallax_Mapping
        rayDirection.xy *= tileResolution;
        rayDirection *= Parallax_Mapping_Depth * invsteps;

        vec2 sampleTexel = floor(coord * vec2(atlasSize));
        vec2 basicTexel = floor(sampleTexel / tileResolution) * tileResolution;
        vec2 stepLength = abs(1.0 / rayDirection.xy);
        vec2 dirSigned = sign(rayDirection.xy);
        vec2 nextLength = (dirSigned * (0.5 - fract(coord * vec2(atlasSize))) + 0.5) * stepLength;
        vec2 nextPixel = vec2(0.0);

        float previousStepHeight = 0.0;

        for(int i = 0; i < steps; i++) {
            float rayLength = min(nextLength.x, nextLength.y);
            float rayHeight = rayLength * -rayDirection.z;
                  rayHeight = 0.0 - rayHeight;

            previousStepHeight = parallaxOffset;

            if(rayHeight < texelHeight) {
                parallaxOffset = texelHeight;
                break;
            }

            nextPixel = step(nextLength, vec2(rayLength));
            nextLength += nextPixel * stepLength;
            sampleTexel += nextPixel * dirSigned;
            sampleTexel = basicTexel + mod(sampleTexel, vec2(tileResolution));

            texelHeight = LowDetailDepthMap((sampleTexel + 0.5) * atlasTexelSize, ddx, ddy);
            //texelHeight = floor((texelHeight / Parallax_Mapping_Depth + 1.0) / invsteps) * invsteps;
            //texelHeight = (texelHeight - 1.0) * Parallax_Mapping_Depth;

            parallaxOffset = rayHeight;

            if(rayHeight < texelHeight) {
                parallaxTexNormal = vec3(-nextPixel * dirSigned, 0.0);
                break;
            }
        }
        
        //parallaxTexNormal.z += max(0.0, 200.0 / imagePixel - 1.0);
        //parallaxTexNormal.z = max(parallaxTexNormal.z, 1.0 - (scale - scaleLimit - 1.0));

        vec2 d1 = abs(ddx);
        vec2 d2 = abs(ddy);

        parallaxTexNormal.z += max(0.0, max(max(d1.x, d2.x) * float(atlasSize.x), max(d1.y, d2.y) * float(atlasSize.y)) - 1.0);
        parallaxNormal = normalize(tbn * parallaxTexNormal);

        parallaxCoord = (sampleTexel + vec2(0.5)) * atlasTexelSize;
        parallaxDepth = parallaxOffset;
#else
        float stepLength = invsteps * Parallax_Mapping_Depth;

        vec2 texelIndex = atlasTexelSize * tileResolution;
        vec2 parallaxDelta = rayDirection.xy / -rayDirection.z * texelIndex * stepLength;

        for(int i = 0; i < steps; i++) {
            parallaxOffset -= stepLength;
            if(parallaxOffset < texelHeight) break;

            parallaxCoord = OffsetCoord(parallaxCoord, parallaxDelta, texelIndex);
            texelHeight = GradHeightmap(parallaxCoord, ddx, ddy);
        }

        parallaxOffset += stepLength;
        parallaxDepth = parallaxOffset;

        if(parallaxOffset < texelHeight) {
            parallaxNormal = normalize(tbn * vec3(normalFromHeight(parallaxCoord, 0.5 / tileResolution, texelIndex, ddx, ddy), 0.0));

        }
#endif
    }

    return parallaxCoord;
}
#endif

uniform int renderStage;

//#define MC_RENDER_STAGE_NONE 0
//#define MC_RENDER_STAGE_TERRAIN_SOLID 2
//#define MC_RENDER_STAGE_TERRAIN_CUTOUT_MIPPED 3
//#define MC_RENDER_STAGE_TERRAIN_CUTOUT 3

void GetTranslucentMaterials(inout GbuffersData data) {
    //data.

    //bbol data1ISWork = 

    if(data.tileID == F_Water) {
        data.albedo = vec4(color.rgb, 1.0);
        data.smoothness = 0.999;
        data.metalness  = 0.02;
    } else if(data.tileID == F_Ice) {
        data.smoothness = 0.999;
        data.metalness  = 0.0179;
    } else {
        data.smoothness = 0.95;
        data.metalness  = 0.02;        
    }
    //IOR=(F(0)-1+2*sqrt[F(0)])/(1-F(0))
    //F(0)=([1-IOR2]/[1+IOR2])^2

    data.emissive = BoolMask(F_Nether_Portal_Start, F_Nether_Portal_End, data.tileID) ? 11.0 / 15.0 : data.emissive;
}

void main() {
    vec2 ddx = dFdx(texcoord);
    vec2 ddy = dFdy(texcoord);

    vec3 parallaxNormal = normal;
    float parallaxDepth = 0.0;
    vec2 coord = texcoord;

#ifdef Parallax_Mapping
    coord = CalculateParallaxMapping(texcoord, ddx, ddy, 16.0, viewPosition, parallaxDepth, parallaxNormal);
#endif

    GbuffersData data = DeafultGbuffersData(coord, lmcoord, normal, color, ddx, ddy);
    GetTranslucentMaterials(data);

    data.geometryNormal = normal;

    mat3 tbn = mat3(tangent, binormal, parallaxNormal);
    //data.texturedNormal = normalize(tbn * vec3(0.0, 0.0, 1.0));
    data.texturedNormal = normalize(tbn * BentNormal(textureGrad(normals, coord, ddx, ddy).xy));

    //#if defined(MC_RENDER_STAGE_TERRAIN_CUTOUT) && defined(MC_RENDER_STAGE_TERRAIN_CUTOUT_MIPPED)
    //if(renderStage == MC_RENDER_STAGE_TERRAIN_CUTOUT || renderStage == MC_RENDER_STAGE_TERRAIN_CUTOUT_MIPPED || renderStage == MC_RENDER_STAGE_TERRAIN_SOLID) data.albedo = vec4(1.0, 0.0, 0.0, 1.0);
    //#endif

    GbuffersDataPacking(data);

    //data.albedo.a = 1.0;

    //data.data0.rgb = mat3(gbufferModelViewInverse) * direction;
    //data.data0.rgb = saturate(dot(direction, data.texturedNormal)) * vec3(1.0);
    //data.data0.rgb = saturate(data.data0.rgb);

#if 1
    vec3 p = viewPosition + normal - normal * lmcoord.x;

    vec3 direction = normalize(cross(dFdx(p), dFdy(p)));
         direction = normalize(direction - normal * 0.95);
         direction = normalize(direction + data.texturedNormal);

    data.data1.a = max(0.0, dot(direction, data.texturedNormal));
    //data.data1.z = max(0.0, dot(dnormal, normal));
#else
    data.data1.a = 1.0;
#endif

    //vec2 d1 = abs(ddx);
    //vec2 d2 = abs(ddy);
    //data.data0.rgb = vec3(max(0.0, max(max(d1.x, d2.x) * float(atlasSize.x), max(d1.y, d2.y) * float(atlasSize.y)) - 1.0));

    //if(!gl_FrontFacing) discard;

    gl_FragData[0] = data.data0;
    gl_FragData[1] = data.data1;
    gl_FragData[2] = data.data2;
    
    gl_FragDepth = LinearToExpDepth(ExpToLinearDepth(gl_FragCoord.z) + CalculateMask(F_Water, data.tileID) / abs(dot(normalize(viewPosition), normal)) * 2e-3);
}
/* DRAWBUFFERS:012 */