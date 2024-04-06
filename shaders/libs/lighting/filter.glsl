#extension GL_ARB_gpu_shader5 : enable


uniform sampler2D colortex2;

uniform sampler2D colortex10;

uniform sampler2D depthtex0;

uniform int hideGUI;

ivec2 iresolution = ivec2(resolution) - 1;

bool Inscreen(in ivec2 texelPosition) {
    return texelPosition.x > 1 && texelPosition.x < iresolution.x && texelPosition.y > 1 && texelPosition.y < iresolution.y;
}

#include "/libs/lighting/brdf.glsl"

vec4 SpatialFliter(in sampler2D tex, in vec2 coord, in int stepSize, in float lum) {
    ivec2 texelPosition = ivec2(coord * resolution);

    vec4 sample0 = texelFetch(tex, texelPosition, 0);
    float variance = sample0.a;

    float   centerDepth     = texelFetch(depthtex0, texelPosition, 0).x;
    vec3    centerNormal    = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).xy);
    vec3    geometryNormal    = DecodeSpheremap(texelFetch(colortex2, texelPosition, 0).zw);
    vec3    viewPosition    = GetViewPosition(texcoord, centerDepth);
    float   luminance       = sum3(sample0.rgb);

    mat3 imv = mat3(gbufferModelViewInverse);
    vec3 worldNormal = imv * centerNormal;
    vec3 worldPosition = imv * viewPosition;

    vec3 v = normalize(viewPosition);
    vec3 e = -v;
    vec3 n = centerNormal;//CalculateVisibleNormals(e, centerNormal);

    float historyLength = texelFetch(colortex10, texelPosition, 0).a * 255.0;
    float accuFrameWeight = min(1.0, historyLength / float(Diffuse_Accumulation_Frame));

    float roughness = 0.2;
    float sigma2 = mix(0.7, 0.3, accuFrameWeight);
    float sigma3 = 0.999;

    float depthWeightPhi = 100.0 * mix(1.0, 0.25, step(centerDepth, 0.7));
    float lumiannceWeightPhi = clamp(0.1 / variance * accuFrameWeight, 1.0, 30.0) * pow(VoxelHDR, 1.0 / 2.2);
    
    //variance *= 0.3;
    //float lWeightPhi = 30.0 * rescale(accuFrameWeight, -0.2, 1.0) * pow(VoxelHDR, 1.0 / 2.2);

    //variance *= 0.3;
    //float lWeightPhi = 40.0 * rescale(accuFrameWeight, -0.2, 1.0) * pow(VoxelHDR, 1.0 / 2.2);

    //variance *= 2.0;
    //float lWeightPhi = clamp(0.5 / variance * accuFrameWeight, 10.0, 100.0) * pow(VoxelHDR, 1.0 / 2.2);

    variance *= 0.2;
    //float lumTermPhi = min(0.01 / variance, 100.0) * rescale(accuFrameWeight, -0.2, 1.0) * pow(VoxelHDR, 1.0 / 2.2);
    float lumTermPhi = min(0.01 / variance, 100.0) * rescale(accuFrameWeight, -0.2, 1.0) * pow(VoxelHDR, 1.0 / 2.2);

    //float lWeightPhi = (0.05 / max(1e-3, variance)) * rescale(accuFrameWeight, -0.01, 1.0) * pow(VoxelHDR, 1.0 / 2.2);
    //float lumiannceWeightPhi = 50.0 * pow(VoxelHDR, 1.0 / 2.2);

    //float weight0 = GetPixelPDF(vec3(0.0), n, n, sigma2) * GetPixelPDF(e, normalize(reflect(v, n)), n, sigma3);
    float weight0 = GetPixelPDF(vec3(0.0), n, n, sigma2);
          weight0 *= GetPixelPDF(1.0, sigma3);
          //weight0 *= GetPixelPDF(vec3(0.0), v, v, sigma3);
          //weight0 = 1.0;

    float totalWeight = weight0;
    vec4 result = sample0 * weight0;

    int radius = 1;

    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j) * stepSize;
            if((i == 0 && j == 0) || !Inscreen(sampleTexel)) continue;

            float sampleDepth = texelFetch(depthtex0, sampleTexel, 0).x;
            vec3 samplePosition = GetViewPosition(vec2(sampleTexel) * texelSize, sampleDepth);
            vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, sampleTexel, 0).xy);
            vec3 sampleVisibleNormal = DecodeSpheremap(texelFetch(colortex2, sampleTexel, 0).zw);//CalculateVisibleNormals(e, sampleNormal);
            vec4 sampleColor = texelFetch(tex, sampleTexel, 0);

            vec3 halfVector = samplePosition - viewPosition;

            //float weight = GetPixelPDF(e, normalize(reflect(normalize(samplePosition), sampleNormal)), n, roughness);
            //float weight = GetPixelPDF(vec3(0.0), sampleNormal, n, sigma2) * GetPixelPDF(e, normalize(reflect(normalize(samplePosition), sampleNormal)), n, sigma3);
            //float weight = GetPixelPDF(sampleNormal, normalize(halfVector), n, sigma3);// * GetPixelPDF(vec3(0.0), sampleNormal, n, sigma2);

            //weight *= saturate(exp(-depthWeightPhi * abs(dot(halfVector, geometryNormal)) / length(samplePosition)));
            //weight *= sampleDepth >= 1.0 ? 0.0 : 1.0;
            
            float sampleLuminance = sum3(sampleColor.rgb);
            //weight *= min(1.0, exp(-abs(luminance - sampleLuminance) * lumiannceWeightPhi));

            float weight = saturate(exp(-depthWeightPhi * abs(dot(halfVector, centerNormal)) / length(samplePosition)));

            //weight *= GetPixelPDF(vec3(0.0), sampleNormal, n, sigma2) * GetPixelPDF(e, normalize(reflect(normalize(samplePosition), sampleNormal)), n, sigma3);
            weight *= GetPixelPDF(vec3(0.0), sampleNormal, n, sigma2);
            //weight *= GetPixelPDF(v, halfVector, v, sigma3);
            //weight *= GetPixelPDF(e, halfVector, e, sigma3);

            //float ndoth = max(0.0, dot(normalize(v+normalize(samplePosition)), v));
            //weight *= DistributionTerm(ndoth, sigma3) * ndoth;

            //weight *= saturate(exp(-0.01 * length(vec2(i, j) * stepSize * length(viewPosition))));
            //weight *= GetPixelPDF(-normalize(samplePosition), vec3(0.0), e, sigma3);
            //weight *= min(1.0, exp(-abs(luminance - sampleLuminance) * lumiannceWeightPhi));

            float visiblity = max(0.0, dot(normalize(samplePosition), v)) / max(1.0, pow2(length(halfVector)));
            weight *= visiblity > 0.0 ? GetPixelPDF(visiblity, sigma3) : 0.0;

            //weight *= min(1.0, exp(-max(0.0, abs(luminance - sampleLuminance) - variance) * lWeightPhi));

            //float luminanceDiff = saturate((abs(sampleLuminance - luminance) - variance) / variance * 100.0);
            //weight *= step(luminanceDiff, variance);
            //weight *= min(1.0, exp(-max(0.0, abs(luminance - sampleLuminance) - variance) * lWeightPhi));
            //weight *= min(1.0, exp(-max(0.0, (abs(sampleLuminance - luminance) - variance) / variance * lumTermPhi)));

            weight *= min(1.0, exp(-max(0.0, (abs(sampleLuminance - luminance) - variance) * lumTermPhi)));
            
            //weight = 1.0;

            result += sampleColor * weight;
            totalWeight += weight;
        }
    }        

    result /= totalWeight;
    //result.a = variance;

    return result;
}