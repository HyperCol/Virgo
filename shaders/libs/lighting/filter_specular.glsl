uniform sampler2D colortex4;

#include "/libs/lighting/brdf.glsl"

vec4 SpatialFilter(in vec3 e, in vec3 n) {
    ivec2 texelPosition = ivec2(gl_FragCoord.xy);

    float roughness = texelFetch(colortex4, texelPosition, 0).a;

#if !defined Iteration_Step_Size
    return vec4(LinearToGamma(texture(colortex4, texcoord).rgb) * MappingToHDR, roughness);
#else
    //74
    vec2 frac = fract(gl_FragCoord.xy) * texelSize;

    vec3 color = vec3(0.0);
    float totalWeight = 0.0;

    const int radius = 1;

    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            ivec2 sampleTexel = texelPosition + ivec2(i, j) * Iteration_Step_Size;
            vec2 screenSpace = texcoord + vec2(i * Iteration_Step_Size, j * Iteration_Step_Size) * texelSize;

            float sampleDepth = texelFetch(depthtex0, sampleTexel, 0).x;
            if(sampleDepth >= 1.0 || any(lessThanEqual(screenSpace, vec2(0.0))) || any(greaterThanEqual(screenSpace, vec2(1.0)))) continue;

            vec3 sampleNormal = DecodeSpheremap(texelFetch(colortex2, sampleTexel, 0).xy);
            vec3 sampleViewPosition = GetViewPosition(screenSpace, sampleDepth);

            float sampleRoughness = texelFetch(colortex4, sampleTexel, 0).a;

            vec3 sampleViewDir = normalize(sampleViewPosition);
            vec3 rayDirection = normalize(reflect(sampleViewDir, sampleNormal));

            float ndoth = saturate(dot(n, normalize(rayDirection + e)));

            float weight = DistributionTerm(ndoth, roughness) * ndoth;
            if(weight <= 0.0) continue;
            //float weight = GetPixelPDF(e, rayDirection, n, 0.0001);
                  weight *= step(abs(sampleRoughness - roughness), 0.02);

            vec3 sampleColor = LinearToGamma(texelFetch(colortex4, sampleTexel, 0).rgb) * MappingToHDR;

            color += sampleColor * weight;
            totalWeight += weight;
        }
    }

    color /= step(totalWeight, 0.0) + totalWeight;

    return vec4(color, roughness);
#endif
}