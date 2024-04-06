uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

#if defined(Shadow_Depth_Mul)
const float defaultShadowBias = 0.0005 * Shadow_Depth_Mul;
#else
const float defaultShadowBias = 0.0005;
#endif

float GetShading(in sampler2D tex, in vec3 coord) {
    #if 0
    vec4 gather = textureGather(tex, coord.xy, 0);
    #else
    vec4 gather = vec4(texture(tex, coord.xy + vec2(shadowTexelSize, 0.0)).x, texture(tex, coord.xy - vec2(shadowTexelSize, 0.0)).x, texture(tex, coord.xy + vec2(0.0, shadowTexelSize)).x, texture(tex, coord.xy - vec2(0.0, shadowTexelSize)).x);
    #endif

    float farz = max(max(maxComponent(gather.xyz), gather.w), texture(tex, coord.xy).x);

    return step(coord.z, farz);
    //return step(coord.z, texture(tex, coord.xy).x);
}

float GetSimpleShading(in sampler2D tex, in vec3 coord) {
    if(abs(coord.x - 0.5) >= 0.5 || abs(coord.y - 0.5) >= 0.5) return 1.0;

    vec4 gather = vec4(texture(tex, coord.xy + vec2(shadowTexelSize, 0.0)).x, texture(tex, coord.xy - vec2(shadowTexelSize, 0.0)).x, texture(tex, coord.xy + vec2(0.0, shadowTexelSize)).x, texture(tex, coord.xy - vec2(0.0, shadowTexelSize)).x);
    float blocker = max(max(maxComponent(gather.xyz), gather.w), texture(tex, coord.xy).x);

    return step(coord.z, blocker);
}

vec3 CalculateTintShadow(in vec3 shadowCoord) {
    vec3 shading = vec3(1.0);

    //if(abs(shadowCoord.x) - 0.5 >= 0.5 || abs(shadowCoord.y - 0.5) >= 0.5) return vec3(1.0);

    //if(shadow1 < 0.5) return vec3(0.0); 

#if 0
    shading = vec3(GetShading(shadowtex1, shadowCoord));
#else
if(abs(shadowCoord.x) - 0.5 < 0.5 && abs(shadowCoord.y - 0.5) < 0.5) {
    float shadow1 = GetShading(shadowtex1, shadowCoord);
    float shadow0 = GetShading(shadowtex0, shadowCoord);
    float alpha = texture(shadowcolor0, shadowCoord.xy).a;

    float d0 = texture(shadowtex0, shadowCoord.xy).x;

    //float tileID = round(texture(shadowcolor1, shadowCoord.xy).x * 255.0);
    //float stained = CalculateMask(15.0, tileID) + CalculateMask(145.0, 160.0, tileID) + CalculateMask(F_Water, tileID);

    float stained = step(0.1 + 1e-5, texture(shadowcolor1, shadowCoord.xy).a);

    vec3 albedo = LinearToGamma(texture(shadowcolor0, shadowCoord.xy).rgb);

    vec3 shadowColor = vec3(1.0);

    vec3 sigma_s = vec3(0.2);
    vec3 sigma_a = albedo / mix(maxComponent(albedo), 1.0, 0.3);
         sigma_a = pow(1.0 - sigma_a * 0.999, vec3(2.7));
         sigma_a *= pow(albedo.g / sum3(albedo), 1.0) * 1.0;

    //if(tileID == F_Water) {
    //    float opticalLength = max(0.0, GetShadowLinearDepth(shadowCoord.z) - GetShadowLinearDepth(d0));
    //    shadowColor *= exp(-opticalLength * (sigma_s + sigma_a));
    //} else {
        shadowColor *= mix(vec3(1.0), albedo, vec3(stained)) * (1.0 - alpha);
    //}

    //shadowColor *= max(0.0, texture(shadowcolor1, shadowCoord.xy).g * 2.0 - 1.0);

    shading = mix(shadowColor, vec3(1.0), vec3(shadow0)) * shadow1;
    //shading = shadowColor;
}
#endif

    return shading;
}