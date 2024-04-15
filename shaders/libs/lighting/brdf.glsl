float SchlickFresnel(in float cosTheta){
	return pow5(1.0 - cosTheta);
}

vec3 SchlickFresnel(in vec3 F0, in float cosTheta){
	return F0 + (1.0 - F0) * SchlickFresnel(cosTheta);
}

float SchlickFresnel(in float F0, in float F90, in float cosTheta){
	return F0 + (F90 - F0) * SchlickFresnel(cosTheta);
}

float DistributionTerm( float ndoth, float roughness ) {
    float a2 = max(1e-7, roughness * roughness);
	float d	 = ( ndoth * a2 - ndoth ) * ndoth + 1.0;
	return a2 / (d * d * Pi);
}

float SmithGGX(float cosTheta, float a2){
    return cosTheta / (cosTheta * (1.0 - a2) + a2);
    //float c2 = cosTheta * cosTheta;
    //return (2.0 * cosTheta) / (cosTheta + sqrt(a2 + (1.0 - a2) * c2));
}

float VisibilityTerm(float cosTheta1, float cosTheta2, float roughness){
    float c = 4.0 * cosTheta1 * cosTheta2 + 1e-5;
    //mark
    float a2 = max(1e-7, roughness * roughness);

    return SmithGGX(cosTheta1, a2) * SmithGGX(cosTheta2, a2) / c;
    
    /*
    float a = roughness;
    
    float Vis_SmithV = cosTheta1 * (cosTheta2 * (1.0 - a) + a);
    float Vis_SmithL = cosTheta2 * (cosTheta1 * (1.0 - a) + a);

    return min(1.0, (1.0 / (Vis_SmithV + Vis_SmithL)) * 0.5 * cosTheta1);
    */
}

vec4 ImportanceSampleGGX(in vec2 E, in float roughness){
    float a2 = max(1e-7, roughness * roughness);

    E.y *= 0.9;

    float Phi = E.x * 2.0 * Pi;
    float CosTheta = sqrt((1.0 - E.y) / ( 1.0 + (a2 - 1.0) * E.y));
    float SinTheta = sqrt(1.0 - CosTheta * CosTheta);

    vec3 H = vec3(cos(Phi) * SinTheta, sin(Phi) * SinTheta, CosTheta);
    float D = DistributionTerm(roughness, CosTheta) * CosTheta;

    return vec4(H, D);
}

float GetPixelPDF(in vec3 e, in vec3 r, in vec3 n, in float roughness) {
    vec3 h = normalize(r + e);

    float ndoth = max(0.0, dot(n, h));
    float d = DistributionTerm(ndoth, roughness) * ndoth;

    return max(1e-6, d);//max(d / (4.0 * abs(dot(e, h)) + 1e-6), 1e-6);
}

float GetPixelPDF(in float cosTheta, in float roughness) {
    float d = DistributionTerm(cosTheta, roughness) * cosTheta;

    return max(1e-6, d);
}
/*
vec3 SpecularLighting(in vec3 L, in vec3 E, in vec3 normal, in vec3 F0, in float roughness) {
    float ndotv = max(0.0, dot(normal, E));
    float ndotl = max(0.0, dot(normal, L));

    vec3 specular = vec3(0.0);

if(ndotl > 0.0 && ndotv > 0.0) {
    vec3 h = normalize(L + E);

    float ndoth = max(0.0, dot(normal, h));
    float hdotl = max(0.0, dot(L, h));

    vec3 f = SchlickFresnel(F0, hdotl);

    float d = DistributionTerm(ndoth, roughness);
    float g = VisibilityTerm(ndotl, ndotv, roughness);

    specular = f * (g * d * ndotl);
}

    return specular;
}
*/
vec3 SpecularLighting(in vec3 L, in vec3 E, in vec3 normal, in vec3 F0, in float roughness, in float clip) {
    float ndotv = max(0.0, dot(normal, E));
    float ndotl = max(0.0, dot(normal, L));

    vec3 specular = vec3(0.0);

if(ndotl > 0.0 && ndotv > 0.0) {
    vec3 h = normalize(L + E);

    float ndoth = saturate(dot(normal, h));
    float hdotl = max(0.0, dot(L, h));

    vec3 f = SchlickFresnel(F0, hdotl);

    float d = DistributionTerm(ndoth, roughness);
    float g = VisibilityTerm(ndotl, ndotv, roughness);

    specular = f * min(clip, g * d * ndotl);
}

    return specular;
}

vec3 BRDFLighting(in vec3 E, in vec3 L, in vec3 albedo, in vec3 normal, in vec3 F0, in float roughness, in float metallic) {
    vec3 lighting = vec3(0.0); 

    float ndotl = saturate(dot(normal, L));

if(ndotl > 0.0) {
    vec3 visivleNormal = CalculateVisibleNormals(normal, E);

    float ndotv = saturate(dot(visivleNormal, E));
    //if(ndotv == 0.0) ndotv = saturate(dot(normalize(normal + E), E));

    vec3 h = normalize(L + E);

    float ndoth = saturate(dot(visivleNormal, h));
    float hdotl = saturate(dot(L, h));

    vec3  f = SchlickFresnel(F0, hdotl);
    float d = DistributionTerm(ndoth, roughness);
    float g = VisibilityTerm(ndotl, ndotv, roughness);

    vec3 specular = f * min(100.0, g * d * ndotl);

    float FD90 = hdotl * hdotl * roughness * 2.0 + 0.5;
    float FDV = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotv);
    float FDL = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotl);

    vec3 diffuse = (albedo * (1.0 - f)) * (ndotl * FDL * FDV * invPi * (1.0 - metallic));
    
    lighting = (diffuse + specular);

    //float caustic = 0.8 + 0.2 * ndotl + pow(1.0 - hdotl, 8.0) * 7.0;
    //      caustic *= (1.05 - SchlickFresnel(hdotl)) + 0.05;
    //      caustic *= min(1.0, DistributionTerm(1.0, roughness) * 0.0001) * (1.0 - metallic);
}

    return lighting /* + caustic * albedo */;
}

vec3 BRDFLighting(in vec3 E, in vec3 L, in float s, in vec3 albedo, in vec3 normal, in vec3 F0, in float roughness, in float metallic) {
    vec3 lighting = vec3(0.0);

    float ndotl = max(0.0, dot(normal, L));

if(ndotl > 0.0) {
    float ndotv = max(0.0, dot(normal, E));
    if(ndotv == 0.0) ndotv = max(0.0, dot(normalize(normal + E), E));

    vec3 h = normalize(L + E);

    float ndoth = max(0.0, dot(normal, h));
    float hdotl = max(0.0, dot(L, h));

    vec3  f = SchlickFresnel(F0, hdotl);
    float d = DistributionTerm(ndoth, roughness);
    float g = VisibilityTerm(ndotl, ndotv, roughness);

    vec3 specular = f * (g * d * ndotl);

    #ifdef Specular_Strength
    specular *= Specular_Strength;
    #endif

    float FD90 = hdotl * hdotl * roughness * 2.0 + 0.5;
    float FDV = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotv);
    float FDL = 1.0 + (FD90 - 1.0) * SchlickFresnel(ndotl);

    //vec3 diffuse = albedo * FDL * FDV * invPi * (1.0 - metallic) * (1.0 - f);
    //lighting = (diffuse + specular) * min(1.0, 1.0 / pow2(s)) * ndotl;

    float fade = 1.0 / pow2(max(1.0, s));

    vec3 diffuse = (albedo * (1.0 - f)) * (ndotl * FDL * FDV * invPi * (1.0 - metallic));
    
    lighting = (diffuse + specular) * fade;
}
    return lighting;
}