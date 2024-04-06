#version 130
/*
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;

uniform sampler2D depthtex0;

uniform sampler2D depthtex2;

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
*/
in vec2 texcoord;
/*
in vec3 SunLightingColor;
in vec3 MoonLightingColor;
in vec3 LightingColor;
in vec3 SkyLightingColor;

in float shadowFade;
*/
#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
//#include "/libs/volumetric/atmospheric_common.glsl"

/*
struct Ray {
    vec3 origin;
    vec3 direction;
    vec3 dirInv;
};

struct PathTrace {
    vec3 voxelPos;
    vec3 stepSize;
    vec3 dirSigned;
    vec3 totalStep;
    vec3 nextBlock;
};

PathTrace startTrace(Ray ray) {
    PathTrace pathTrace;
    pathTrace.voxelPos = floor(ray.origin);//floor(floor(ray.origin) - cameraPosition);
    pathTrace.stepSize = abs(1.0 / (ray.direction + 1e-07));
    pathTrace.dirSigned = sign(ray.direction);
    pathTrace.totalStep = (pathTrace.dirSigned * (pathTrace.voxelPos - ray.origin + 0.5) + 0.5) * pathTrace.stepSize;
    pathTrace.nextBlock = vec3(0.0);
    return pathTrace;
}

void stepTrace(inout PathTrace pathTrace) {
    pathTrace.nextBlock = step(pathTrace.totalStep.xyz, vec3(min(min(pathTrace.totalStep.x, pathTrace.totalStep.y), pathTrace.totalStep.z)));
    pathTrace.totalStep += pathTrace.nextBlock * pathTrace.stepSize;
    pathTrace.voxelPos += pathTrace.nextBlock * pathTrace.dirSigned;
}

void backTrace(inout PathTrace pathTrace) {
    pathTrace.nextBlock = step(pathTrace.totalStep.xyz, vec3(min(min(pathTrace.totalStep.x, pathTrace.totalStep.y), pathTrace.totalStep.z)));
    pathTrace.totalStep -= pathTrace.nextBlock * pathTrace.stepSize;
    pathTrace.voxelPos -= pathTrace.nextBlock * pathTrace.dirSigned;
}

bool isRayHitElement(vec3 fromPos, vec3 toPos, Ray ray, inout float rayLength, inout float tmax, inout vec3 voxelNormal) {
    
    vec3 boundingA = ray.dirInv * (fromPos - 1e-5 - ray.origin);
    vec3 boundingB = ray.dirInv * (toPos + 1e-5 - ray.origin);

    vec3 n1 = min(boundingB, boundingA);
    vec3 n2 = max(boundingB, boundingA);

    float near = max(max(n1.x, n1.y), n1.z); 
    float far = min(min(n2.x, n2.y), n2.z);

    bool hit = far > 0.0 && far > near;

    vec3 front = -sign(ray.direction) * step(n1.zxy, n1.xyz) * step(n1.yzx, n1.xyz);

    if(hit) {
        voxelNormal = (front);
        rayLength = near;
        tmax = far;
    }

    return hit;
}

bool isRayHitBlock(vec3 origin, Ray ray, inout float rayLength, inout float tmax, inout vec3 voxelNormal) {
    return isRayHitElement(origin, origin + vec3(1.0), ray, rayLength, tmax, voxelNormal);
}
*/

vec2 RaySphereIntersection(vec3 rayOrigin, vec3 rayDirection, vec3 sphereCenter, float sphereRadius) {
	rayOrigin -= sphereCenter;

	float a = dot(rayDirection, rayDirection);
	float b = 2.0 * dot(rayOrigin, rayDirection);
	float c = dot(rayOrigin, rayOrigin) - (sphereRadius * sphereRadius);
	float d = b * b - 4.0 * a * c;

	if (d < 0) return vec2(-1.0);

	d = sqrt(d);
	return vec2(-b - d, -b + d) / (2.0 * a);
}

vec3 CalculateSunLighting(in vec3 rayOrigin, in vec3 L, in float density) {
    vec2 tracingAtmosphere = RaySphereIntersection(rayOrigin, L, vec3(0.0), atmosphere_radius);
    if(tracingAtmosphere.y < 0.0) return vec3(1.0);

    vec2 tracingPlanet = RaySphereIntersection(rayOrigin, L, vec3(0.0), planet_radius);
    float planetShadow = tracingPlanet.x > 0.0 ? exp(-(tracingPlanet.y - tracingPlanet.x) * 0.00001) : 1.0;
    //if(tracingPlanet.x > 0.0) return vec3(0.0);

    float exposure = 1.0 / (exp2(Shadow_Light_Exposure));

    float mu = 0.999;
    float phaser = (3.0 / 16.0 * invPi) * (1.0 + mu * mu);
    float phasem = min(1.0, HG(mu, 0.76));

    float depthPoint = 0.5;
    float heightPoint = 0.02;

    float stepLength = tracingAtmosphere.y;

    float height = length(rayOrigin + L * stepLength * heightPoint) - planet_radius;
          height = max(1e-5, height);

    float density_rayleigh  = exp(-height / rayleigh_distribution) * density;
    float density_mie       = exp(-height / mie_distribution) * density;

    vec3 tau = (rayleigh_scattering + rayleigh_absorption) * density_rayleigh + (mie_scattering + mie_absorption) * density_mie;
    vec3 transmittance = exp(-tau * stepLength * depthPoint);

    vec3 scattering = phaser * rayleigh_scattering * density_rayleigh / sum3(rayleigh_scattering) + phasem * mie_scattering * density_mie / sum3(mie_scattering);
         scattering *= transmittance;

    return (transmittance * phasem + scattering) * planetShadow * exposure;
}

struct SkyAtmosphereStruct {
    vec3 r_scattering;
    vec3 r_extinction;

    vec3 m_scattering;
    vec3 m_extinction;

    vec3 o_extinction;

    vec3 extinction;
};

SkyAtmosphereStruct GetSkyAtmosphere(in float h) {
    SkyAtmosphereStruct a;

    float rdensity = exp(-h / rayleigh_distribution);
    float mdensity = exp(-h / mie_distribution);
    float odensity = saturate(1.0 - abs(h - 25000.0) / 15000.0);

    a.r_scattering = rdensity * rayleigh_scattering;
    a.r_extinction = rdensity * rayleigh_absorption + a.r_scattering;

    a.m_scattering = mdensity * mie_scattering;
    a.m_extinction = mdensity * mie_absorption + a.m_scattering;

    a.o_extinction = odensity * ozone_absorption;

    a.extinction = a.r_extinction + a.m_extinction + a.o_extinction;

    return a;
}

vec3 CalculateLocalInScattering(in vec3 rayOrigin, in vec3 rayDirection) {
    const int steps = 6;
    const float invsteps = 1.0 / float(steps);

    float planetShadow = 1.0;

    vec2 tracingAtmosphere = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0), atmosphere_radius);
    if(tracingAtmosphere.y < 0.0) return vec3(1.0);

    vec2 tracingPlanet = RaySphereIntersection(rayOrigin, rayDirection, vec3(0.0, -1.0, 0.0), planet_radius);

    planetShadow = tracingPlanet.x > 0.0 ? exp(-(tracingPlanet.y - tracingPlanet.x) * 0.00001) : 1.0;
    if(planetShadow < 1e-5) return vec3(0.0);

    float stepLength = tracingAtmosphere.y * invsteps;

    vec3 tau = vec3(0.0);

    for(int i = 0; i < steps; i++) {
        vec3 p = rayOrigin + rayDirection * (stepLength * (0.5 + float(i)));
        float h = max(1e-5, length(p) - planet_radius);

        SkyAtmosphereStruct a = GetSkyAtmosphere(h);
        tau += a.extinction;
    }

    vec3 transmittance = exp(-tau * stepLength);

    return transmittance * planetShadow;
}

vec3 planetOrigin = vec3(0.0, planet_radius, 0.0);

struct AtmosphereDataStruct {
    float r_density;
    vec3  r_scattering;
    vec3  r_extinction;

    float m_density;
    vec3  m_scattering;
    vec3  m_extinction;

    vec3 extinction;
};

AtmosphereDataStruct GetAtmosphereData(in float height) {
    AtmosphereDataStruct a;

    a.r_density = exp(-height / rayleigh_distribution);
    a.m_density = exp(-height / mie_distribution);

    a.r_scattering = rayleigh_scattering * a.r_density;
    a.r_extinction = a.r_scattering + rayleigh_absorption * a.r_density;

    a.m_scattering = mie_scattering * a.m_density;
    a.m_extinction = a.m_scattering + mie_absorption * a.m_density;

    a.extinction = a.m_extinction + a.r_extinction;

    return a;
}

void CalculateAtmosphericScattering(inout vec3 color, in vec3 rayOrigin, in vec3 rayDirection, in vec3 L) {
    const int steps = 20;
    const float invsteps = 1.0 / float(steps);

    vec2 tracingAtmosphere = RaySphereIntersection(planetOrigin + rayOrigin, rayDirection, vec3(0.0), atmosphere_radius);
    vec2 tracingPlanet = RaySphereIntersection(planetOrigin + rayOrigin, rayDirection, vec3(0.0), planet_radius);

    float exposure = 1.0 / exp2(Sky_Texture_Exposure);

    float theta = dot(rayDirection, L);
    float miePhase = HG(theta, 0.76);
    float miePhase2 = HG(-theta, 0.76);
    float rayleighPhase = RayleighPhase(theta);

    if(tracingPlanet.x > 0.0) {
        //color = vec3(0.0);//LightingColor * invPi * exposure * 0.05;
        color = vec3(0.0);

    #if 1
        float t = max(0.0, tracingPlanet.x) - max(0.0, tracingAtmosphere.x) + 0.05 + 20000.0;
        float h = Altitude_Start + 0.01;
    #else
        float t = sqrt(dot(v.worldPosition.xz, v.worldPosition.xz));
        float h = max(0.0, v.worldPosition.y + cameraPosition.y - 63.0) * 1.0
    #endif

        AtmosphereDataStruct a = GetAtmosphereData(h);

        vec3 SunLightingColor   = CalculateSunLighting(planetOrigin + rayOrigin, L, 0.5) * Sun_Light_Luminance;
        vec3 MoonLightingColor  = CalculateSunLighting(planetOrigin + rayOrigin, -L, 0.5) * Moon_Light_Luminance;
        vec3 LightingColor      = SunLightingColor + MoonLightingColor;

        vec3 fogColor = a.r_scattering * RayleighPhase(theta) * LightingColor;
             fogColor += a.m_scattering * HG(theta, 0.76) * SunLightingColor;
             fogColor += a.m_scattering * HG(-theta, 0.76) * MoonLightingColor;

        vec3 fogAlpha = exp(-a.extinction * t);

        fogColor = (fogColor - fogColor * fogAlpha) / (a.extinction + step(a.extinction, vec3(0.0)));

        //float weatherLight = mix(HG(-0.9999, Fog_BackScattering_Phase) * (1.0 - Fog_Front_Scattering_Weight), 1.0, saturate(exp(-Fog_Light_Extinction_Distance * Rain_Fog_Density - Biome_Fog_Density * Fog_Light_Extinction_Distance)));
        float weatherAlpha = saturate(exp(-Rain_Fog_Density * t - Biome_Fog_Density * t));
        vec3 weatherColor = (SunLightingColor * miePhase + MoonLightingColor * miePhase2) * (HG(-0.9999, Fog_BackScattering_Phase) * (1.0 - Fog_Front_Scattering_Weight));
        fogColor = fogColor * weatherAlpha + (weatherColor - weatherColor * weatherAlpha);

        color = fogColor;

        /*
        float md = exp(-0.5 / mie_distribution);
        float rd = exp(-0.5 / rayleigh_distribution);
        vec3 ms = mie_scattering * md;
        vec3 mt = ms + mie_absorption * md;
        vec3 rs = rayleigh_scattering * rd;
        vec3 rt = rs + rayleigh_absorption * rd;
        vec3 t = mt + rt;
        float invt = 1.0 / sum3(t);

        vec3 s = (ms + rs) / sum3(ms + rs);
             s = mix(vec3(1.0), s, vec3(1.0));

        vec3 l1 = ms * miePhase + rs * RayleighPhase(max(0.0, theta));
        vec3 l2 = ms * miePhase2 + rs * RayleighPhase(max(0.0, -theta));
        vec3 trans = exp(-tracingPlanet.x * t);

        color += CalculateSunLighting(planetOrigin + rayOrigin, L, 0.5) * Sun_Light_Luminance * l1 * invt * trans * invPi;
        color += CalculateSunLighting(planetOrigin + rayOrigin, -L, 0.5) * Moon_Light_Luminance * l2 * invt * trans * invPi;
        */
    }

    float end = tracingPlanet.x > 0.0 ? tracingPlanet.x : tracingAtmosphere.y;
    float start = tracingAtmosphere.x > 0.0 ? tracingAtmosphere.x : 0.0;

    float stepLength = (end - start) * invsteps;

    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);
    vec3 m2 = vec3(0.0);

    vec3 transmittance = vec3(1.0);

    vec3 rayStart = planetOrigin + rayOrigin + rayDirection * start + rayDirection * stepLength * 0.5;

    for(int i = 0; i < steps; i++) {
        vec3 p = rayStart + rayDirection * stepLength * float(i);
        float h = max(1e-5, length(p) - planet_radius);
/*
        float density_rayleigh  = exp(-h / rayleigh_distribution);
        float density_mie       = exp(-h / mie_distribution);
        float density_ozone     = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);

        vec3 tau = (rayleigh_scattering + rayleigh_absorption) * (density_rayleigh) + (mie_scattering + mie_absorption) * (density_mie) + ozone_absorption * density_ozone;
        vec3 attenuation = exp(-tau * stepLength);
*/
        SkyAtmosphereStruct a = GetSkyAtmosphere(h);

        vec3 invt = 1.0 / a.extinction;
        vec3 attenuation = exp(-a.extinction * stepLength);

        vec3 L1 = CalculateLocalInScattering(p, L) * Sun_Light_Luminance;
        vec3 S1 = (L1 - L1 * attenuation) * transmittance * invt;

        vec3 L2 = CalculateLocalInScattering(p, -L) * Moon_Light_Luminance;
        vec3 S2 = (L2 - L2 * attenuation) * transmittance * invt;

        r += (S1 + S2) * a.r_scattering;
        m += S1 * a.m_scattering;
        m2 += S2 * a.m_scattering; 

        transmittance *= attenuation;
    }
    //color += -theta - (1.0 - 0.001) > 0.0 ? vec3(1.0) : vec3(0.0);
    color *= transmittance;

    vec3 scattering = (r * rayleighPhase + m * miePhase + m2 * miePhase2);

    //float weatherAlpha = exp(-Rain_Fog_Density * 4000.0);
    //vec3 weatherColor = scattering;
    //scattering = weatherAlpha * scattering * 0.0 + (scattering - scattering * weatherAlpha) * 10.0 * (HG(-0.9999, Fog_BackScattering_Phase) * 0.7 * invPi);

    color += scattering * exposure;
}

in vec3 direction;

void main() {
    vec3 color = vec3(0.0);
    float alpha = 1.0;

    vec3 rayDirection = DecodeOctahedralmap(texcoord * 2.0 - 1.0);
/*
    vec3 rayStart = cameraPosition;

    Ray tracedRay = Ray(rayStart, rayDirection, 1.0 / rayDirection);
    PathTrace pathTrace = startTrace(tracedRay);
    stepTrace(pathTrace);

    float rayLength = 3000.0;
    vec3 voxelNormal = vec3(0.0);
    float lastRayLength = 0.0;

    vec3 attenuation = vec3(1.0);
    vec3 throughput = vec3(1.0);

    bool hit = false;

    for(int i = 0; i < 32; i++) {
        vec3 voxelPosition = floor(pathTrace.voxelPos - cameraPosition);

        if(!HitVoxelBoundingBox(voxelPosition)) break;

        vec2 texelCoord = CalculateVoxelCoord(voxelPosition);
             texelCoord *= shadowTexelSize;

        float tileID = round(texture(shadowcolor1, texelCoord).b * 255.0);        

        if(tileID < 255.0) {
            hit = true;
        } else {
            for(int j = 0; j < 3; j++) {
                stepTrace(pathTrace);
                vec2 texelCoord = CalculateVoxelCoord(floor(pathTrace.voxelPos - cameraPosition));
                     texelCoord *= shadowTexelSize;

                float tileID = round(texture(shadowcolor1, texelCoord).b * 255.0);

                if(tileID < 255.0) {
                    backTrace(pathTrace);
                    break;
                }
            }            
        }

        stepTrace(pathTrace);
    }

    if(!hit) {
    }
*/

    vec3 rayOrigin = vec3(0.0, Altitude_Start, 0.0);
         //rayOrigin.y += max(0.0, (cameraPosition.y - 63.0) * (8848.43 / (220.0 - 63.0)));

    CalculateAtmosphericScattering(color, rayOrigin, rayDirection, worldSunVector);

    vec3 fog_scattering = vec3(0.001 * rainStrength);
    float stepLength = 4000.0;

    vec3 transmittance = exp(-stepLength * fog_scattering);

    float theta = dot(rayDirection, worldSunVector);
    float Pfog = mix(HG(0.85, 0.8), HG(theta, 0.8), 0.3);

    float theta2 = dot(rayDirection, worldMoonVector);
    float Pfog2 = mix(HG(0.85, 0.8), HG(theta2, 0.8), 0.3);

    //color = color * transmittance + fog_scattering * min(1000.0, stepLength) * (Pfog * SunLightingColor + Pfog2 * MoonLightingColor + color) * shadowFade;
    //color-=color;

    color = GammaToLinear(color * MappingToSDR);

    gl_FragData[0] = vec4(color, alpha);
}
/* DRAWBUFFERS:4 */
/* RENDERTARGETS: 4 */