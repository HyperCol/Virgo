const float     clouds_height       = 1500.0;
const float     clouds_thickness    = 800.0;

#ifndef Clouds_Shadow_On_Atmosphric_Scattering
const vec3      clouds_scattering   = vec3(0.08);
#else
const vec3      clouds_scattering   = vec3(0.05);
#endif

vec2 TracingCloudsLayer(in vec3 origin, in vec3 direction) {
    vec2 tracingBottom = RaySphereIntersection(origin, direction, vec3(0.0), planet_radius + clouds_height);
    vec2 tracingTop = RaySphereIntersection(origin, direction, vec3(0.0), planet_radius + clouds_height + clouds_thickness);

    float rayStart = max(0.0, tracingBottom.y);
    float rayEnd = max(0.0, tracingTop.y);

    if(rayStart > rayEnd) {
        float v = rayStart;
        rayStart = rayEnd;
        rayEnd = v;
    }

    return vec2(rayStart, rayEnd);
}

float GetCloudsMap(in vec3 position, in float linearHeight) {
    vec3 worldPosition = vec3(position.x, position.z, position.y - planet_radius);

    float t = frameTimeCounter * Clouds_Speed;

    worldPosition.x += t * Clouds_X_Speed;
    worldPosition.x += linearHeight / (clouds_thickness) * Clouds_X_Speed;

    worldPosition.z += t * Clouds_Y_Speed;
    worldPosition.z += linearHeight / (clouds_thickness) * Clouds_Y_Speed;

    vec3 shapeCoord = worldPosition * 0.0005;
    float shape = (noise(shapeCoord.xy) + noise(shapeCoord.xy * 2.0) * 0.5) / 1.5;
    float shape2 = (noise(shapeCoord * 4.0) + noise(shapeCoord.xy * 8.0) * 0.5) / 1.5;

    float density = max(0.0, rescale((shape + shape2 * 0.25) / 1.25, 0.0, 1.0));

    return density;
}

float GetCloudsMapDetail(in vec3 position, in float shape, in float distortion) {
    vec3 worldPosition = vec3(position.x, position.z, position.y - planet_radius);

    vec3 noiseCoord0 = worldPosition * 0.01;
    float noise0 = (noise(noiseCoord0) + noise(noiseCoord0 * 2.0) * 0.5 + noise(noiseCoord0 * 4.0) * 0.25) / (1.75);

    return saturate(rescale(shape - noise0 * distortion, 0.0, 1.0 - distortion));
} 

float GetCloudsCoverage(in float linearHeight) { 
    return pow(0.75, remap(linearHeight, 0.7, 0.8, 1.0, mix(1.0, 0.5, 0.3)) * saturate(rescale(linearHeight, -0.05, 0.1)) * saturate(remap(linearHeight, 0.95, 1.0, 1.0, 0.0)));
}

float CalculateCloudsCoverage(in float height, in float clouds) {
    float linearHeight = (height - clouds_height) / clouds_thickness;    

    return saturate(rescale(clouds, GetCloudsCoverage(linearHeight), 1.0) * 2.0);
}

vec3 CloudsPowderEffect(in vec3 opticalDepth) {
    return 1.0 - exp(-opticalDepth * 2.0);
}

vec3 CloudsLocalLighting(in vec3 opticalDepth) {
    vec3 extinction = (exp(-opticalDepth) + exp(-opticalDepth * 0.25) * 0.7 + exp(-opticalDepth * 0.03) * 0.24) / (1.7 + 0.24);

    return extinction;
}

vec4 CalculateCloudsMedia(in vec3 rayPosition, in vec3 origin) {
    float height = length(rayPosition - vec3(origin.x, 0.0, origin.z)) - planet_radius;

    float density = GetCloudsMap(rayPosition, height);
          density = GetCloudsMapDetail(rayPosition, density, 0.2);
          density = CalculateCloudsCoverage(height, density);

    return vec4(clouds_scattering * density, density);
}