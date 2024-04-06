uniform sampler2D colortex5;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform vec3 cameraPosition;
//uniform float eyeAltitude;

uniform float rainStrength;

uniform float frameTimeCounter;

uniform int frameCounter;

uniform int heldBlockLightValue;
uniform int heldBlockLightValue2;

uniform ivec2 eyeBrightness;
uniform ivec2 eyeBrightnessSmooth;

uniform int isEyeInWater;

uniform int worldTime;

float GetAltitude(in vec3 p) {
    return p.y + cameraPosition.y - 63.0;
}

float GetAltitudeClip(in vec3 p, in float clip) {
    return max(clip, GetAltitude(p));
}

vec3 GetViewPosition(in vec3 coord) {
    vec4 p = gbufferProjectionInverse * vec4(ApplyTAAJitter(coord.xy) * 2.0 - 1.0, coord.z * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}

vec3 NonJitterViewPosition(in vec3 coord) {
    vec4 p = gbufferProjectionInverse * vec4(coord.xy * 2.0 - 1.0, coord.z * 2.0 - 1.0, 1.0);
    return p.xyz / p.w;
}

vec3 GetViewPosition(in vec2 coord, in float depth) {
    return GetViewPosition(vec3(coord, depth));
}

vec3 GetFragCoord(in vec3 position) {
    vec4 p = gbufferProjection * vec4(position, 1.0);
    return p.xyz / p.w * 0.5 + 0.5;
}

float ExpToLinearDepth(float depth) {
    vec2 viewDepth = mat2(gbufferProjectionInverse[2].zw, gbufferProjectionInverse[3].zw) * vec2(depth * 2.0 - 1.0, 1.0);
    return -viewDepth.x / viewDepth.y;
}

float LinearToExpDepth(float linerDepth) {
    vec2 expDepth = mat2(gbufferProjection[2].zw, gbufferProjection[3].zw) * vec2(-linerDepth, 1.0);

    return (expDepth.x / expDepth.y) * 0.5 + 0.5;
}

vec3 CalculateVisibleNormals(in vec3 n, in vec3 e) {
    float visible = dot(e, n);
    float cosTheta = max(1e-2, -visible);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return visible > 0.0 ? n : normalize(n * sinTheta + e * cosTheta);
}

vec3 CalculateVisibleNormals(in vec3 n, in vec3 e, in float t) {
    float visible = dot(e, n);
    float cosTheta = max(1e-2, -visible);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return visible > t ? n : normalize(n * sinTheta + e * cosTheta);
}

float SimpleGeometryTerm(in float angle){
    return saturate(rescale(abs(angle), 0.05, 0.15));
}

uniform mat4 gbufferPreviousModelView; 
uniform mat4 gbufferPreviousProjection;

uniform vec3 previousCameraPosition;

vec2 GetVelocity(in vec3 coord) {
    //vec4 velocityMap = texelFetch(colortex5, ivec2(coord.xy * resolution), 0);

    coord.xy = ApplyTAAJitter(coord.xy);

    vec4 p = gbufferProjectionInverse * vec4(coord * 2.0 - 1.0, 1.0);
         p = gbufferModelViewInverse * vec4(p.xyz / p.w, 1.0);
         p.xyz += cameraPosition - previousCameraPosition;
         p = gbufferPreviousModelView * p;
         p = gbufferPreviousProjection * p;

    //vec2 c = p.xy / p.w * 0.5 + 0.5;
    
    vec2 velocity = coord.xy - (p.xy / p.w * 0.5 + 0.5);
    //if(velocityMap.a > 0.5) velocity = velocityMap.xy;

    return velocity;
}

void RotateDirection(inout vec2 direction, in float angle) {
    float cosTheta = cos(angle * 2.0 * Pi);
    float sinTheta = sin(angle * 2.0 * Pi);

    mat2 rotate = mat2(cosTheta, sinTheta, -sinTheta, cosTheta);

    direction *= rotate;
}

/*
uniform vec3 lightVector;
uniform vec3 worldLightVector;

uniform vec3 sunVector;
uniform vec3 worldSunVector;

uniform vec3 moonVector;
uniform vec3 worldMoonVector;

uniform vec3 upVector;
uniform vec3 worldUpVector;
*/
/*
uniform vec3 shadowLightPosition;
uniform vec3 sunPosition;
uniform vec3 moonPosition;
uniform vec3 upPosition;

vec3 lightVector = normalize(shadowLightPosition);
vec3 worldLightVector = mat3(gbufferModelViewInverse) * lightVector;

vec3 sunVector = normalize(sunPosition);
vec3 worldSunVector = mat3(gbufferModelViewInverse) * sunVector;

vec3 moonVector = normalize(moonPosition);
vec3 worldMoonVector = mat3(gbufferModelViewInverse) * moonVector;

vec3 upVector = normalize(upPosition);
vec3 worldUpVector = vec3(0.0, 1.0, 0.0);
*/