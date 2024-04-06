uniform sampler2D depthtex2;

const int BlueNoiseResolution = 128;

float GetBlueNoise(in ivec2 coord) {
    return texelFetch(depthtex2, coord % BlueNoiseResolution, 0).x;
}

float GetBlueNoise(in vec2 coord) {
    return GetBlueNoise(ivec2(coord * resolution));
}

float GetBlueNoise1(in ivec2 coord) {
    return texelFetch(depthtex2, coord % BlueNoiseResolution, 0).y;
}

float GetBlueNoise1(in vec2 coord) {
    return GetBlueNoise1(ivec2(coord * resolution));
}

float GetBlueNoise2(in ivec2 coord) {
    return texelFetch(depthtex2, coord % BlueNoiseResolution, 0).z;
}

float GetBlueNoise2(in vec2 coord) {
    return GetBlueNoise2(ivec2(coord * resolution));
}

uniform sampler2D noisetex;

const int noiseTextureResolution = 64;

float noise(in vec2 x){
    return texture(noisetex, x / noiseTextureResolution).x;
}

float noise(in vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);

	f = f*f*(3.0-2.0*f);

	vec2 uv = (i.xy + i.z * vec2(17.0)) + f.xy;
    uv += 0.5;

	vec2 rg = vec2(noise(uv), noise(uv+17.0));

	return mix(rg.x, rg.y, f.z);
}

float hash( in vec2 p ) {
	// replace this by something better
    p  = 50.0*fract( p*0.3183099 + vec2(0.71,0.113));
    return fract( p.x*p.y*(p.x+p.y) );
}

float hash(vec3 p) {
    p  = fract(p * 0.3183099 + 0.1);
	p *= 17.0;
    return fract(p.x*p.y*p.z*(p.x+p.y+p.z));
}

float R2dither(in vec2 seed) {
    float g = 1.32471795724474602596;
    vec2  a = 1.0 / vec2(g, g * g);

    return mod(dot(a, seed), 1.0);
}

float R2dither(in float vx, in float vy) {
    float g = 1.32471795724474602596;
    vec2  a = 1.0 / vec2(g, g * g);

    return mod(a.x * vx + a.x * vy , 1.0);
}

float R2dither(in vec3 seed) {
    float g = 1.22074408460575947536;
    vec3  a = 1.0 / vec3(g, g * g, g * g * g);

    return mod(dot(a, seed), 1.0);
}

float R2dither(in vec2 seed, in float t) {
    float g = 1.22074408460575947536;
    vec3  a = 1.0 / vec3(g, g * g, g * g * g);

    return mod(dot(a.xy, seed) + t * a.z, 1.0);
}

vec2 float2R2(in float n) {
	float g = 1.32471795724474602596;
	vec2  a = 1.0 / vec2(g, g * g);

	return mod(0.5 + n * a, 1.0);
}

vec3 vector3R2(in float n) {
    float g = 1.22074408460575947536;
    vec3  a = 1.0 / vec3(g, g * g, g * g * g);

    return mod(0.5 + n * a, 1.0);
}