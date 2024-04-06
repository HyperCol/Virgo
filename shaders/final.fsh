#version 130

const int noiseTextureResolution = 64;

/*
float noiseTexelSize = 1.0 / float(noiseTextureResolution);

const int colortex0Format 	= RGBA16;		//update 0.01
const int colortex1Format 	= RGBA16;
const int colortex2Format 	= RGBA16;
const int colortex3Format 	= RGBA16F;		//color
const int colortex4Format 	= RGBA16F;		//sky texture
const int colortex5Format 	= RGBA32F;		//down scale data
const int colortex6Format 	= R32F;			//
const int colortex7Format 	= RGBA16;		//taa
const int colortex8Format 	= RGBA16F;		//cahce
const int colortex9Format 	= RGBA16F;		//
const int colortex10Format 	= RGBA16;		//diffuse denoise
const int colortex11Format 	= RGB32F;		//diffuse denoise
const int colortex12Format 	= RGBA32F;		//specluar denoise
const int colortex13Format 	= RGBA16;		//PT:atla, SS:envmap
const int colortex14Format 	= RGBA32F;			//PT:atla_s, SS:envmap
const int colortex15Format 	= RGBA16F;		//specluar denoise

const bool shadowcolor0Nearest = true;
const bool shadowcolor1Nearest = true;
const bool shadowtex0Nearest = true;
const bool shadowtex1Nearest = true;

const float ambientOcclusionLevel = 0.0;

const float sunPathRotation = -35.0;

const bool colortex4MipmapEnabled = false;
*/

uniform float screenBrightness;

uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex7;

uniform sampler2D noisetex;
uniform sampler2D depthtex2;

uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;

//uniform float frameTimeCounter;
//uniform int frameCounter;
uniform int hideGUI;

in vec2 texcoord;

//uniform mat4 gbufferModelViewInverse;

#include "/libs/setting.glsl"
#include "/libs/common.glsl"
#include "/libs/function.glsl"
#include "/libs/uniform.glsl"
#include "/libs/materialid.glsl"
//#include "/libs/lighting/lighting_common.glsl"

vec3 Uncharted2Tonemap(vec3 x) {
	const float A = 0.22;
	const float B = 0.30;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.01;
	const float F = 0.30;

	return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 ACESToneMapping(in vec3 color) {
	const float A = 2.51;
	const float B = 0.03;
	const float C = 2.43;
	const float D = 0.59;
	const float E = 0.14;

	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

vec3 saturation(in vec3 color, in float s) {
	float lum = dot(color, vec3(1.0 / 3.0));
	return max(vec3(0.0), lum + (color - lum) * s);
}

vec2 float2R2(in float n) {
	float g = 1.32471795724474602596;
	vec2  a = 1.0 / vec2(g, g * g);

	return mod(n * a, 1.0);
}

float R2dither(in vec2 seed) {
	float g = 1.32471795724474602596;
	vec2  a = 1.0 / vec2(g, g * g);

	return fract(seed.x * a.x + seed.y * a.y);	
}

void sharpeness(inout vec3 color, in vec2 coord) {
	vec4 neighborColor = vec4(0.0);

	ivec2 texelPosition = ivec2(clamp(coord * resolution, vec2(1.5), resolution - 1.5));
	neighborColor.rgb += texelFetch(colortex3, texelPosition + ivec2(1, 0), 0).rgb;
	neighborColor.rgb += texelFetch(colortex3, texelPosition + ivec2(0, 1), 0).rgb;
	neighborColor.rgb += texelFetch(colortex3, texelPosition + ivec2(-1, 0), 0).rgb;
	neighborColor.rgb += texelFetch(colortex3, texelPosition + ivec2(0, -1), 0).rgb;
	neighborColor.rgb *= 0.25;

	vec3 sharpen = color - LinearToGamma(neighborColor.rgb) * MappingToHDR;
		 sharpen *= float(TAA_Post_Processing_Sharpeness) * 0.01 * 3.0;

	color = saturate(color + clamp(sharpen, vec3(-TAA_Post_Sharpen_Limit), vec3(TAA_Post_Sharpen_Limit)));	


	/*
	float v = pow(MappingToHDR, 1.0 / 2.2);
	float iv = 1.0 / v;

	for(float i = -1.0; i <= 1.0; i += 1.0) {
		for(float j = -1.0; j <= 1.0; j += 1.0) {
			if(i == 0.0 && j == 0.0) continue;

			neighborColor += vec4(texture(colortex3, coord + texelSize * vec2(i, j)).rgb, 1.0);
		}
	}

	vec3 sharpen = color - LinearToGamma(neighborColor.rgb / neighborColor.a) * MappingToHDR;
		 sharpen *= float(TAA_Post_Processing_Sharpeness) * 0.01 * 3.0;

	color = saturate(color + clamp(sharpen, vec3(-TAA_Post_Sharpen_Limit), vec3(TAA_Post_Sharpen_Limit)));
	
	//if(maxComponent(abs(sharpen)) > 0.01) color = vec3(mod(frameTimeCounter, 4.0) / 4.0 * 0.1, 0.0, 0.0);
	*/
	/*
	float K = pow(MappingToHDR, 1.0 / 2.2);
	float invK = 1.0 / K;
	
	vec3 sharpen = color - neighborColor.rgb / neighborColor.a;
		 sharpen *= float(TAA_Post_Processing_Sharpeness) * 0.01;
		 sharpen = clamp(sharpen * 10.0, vec3(-0.005 * invK), vec3(0.005 * invK));
		 
	color = saturate(color + sharpen);
	*/
	/*
	color =  LinearToGamma(color) * MappingToHDR;

	vec3 sharpen = color - LinearToGamma(neighborColor.rgb / neighborColor.a) * MappingToHDR;
		 sharpen *= float(TAA_Post_Processing_Sharpeness) * 0.01;

	color = max(vec3(0.0), color + clamp(sharpen, vec3(-0.01), vec3(0.01)));
	color = GammaToLinear(color * MappingToSDR);
	*/
}

vec3 UpScaleDualBlur(in sampler2D tex, in vec2 coord, in vec2 texel) {
	vec3 color = vec3(0.0);

	vec2 texel2x = texel * 2.0;

	color += texture(tex, coord + vec2( texel.x,  texel.x)).rgb * 2.0;
	color += texture(tex, coord + vec2( texel.x, -texel.y)).rgb * 2.0;
	color += texture(tex, coord - vec2( texel.x,  texel.x)).rgb * 2.0;
	color += texture(tex, coord - vec2( texel.x, -texel.y)).rgb * 2.0;

	color += texture(tex, coord + vec2(texel2x.x, 0.0)).rgb;
	color += texture(tex, coord - vec2(texel2x.x, 0.0)).rgb;
	color += texture(tex, coord + vec2(0.0, texel2x.y)).rgb;
	color += texture(tex, coord - vec2(0.0, texel2x.y)).rgb;

	color /= 12.0;

	return color;
}

float CalculateEV100(in float aperture, in float shutterTime, in float ISO) {
	return log2(sqrt(aperture) * shutterTime * ISO * 100.0);
}

float EV100ToExposure(in float ev100) {
	return 1.0 / (1.2 / exp2(ev100));
}

/*
float GetEntityEdge() {
	float radius = 2.0;

	float entity = 0.0;
	float weight = 0.0;

	for(float i = -radius; i <= radius; i += 1.0) {
		for(float j = -radius; j <= radius; j += 1.0) {
			vec2 texelPosition = texcoord + vec2(i, j) * texelSize * 4.0;

			entity += CalculateMask(ENTITY, round(texture(colortex1, texelPosition).b * 255.0));
			weight += 1.0;
		}
	}

	return entity / weight;
}
*/

float hash( in vec2 p ) {
	// replace this by something better
    p  = 50.0*fract( p*0.3183099 + vec2(0.71,0.113));
    return fract( p.x*p.y*(p.x+p.y) );
}

float noise(in vec2 p) {
	return texture(noisetex, p / 64.0).x;
	//return step(0.99, hash(floor(p)));
}

vec4 cubic(float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec3 textureBicubic(sampler2D tex, vec2 texCoords){
    texCoords = texCoords * resolution - 0.5;

    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= texelSize.xxyy;

    vec3 sample0 = texture(tex, offset.xz).rgb;
    vec3 sample1 = texture(tex, offset.yz).rgb;
    vec3 sample2 = texture(tex, offset.xw).rgb;
    vec3 sample3 = texture(tex, offset.yw).rgb;

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

vec3 colorTemperatureToRGB(in float temperature){
  // Values from: http://blenderartists.org/forum/showthread.php?270332-OSL-Goodness&p=2268693&viewfull=1#post2268693   
  mat3 m = (temperature <= 6500.0) ? mat3(vec3(0.0, -2902.1955373783176, -8257.7997278925690),
	                                      vec3(0.0, 1669.5803561666639, 2575.2827530017594),
	                                      vec3(1.0, 1.3302673723350029, 1.8993753891711275)) : 
	 								 mat3(vec3(1745.0425298314172, 1216.6168361476490, -8257.7997278925690),
   	                                      vec3(-2666.3474220535695, -2173.1012343082230, 2575.2827530017594),
	                                      vec3(0.55995389139931482, 0.70381203140554553, 1.8993753891711275)); 
  return mix(clamp(vec3(m[0] / (vec3(clamp(temperature, 1000.0, 40000.0)) + m[1]) + m[2]), vec3(0.0), vec3(1.0)), vec3(1.0), smoothstep(1000.0, 0.0, temperature));
}

void main() {
	vec3 color = LinearToGamma(texture(colortex3, texcoord).rgb) * MappingToHDR;
	vec3 rawcolor = color;

	#ifdef Enabled_Temporal_AA
	//sharpeness(color, texcoord);
	#endif

	float defualtExposure = 100.3992;
	float ev = CalculateEV100(1.4, 1.0 / 200.0, float(Camera_ISO)) + Camera_Exporsure_Value;

	#ifdef Camera_Auto_Exposure
		float minEV = Camera_Auto_Min_EV;
		float maxEV = Camera_Auto_Max_EV;

		if(minEV > maxEV) {
			minEV = Camera_Auto_Max_EV;
			maxEV = Camera_Auto_Min_EV;
		}

		float luminanceWhiteScale = 0.8;

		////float averageLuminance = texture(colortex3, vec2(0.5)).a * pow(MappingToHDR, 1.0 / 2.2);
		//float averageLuminance = texture(colortex7, texcoord).a * pow(MappingToHDR, 1.0 / 2.2);
		//	  averageLuminance = pow(averageLuminance, 0.8);

		//ev -= clamp(log2(averageLuminance * 100.0 / 12.5) - 0.0, -5.0, 5.0);

		float averageLuminance = texelFetch(colortex7, ivec2(1), 0).a;
			  averageLuminance = averageLuminance * 10.0 - 5.0;

		const float autoEVbias = -0.5;

		ev -= averageLuminance;
		ev += autoEVbias;
		//ev -= 5.0;

		//if(hideGUI == 0)
	#endif

#if Film_Grain > OFF
	vec2 jitter = float2R2(frameTimeCounter * float(Camera_FPS)) - 0.5;

	float dither = hash(floor((texcoord + jitter) * resolution));

	color = max(vec3(0.0), color - dither / defualtExposure * 0.0002 * float(Film_Grain));
#endif

	color *= EV100ToExposure(ev);

	//color = Uncharted2Tonemap(color) / Uncharted2Tonemap(vec3(1.0));
	color = ACESToneMapping(color);
	color = pow(color, vec3(1.0 + step(color, vec3(1.0)) * 0.2));

#if White_Balance_Adjustment > OFF
	color *= colorTemperatureToRGB(float(White_Balance_Adjustment));
#endif

	float bandJitter = hash(floor(texcoord * resolution));

	#if 0
	bandJitter = rescale(bandJitter, -0.5, 1.0) / 255.0;
	#else
	bandJitter = bandJitter / 510.0;
	#endif

	float range = 16383.0;

	vec3 color0 = rawcolor;
	vec3 color1 = LinearToGamma(texture(colortex3, texcoord + texelSize).rgb) * MappingToHDR;
	vec3 color2 = LinearToGamma(texture(colortex3, texcoord + vec2(texelSize.x, 0.0)).rgb) * MappingToHDR;
	vec3 color3 = LinearToGamma(texture(colortex3, texcoord + vec2(0.0, texelSize.y)).rgb) * MappingToHDR;

	vec3 diff0 = abs(color0 - color1);// * step(vec3(1e-5), bandCheck) * step(bandCheck, vec3(1.0)) * saturate(band * 10.0)
	vec3 diff1 = abs(color0 - color2);
	vec3 diff2 = abs(color0 - color3);
	float bandCheck = maxComponent(max(max(diff0, diff1), diff2)) * range;

	float band = step((bandCheck), 1.0);
	//color = saturate((color - (bandJitter * band)) / (1.0 - 0.5 / 255.0));
	color = saturate(color - band * bandJitter);

    color = GammaToLinear(color);

	//if(hideGUI == 1) color = texture(shadowcolor0, texcoord).rgb;

/*
	//color = saturate(texture(shadowcolor1, texcoord * vec2(aspectRatio, 1.0)).rgb * 2.0 - 1.0);
	vec2 coord0 = (texcoord * resolution * 0.0625) + vec2(floor(frameTimeCounter) * 0.0, 0.0);

    float n = noise(coord0);//(noise(coord0) + noise(coord0 * 2.0) * 0.5 + noise(coord0 * 4.0) * 0.25)/1.75;
		  //n = 1.0 - abs(n * 2.0 - 1.0);
		  //n = abs(n - noise(coord0 + vec2(1.0, 0.0))) * abs(n - noise(coord0 + vec2(0.0, 1.0)));
		  n = (noise(coord0) * 2.0 + noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(1.0, -1.0)) + noise(coord0 + vec2(-1.0, 1.0)) + noise(coord0 + vec2(-1.0, -1.0))) / 6.0;
		  //n = noise(coord0);
		  n = saturate(rescale(n, 0.5, 1.0));

	color = vec3(n);

    //float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(1.0, 1.0)) + noise(coord0 + vec2(0.0, 1.0))) / (4.0);
    float shape = (noise(coord0) + noise(coord0 + vec2(1.0, 0.0)) + noise(coord0 + vec2(-1.0, -1.0)) + noise(coord0 + vec2(-1.0, 1.0))) / (4.0);
    //float shape = 	max(noise(coord0), max(noise(coord0 + vec2(-1.0, 1.0)), noise(coord0 + vec2(-1.0, -1.0))));
		  shape = saturate(rescale(shape, 0.5, 1.0));
	color = vec3(shape);
*/
/*
    const vec2[4] offset = vec2[4](vec2(0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

    vec2 cbCoord = 1.0 + floor(texcoord * resolution) + vec2(0.0, 0.0) + offset[int(frameTimeCounter) % 4];
    float checkerBoardHit = mod(floor(cbCoord.x), 2.0) * mod(floor(cbCoord.y), 2.0);
	
	color = vec3(checkerBoardHit);
*/
/*
    vec2 stageJitter = (float2R2(float(frameTimeCounter) + 1011.5)) * (1.0 / 0.25 - 1.0);
    
    vec2 coord = (texcoord * resolution);
         coord = floor(coord + stageJitter) + 4.0;
    float hit = step(mod(coord.x, 4.0), 1e-5) * step(mod(coord.y, 4.0), 1e-5);

	color = vec3(hit);
*/

	//color = texture(colortex4, texcoord).rgb * pow(MappingToHDR, 1.0 / 2.2);

	//color.g = pow(color.g, 1.05);

	//color = floor(color * vec3(exp2(6.0), exp2(5.0), exp2(6.0))) / vec3(exp2(6.0), exp2(5.0), exp2(6.0));
	//color *= 100.0;

    gl_FragColor = vec4(color, 1.0);
}