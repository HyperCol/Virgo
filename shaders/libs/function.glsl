float pow2(in float v) {
    return v * v;
}

float pow5(in float v) {
    return v * v * v * v * v;
}

float rescale(in float v, in float vmin, in float vmax) {
    return (v - vmin) / (vmax - vmin);
}

float remap(in float v, in float vmin, in float vmax, in float nmin, in float nmax) {
    return nmin + (((v - vmin) / (vmax - vmin)) * (nmax - nmin));
}

float rescaleClamp(in float v, in float vmin, in float vmax, in float nmin, in float nmax) {
    return nmin + max(0.0, (v - vmin) / (vmax - vmin)) * (nmax - nmin);
}

float sum3(in vec3 v) {
    return (v.r + v.g + v.b) / 3.0;
}

float maxComponent(in vec3 v) {
    return max(v.r, max(v.g, v.b));
}

float maxComponent(in vec4 v) {
    return max(v.r, max(v.g, max(v.b, v.a)));
}

float minComponent(in vec3 v) {
    return min(v.r, min(v.g, v.b));
}

vec3 BentNormal(in vec2 t) {
    t = t * 2.0 - 1.0;
    return vec3(t, max(sqrt(1.0 - dot(t, t)), 1e-5));
}

mat3 tbnNormalTangent(vec3 normal, vec3 tangent) {
    vec3 bitangent = cross(normal, tangent);
    return mat3(tangent, bitangent, normal);
}

mat3 tbnNormal(vec3 normal) {
    vec3 tangent = normalize(cross(normal, vec3(0, 1, 1)));
    return tbnNormalTangent(normal, tangent);
}

float sdSphere( vec3 p, float s ) {
  return length(p) - s;
}

float sdBox( vec3 p, vec3 b ) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

vec2 signNotZero(vec2 v) {
    return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}
// Assume normalized input. Output is on [-1, 1] for each component.
vec2 EncodeOctahedralmap(in vec3 v) {
    //v = v.xzy;
    // Project the sphere onto the octahedron, and then onto the xy plane
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}

vec3 DecodeOctahedralmap(vec2 e) {
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    return normalize(v);
}

vec2 EncodeSpheremap(vec3 n) {
    float f = sqrt(8.0 * n.z + 8.0);
    return n.xy / f + 0.5;
}

vec3 DecodeSpheremap(vec2 enc) {
    vec2 fenc = enc * 4.0 - 2.0;
    float f = dot(fenc, fenc);
    float g = sqrt(1.0 - f / 4.0);
    vec3 n;
    n.xy = fenc * g;
    n.z = 1.0 - f / 2.0;
    return n;
}

float pack2x8(in vec2 x) {
    float pack = dot(round(x * 255.0), vec2(1.0, 256.0));
    return pack / 65535.0;
}

float pack2x8(in float x, in float y){
    return pack2x8(vec2(x, y));
}

vec2 unpack2x8(in float x) {
    x *= 65535.0;

    return vec2(floor(mod(x, 256.0)), floor(x / 256.0)) / 255.0;
}

float unpack2x8X(in float packge) {
    return fract(packge * 65535.0 / 256.0);
}

float unpack2x8Y(in float packge) {
    return (1.0 / 255.0) * floor(packge * 65535.0 / 256.0);
}

float pack2x4(in vec2 x) {
    return dot(round(x * 15.0), vec2(1.0, 16.0)) / 255.0;
}

vec2 unpack2x4(in float x) {
    x *= 255.0;
    return vec2(mod(x, 16.0), floor(x / 16.0)) / 15.0;
}

float pack2x16(in vec2 v) {
    return dot(round(v * 65535.0), vec2(1.0, 65536.0)) / 4294967295.0;
}

float pack2x16(in float vx, in float vy) {
    return pack2x16(vec2(vx, vy));
}

vec2 unpack2x16(in float x) {
    x *= 4294967295.0;
    return vec2(floor(mod(x, 65536.0)), floor(x / 65536.0)) / 65535.0;
}

vec4 unpack4x8(in float v) {
    v *= exp2(32.0) - 1.0;

    vec4 res = vec4(mod(v, 256.0), mod(v / 256.0, 256.0), mod(v / 65536.0, 256.0), 0.0);

	return floor(res) / 255.0;
}

float HG(in float m, in float g) {
  return (0.25 / Pi) * ((1.0 - g*g) / pow(1.0 + g*g - 2.0 * g * m, 1.5));
}

float RayleighPhase(in float theta) {
    return (3.0 / 16.0 / Pi) * (1.0 + theta * theta);
}

float luminance3(in vec3 color){
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float CalculateMask(in float matchID, in float tileID) {
    return step(matchID - 0.5, tileID) * step(tileID, matchID + 0.5);
}

float CalculateMask(in float startID, in float endID, in float tileID) {
    return step(startID - 0.5, tileID) * step(tileID, endID + 0.5);
}

float CalculateMaskBetween(in float startID, in float endID, in float tileID) {
    return step(startID - 0.5, tileID) * step(tileID, endID + 0.5);
}

bool BoolMask(in float matchID, in float tileID) {
    return tileID == matchID;
}

bool BoolMask(in float startID, in float endID, in float tileID) {
    return tileID >= startID && tileID <= endID;
}

bool BoolMaskBetween(in float startID, in float endID, in float tileID) {
    return tileID >= startID && tileID <= endID;
}

//float GaussianBlurWeight(in float l, in float sigma) {
//    return exp(-(l * l) / (2.0 * sigma * sigma));
//}

float GaussianBlurWeight(in float l, in float phi) {
    return exp(-(l * l) / phi);
}