#ifndef INCLUDED_INTERSECTION
#define INCLUDED_INTERSECTION

float IntersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
    return dot(origin - point, normal) / -dot(direction, normal);
}

float IntersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal, inout float front) {
    float ndotv = -dot(direction, normal);
    front = ndotv > 0.0 ? 1.0 : -1.0;

    return dot(origin - point, normal) / ndotv;
}

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

bool RaySphereIntersection(inout vec2 p, vec3 rayOrigin, vec3 rayDirection, vec3 sphereCenter, float sphereRadius) {
	rayOrigin -= sphereCenter;

	float a = dot(rayDirection, rayDirection);
	float b = 2.0 * dot(rayOrigin, rayDirection);
	float c = dot(rayOrigin, rayOrigin) - (sphereRadius * sphereRadius);
	float d = b * b - 4.0 * a * c;

	if (d < 0) return false;
	d = sqrt(d);

    vec2 t = vec2(-b - d, -b + d) / (2.0 * a);
    float hit = (t.x > 0.0 ? t.x : t.y);

    if(hit < p.x && t.y > 0.0 && t.y > t.x) {
        p = t;
	    return true;
    } else {
        return false;
    }
}

vec2 IntersectCube(vec3 rayOrigin, in vec3 rayDirection, in vec3 shapeCenter, in vec3 size, inout vec3 normal) {
    vec3 dr = 1.0 / rayDirection;
    vec3 n = (rayOrigin - shapeCenter) * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    vec3 front = -sign(rayDirection) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    vec3 back = -sign(rayDirection) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);

    normal = front;

    if(far > near && far > 0.0) {
        return vec2(near, far);
    }else{
        return vec2(-1.0);
    }
}

vec2 IntersectCube(vec3 rayOrigin, in vec3 rayDirection, in vec3 shapeCenter, in vec3 size) {
    vec3 dr = 1.0 / rayDirection;
    vec3 n = (rayOrigin - shapeCenter) * dr;
    vec3 k = size * abs(dr);

    vec3 pin = -k - n;
    vec3 pout = k - n;

    float near = max(pin.x, max(pin.y, pin.z));
    float far = min(pout.x, min(pout.y, pout.z));

    if(far > near && far > 0.0) {
        return vec2(near, far);
    }else{
        return vec2(-1.0);
    }
}

bool IntersectBox(inout vec2 t, inout vec3 normal, in vec3 origin, in vec3 direction, in vec3 start, in vec3 end) {
    /*
    vec3 dr = 1.0 / direction;
    vec3 nor = (origin - 0.5 - start * 0.5) * dr;
    vec3 k = (end * 0.5) * abs(dr);

    vec3 pin = -k - nor;
    vec3 pout = k - nor;

    float n = max(pin.x, max(pin.y, pin.z));
    float f = min(pout.x, min(pout.y, pout.z));
    */

    vec3 boundingA = (start - origin) / direction;
    vec3 boundingB = (end - origin) / direction;

    vec3 pin = min(boundingB, boundingA);
    vec3 pout = max(boundingB, boundingA);

    float n = max(max(pin.x, pin.y), pin.z); 
    float f = min(min(pout.x, pout.y), pout.z);

    bool front = n > 0.0;

    float p = front ? n : f;

    bool hit = f > 0.0 && f > n && t.x > p;

    if(hit) {
        if(front) {
            t = vec2(n, f - n);
            normal = step(vec3(n), pin);
        } else {
            t = vec2(f, f);
            normal = step(pout, vec3(f));
        }
        normal *= -sign(direction);
    }

    return hit;
}
/*
bool IntersectBox(inout float hitLength, inout float opticalLength, inout vec3 normal, in vec3 origin, in vec3 direction, in vec3 start, in vec3 end) {
    vec3 boundingA = (start - origin) / direction;
    vec3 boundingB = (end - origin) / direction;

    vec3 pin = min(boundingB, boundingA);
    vec3 pout = max(boundingB, boundingA);

    float n = max(max(pin.x, pin.y), pin.z); 
    float f = min(min(pout.x, pout.y), pout.z);

    bool front = n > 0.0;

    float p = front ? n : f;

    bool hit = f > 0.0 && f > n && hitLength > p;

    if(hit) {
        hitLength = p;
        opticalLength += f - max(0.0, n);
        normal = front ? step(vec3(n), pin) : step(pout, vec3(f));
        normal *= -sign(direction);
    }

    return hit;
}

bool IntersectBox(inout float hitLength, inout float opticalLength, inout vec3 normal, in vec3 origin, in vec3 direction, in vec3 signedDirection, in vec3 start, in vec3 end) {
    vec3 boundingA = (start - origin) / direction;
    vec3 boundingB = (end - origin) / direction;

    vec3 pin = min(boundingB, boundingA);
    vec3 pout = max(boundingB, boundingA);

    float n = max(max(pin.x, pin.y), pin.z); 
    float f = min(min(pout.x, pout.y), pout.z);

    bool front = n > 0.0;

    float p = front ? n : f;

    bool hit = f > 0.0 && f > n && hitLength > p;

    if(hit) {
        hitLength = p;
        opticalLength += f - max(0.0, n);
        normal = front ? step(vec3(n), pin) : step(pout, vec3(f));
        normal *= -signedDirection;
    }

    return hit;
}
*/
#endif