vec3 SimpleLightTransmittance(in float rayLength, in float height, in float density) {
    float density_rayleigh  = exp(-height / rayleigh_distribution) * density;
    float density_mie       = exp(-height / mie_distribution) * density;

    vec3 tau = (rayleigh_scattering + rayleigh_absorption) * density_rayleigh + (mie_scattering + mie_absorption) * density_mie;
    vec3 transmittance = exp(-tau * rayLength);

    return transmittance;
}

vec3 CalculateSunLighting(in vec3 rayOrigin, in vec3 L, in float density) {
    vec3 lightColor = vec3(1.0);

    float exposure = 1.0 / exp2(Shadow_Light_Exposure);

    float depthPoint = 0.5;
    float heightPoint = 0.02;
    float mu = 0.9999;

    float phaser = RayleighPhase(mu);
    float phasem = min(1.0, HG(mu, 0.76));

    vec2 tracingAtmosphere = RaySphereIntersection(rayOrigin, L, vec3(0.0), atmosphere_radius);
    //if(tracingAtmosphere.y < 0.0) return vec3(1.0);

    vec2 tracingPlanet = RaySphereIntersection(rayOrigin, L, vec3(0.0), planet_radius);
    float planetShadow = tracingPlanet.x > 0.0 ? exp(-(tracingPlanet.y - tracingPlanet.x) * 0.00001) : 1.0;
    //if(tracingPlanet.x > 0.0) return vec3(1.0, 0.0, 0.0);

if(tracingAtmosphere.y > 0.0) {
#if 1
    float stepLength = tracingAtmosphere.y;

    float height = length(rayOrigin + L * stepLength * heightPoint) - planet_radius;
          height = max(1e-5, height);

    float density_rayleigh  = exp(-height / rayleigh_distribution) * density;
    float density_mie       = exp(-height / mie_distribution) * density;

    vec3 tau = (rayleigh_scattering + rayleigh_absorption) * density_rayleigh + (mie_scattering + mie_absorption) * density_mie;

    vec3 transmittance = exp(-tau * stepLength * depthPoint);

    //vec3 scattering = phaser * rayleigh_scattering * density_rayleigh / sum3(rayleigh_scattering) + phasem * mie_scattering * density_mie / sum3(mie_scattering);
    //     scattering *= transmittance;
    vec3 scattering = rayleigh_scattering * (phaser * density_rayleigh) + mie_scattering * (phasem * density_mie);
         scattering = (scattering - scattering * transmittance) / tau * exp(-tau * (0.25 * stepLength * depthPoint));

    //vec3 m = mie_scattering * (density_mie * phasem);
    //vec3 r = rayleigh_scattering * (density_rayleigh * phaser);
    //vec3 scattering = r + m;
    //     scattering = (scattering - scattering * transmittance) / tau * invPi;

    //return (transmittance * phasem + scattering) * planetShadow * exposure;
    lightColor = (transmittance * phasem + scattering) * planetShadow * exposure;
#else
    int steps = 20;
    float invsteps = 1.0 / float(steps);

    float stepLength = tracingAtmosphere.y * invsteps;

    vec3 transmittance = vec3(1.0);
    vec3 r = vec3(0.0);
    vec3 m = vec3(0.0);

    for(int i = 0; i < steps; i++) {
        vec3 p = rayOrigin + L * stepLength * (float(i) + 0.5);
        float height = max(1e-5, length(p) - planet_radius);

        float density_rayleigh  = exp(-height / rayleigh_distribution);
        float density_mie       = exp(-height / mie_distribution);

        vec3 tau = (rayleigh_scattering + rayleigh_absorption) * density_rayleigh + (mie_scattering + mie_absorption) * density_mie;
        vec3 attenuation = exp(-tau * stepLength);

        vec3 s = (1.0 - attenuation) * transmittance / sum3(tau);

        r += s * density_rayleigh * rayleigh_scattering;
        m += s * density_mie * mie_scattering;

        transmittance *= attenuation;
    }

    vec3 scattering = r * phaser + m * phasem;

    //return (transmittance * phasem + scattering) * planetShadow  * exposure;
    lightColor = (transmittance * phasem + scattering) * planetShadow  * exposure;
#endif
}

    return lightColor;
}