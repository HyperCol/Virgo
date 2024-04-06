vec3 KarisToneMapping(in vec3 color){
	float a = 0.001;
	float b = 1.0;

	float lum = maxComponent(color);
	
	if(lum < a) return color;

	return color/lum*((a*a-b*lum)/(2.0*a-b-lum));
}

vec3 InverseKarisToneMapping(in vec3 color) {
	float a = 0.001;
	float b = 1.0;

	float lum = maxComponent(color);

	if(lum < a) return color;

	return color/lum*((a*a-(2.0*a-b)*lum)/(b-lum));
}