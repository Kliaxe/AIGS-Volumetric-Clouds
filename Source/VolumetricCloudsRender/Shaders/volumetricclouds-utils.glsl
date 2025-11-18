#ifndef VOLUMETRICCLOUDS_UTILS_GLSL
#define VOLUMETRICCLOUDS_UTILS_GLSL

#define FLT_MAX 3.402823466e+38
#define HALF_FLT_MAX 65504.0

#define M_TO_SKY_UNIT 0.001
#define SKY_UNIT_TO_M (1.0 / M_TO_SKY_UNIT)

// -------------------------------------------------------------------------
//    Noise Functions
// -------------------------------------------------------------------------

// Amazing noise and weather creation, modified from: https://github.com/greje656/clouds

// Perlin noise functions from: https://github.com/BrianSharpe/GPU-Noise-Lib/blob/master/gpu_noise_lib.glsl
vec3 Interpolation_C2(vec3 x)
{
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

void PerlinHash(vec3 gridcell, float s, bool tile,
								inout vec4 lowz_hash_0,
								inout vec4 lowz_hash_1,
								inout vec4 lowz_hash_2,
								inout vec4 highz_hash_0,
								inout vec4 highz_hash_1,
								inout vec4 highz_hash_2)
{
    const vec2 OFFSET = vec2(50.0, 161.0);
    const float DOMAIN = 69.0;
    const vec3 SOMELARGEFLOATS = vec3(635.298681, 682.357502, 668.926525);
    const vec3 ZINC = vec3(48.500388, 65.294118, 63.934599);

    gridcell.xyz = gridcell.xyz - floor(gridcell.xyz * (1.0 / DOMAIN)) * DOMAIN;
    float d = DOMAIN - 1.5;
    vec3 gridcell_inc1 = step(gridcell, vec3(d, d, d)) * (gridcell + 1.0);

    gridcell_inc1 = tile ? mod(gridcell_inc1, s) : gridcell_inc1;

    vec4 P = vec4(gridcell.xy, gridcell_inc1.xy) + OFFSET.xyxy;
    P *= P;
    P = P.xzxz * P.yyww;
    vec3 lowz_mod = vec3(1.0 / (SOMELARGEFLOATS.xyz + gridcell.zzz * ZINC.xyz));
    vec3 highz_mod = vec3(1.0 / (SOMELARGEFLOATS.xyz + gridcell_inc1.zzz * ZINC.xyz));
    lowz_hash_0 = fract(P * lowz_mod.xxxx);
    highz_hash_0 = fract(P * highz_mod.xxxx);
    lowz_hash_1 = fract(P * lowz_mod.yyyy);
    highz_hash_1 = fract(P * highz_mod.yyyy);
    lowz_hash_2 = fract(P * lowz_mod.zzzz);
    highz_hash_2 = fract(P * highz_mod.zzzz);
}

float Perlin(vec3 P, float s, bool tile)
{
    P *= s;

    vec3 Pi = floor(P);
    vec3 Pi2 = floor(P);
    vec3 Pf = P - Pi;
    vec3 Pf_min1 = Pf - 1.0;

    vec4 hashx0, hashy0, hashz0, hashx1, hashy1, hashz1;
    PerlinHash(Pi2, s, tile, hashx0, hashy0, hashz0, hashx1, hashy1, hashz1);

    vec4 grad_x0 = hashx0 - 0.49999;
    vec4 grad_y0 = hashy0 - 0.49999;
    vec4 grad_z0 = hashz0 - 0.49999;
    vec4 grad_x1 = hashx1 - 0.49999;
    vec4 grad_y1 = hashy1 - 0.49999;
    vec4 grad_z1 = hashz1 - 0.49999;
    vec4 grad_results_0 = inversesqrt(grad_x0 * grad_x0 + grad_y0 * grad_y0 + grad_z0 * grad_z0) * (vec2(Pf.x, Pf_min1.x).xyxy * grad_x0 + vec2(Pf.y, Pf_min1.y).xxyy * grad_y0 + Pf.zzzz * grad_z0);
    vec4 grad_results_1 = inversesqrt(grad_x1 * grad_x1 + grad_y1 * grad_y1 + grad_z1 * grad_z1) * (vec2(Pf.x, Pf_min1.x).xyxy * grad_x1 + vec2(Pf.y, Pf_min1.y).xxyy * grad_y1 + Pf_min1.zzzz * grad_z1);

    vec3 blend = Interpolation_C2(Pf);
    vec4 res0 = mix(grad_results_0, grad_results_1, blend.z);
    vec4 blend2 = vec4(blend.xy, vec2(1.0 - blend.xy));
    float final = dot(res0, blend2.zxzx * blend2.wwyy);
    final *= 1.0 / sqrt(0.75);
    return ((final * 1.5) + 1.0) * 0.5;
}

float GetPerlin_5_Octaves(vec3 p, bool tile)
{
    vec3 xyz = p;
    float amplitude_factor = 0.5;
    float frequency_factor = 2.0;

    float a = 1.0;
    float perlin_value = 0.0;
    perlin_value += a * Perlin(xyz, 1, tile).r;
    a *= amplitude_factor;
    xyz *= (frequency_factor + 0.02);
    perlin_value += a * Perlin(xyz, 1, tile).r;
    a *= amplitude_factor;
    xyz *= (frequency_factor + 0.03);
    perlin_value += a * Perlin(xyz, 1, tile).r;
    a *= amplitude_factor;
    xyz *= (frequency_factor + 0.01);
    perlin_value += a * Perlin(xyz, 1, tile).r;
    a *= amplitude_factor;
    xyz *= (frequency_factor + 0.01);
    perlin_value += a * Perlin(xyz, 1, tile).r;

    return perlin_value;
}

float GetPerlin_5_Octaves(vec3 p, float s, bool tile)
{
    vec3 xyz = p;
    float f = 1.0;
    float a = 1.0;

    float perlin_value = 0.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;

    return perlin_value;
}

float GetPerlin_3_Octaves(vec3 p, float s, bool tile)
{
    vec3 xyz = p;
    float f = 1.0;
    float a = 1.0;

    float perlin_value = 0.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;

    return perlin_value;
}

float GetPerlin_7_Octaves(vec3 p, float s, bool tile)
{
    vec3 xyz = p;
    float f = 1.0;
    float a = 1.0;

    float perlin_value = 0.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;
    a *= 0.5;
    f *= 2.0;
    perlin_value += a * Perlin(xyz, s * f, tile).r;

    return perlin_value;
}

// -------------------------------------------------------------------------
//    Worley Noise
// -------------------------------------------------------------------------

vec3 VoronoiHash(vec3 x, float s)
{
    x = mod(x, s);
    x = vec3(dot(x, vec3(127.1, 311.7, 74.7)),
							dot(x, vec3(269.5, 183.3, 246.1)),
							dot(x, vec3(113.5, 271.9, 124.6)));
				
    return fract(sin(x) * 43758.5453123);
}

vec3 Voronoi(in vec3 x, float s, float seed, bool inverted)
{
    x *= s;
    x += 0.5;
    vec3 p = floor(x);
    vec3 f = fract(x);

    float id = 0.0;
    vec2 res = vec2(1.0, 1.0);
    for (int k = -1; k <= 1; k++)
    {
        for (int j = -1; j <= 1; j++)
        {
            for (int i = -1; i <= 1; i++)
            {
                vec3 b = vec3(i, j, k);
                vec3 r = b - f + VoronoiHash(p + b + seed * 10, s);
                float d = dot(r, r);

                if (d < res.x)
                {
                    id = dot(p + b, vec3(1.0, 57.0, 113.0));
                    res = vec2(d, res.x);
                }
                else if (d < res.y)
                {
                    res.y = d;
                }
            }
        }
    }

    vec2 result = res;
    id = abs(id);

    if (inverted)
        return vec3(1.0 - result, id);
    else
        return vec3(result, id);
}

float GetWorley_2_Octaves(vec3 p, float s, float seed)
{
    vec3 xyz = p;

    float worley_value1 = Voronoi(xyz, 1.0 * s, seed, true).r;
    float worley_value2 = Voronoi(xyz, 2.0 * s, seed, false).r;

    worley_value1 = clamp(worley_value1, 0.0, 1.0);
    worley_value2 = clamp(worley_value2, 0.0, 1.0);

    float worley_value = worley_value1;
    worley_value = worley_value - worley_value2 * 0.25;

    return worley_value;
}

float GetWorley_2_Octaves(vec3 p, float s)
{
    return GetWorley_2_Octaves(p, s, 0);
}

float GetWorley_3_Octaves(vec3 p, float s, float seed)
{
    vec3 xyz = p;

    float worley_value1 = Voronoi(xyz, 1.0 * s, seed, true).r;
    float worley_value2 = Voronoi(xyz, 2.0 * s, seed, false).r;
    float worley_value3 = Voronoi(xyz, 4.0 * s, seed, false).r;

    worley_value1 = clamp(worley_value1, 0.0, 1.0);
    worley_value2 = clamp(worley_value2, 0.0, 1.0);
    worley_value3 = clamp(worley_value3, 0.0, 1.0);

    float worley_value = worley_value1;
    worley_value = worley_value - worley_value2 * 0.3;
    worley_value = worley_value - worley_value3 * 0.3;

    return worley_value;
}

float GetWorley_3_Octaves(vec3 p, float s)
{
    return GetWorley_3_Octaves(p, s, 0);
}

// -------------------------------------------------------------------------
//    Curl Noise
// -------------------------------------------------------------------------

vec3 CurlNoise(vec3 pos)
{
    float e = 0.05;
    float n1, n2, a, b;
    vec3 c;

    n1 = GetPerlin_5_Octaves(pos.xyz + vec3(0, e, 0), false);
    n2 = GetPerlin_5_Octaves(pos.xyz + vec3(0, -e, 0), false);
    a = (n1 - n2) / (2 * e);
    n1 = GetPerlin_5_Octaves(pos.xyz + vec3(0, 0, e), false);
    n2 = GetPerlin_5_Octaves(pos.xyz + vec3(0, 0, -e), false);
    b = (n1 - n2) / (2 * e);

    c.x = a - b;

    n1 = GetPerlin_5_Octaves(pos.xyz + vec3(0, 0, e), false);
    n2 = GetPerlin_5_Octaves(pos.xyz + vec3(0, 0, -e), false);
    a = (n1 - n2) / (2 * e);
    n1 = GetPerlin_5_Octaves(pos.xyz + vec3(e, 0, 0), false);
    n2 = GetPerlin_5_Octaves(pos.xyz + vec3(-e, 0, 0), false);
    b = (n1 - n2) / (2 * e);

    c.y = a - b;

    n1 = GetPerlin_5_Octaves(pos.xyz + vec3(e, 0, 0), false);
    n2 = GetPerlin_5_Octaves(pos.xyz + vec3(-e, 0, 0), false);
    a = (n1 - n2) / (2 * e);
    n1 = GetPerlin_5_Octaves(pos.xyz + vec3(0, e, 0), false);
    n2 = GetPerlin_5_Octaves(pos.xyz + vec3(0, -e, 0), false);
    b = (n1 - n2) / (2 * e);

    c.z = a - b;

    return c;
}

vec3 DecodeCurlNoise(vec3 c)
{
    return (c - 0.5) * 2.0;
}

vec3 EncodeCurlNoise(vec3 c)
{
    return (c + 1.0) * 0.5;
}

// -------------------------------------------------------------------------
//    Shared Utils
// -------------------------------------------------------------------------

// Remap an original value from an old range (minimum and maximum) to a new one.
float Remap(float original_value, float original_min, float original_max, float new_min, float new_max)
{
    return new_min + (((original_value - original_min) / (original_max - original_min)) * (new_max - new_min));
}

// Remap an original value from an old range (minimum and maximum) to a new one, clamped to the new range.
float RemapClamped(float original_value, float original_min, float original_max, float new_min, float new_max)
{
    return new_min + (clamp((original_value - original_min) / (original_max - original_min), 0.0, 1.0) * (new_max - new_min));
}

float Remap01(float value, float low, float high)
{
    return clamp((value - low) / (high - low), 0.0, 1.0);
}

float Remap01Unsaturated(float value, float low, float high)
{
    return (value - low) / (high - low);
}

vec3 Remap01Unsaturated(vec3 value, float low, float high)
{
    return (value - vec3(low)) / (high - low);
}

float DilatePerlinWorley(float p, float w, float x)
{
    float curve = 0.75;
    if (x < 0.5)
    {
        x = x / 0.5;
        float n = p + w * x;
        return n * mix(1, 0.5, pow(x, curve));
    }
    else
    {
        x = (x - 0.5) / 0.5;
        float n = w + p * (1.0 - x);
        return n * mix(0.5, 1.0, pow(x, 1.0 / curve));
    }
}

// -------------------------------------------------------------------------
//    Shared Utils
// -------------------------------------------------------------------------

struct CloudLayer
{
	vec3 windDirection;
	vec3 windOffset;
	vec2 coverageWindDirection;
	vec2 coverageWindOffset;
};

CloudLayer SampleCloudLayer()
{
	CloudLayer cloudLayer;
	
	float animation = uAnimationMultiplier * uTime;

	cloudLayer.windDirection = vec3(cos(uWindAngle), -uWindUpAmount, sin(uWindAngle));
	cloudLayer.windOffset = uWindSpeed * cloudLayer.windDirection * animation;

	cloudLayer.coverageWindDirection = vec2(cos(uCoverageWindAngle), sin(uCoverageWindAngle));
	cloudLayer.coverageWindOffset = uCoverageWindSpeed * cloudLayer.coverageWindDirection * animation;

	return cloudLayer;
}

float GetHeightFractionForPoint(vec3 pos)
{
    float planetRadiusM = uEarthRadius * SKY_UNIT_TO_M;
	vec3 planetCenterWorldM = uEarthCenter * SKY_UNIT_TO_M;

	return clamp((distance(pos, planetCenterWorldM) - (planetRadiusM + uCloudStartHeight)) / uCloudThickness, 0.0, 1.0);
}

float SampleGradient(vec4 gradient, float heightFraction)
{
	return smoothstep(gradient.x, gradient.y, heightFraction) - smoothstep(gradient.z, gradient.w, heightFraction);
}

vec3 GetWeatherTypeMask(vec3 weatherData)
{
	float weatherType = weatherData.g;

	float smallType = 1.0f - clamp(weatherType * 2.0f, 0.0, 1.0);
	float mediumType = 1.0f - abs(weatherType - 0.5f) * 2.0f;
	float largeType = clamp(weatherType - 0.5f, 0.0, 1.0) * 2.0f;

	return vec3(smallType, mediumType, largeType);
}

vec4 GetHeightGradientType(vec3 weatherTypeMask)
{
	return (uGradientSmall * weatherTypeMask.r) + (uGradientMedium * weatherTypeMask.g) + (uGradientLarge * weatherTypeMask.b);
}

float SampleHeightGradient(float heightFraction, vec3 weatherData)
{
	vec3 weatherTypeMask = GetWeatherTypeMask(weatherData);

	vec4 heightGradient = GetHeightGradientType(weatherTypeMask);

	return SampleGradient(heightGradient, heightFraction);
}

vec3 SampleWeather(sampler2D texture_weatherMap, vec3 p, float heightFraction, CloudLayer layer)
{
	vec2 pos = p.xz + layer.coverageWindOffset;
	pos += heightFraction * layer.coverageWindDirection * uSkewAlongCoverageWindDirection;

	vec3 weatherData = texture(texture_weatherMap, pos * uWeatherScale).rgb;

    // Apply effects for coverage
	weatherData.r = RemapClamped(weatherData.r * uCoverageAmount, 0.0, 1.0, uCoverageMinimum, 1.0);
	weatherData.g = RemapClamped(weatherData.g * uTypeAmount,     0.0, 1.0, uTypeMinimum,     1.0);
	weatherData.b = RemapClamped(weatherData.b * uRainAmount,     0.0, 1.0, uRainMinimum,     1.0);
	
	return weatherData;
}

bool ValidCloudDensity(float heightFraction, vec3 weatherData)
{
	vec3 weatherTypeMask = GetWeatherTypeMask(weatherData);

	vec4 heightGradient = GetHeightGradientType(weatherTypeMask);

	float coverage = weatherData.r;
	
	return coverage > 0.05 && heightGradient.r < heightFraction && heightGradient.a > heightFraction;
}

float SampleCloudDensity(sampler3D texture_shapeNoise, sampler3D texture_detailNoise, sampler2D texture_curlNoise, vec3 p, float heightFraction, CloudLayer layer, vec3 weatherData, float lod, bool sampleDetail)
{
	// When can estimate the bounding area of the clouds if we know the height on the heightfraction and the cloud coverage area
	if (!ValidCloudDensity(heightFraction, weatherData)) return 0.0;

	vec3 lowFrequencyPos = vec3(p.x, p.y + (p.x / 3.0f + p.z / 7.0f), p.z); // Offset to avoid repeting pattern for top-down view
	lowFrequencyPos += layer.windOffset;
	lowFrequencyPos += heightFraction * layer.windDirection * uSkewAlongWindDirection;

	vec4 lowFrequencyNoises = textureLod(texture_shapeNoise, lowFrequencyPos * uTotalNoiseScale, lod);

	// Create an FBM out of the low-frequency Perlin-Worley Noises
	float lowFrequencyFBM = (lowFrequencyNoises.g * 0.625) + (lowFrequencyNoises.b * 0.25) + (lowFrequencyNoises.a * 0.125);
	lowFrequencyFBM = clamp(lowFrequencyFBM, 0.0, 1.0);

	float cloudSample = Remap(lowFrequencyNoises.r, -(1.0 - lowFrequencyFBM), 1.0, 0.0, 1.0);

	// Apply height gradients
	float densityHeightGradient = SampleHeightGradient(heightFraction, weatherData);
	cloudSample *= densityHeightGradient;

	float cloudCoverage = weatherData.r;
	
	// Apply Coverage to sample
	cloudSample = Remap(cloudSample, 1.0 - cloudCoverage, 1.0, 0.0, 1.0);
	cloudSample *= cloudCoverage;

    // Erode with detail noise if cloud sample > 0
	if (cloudSample > 0.0 && sampleDetail)
	{
		vec3 highFrequencyPos = p + layer.windOffset;
		highFrequencyPos += heightFraction * layer.windDirection * uSkewAlongWindDirection;
		
        // Apply our curl noise to erode with tiny details.
		vec3 curlNoise = DecodeCurlNoise(texture(texture_curlNoise, p.xz * uCurlScale * uTotalNoiseScale).rgb);
		highFrequencyPos += vec3(curlNoise.r, curlNoise.b, curlNoise.g) * clamp(pow(1.0 - heightFraction, uCurlNoiseHeightFraction), 0.0, 1.0) * uCurlNoiseModifier;

		vec3 highFrequencyNoises = textureLod(texture_detailNoise, highFrequencyPos * uDetailScale * uTotalNoiseScale, lod).rgb;

        // Create an FBM out of the high-frequency Worley Noises
		float highFrequencyFBM = (highFrequencyNoises.r * 0.625) + (highFrequencyNoises.g * 0.25) + (highFrequencyNoises.b * 0.125);
		highFrequencyFBM = clamp(highFrequencyFBM, 0.0, 1.0);
		
        // Dilate detail noise based on height
		float highFrequenceNoiseModifier = mix(1.0 - highFrequencyFBM, highFrequencyFBM, clamp(heightFraction * uDetailNoiseHeightFraction, 0.0, 1.0));

        // Erode with base of clouds
		cloudSample = Remap(cloudSample, highFrequenceNoiseModifier * uDetailNoiseModifier, 1.0, 0.0, 1.0);
	}
	
	return max(cloudSample, 0.0);
}

vec3 SampleAlbedo(float density, vec3 weatherData)
{
	vec3 albedo = density * uAlbedo;
	albedo  = pow(clamp(albedo * uBeerPowder, 0.0, 1.0), vec3(uBeerPowderPower)); // Artistic approach
	albedo *= (1.0 - weatherData.b);

	return albedo;
}

vec3 SampleExtinction(float density)
{
	vec3 extinction = density * uExtinctionCoefficient;
	return extinction;
}

#endif // VOLUMETRICCLOUDS_UTILS_GLSL