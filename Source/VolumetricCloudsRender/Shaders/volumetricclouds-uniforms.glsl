
// -------------------------------------------------------------------------
//    Lighting
// -------------------------------------------------------------------------

uniform vec3  uAlbedo = vec3(0.9f, 0.9f, 0.9f); // Cloud albedo is normally very close to white
uniform vec3  uExtinctionCoefficient = vec3(0.71f, 0.86f, 1.0f);
uniform float uExtinctionCoefficientMultiplier = 0.025f;
uniform vec3  uEarthGroundAlbedo = vec3(0.3f, 0.3f, 0.3f);

uniform vec3  uSunDirection = vec3(0.0f, 1.0f, 0.0f);
uniform vec3  uSunIlluminance = vec3(8.0f, 8.0f, 7.0f);
uniform vec3  uAmbientLuminance = vec3(0.5f, 0.6f, 0.7f);

uniform float uBeerPowder = 20.0f;
uniform float uBeerPowderPower = 0.5f;
uniform float uAmbientGroundMultiplier = 0.75f; // [0; 1] Amount of ambient light to reach the bottom of clouds
uniform float uPhaseG = 0.5f; // [-0.999; 0.999]

uniform float uPhaseG2 = -0.5f; // [-0.999; 0.999]
uniform float uPhaseBlend = 0.2f; // [0; 1]
uniform float uMultiScatteringScattering = 1.0f;
uniform float uMultiScatteringExtinction = 0.1f;

uniform float uMultiScatteringEccentricity = 0.2f;
uniform float uShadowStepLength = 3000.0f;
uniform float uHorizonBlendAmount = 0.0000125f;
uniform float uHorizonBlendPower = 2.0f;

uniform bool  uJitterEnabled = true; // Enable per-ray white-noise jitter in the raymarcher

// -------------------------------------------------------------------------
//    Modelling
// -------------------------------------------------------------------------

uniform float uCloudStartHeight = 1500.0f;
uniform float uCloudThickness = 5000.0f;
uniform float uEarthRadius = 6360.0f;
uniform vec3  uEarthCenter = vec3(0.0f, -6360.0f, 0.0f);

uniform float uSkewAlongWindDirection = 700.0f;
uniform float uTotalNoiseScale = 0.0006f;
uniform float uCurlScale = 0.3f;
uniform float uCurlNoiseHeightFraction = 5.0f;

uniform float uCurlNoiseModifier = 500.0f;
uniform float uDetailScale = 4.0f;
uniform float uDetailNoiseHeightFraction = 10.0f;
uniform float uDetailNoiseModifier = 0.3f;

uniform float uSkewAlongCoverageWindDirection = 2500.0f;
uniform float uWeatherScale = 0.00002f;
uniform float uCoverageAmount = 1.0f;
uniform float uCoverageMinimum = 0.0f;

uniform float uTypeAmount = 1.0f;
uniform float uTypeMinimum = 0.0f;
uniform float uRainAmount = 0.0f; // Rain clouds disabled by default.
uniform float uRainMinimum = 0.0f;

// Cloud types: 4 positions of a black, white, white, black gradient
uniform vec4  uGradientSmall = vec4(0.01f, 0.1f, 0.11f, 0.2f);
uniform vec4  uGradientMedium = vec4(0.01f, 0.08f, 0.3f, 0.4f);
uniform vec4  uGradientLarge = vec4(0.01f, 0.06f, 0.75f, 0.95f);

// -------------------------------------------------------------------------
//    Animation
// -------------------------------------------------------------------------

uniform float uWindSpeed = 15.0f;
uniform float uWindAngle = 0.75f;
uniform float uWindUpAmount = 0.5f;
uniform float uCoverageWindSpeed = 30.0f;

uniform float uCoverageWindAngle = 0.0f;
uniform float uAnimationMultiplier = 10.0f;
uniform float uTime;

// -------------------------------------------------------------------------
//    Performance
// -------------------------------------------------------------------------

uniform int   uMaxStepCount = 256; // Maximum number of iterations. Higher gives better images but may be slow.
uniform float uMaxMarchingDistance = 90000.0f; // Clamping the marching steps to be within a certain distance.
uniform float uInverseDistanceStepCount = 15000.0f; // Distance over which the raymarch steps will be evenly distributed.
uniform float uRenderDistance = 70000.0f; // Maximum distance to march before returning a miss.

uniform float uLODDistance = 30000.0f; // After a certain distance, noises will get higher LOD
uniform float uLODMin = 0.0f; // 
uniform float uBigStepMarch = 2.0f; // How long inital rays should be until they hit something. Lower values may give a better image but may be slower.
uniform float uTransmittanceThreshold = 0.005f; // Default: 0.005. If the clouds transmittance has reached it's desired opacity, there's no need to keep raymarching for performance.

uniform float uShadowSampleCount = 5.0f;
uniform float uGroundContributionSampleCount = 3.0f;