#include "VolumetricCloudsRenderUtils.h"

#include <array>
#include <algorithm>
#include <cmath>

#include <glm/gtc/constants.hpp>

namespace
{

    // ----------------------------------------------------------------------------------------------
    // Constants borrowed from the previous Three.js implementation ---------------------------------
    // ----------------------------------------------------------------------------------------------
    //  The ranges have been clamped to safe intervals so that accidental parameter tweaks never
    //  explode the colour computation.
    // ----------------------------------------------------------------------------------------------
    constexpr float kDefaultTurbidity = 1.7f;                  // ~2 (clear) .. 10 (hazy)
    constexpr float kDefaultAngstromAlpha = 1.6f;              // 1.0 .. 2.0 typical

    constexpr float kMinTurbidity = 1.0f;
    constexpr float kMinAlpha = 0.5f;
    constexpr float kMaxAlpha = 2.5f;

    constexpr float kRayleighScale = 0.008735f;                // λ^-4
    constexpr float kMieScaleBase = 0.00021f;                  // λ^-α, scaled by (turbidity - 1)

    constexpr std::array<float, 3> kWavelengthsMicrometers = { 0.68f, 0.55f, 0.44f };
    constexpr std::array<float, 3> kLumaWeights = { 0.2126f, 0.7152f, 0.0722f };

    constexpr float kMinimumLuma = 1.0e-4f;
}

// --------------------------------------------------------------------------------------------------
// ComputeAtmosphericSunLight
// --------------------------------------------------------------------------------------------------
glm::vec3 VolumetricCloudsRenderUtils::ComputeAtmosphericSunLight(const glm::vec3& sunDirection)
{
    // ----------------------------------------------------------------------------------------------
    // Normalise the direction ---------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    const float directionLength = glm::length(sunDirection);

    if (directionLength <= 0.0f)
    {
        return glm::vec3(0.0f);
    }

    const glm::vec3 unitDirection = sunDirection / directionLength;

    // Clamp the dot product so asin stays in range.
    const float clampedUpDot = glm::clamp(unitDirection.y, -1.0f, 1.0f);

    // ----------------------------------------------------------------------------------------------
    // Elevation / horizon handling ----------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    const float elevationRadians = std::asin(clampedUpDot);

    if (elevationRadians <= 0.0f)
    {
        // Sun is below the horizon – return darkness.
        return glm::vec3(0.0f);
    }

    // ----------------------------------------------------------------------------------------------
    // Air mass computation (Kasten–Young 1989) ----------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    const float elevationDegrees = elevationRadians * (180.0f / glm::pi<float>());
    const float sineElevation = std::sin(elevationRadians);

    const float airMass = 1.0f / (sineElevation + 0.50572f * std::pow(elevationDegrees + 6.07995f, -1.6364f));

    // ----------------------------------------------------------------------------------------------
    // Atmospheric parameters ----------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    const float turbidity = std::max(kMinTurbidity, kDefaultTurbidity);
    constexpr float alpha = glm::clamp(kDefaultAngstromAlpha, kMinAlpha, kMaxAlpha);

    glm::vec3 transmittance(0.0f);
    glm::vec3 zenithTransmittance(0.0f);

    // ----------------------------------------------------------------------------------------------
    // Per-channel scattering terms ----------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    for (std::size_t channel = 0; channel < 3U; ++channel)
    {
        const float wavelength = kWavelengthsMicrometers[channel];

        const float betaRayleigh = kRayleighScale * std::pow(wavelength, -4.0f);
        const float betaMie = kMieScaleBase * (turbidity - 1.0f) * std::pow(wavelength, -alpha);

        const float opticalDepth = (betaRayleigh + betaMie) * airMass;
        const float opticalDepthZenith = (betaRayleigh + betaMie);

        const float transmittanceAtMass = std::exp(-opticalDepth);
        const float transmittanceZenith = std::exp(-opticalDepthZenith);

        transmittance[channel] = transmittanceAtMass;
        zenithTransmittance[channel] = transmittanceZenith;
    }

    // ----------------------------------------------------------------------------------------------
    // Exposure normalisation ----------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    const float zenithLuma = std::max(kMinimumLuma,
                                      zenithTransmittance.x * kLumaWeights[0] +
                                      zenithTransmittance.y * kLumaWeights[1] +
                                      zenithTransmittance.z * kLumaWeights[2]);

    const float exposure = 1.0f / zenithLuma;

    // ----------------------------------------------------------------------------------------------
    // Final colour --------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    glm::vec3 sunlightColour = exposure * transmittance;

    sunlightColour.x = std::max(0.0f, sunlightColour.x);
    sunlightColour.y = std::max(0.0f, sunlightColour.y);
    sunlightColour.z = std::max(0.0f, sunlightColour.z);

    return sunlightColour;
}


