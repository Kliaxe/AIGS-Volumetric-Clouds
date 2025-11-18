#pragma once

#include <glm/glm.hpp>

// --------------------------------------------------------------------------------------------------
// VolumetricCloudsRenderUtils
// --------------------------------------------------------------------------------------------------
//  Collection of helper utilities shared across the volumetric clouds sample.  Everything here is
//  defined as static so we can call these helpers from anywhere without instantiating stateful
//  objects.  Think of this header as the grab bag for math helpers that do not belong to a single
//  subsystem.
// --------------------------------------------------------------------------------------------------
class VolumetricCloudsRenderUtils
{
public:

    // ----------------------------------------------------------------------------------------------
    // Atmospheric lighting helpers ----------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------
    //  computeAtmosphericSunLight                                                             
    //  -----------------------------------------------------------------------------------------
    //  Approximates the visible sunlight colour for a given sun direction using a lightweight
    //  Rayleigh/Mie scattering model.  The input direction is expected to be a unit vector pointing
    //  FROM the ground TOWARDS the sun (y-up).  The returned RGB values are linear and include the
    //  exposure so the caller can feed them directly into the renderer.
    // ----------------------------------------------------------------------------------------------
    static glm::vec3 ComputeAtmosphericSunLight(const glm::vec3& sunDirection);
};


