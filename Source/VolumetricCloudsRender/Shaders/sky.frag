#version 460 core

// Inputs --------------------------------------------------------------
in vec2 TexCoord;

// Outputs -------------------------------------------------------------
out vec4 FragColor;

// Uniforms ------------------------------------------------------------
uniform sampler2D SkyTexture;
uniform mat4 InvViewProjMatrix;
uniform vec3 CameraPosition;

// Constants -----------------------------------------------------------
const float PI = 3.1415926535897932384626433832795;

// Utility -------------------------------------------------------------
vec2 dirToEquirectUV(vec3 d)
{
    d = normalize(d);

    const float phi = atan(d.z, d.x);
    const float theta = asin(clamp(d.y, -1.0, 1.0));

    const float u = 0.5 + (phi / (2.0 * PI));
    const float v = 0.5 + (theta / PI);

    return vec2(u, v);
}

// Main ----------------------------------------------------------------
void main()
{
    // Reconstruct normalized device coordinates from the fullscreen vertex shader
    vec2 ndc = TexCoord * 2.0 - 1.0;

    // Push the sample ray towards the far plane to avoid precision issues
    const vec4 clip = vec4(ndc, 1.0, 1.0);

    // Convert the clip position back to world-space
    vec4 worldPosition = InvViewProjMatrix * clip;
    worldPosition /= worldPosition.w;

    // Build the view direction (flip Z to reconcile GL right-handed space with panorama convention)
    vec3 viewDir = worldPosition.xyz - CameraPosition;
    viewDir.z = -viewDir.z;

    // Convert direction to equirectangular texture coordinates
    vec2 uv = dirToEquirectUV(viewDir);
    uv.y = 1.0 - uv.y;

    // Sample the HDR environment at the highest fidelity
    FragColor = textureLod(SkyTexture, uv, 0.0);
}


