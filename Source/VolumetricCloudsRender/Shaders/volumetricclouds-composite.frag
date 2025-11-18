#version 460 core

// Inputs --------------------------------------------------------------
in vec2 TexCoord;

// Outputs -------------------------------------------------------------
out vec4 FragColor;

// Uniforms ------------------------------------------------------------
uniform sampler2D CloudsTexture;

// Main ----------------------------------------------------------------
void main()
{
    vec4 cloudSample = texture(CloudsTexture, TexCoord);

    // Cloud color is already stored as premultiplied RGB with alpha = 1 - transmittance.
    FragColor = cloudSample;
}


