#include "PostFXRenderPass.h"

#include "Renderer.h"
#include "Shader/Material.h"
#include "Texture/FramebufferObject.h"

PostFXRenderPass::PostFXRenderPass(std::shared_ptr<Material> material, std::shared_ptr<const FramebufferObject> framebuffer)
    : RenderPass(framebuffer), m_material(material)
{
}

void PostFXRenderPass::Render()
{
    Renderer& renderer = GetRenderer();

    assert(m_material);
    m_material->Use();

    const Mesh* mesh = &renderer.GetFullscreenMesh();
    mesh->DrawSubmesh(0);
}
