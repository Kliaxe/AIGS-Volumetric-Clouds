#include "RendererSceneVisitor.h"

#include "Renderer/Renderer.h"
#include "SceneCamera.h"
#include "SceneLight.h"
#include "SceneModel.h"
#include "Transform.h"

RendererSceneVisitor::RendererSceneVisitor(std::shared_ptr<Renderer> renderer) : m_renderer(renderer)
{
}

void RendererSceneVisitor::VisitCamera(SceneCamera& sceneCamera)
{
    assert(!m_renderer->HasCamera()); // Currently, only one camera per scene supported
    m_renderer->SetCurrentCamera(*sceneCamera.GetCamera());
}

void RendererSceneVisitor::VisitLight(SceneLight& sceneLight)
{
    m_renderer->AddLight(*sceneLight.GetLight());
}

void RendererSceneVisitor::VisitModel(SceneModel& sceneModel)
{
    assert(sceneModel.GetTransform());
    m_renderer->AddModel(*sceneModel.GetModel(), sceneModel.GetTransform()->GetTransformMatrix());
}
