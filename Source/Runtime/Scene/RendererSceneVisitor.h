#pragma once

#include "SceneVisitor.h"
#include <memory>

class Renderer;
class SceneCamera;
class SceneLight;
class SceneModel;
class Transform;

class RendererSceneVisitor : public SceneVisitor
{
public:
    RendererSceneVisitor(std::shared_ptr<Renderer> renderer);

    void VisitCamera(SceneCamera& sceneCamera) override;

    void VisitLight(SceneLight& sceneLight) override;

    void VisitModel(SceneModel& sceneModel) override;

private:
    std::shared_ptr<Renderer> m_renderer;
};
