#include "SceneModel.h"

#include "Geometry/Model.h"
#include "Geometry/Mesh.h"
#include "Transform.h"
#include "SceneVisitor.h"
#include <cassert>

SceneModel::SceneModel(const std::string& name, std::shared_ptr<Model> model) : SceneNode(name), m_model(model)
{
}

SceneModel::SceneModel(const std::string& name, std::shared_ptr<Model> model, std::shared_ptr<Transform> transform) : SceneNode(name, transform), m_model(model)
{
}

std::shared_ptr<Model> SceneModel::GetModel() const
{
    return m_model;
}

void SceneModel::SetModel(std::shared_ptr<Model> model)
{
    m_model = model;
}

SphereBounds SceneModel::GetSphereBounds() const
{
    return SphereBounds(GetBoxBounds());
}

AabbBounds SceneModel::GetAabbBounds() const
{
    return AabbBounds(GetBoxBounds());
}

BoxBounds SceneModel::GetBoxBounds() const
{
    assert(m_transform);
    assert(m_model);
    return BoxBounds(m_transform->GetTranslation(), m_transform->GetRotationMatrix(), m_transform->GetScale() /* * m_model->GetSize()*/);
}

void SceneModel::AcceptVisitor(SceneVisitor& visitor)
{
    visitor.VisitModel(*this);
    //visitor.VisitRenderable(*this);
}

void SceneModel::AcceptVisitor(SceneVisitor& visitor) const
{
    visitor.VisitModel(*this);
    //visitor.VisitRenderable(*this);
}
