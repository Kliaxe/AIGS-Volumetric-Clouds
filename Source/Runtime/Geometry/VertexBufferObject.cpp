#include "VertexBufferObject.h"

VertexBufferObject::VertexBufferObject()
{
    // Nothing to do here, it is done by the base class
}

// Call the base implementation with Usage::StaticDraw
void VertexBufferObject::AllocateData(size_t size)
{
    AllocateData(size, Usage::StaticRead);
}

// Call the base implementation with Usage::StaticDraw
void VertexBufferObject::AllocateData(std::span<const std::byte> data)
{
    AllocateData(data, Usage::StaticRead);
}
