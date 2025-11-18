#include "ShaderStorageBufferObject.h"

ShaderStorageBufferObject::ShaderStorageBufferObject()
{
    // Nothing to do here, it is done by the base class
}

// Call the base implementation with Usage::DynamicCopy
void ShaderStorageBufferObject::AllocateData(size_t size)
{
    AllocateData(size, Usage::StaticCopy);
}

// Call the base implementation with Usage::DynamicCopy
void ShaderStorageBufferObject::AllocateData(std::span<const std::byte> data)
{
    AllocateData(data, Usage::StaticCopy);
}
