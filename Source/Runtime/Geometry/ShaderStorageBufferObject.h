#pragma once

#include "Core/BufferObject.h"
#include "Core/Data.h"

// Shader Storage Buffer Object (SSBO)
class ShaderStorageBufferObject : public BufferObjectBase<BufferObject::ShaderStorageBuffer>
{
public:
    ShaderStorageBufferObject();

    // (C++) 3
    // Use the same AllocateData methods from the base class
    using BufferObject::AllocateData;
    // Additionally, provide AllocateData methods with StaticCopy as default usage
    void AllocateData(size_t size);
    void AllocateData(std::span<const std::byte> data);
    // Additionally, provide AllocateData template method for any type of data span, specifying the VertexAttribute that it contains
    template<typename T>
    void AllocateData(std::span<const T> data, Usage usage = Usage::StaticCopy);
    template<typename T>
    inline void AllocateData(std::span<T> data, Usage usage = Usage::StaticCopy) { AllocateData(std::span<const T>(data), usage); }

    // (C++) 3
    // Use the same UpdateData methods from the base class
    using BufferObject::UpdateData;
    // Additionally, provide UpdateData template method for any type of data span
    template<typename T>
    void UpdateData(std::span<const T> data, size_t offsetBytes = 0);
    template<typename T>
    inline void UpdateData(std::span<T> data, size_t offsetBytes = 0) { UpdateData(std::span<const T>(data), offsetBytes); }
};


// Call the base implementation with the span converted to bytes
template<typename T>
void ShaderStorageBufferObject::AllocateData(std::span<const T> data, Usage usage)
{
    AllocateData(Data::GetBytes(data), usage);
}

// Call the base implementation with the span converted to bytes
template<typename T>
void ShaderStorageBufferObject::UpdateData(std::span<const T> data, size_t offsetBytes)
{
    UpdateData(Data::GetBytes(data), offsetBytes);
}
