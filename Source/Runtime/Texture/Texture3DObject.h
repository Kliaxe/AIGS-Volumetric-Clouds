#pragma once

#include "TextureObject.h"
#include "Core/Data.h"

// Texture object in3 dimensions
class Texture3DObject : public TextureObjectBase<TextureObject::Texture3D>
{
public:
 Texture3DObject();

 // Initialize the texture3D with a specific format
 void SetImage(GLint level,
 GLsizei width, GLsizei height, GLsizei depth,
 Format format, InternalFormat internalFormat);

 // Initialize the texture3D with a specific format and initial data
 template <typename T>
 void SetImage(GLint level,
 GLsizei width, GLsizei height, GLsizei depth,
 Format format, InternalFormat internalFormat,
 std::span<const T> data, Data::Type type = Data::Type::None);

 void BindImageTexture(GLuint unit, GLuint level, GLboolean layered, GLint layer, GLenum access, InternalFormat internalFormat);
};

// Set image with data in bytes
template <>
void Texture3DObject::SetImage<std::byte>(GLint level, GLsizei width, GLsizei height, GLsizei depth, Format format, InternalFormat internalFormat, std::span<const std::byte> data, Data::Type type);

// Template method to set image with any kind of data
template <typename T>
inline void Texture3DObject::SetImage(GLint level, GLsizei width, GLsizei height, GLsizei depth,
 Format format, InternalFormat internalFormat, std::span<const T> data, Data::Type type)
{
 if (type == Data::Type::None)
 {
 type = Data::GetType<T>();
 }
 SetImage(level, width, height, depth, format, internalFormat, Data::GetBytes(data), type);
}
