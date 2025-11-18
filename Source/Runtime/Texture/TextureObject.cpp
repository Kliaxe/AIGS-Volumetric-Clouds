#include "TextureObject.h"

#include <cassert>
#include <algorithm>

TextureObject::TextureObject() : Object(NullHandle)
{
    Handle& handle = GetHandle();
    glGenTextures(1, &handle);
}

TextureObject::~TextureObject()
{
    Handle& handle = GetHandle();
    glDeleteTextures(1, &handle);
}

#ifndef NDEBUG
GLint TextureObject::GetActiveTexture()
{
    GLint activeTexture;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
    return activeTexture - GL_TEXTURE0;
}
#endif

void TextureObject::SetActiveTexture(GLint textureUnit)
{
    glActiveTexture(GL_TEXTURE0 + textureUnit);
}

void TextureObject::Bind(Target target) const
{
    Handle handle = GetHandle();
    glBindTexture(target, handle);
}

void TextureObject::Unbind(Target target)
{
    Handle handle = NullHandle;
    glBindTexture(target, handle);
}

void TextureObject::GenerateMipmap()
{
    assert(IsBound());
    glGenerateMipmap(GetTarget());
}

void TextureObject::GenerateMipChain(MipGenFilter filter)
{
    assert(IsBound());

    // Query the dimensions of the base level so we can compute how many mip levels we need
    GLint width = 1;
    GLint height = 1;
    GLint depth = 1;

    GetParameter(0, ParameterInt::Width, width);

    if (GetTarget() != TextureObject::Texture1D)
    {
        GetParameter(0, ParameterInt::Height, height);
    }

    // Only array and volume textures expose a depth we need to inspect
    switch (GetTarget())
    {
    case TextureObject::Texture3D:
    case TextureObject::Texture2DArray:
    case TextureObject::TextureCubemapArray:
        GetParameter(0, ParameterInt::Depth, depth);
        break;

    default:
        depth = 1;
        break;
    }

    width = std::max(width, 1);
    height = std::max(height, 1);
    depth = std::max(depth, 1);

    // Walk the mip chain dimensions down to 1x1x1 (or until all axes reach 1)
    GLint maxLevel = 0;

    GLint mipWidth = width;
    GLint mipHeight = height;
    GLint mipDepth = depth;

    while (mipWidth > 1 || mipHeight > 1 || mipDepth > 1)
    {
        mipWidth = std::max(mipWidth / 2, 1);
        mipHeight = std::max(mipHeight / 2, 1);
        mipDepth = std::max(mipDepth / 2, 1);

        ++maxLevel;
    }

    SetParameter(ParameterInt::BaseLevel, 0);
    SetParameter(ParameterInt::MaxLevel, maxLevel);

    SetParameter(ParameterFloat::MinLod, 0.0f);
    SetParameter(ParameterFloat::MaxLod, static_cast<GLfloat>(maxLevel));

    const GLenum glMinFilter = static_cast<GLenum>(filter);
    SetParameter(ParameterEnum::MinFilter, glMinFilter);

    // Finally ask OpenGL to populate the derived mip levels from level zero
    GenerateMipmap();
}

void TextureObject::GetParameter(ParameterFloat pname, GLfloat& param) const
{
    assert(IsBound());
    glGetTexParameterfv(GetTarget(), static_cast<GLenum>(pname), &param);
}

void TextureObject::GetParameter(GLint level, ParameterFloat pname, GLfloat& param) const
{
    assert(IsBound());
    glGetTexLevelParameterfv(GetTarget(), level, static_cast<GLenum>(pname), &param);
}

void TextureObject::SetParameter(ParameterFloat pname, GLfloat param)
{
    assert(IsBound());
    glTexParameterf(GetTarget(), static_cast<GLenum>(pname), param);
}

void TextureObject::GetParameter(ParameterInt pname, GLint& param) const
{
    assert(IsBound());
    glGetTexParameteriv(GetTarget(), static_cast<GLenum>(pname), &param);
}

void TextureObject::GetParameter(GLint level, ParameterInt pname, GLint& param) const
{
    assert(IsBound());
    glGetTexLevelParameteriv(GetTarget(), level, static_cast<GLenum>(pname), &param);
}

void TextureObject::SetParameter(ParameterInt pname, GLint param)
{
    assert(IsBound());
    glTexParameteri(GetTarget(), static_cast<GLenum>(pname), param);
}

void TextureObject::GetParameter(ParameterEnum pname, GLenum& param) const
{
    assert(IsBound());
    glGetTexParameterIuiv(GetTarget(), static_cast<GLenum>(pname), &param);
}

void TextureObject::SetParameter(ParameterEnum pname, GLenum param)
{
    assert(IsBound());
    glTexParameteri(GetTarget(), static_cast<GLenum>(pname), param);
}

void TextureObject::GetParameter(ParameterEnumVector pname, std::span<GLenum> params) const
{
    assert(IsBound());
    glGetTexParameterIuiv(GetTarget(), static_cast<GLenum>(pname), params.data());
}

void TextureObject::SetParameter(ParameterEnumVector pname, std::span<const GLenum> params)
{
    assert(IsBound());
    glTexParameterIuiv(GetTarget(), static_cast<GLenum>(pname), params.data());
}

void TextureObject::GetParameter(ParameterColor pname, std::span<GLfloat, 4> params) const
{
    assert(IsBound());
    glGetTexParameterfv(GetTarget(), static_cast<GLenum>(pname), params.data());
}

void TextureObject::GetParameter(GLint level, ParameterColor pname, std::span<GLfloat, 4> params) const
{
    assert(IsBound());
    glGetTexLevelParameterfv(GetTarget(), level, static_cast<GLenum>(pname), params.data());
}

void TextureObject::SetParameter(ParameterColor pname, std::span<const GLfloat, 4> params)
{
    assert(IsBound());
    glTexParameterfv(GetTarget(), static_cast<GLenum>(pname), params.data());
}

GLuint64 TextureObject::GetBindlessTextureHandle()
{
    Handle handle = GetHandle();
    return glGetTextureHandleARB(handle);
}

void TextureObject::GetTextureData(GLint level, Format format, Data::Type type, void* pixels) const
{
    assert(IsBound());
    glGetTexImage(GetTarget(), level, format, (GLenum)type, pixels);
}

void TextureObject::ClearTexture(GLint level, Format format, Data::Type type, const void* clearColor)
{
    assert(IsBound());
    Handle handle = GetHandle();
    glClearTexImage(handle, level, format, (GLenum)type, clearColor);
}

#ifndef NDEBUG
bool TextureObject::IsValidFormat(Format format, InternalFormat internalFormat)
{
    switch (internalFormat)
    {
    case InternalFormatR:
    case InternalFormatR8:
    case InternalFormatR16:
    case InternalFormatR8SNorm:
    case InternalFormatR16SNorm:
    case InternalFormatR16F:
    case InternalFormatR32F:
    case InternalFormatRCompressed:
        return format == FormatR;
    case InternalFormatRG:
    case InternalFormatRG8:
    case InternalFormatRG16:
    case InternalFormatRG8SNorm:
    case InternalFormatRG16SNorm:
    case InternalFormatRG16F:
    case InternalFormatRG32F:
    case InternalFormatRGCompressed:
        return format == FormatRG;
    case InternalFormatRGB:
    case InternalFormatRGB8:
    case InternalFormatRGB16:
    case InternalFormatRGB8SNorm:
    case InternalFormatRGB16SNorm:
    case InternalFormatRGB16F:
    case InternalFormatRGB32F:
    case InternalFormatSRGB8:
    case InternalFormatRGBCompressed:
    case InternalFormatSRGBCompressed:
    case InternalFormatR11G11B10:
        return format == FormatRGB || format == FormatBGR;
    case InternalFormatRGBA:
    case InternalFormatRGBA8:
    case InternalFormatRGBA16:
    case InternalFormatRGBA8SNorm:
    case InternalFormatRGBA16SNorm:
    case InternalFormatRGBA16F:
    case InternalFormatRGBA32F:
    case InternalFormatSRGBA8:
    case InternalFormatRGBACompressed:
    case InternalFormatSRGBACompressed:
    case InternalFormatRGB10A2:
        return format == FormatRGBA || format == FormatBGRA;
    case InternalFormatDepth:
    case InternalFormatDepth16:
    case InternalFormatDepth24:
    case InternalFormatDepth32:
    case InternalFormatDepth32F:
        return format == FormatDepth;
    case InternalFormatDepthStencil:
    case InternalFormatDepth24Stencil8:
    case InternalFormatDepth32FStencil8:
        return format == FormatDepthStencil;
    default:
        //Unknown format
        return false;
    }
}
#endif

int TextureObject::GetComponentCount(Format format)
{
    switch (format)
    {
    case FormatR:
    case FormatDepth:
        return 1;
    case FormatRG:
    case FormatDepthStencil:
        return 2;
    case FormatRGB:
    case FormatBGR:
        return 3;
    case FormatRGBA:
    case FormatBGRA:
        return 4;
    default:
        //Unknown format
        return 0;
    }
}

int TextureObject::GetDataComponentCount(InternalFormat internalFormat)
{
    switch (internalFormat)
    {
    case InternalFormatR:
    case InternalFormatR8:
    case InternalFormatR8SNorm:
    case InternalFormatR16:
    case InternalFormatR16SNorm:
    case InternalFormatR16F:
    case InternalFormatR32F:
    case InternalFormatRCompressed:
    case InternalFormatR11G11B10:
    case InternalFormatRGB10A2:
    case InternalFormatDepth:
    case InternalFormatDepth16:
    case InternalFormatDepth24:
    case InternalFormatDepth32:
    case InternalFormatDepth32F:
    case InternalFormatDepthStencil:
    case InternalFormatDepth24Stencil8:
        return 1;
    case InternalFormatRG:
    case InternalFormatRG8:
    case InternalFormatRG8SNorm:
    case InternalFormatRG16:
    case InternalFormatRG16SNorm:
    case InternalFormatRG16F:
    case InternalFormatRG32F:
    case InternalFormatRGCompressed:
        return 2;
    case InternalFormatRGB:
    case InternalFormatRGB8:
    case InternalFormatRGB8SNorm:
    case InternalFormatRGB16:
    case InternalFormatRGB16SNorm:
    case InternalFormatRGB16F:
    case InternalFormatRGB32F:
    case InternalFormatSRGB8:
    case InternalFormatRGBCompressed:
    case InternalFormatSRGBCompressed:
        return 3;
    case InternalFormatRGBA:
    case InternalFormatRGBA8:
    case InternalFormatRGBA8SNorm:
    case InternalFormatRGBA16:
    case InternalFormatRGBA16SNorm:
    case InternalFormatRGBA16F:
    case InternalFormatRGBA32F:
    case InternalFormatSRGBA8:
    case InternalFormatRGBACompressed:
    case InternalFormatSRGBACompressed:
        return 4;
    default:
        //Unknown format
        return 0;
    }
}
