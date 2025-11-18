#include "ShaderLoader.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cassert>

#include <iostream>

ShaderLoader::ShaderLoader(Shader::Type type) : m_type(type)
{
}

bool ShaderLoader::IsValid(std::span<const char*> paths)
{
    bool valid = true;
    for (auto& path : paths)
    {
        valid &= IsValid(path);
    }
    return valid;
}

Shader ShaderLoader::Load(const char* path)
{
    Shader shader(m_type);
    std::ifstream file(path);
    assert(file.is_open());
    std::stringstream stringStream;
    stringStream << file.rdbuf();

    std::string shaderSource = stringStream.str();

    if (!shaderSource.empty() && shaderSource.back() != '\n')
    {
        shaderSource.push_back('\n');
    }

    shader.SetSource(shaderSource.c_str());
    Compile(shader);
    return shader;
}

Shader ShaderLoader::Load(std::span<const char*> paths)
{
    Shader shader(m_type);
    std::vector<std::string> sourceCodeStrings(paths.size());
    std::vector<const char*> sourceCode(paths.size());
    for (int i = 0; i < paths.size(); ++i)
    {
        std::ifstream file(paths[i]);
        assert(file.is_open());

        std::stringstream stringStream;
        stringStream << file.rdbuf();

        sourceCodeStrings[i] = stringStream.str();

        if (!sourceCodeStrings[i].empty() && sourceCodeStrings[i].back() != '\n')
        {
            sourceCodeStrings[i].push_back('\n');
        }

        sourceCode[i] = sourceCodeStrings[i].c_str();
    }
    shader.SetSource(sourceCode);
    Compile(shader);
    return shader;
}

Shader* ShaderLoader::LoadNew(std::span<const char*> paths)
{
    Shader* shader = nullptr;
    if (IsValid(paths))
    {
        shader = new Shader(Load(paths));
    }
    return shader;
}

bool ShaderLoader::LoadInto(Shader& shader, std::span<const char*> paths)
{
    bool valid = IsValid(paths);
    if (valid)
    {
        shader = Load(paths);
    }
    return valid;
}

void ShaderLoader::Compile(Shader& shader)
{
    if (!shader.Compile())
    {
        std::array<char, 512> infoLog;
        shader.GetCompilationErrors(infoLog);

        const char* typeName = "UNKNOWN";
        switch (shader.GetType())
        {
        case Shader::ComputeShader:
            typeName = "COMPUTE";
            break;
        case Shader::VertexShader:
            typeName = "VERTEX";
            break;
        case Shader::TesselationControlShader:
            typeName = "TCS";
            break;
        case Shader::TesselationEvaluationShader:
            typeName = "TES";
            break;
        case Shader::GeometryShader:
            typeName = "GEOMETRY";
            break;
        case Shader::FragmentShader:
            typeName = "FRAGMENT";
            break;
        }
        std::cout << "ERROR::SHADER::" << typeName << "::COMPILATION_FAILED\n" << infoLog.data() << std::endl;
    }
}

Shader ShaderLoader::Load(Shader::Type type, const char* path)
{
    ShaderLoader shaderLoader(type);
    return shaderLoader.Load(path);
}
