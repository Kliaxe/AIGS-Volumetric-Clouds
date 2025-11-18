#pragma once

#include "AssetLoader.h"

#include "Geometry/Model.h"
#include "Geometry/Mesh.h"
#include "Texture2DLoader.h"
#include <vector>
#include <set>

struct aiMesh;
struct aiMaterial;
class VertexFormat;

// Asset loader for Models. Contains a pointer to a reference material for loaded submeshes
class ModelLoader : public AssetLoader<Model>
{
public:
    // Enum to read material properties from the file
    enum class MaterialProperty;

public:
    ModelLoader(std::shared_ptr<Material> referenceMaterial = nullptr);

    std::shared_ptr<Material> GetReferenceMaterial() const;
    void SetReferenceMaterial(std::shared_ptr<Material> referenceMaterial);

    bool GetCreateMaterials() const;
    void SetCreateMaterials(bool createMaterials);

    Texture2DLoader& GetTexture2DLoader();
    const Texture2DLoader& GetTexture2DLoader() const;

    // Load the model from the path
    Model Load(const char* path) override;

    // Maps a semantic to an attribute in the shader program used by the material
    bool SetMaterialAttribute(VertexAttribute::Semantic semantic, const char* attributeName);

    // Tell the loader only to apply these specific semantics
    void SetSemanticAttribute(VertexAttribute::Semantic semantic);

    // Maps a material property to a uniform in the shader program used by the material
    bool SetMaterialProperty(MaterialProperty materialProperty, const char* uniformName);

private:
    // Generate a submesh from the loaded mesh data
    void GenerateSubmesh(Mesh& mesh, const aiMesh& meshData);

    // Generate a material from the loaded material data
    std::shared_ptr<Material> GenerateMaterial(const aiMaterial& materialData);

    // Load a texture of the specific type in the location
    void LoadTexture(const aiMaterial& materialData, int textureType, Material& material, ShaderProgram::Location location,
        int materialTextureSlot, TextureObject::Format format, TextureObject::InternalFormat internalFormat) const;

    // Load a texture to a specific material texture slot
    void LoadMaterialTexture(const aiMaterial& materialData, int textureType, int textureTypeIndex, Material& material,
        int materialTextureSlot, TextureObject::Format format, TextureObject::InternalFormat internalFormat) const;

    // Build the vertex data from the mesh data
    static std::vector<GLubyte> CollectVertexData(const aiMesh& meshData, VertexFormat& vertexFormat, bool interleaved, std::set<VertexAttribute::Semantic> allowedSemantics);

    // Build the element data from the mesh data
    static std::vector<GLubyte> CollectElementData(const aiMesh& meshData, Data::Type& elementType,
        std::vector<Drawcall::Primitive>& primitives, std::vector<int>& elementCounts);

    // Get the correct vertex data pointer for a specific semantic
    static const void* GetVertexDataPointer(const aiMesh& meshData, VertexAttribute::Semantic semantic, int& stride);

    // Copy one buffer to another preserving the stride
    static void CopyBuffer(void* dstBuffer, size_t dstStride, const void* srcBuffer, size_t srcStride, size_t count, size_t size);

    // Get the type of primitive depending on the number of elements
    static Drawcall::Primitive GetPrimitiveType(int elementCount);

private:
    // Path to the base folder where we are loading the current model
    std::string m_baseFolder;

    // Pointer to the reference material
    std::shared_ptr<Material> m_referenceMaterial;

    // Maps the semantic to attributes in the reference material
    Mesh::SemanticMap m_materialAttributeMap;

    // Specify which semantics that are allowed for vertex parsing
    std::set<VertexAttribute::Semantic> m_allowedSemantics;

    // Maps material properties to uniforms in the reference material
    std::unordered_map <MaterialProperty, ShaderProgram::Location> m_materialPropertyMap;

    // Should create new materials for each submesh or use the reference material
    bool m_createMaterials;

    // Texture loader to cache already loaded shared textures
    mutable Texture2DLoader m_textureLoader;
};

enum class ModelLoader::MaterialProperty
{
    AmbientColor,
    DiffuseColor,
    SpecularColor,
    SpecularExponent,
    DiffuseTexture,
    NormalTexture,
    SpecularTexture,
};
