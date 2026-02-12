from array import array

import cffi
import numpy as np
from ai4animation.AI4Animation import AI4Animation
from ai4animation.Math import Tensor
from pyray import load_model_from_mesh, Matrix, Mesh
from raylib import (
    MATERIAL_MAP_DIFFUSE,
    MatrixIdentity,
    MemAlloc,
    RAYWHITE,
    UploadMesh,
    WHITE,
)

ffi = cffi.FFI()


class SkinnedMesh:
    def __init__(self, actor, glb):
        self.Actor = actor

        self.SkinnedMeshes = [mesh for mesh in glb.Meshes if mesh.HasSkinning]

        self.BindMatrices = np.transpose(
            Tensor.Create(glb.Skin.Inverse_bind_matrices), axes=(0, 2, 1)
        )

        self.Models = []
        self.BoneMatrixViews = []
        self.Color = RAYWHITE

        print(
            f"Loading {len(self.SkinnedMeshes)} skinned meshes (skipping {len(glb.Meshes) - len(self.SkinnedMeshes)} non-skinned meshes)"
        )

        boneCount = len(glb.JointNames)
        self.BoneCount = boneCount

        MAX_BONES_SUPPORTED = 254
        if boneCount > MAX_BONES_SUPPORTED:
            raise ValueError(
                f"Character has {boneCount} bones, but shader only supports {MAX_BONES_SUPPORTED}. "
                f"Increase MAX_BONE_NUM in skinnedShadow.vs and skinnedBasic.vs"
            )

        for mesh in self.SkinnedMeshes:
            vertexCount = len(mesh.Vertices)

            # Create Raylib mesh for this mesh
            vertices = array("f", mesh.Vertices.flatten())
            normals = array("f", mesh.Normals.flatten())
            triangles = array("H", mesh.Triangles.flatten().astype(np.uint16))
            texcoords = array("f", [0.5, 0.5] * vertexCount)

            # 4 bones per vertex
            boneIds = np.zeros((vertexCount, 4), dtype=np.uint8)
            currentSkinBones = min(mesh.SkinIndices.shape[1], 4)
            boneIds[:, :currentSkinBones] = mesh.SkinIndices[
                :, :currentSkinBones
            ].astype(np.uint8)
            bone_ids = array("B", boneIds.flatten())

            # Bone weights
            boneWeights = np.zeros((vertexCount, 4), dtype=np.float32)
            boneWeights[:, :currentSkinBones] = mesh.SkinWeights[:, :currentSkinBones]
            bone_weights = array("f", boneWeights.flatten())

            raylib_mesh = Mesh()
            raylib_mesh.vertexCount = vertexCount
            raylib_mesh.triangleCount = int(len(triangles) / 3)
            raylib_mesh.vertices = ffi.cast("float*", vertices.buffer_info()[0])
            raylib_mesh.texcoords = ffi.cast("float*", texcoords.buffer_info()[0])
            raylib_mesh.normals = ffi.cast("float*", normals.buffer_info()[0])
            raylib_mesh.indices = ffi.cast(
                "unsigned short*", triangles.buffer_info()[0]
            )
            raylib_mesh.boneIds = ffi.cast("unsigned char*", bone_ids.buffer_info()[0])
            raylib_mesh.boneWeights = ffi.cast("float*", bone_weights.buffer_info()[0])
            raylib_mesh.boneCount = boneCount
            raylib_mesh.vaoId = 0

            # Allocate bone matrices
            raylib_mesh.boneMatrices = MemAlloc(boneCount * ffi.sizeof(Matrix()))
            for i in range(boneCount):
                raylib_mesh.boneMatrices[i] = MatrixIdentity()

            # Upload mesh with dynamic flag for bone updates
            UploadMesh(ffi.addressof(raylib_mesh), True)

            # Create Model for this mesh
            model = load_model_from_mesh(raylib_mesh)
            model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = WHITE

            self.Models.append(model)

            # Cache numpy view of bone matrices for efficient updates
            mesh = model.meshes[0]
            matView = np.frombuffer(
                ffi.buffer(mesh.boneMatrices, mesh.boneCount * ffi.sizeof(Matrix())),
                dtype=np.float32,
            ).reshape(mesh.boneCount, 4, 4)
            self.BoneMatrixViews.append(matView)

        print(
            f"Initialized {len(self.Models)} skinned submeshes with {boneCount} bones"
        )

        AI4Animation.Standalone.RenderPipeline.RegisterModel(
            name=self.Actor.Entity.Name,
            model=self.Models,
            skinned_mesh=self,
            color=self.Color,
        )

    def SetColor(self, color):
        self.Color = color
        self.Unregister()
        self.Register()

    def Register(self):
        if not AI4Animation.Standalone.RenderPipeline.HasModel(self.Models):
            AI4Animation.Standalone.RenderPipeline.RegisterModel(
                name=self.Actor.Entity.Name,
                model=self.Models,
                skinned_mesh=self,
                color=self.Color,
            )

    def Unregister(self):
        if AI4Animation.Standalone.RenderPipeline.HasModel(self.Models):
            AI4Animation.Standalone.RenderPipeline.UnregisterModel(self.Models)

    def Update(self):
        # GPU skinning - compute and update bone matrices
        if not self.Models:
            return

        # Update bone matrices for all meshes (GPU will use these in shaders)
        transforms = np.matmul(
            AI4Animation.Scene.GetSkinningTransforms(self.Actor.Entities),
            self.BindMatrices,
        )
        for matView in self.BoneMatrixViews:
            matView[:] = transforms
